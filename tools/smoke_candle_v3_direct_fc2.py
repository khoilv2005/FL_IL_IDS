"""Bounded structural/control smoke, not an accuracy benchmark or TEST evaluation."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
RUNTIME = REPO/'candle_v3_runtime'
sys.path[:] = [str(RUNTIME)]+[p for p in sys.path if Path(p or '.').resolve() not in (REPO,REPO/'tools')]

import numpy as np
import torch
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector
from fed_learning.training.checkpoint_state import snapshot_denice_state, restore_denice_state
from fed_learning.training.denice_provenance import source_identity
from fed_learning.strategies.fed_incremental.nice import update_freeze_masks
from fed_learning.strategies.decentralized.candle_v3.contracts import TeachingConfig
from fed_learning.strategies.decentralized.candle_v3.knowledge import announce, record_local_update, state
from fed_learning.strategies.decentralized.candle_v3.snapshot import freeze_receiver, reference_payload, digest, canonical_features
from fed_learning.strategies.decentralized.candle_v3.direct_fc2 import export_row, decode_row, aggregate_rows, RowCompatibilityError
from fed_learning.strategies.decentralized.candle_v3.transaction import stage_head, commit
from fed_learning.strategies.decentralized.candle_v3.communication import WireLedger, CommunicationBudgetExceeded
from fed_learning.strategies.decentralized.candle_v3.data import CurrentTaskRoles
from fed_learning.strategies.decentralized.candle_v3.service import run_task_teaching
from fed_learning.strategies.decentralized.candle_v3.verification import accounting, acceptance, predict

PREPROCESS = 'a'*64


def fixture():
    torch.manual_seed(42)
    model = DeNICEModel((39,1),34).eval()
    for ranks in model.unit_ranks.values():
        ranks[:] = 2
    model.unit_ranks['fc2'][:] = 0
    model.unit_ranks['fc2'][[6,7]] = 2
    update_freeze_masks(model)
    detector = ContextDetector(router_mode='binary_cosine')
    detector.retain_reference_inputs = False
    detector.episode_classes = {0:[0,1],1:[6,7]}
    detector.binarize_thresholds = {name:-.1 for name in ('conv1','conv2','conv3','gru')}
    with torch.no_grad():
        width = sum(v.shape[1] for v in model.get_context_activations_per_sample(torch.zeros(1,39,1)).values())
    detector.activation_memory = {0:np.zeros((2,width),np.float32),1:np.ones((2,width),np.float32)}
    detector.mark_router_fresh(task_id=1)
    return model,detector


def donor_fixture(receiver):
    donor = deepcopy(receiver)
    donor.unit_ranks['fc2'][8] = 2
    with torch.no_grad():
        donor.fc2.weight[8].copy_(torch.linspace(-1,1,256))
        donor.fc2.bias[8] = .25
    record_local_update(donor,1,1,torch.tensor([7,8]*64),1,successful_classes={7,8})
    return donor


def packet_for(model, owner=1):
    ann = announce(model,fixture()[1],owner,1,torch.tensor([7,8]*64),PREPROCESS)
    return export_row(model,donor=owner,task=1,class_id=8,announcement=ann)


def receipt():
    return dict(status='PASS',targets=[8],donors=[1],session_id='controlled_structural_receipt',
                scope='synthetic smoke only',totals=dict(positive=64,negative=64))


class DirectSmoke(unittest.TestCase):
    controlled_service_receipt = None
    def setUp(self):
        self.model,self.detector = fixture()
        self.donor = donor_fixture(self.model)
        self.reference = freeze_receiver(self.model,self.detector,0,1,PREPROCESS,'cpu')
        self.packet = packet_for(self.donor)

    def imported(self):
        return aggregate_rows(self.reference,[8],[self.packet],quality={(1,8):1.})

    def test_private_runtime_import(self):
        import fed_learning
        self.assertTrue(Path(fed_learning.__file__).resolve().is_relative_to(RUNTIME))

    def test_canonical_features_equal_inference(self):
        self.model.set_active_context(1)
        self.model.eval()
        x = torch.randn(4,39,1)
        with torch.no_grad():
            z = canonical_features(self.model,x)
            torch.testing.assert_close(self.model(x),self.model._apply_masked_linear(z,self.model.fc2,'fc2'))

    def test_training_mode_features_rejected(self):
        self.model.train()
        with self.assertRaises(ValueError):
            canonical_features(self.model,torch.randn(2,39,1))

    def test_snapshot_does_not_mutate_live(self):
        before = digest(reference_payload(self.model,self.detector))
        freeze_receiver(self.model,self.detector,0,1,PREPROCESS,'cpu')
        self.assertEqual(before,digest(reference_payload(self.model,self.detector)))

    def test_codec_exact_1028_bytes(self):
        values,mask = decode_row(self.packet)
        self.assertEqual(len(self.packet['row_fp32']),1028)
        self.assertEqual(len(self.packet['weight_mask']),32)
        self.assertTrue(torch.equal(values[:-1],self.donor.fc2.weight[8]))
        self.assertEqual(float(values[-1]),float(self.donor.fc2.bias[8].detach()))
        self.assertTrue(torch.equal(mask,self.donor.weight_masks['fc2'][8].bool()))

    def test_packet_corruption_rejected(self):
        bad = deepcopy(self.packet); bad['row_fp32'] = b'!'+bad['row_fp32'][1:]
        with self.assertRaises(ValueError): decode_row(bad)

    def test_missing_local_provenance_rejected(self):
        delattr(self.donor,'candle_v3_state')
        with self.assertRaises(ValueError): packet_for(self.donor)

    def test_nonmature_donor_rejected(self):
        self.donor.unit_ranks['fc2'][8] = 1
        with self.assertRaises(ValueError): packet_for(self.donor)

    def test_exact_copy_without_receiver_average(self):
        imported = self.imported()
        self.assertTrue(torch.equal(imported['head'][0,:-1],self.donor.fc2.weight[8]))
        self.assertEqual(imported['receiver_row_weight'],0.)
        self.assertEqual(imported['optimizer_steps'],0)

    def test_staging_preserves_every_other_tensor(self):
        candidate = stage_head(self.reference,self.imported())
        for key,value in self.model.state_dict().items():
            if key not in ('fc2.weight','fc2.bias'):
                self.assertTrue(torch.equal(value,candidate.state_dict()[key]),key)
        self.assertTrue(torch.equal(self.model.fc2.weight[:8],candidate.fc2.weight[:8]))
        self.assertTrue(torch.equal(self.model.fc2.weight[9:],candidate.fc2.weight[9:]))
        self.assertEqual(int(self.model.unit_ranks['fc2'][8]),0)

    def test_masked_bias_copied_exactly(self):
        self.donor.bias_masks['fc2'][8] = 0
        packet = packet_for(self.donor)
        imported = aggregate_rows(self.reference,[8],[packet],quality={(1,8):1.})
        candidate = stage_head(self.reference,imported)
        self.assertEqual(float(candidate.fc2.bias[8]),float(self.donor.fc2.bias[8].detach()))
        self.assertEqual(float(candidate.bias_masks['fc2'][8]),0.)

    def test_unprotected_feature_rejected(self):
        self.reference.model.unit_ranks['fc1'][0] = 0
        with self.assertRaises(RowCompatibilityError): self.imported()

    def test_mature_receiver_row_rejected(self):
        self.reference.model.unit_ranks['fc2'][8] = 2
        with self.assertRaises(ValueError): self.imported()

    def test_multidonor_different_feature_frames_rejected(self):
        other = deepcopy(self.donor)
        with torch.no_grad(): other.conv1.weight.add_(.1)
        record_local_update(other,2,1,torch.tensor([7,8]*64),1,successful_classes={7,8})
        p = packet_for(other,2)
        with self.assertRaises(RowCompatibilityError):
            aggregate_rows(self.reference,[8],[self.packet,p],quality={(1,8):.6,(2,8):.4})

    def test_multidonor_normalization_and_zero_contributor(self):
        other = deepcopy(self.donor)
        record_local_update(other,2,1,torch.tensor([7,8]*64),1,successful_classes={7,8})
        with torch.no_grad(): other.fc2.weight[8].add_(2.)
        self.donor.weight_masks['fc2'][8,4] = 0
        other.weight_masks['fc2'][8,4] = 0
        p1,p2 = packet_for(self.donor),packet_for(other,2)
        imported = aggregate_rows(self.reference,[8],[p1,p2],quality={(1,8):.6,(2,8):.4})
        torch.testing.assert_close(imported['head'][0,0],.6*self.donor.fc2.weight[8,0]+.4*other.fc2.weight[8,0])
        self.assertTrue(torch.equal(imported['head'][0,4],self.model.fc2.weight[8,4]))

    def test_commit_and_checkpoint_restore(self):
        candidate = stage_head(self.reference,self.imported())
        router = deepcopy(self.detector); router.episode_classes[1].append(8)
        commit(self.model,self.detector,self.reference,candidate,router,owner=0,receipt=receipt())
        self.assertIn('8',self.model.candle_v3_state['peer_acquired'])
        self.assertNotIn('8',self.model.candle_v3_state['local_seen'].get('1',{}))
        buffer = io.BytesIO()
        torch.save(dict(model=self.model.state_dict(),algorithm=snapshot_denice_state(self.model,self.detector)),buffer)
        buffer.seek(0); saved = torch.load(buffer,weights_only=False)
        restored,rd = fixture()
        restore_denice_state(restored,rd,saved['algorithm']); restored.load_state_dict(saved['model']); restored.eval()
        self.assertEqual(digest(reference_payload(restored,rd)),digest(reference_payload(self.model,self.detector)))
        x = torch.randn(6,39,1)
        a = predict(self.model,self.detector,x,seen_classes=list(range(12)),batch_size=3)
        b = predict(restored,rd,x,seen_classes=list(range(12)),batch_size=3)
        self.assertTrue(all(np.array_equal(v,w) for v,w in zip(a,b)))

    def test_stale_reference_rejected_without_commit(self):
        candidate = stage_head(self.reference,self.imported())
        with torch.no_grad(): self.model.fc1.bias[0].add_(.01)
        before = digest(reference_payload(self.model,self.detector))
        with self.assertRaises(ValueError):
            commit(self.model,self.detector,self.reference,candidate,self.detector,owner=0,receipt=receipt())
        self.assertEqual(before,digest(reference_payload(self.model,self.detector)))

    def test_imported_row_survives_local_optimizer(self):
        candidate = stage_head(self.reference,self.imported())
        commit(self.model,self.detector,self.reference,candidate,self.detector,owner=0,receipt=receipt())
        self.model.unit_ranks['fc2'][9] = 1
        update_freeze_masks(self.model)
        before = self.model.fc2.weight[8].detach().clone()
        self.model.eval()
        optimizer = torch.optim.SGD(self.model.parameters(),lr=.01)
        optimizer.zero_grad()
        loss = torch.nn.functional.cross_entropy(self.model(torch.randn(4,39,1)),torch.full((4,),9))
        loss.backward()
        self.assertGreater(float(self.model.fc2.weight.grad[8].abs().sum()),0.)
        self.model.reset_frozen_gradients()
        optimizer.step()
        self.assertTrue(torch.equal(before,self.model.fc2.weight[8]))

    def test_imported_row_survives_native_aggregation(self):
        from collections import OrderedDict
        from fed_learning.strategies.decentralized.denice_aggregation import age_aware_aggregate, AggregationConfig
        candidate = stage_head(self.reference,self.imported())
        ages = candidate.get_neuron_ages_state(); ages['fc2'][9] = 1
        params = OrderedDict((key,value.detach().clone()) for key,value in candidate.state_dict().items()
                             if key in ('fc2.weight','fc2.bias'))
        delta = OrderedDict((key,torch.ones_like(value)*10) for key,value in params.items())
        result = age_aware_aggregate(params,ages,[delta],np.array([1.]),AggregationConfig(),
                                    neighbor_ages=[ages],neighbor_labels=[[8,9]],target_labels=[6,7,8,9])
        self.assertTrue(torch.equal(result['fc2.weight'][8],params['fc2.weight'][8]))
        self.assertTrue(torch.equal(result['fc2.bias'][8],params['fc2.bias'][8]))
        self.assertFalse(torch.equal(result['fc2.weight'][9],params['fc2.weight'][9]))

    def test_fail_receipt_does_not_install(self):
        before = digest(reference_payload(self.model,self.detector))
        candidate = stage_head(self.reference,self.imported())
        with self.assertRaises(ValueError):
            commit(self.model,self.detector,self.reference,candidate,self.detector,owner=0,
                   receipt=dict(receipt(),status='FAIL'))
        self.assertEqual(before,digest(reference_payload(self.model,self.detector)))

    def test_exception_after_write_rolls_back(self):
        before = digest(reference_payload(self.model,self.detector))
        candidate = stage_head(self.reference,self.imported())
        with patch('fed_learning.strategies.decentralized.candle_v3.transaction.state',side_effect=RuntimeError('injected')):
            with self.assertRaises(RuntimeError):
                commit(self.model,self.detector,self.reference,candidate,self.detector,owner=0,receipt=receipt())
        self.assertEqual(before,digest(reference_payload(self.model,self.detector)))

    def test_successful_step_ownership_only(self):
        record_local_update(self.model,0,1,torch.tensor([6,7]*4),0,successful_classes=set())
        self.assertFalse(state(self.model,0)['local_seen'])
        record_local_update(self.model,0,1,torch.tensor([6,7]*4),1,successful_classes={6})
        self.assertEqual(set(state(self.model,0)['local_seen']['1']),{'6'})

    def test_wire_budget_and_restore(self):
        ledger = WireLedger(100000)
        ledger.transmit(1,0,'fc2_row',self.packet)
        self.assertGreater(ledger.bytes_sent,1028)
        restored = WireLedger.from_state(ledger.state_dict())
        self.assertEqual(restored.summary(),ledger.summary())
        small = WireLedger(1)
        with self.assertRaises(CommunicationBudgetExceeded): small.transmit(1,0,'fc2_row',self.packet)
        self.assertEqual(small.bytes_sent,0)

    def test_current_only_role_guard(self):
        data = CurrentTaskRoles(None,1,[6,7,8],{0:(torch.randn(4,39,1),torch.tensor([0,6,7,8]))},42)
        _,y = data.get(0,'base',task=1,per_class=32)
        self.assertEqual(set(y.tolist()),{6,7,8})
        with self.assertRaises(ValueError): data.get(0,'base',task=0,per_class=32)
        with self.assertRaises(ValueError): data.get(0,'validation',task=1,per_class=32)
        data.close()
        with self.assertRaises(ValueError): data.get(0,'base',task=1,per_class=32)

    def test_acceptance_and_accounting(self):
        y = np.array([6]*64+[8]*64); baseline = np.array([6]*128)
        candidate = y.copy(); routes = np.ones(128,int)
        rows = {0:accounting(y[:64],baseline[:64],candidate[:64],routes[:64],targets=[8],task=1),
                1:accounting(y[64:],baseline[64:],candidate[64:],routes[64:],targets=[8],task=1)}
        controls = {name:{0:accounting(y[:64],baseline[:64],baseline[:64],routes[:64],targets=[8],task=1),
                          1:accounting(y[64:],baseline[64:],baseline[64:],routes[64:],targets=[8],task=1)}
                    for name in ('availability_only','router_only','mask_only')}
        result = acceptance(rows,controls,config=TeachingConfig(),receiver=0,targets=[8])
        self.assertEqual(result['status'],'PASS')
        self.assertEqual(result['totals']['rescue'],64)
        controls['mask_only'] = rows
        self.assertEqual(acceptance(rows,controls,config=TeachingConfig(),receiver=0,targets=[8])['status'],'FAIL')

    def test_negative_break_rejects(self):
        y = np.array([6]*64+[8]*64); baseline = np.array([6]*128)
        candidate = y.copy(); candidate[0] = 8
        row = accounting(y,baseline,candidate,np.ones(128,int),targets=[8],task=1)
        control = accounting(y,baseline,baseline,np.ones(128,int),targets=[8],task=1)
        result = acceptance({0:row},{name:{0:control} for name in ('availability_only','router_only','mask_only')},
                            config=TeachingConfig(),receiver=0,targets=[8])
        self.assertEqual(result['status'],'FAIL')
        self.assertIn('Negative break detected',result['reasons'])

    def test_controlled_service_real_forward_commit_and_no_replay(self):
        import tempfile
        rng = np.random.default_rng(123)
        samples = {}
        for owner,classes in ((0,[6]),(1,[7,8])):
            for role in ('base','fit','calibration'):
                x,y = [],[]
                for c in classes:
                    x.append(torch.from_numpy((rng.normal(0,1e-4,(64,39,1))+(1. if c==8 else -1.)).astype(np.float32)))
                    y.append(torch.full((64,),c,dtype=torch.long))
                samples[(owner,role)] = (torch.cat(x),torch.cat(y))
        with torch.no_grad():
            self.model.fc2.weight.zero_(); self.model.fc2.bias.fill_(-100.)
            self.model.fc2.bias[6] = 0.
            negative = canonical_features(self.model,samples[(0,'base')][0]).mean(0)
            xp,yp = samples[(1,'base')]
            positive = canonical_features(self.model,xp[yp==8]).mean(0)
            delta = positive-negative
            weight = 20*delta/delta.square().sum()
            self.donor = deepcopy(self.model)
            self.donor.fc2.weight[8].copy_(weight)
            self.donor.fc2.bias[8] = -weight.dot((positive+negative)/2)
        self.donor.unit_ranks['fc2'][8] = 2
        record_local_update(self.model,0,1,samples[(0,'base')][1],1,successful_classes={6})
        record_local_update(self.donor,1,1,samples[(1,'base')][1],1,successful_classes={7,8})
        detector1 = deepcopy(self.detector); detector1.episode_classes[1] = [7,8]
        class Panel:
            def __init__(self):
                self.base = {owner:samples[(owner,'base')] for owner in (0,1)}
                self.access = []
            def get(self,owner,role,*,task,per_class):
                if task != 1 or role not in ('base','fit','calibration'):
                    raise AssertionError('Historical or evaluation data access')
                self.access.append(dict(owner=owner,role=role,task=task))
                return samples[(owner,role)]
        data = Panel()
        cfg = dict(candle_v3_teaching=dict(enabled=True,transfer_mode='direct_fc2',max_donors=1,max_targets=1))
        with tempfile.TemporaryDirectory(prefix='candle-v3-controlled-') as directory:
            result = run_task_teaching(config=cfg,task=1,models={0:self.model,1:self.donor},
                detectors={0:self.detector,1:detector1},active_ids=[0,1],data=data,
                seen_classes=list(range(12)),preprocessing_sha256=PREPROCESS,output_dir=directory,
                device='cpu',allowed_edges={0:[1],1:[0]},native_neighbors={0:[0,1],1:[0,1]})
            successes = [r for r in result['receipts'] if r['status']=='PASS']
            self.assertEqual(len(successes),1,result['receipts'])
            self.assertEqual(successes[0]['optimizer_steps'],0)
            self.assertEqual(successes[0]['totals']['negative_break'],0)
            self.assertIn('8',self.model.candle_v3_state['peer_acquired'])
            self.assertTrue(all(row['task']==1 for row in data.access))
            self.assertGreater(result['wire']['by_kind']['receiver_reference'],1028)
            self.assertGreater(result['wire']['by_kind']['fc2_row'],1028)
            previous = deepcopy(self.model.candle_v3_state)
            repeated = run_task_teaching(config=cfg,task=1,models={0:self.model,1:self.donor},
                detectors={0:self.detector,1:detector1},active_ids=[0,1],data=data,
                seen_classes=list(range(12)),preprocessing_sha256=PREPROCESS,output_dir=directory,
                device='cpu',allowed_edges={0:[1],1:[0]},native_neighbors={0:[0,1],1:[0,1]})
            self.assertEqual(repeated['receipts'],[])
            self.assertEqual(repeated['wire']['teaching_bytes'],result['wire']['teaching_bytes'])
            self.assertEqual(previous,self.model.candle_v3_state)
            type(self).controlled_service_receipt = dict(receipt=successes[0],wire=result['wire'],
                fixture='engineered synthetic BASE head; actual CNN-GRU forward, no inference/acceptance mocks',
                scientific_transfer_evidence=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2)
    started = time.monotonic()
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(DirectSmoke)
    with (args.out/'tests.log').open('w',encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    payload = dict(schema='candle_v3_direct_structural_smoke_v1',completed=result.wasSuccessful(),
        tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=len(result.skipped),
        elapsed_seconds=time.monotonic()-started,synthetic=True,accuracy_benchmark=False,
        test_dataset_accessed=False,torch=str(torch.__version__),device='cpu',
        runtime=str(RUNTIME),source_sha256=source_identity(RUNTIME)['source_sha256'],
        failures_detail=[dict(test=str(test),traceback=trace) for test,trace in result.failures+result.errors])
    payload['controlled_service'] = DirectSmoke.controlled_service_receipt
    (args.out/'receipt.json').write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in payload.items() if k not in ('failures_detail','controlled_service')},indent=2))
    if not result.wasSuccessful():
        print((args.out/'tests.log').read_text(encoding='utf-8'))
        raise SystemExit(1)


if __name__ == '__main__':
    main()
