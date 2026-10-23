"""Receiver-encoder head smoke: synthetic mechanism checks, never an accuracy benchmark."""
import argparse
from copy import deepcopy
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import tempfile
import time
import unittest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location('direct_smoke_helpers', REPO/'tools/smoke_candle_v3_direct_fc2.py')
shared = importlib.util.module_from_spec(spec)
spec.loader.exec_module(shared)

import numpy as np
import torch
import torch.nn.functional as F
from fed_learning.strategies.decentralized.candle_v3.head_learning import (
    learn_head, head_permissions, initial_head, participant_delta)
from fed_learning.strategies.decentralized.candle_v3.head_aggregation import aggregate_head
from fed_learning.strategies.decentralized.candle_v3.contracts import TeachingConfig
from fed_learning.strategies.decentralized.candle_v3.snapshot import digest, reference_payload, encode_panel, freeze_receiver
from fed_learning.strategies.decentralized.candle_v3.communication import WireLedger
from fed_learning.strategies.decentralized.candle_v3.service import run_task_teaching
from fed_learning.strategies.decentralized.candle_v3.knowledge import record_local_update
from fed_learning.strategies.decentralized.candle_v3.discovery import TeacherSelection
from fed_learning.strategies.decentralized.candle_v3.service import _session


class ReceiverHeadSmoke(shared.DirectSmoke):
    controlled_service_receipt = None

    def setUp(self):
        super().setUp()
        with torch.no_grad():
            self.model.fc2.weight[8].zero_()
            self.model.fc2.bias[8] = -.001
        self.reference = freeze_receiver(self.model, self.detector, 0, 1, shared.PREPROCESS, 'cpu')
        self.config = TeachingConfig(enabled=True,discovery_policy='control_fit_preflight_v1')
        self.panels = []
        # Deliberately separable feature fixture for gradient algebra only.
        for owner, label, coordinate, count in ((0,6,1,48),(1,8,0,64)):
            z = torch.zeros(count,256); z[:,coordinate] = 4.
            logits = torch.full((count,34),-100.); logits[:,6] = 0.
            self.panels.append(dict(owner=owner, identity=asdict(self.reference.identity),
                features=z, logits=logits, labels=torch.full((count,),label)))

    def imported(self):
        return learn_head(self.reference,self.panels,[8],config=self.config,
                          ledger=WireLedger(128*1024**2),include_receiver=True)

    def test_c_eight_rounds_freeze_and_wire(self):
        before = digest(reference_payload(self.reference.model,self.reference.detector))
        ledger = WireLedger(128*1024**2)
        fitted = learn_head(self.reference,self.panels,[8],config=self.config,ledger=ledger)
        self.assertEqual(len(fitted['history']),8)
        self.assertTrue(all({p['owner'] for p in row['participants']} == {0,1} for row in fitted['history']))
        self.assertFalse(torch.equal(fitted['head'],initial_head(self.reference,[8],fitted['permissions'])))
        self.assertEqual(before,digest(reference_payload(self.reference.model,self.reference.detector)))
        self.assertTrue(all(p.grad is None and not p.requires_grad for p in self.reference.model.parameters()))
        self.assertGreater(ledger.summary()['by_kind']['head_delta'],0)
        self.assertGreater(ledger.summary()['by_kind']['head_reference'],0)

    def test_c_gain_on_separable_features(self):
        fitted = self.imported()
        for panel in self.panels:
            scores = F.linear(panel['features'],fitted['head'][:,:-1],fitted['head'][:,-1])[:,0]
            self.assertTrue(torch.all(scores > 0) if panel['owner']==1 else torch.all(scores < 0))

    def test_c_pooled_gradient_matches_local_aggregation(self):
        permissions = head_permissions(self.reference,[8])
        phi = initial_head(self.reference,[8],permissions)
        arguments = dict(targets=[8],allowed=[6,7,8],class_weights={6:1/48,8:1/64},
                         permissions=permissions,rate=.01,round_id=0,batch_size=13)
        packets = [participant_delta(p,phi,**arguments) for p in self.panels]
        actual = aggregate_head(phi,packets)
        merged = dict(self.panels[0])
        for key in ('features','logits','labels'):
            merged[key] = torch.cat([p[key] for p in self.panels])
        expected = phi+participant_delta(merged,phi,**arguments)['delta']
        torch.testing.assert_close(actual,expected,rtol=1e-6,atol=1e-7)

    def test_c_minibatch_partition_invariance(self):
        a = self.imported()['head']
        b = learn_head(self.reference,self.panels,[8],config=replace(self.config,batch_size=7),
                       ledger=WireLedger(128*1024**2))['head']
        torch.testing.assert_close(a,b,rtol=1e-5,atol=1e-7)

    def test_c_inactive_coordinates_stay_zero(self):
        self.reference.model.unit_ranks['fc1'][0] = 1
        fitted = self.imported()
        self.assertFalse(bool(fitted['permissions'][0]))
        self.assertEqual(float(fitted['head'][0,0]),0.)

    def test_c_stale_feature_cache_rejected(self):
        self.panels[1]['identity']['reference_sha256'] = 'b'*64
        with self.assertRaises(ValueError): self.imported()

    def test_c_duplicate_and_mixed_round_packets_rejected(self):
        permission = head_permissions(self.reference,[8]); phi = initial_head(self.reference,[8],permission)
        packets = [participant_delta(p,phi,targets=[8],allowed=[6,7,8],class_weights={6:1/48,8:1/64},
            permissions=permission,rate=.01,round_id=0,batch_size=32) for p in self.panels]
        with self.assertRaises(ValueError): aggregate_head(phi,[packets[0],packets[0]])
        packets[1]['round'] = 1
        with self.assertRaises(ValueError): aggregate_head(phi,packets)

    def test_c_missing_target_base_rejected(self):
        with self.assertRaises(ValueError):
            learn_head(self.reference,self.panels[:1],[8],config=self.config,ledger=WireLedger(100000))

    def test_c_real_encoder_panel_does_not_use_donor_encoder(self):
        x = torch.randn(8,39,1); labels = torch.full((8,),8)
        before = digest(reference_payload(self.model,self.detector))
        panel = encode_panel(self.reference,x,labels,1,4)
        with torch.no_grad(): expected = self.reference.model.penultimate_features(x)
        torch.testing.assert_close(panel['features'],expected)
        self.assertEqual(panel['identity'],asdict(self.reference.identity))
        self.assertEqual(before,digest(reference_payload(self.model,self.detector)))

    def test_c_missing_cal_stops_before_snapshot_or_training(self):
        class Data:
            def count_upper_bounds(self,owner,role,*,task,per_class):
                return {6:64,8:0 if owner==0 else (3 if role=='calibration' else 64)}
            def get(self,*a,**kw): raise AssertionError('Preflight must not read examples')
        ledger = WireLedger(128*1024**2)
        from unittest.mock import patch
        with patch('fed_learning.strategies.decentralized.candle_v3.service.freeze_receiver',side_effect=AssertionError('No snapshot')):
            row = _session(0,TeacherSelection(0,(1,),(8,),((8,(1,)),)),{}, {},Data(),1,list(range(12)),
                           shared.PREPROCESS,self.config,ledger,'cpu',{})
        self.assertEqual(row['status'],'UNKNOWN'); self.assertEqual(row['stage'],'evidence_preflight')
        self.assertEqual(row['optimizer_steps'],0)
        self.assertNotIn('receiver_reference',ledger.summary()['by_kind'])

    def test_c_metadata_count_guard_and_upper_bound(self):
        class Roles:
            verified = {0}
            manifest = {'clients':{'0':{'role_class_counts':{'calibration':{'0':999,'6':90,'8':3}}}}}
        view = shared.CurrentTaskRoles(Roles(),1,[6,7,8],{},42)
        self.assertEqual(view.count_upper_bounds(0,'calibration',task=1,per_class=64),{6:64,7:0,8:3})
        with self.assertRaises(ValueError): view.count_upper_bounds(0,'calibration',task=0,per_class=64)
        with self.assertRaises(ValueError): view.count_upper_bounds(1,'calibration',task=1,per_class=64)
        view.close()
        with self.assertRaises(ValueError): view.count_upper_bounds(0,'calibration',task=1,per_class=64)

    def test_c_discovery_policy_is_versioned_for_resume(self):
        old = TeachingConfig.from_config({'candle_v3_teaching':{'enabled':True}})
        self.assertEqual(old.discovery_policy,'availability_only_v1')
        self.assertNotEqual(old,self.config)
        with self.assertRaises(ValueError): replace(self.config,discovery_policy='unversioned')

    def test_c_controlled_real_forward_service(self):
        rng = np.random.default_rng(42)
        samples = {}
        for owner,label in ((0,6),(1,8)):
            for role in ('base','fit','calibration'):
                x = torch.from_numpy((rng.normal(0,1e-4,(64,39,1))+(1 if label==8 else -1)).astype(np.float32))
                samples[owner,role] = x,torch.full((64,),label,dtype=torch.long)
        # Engineer a separable synthetic encoder using BASE alone, before any FIT/CAL access.
        # No gate, optimizer setting, inference or acceptance implementation is replaced.
        with torch.no_grad():
            neg = self.model._forward_backbone(samples[0,'base'][0]).mean(0)
            pos = self.model._forward_backbone(samples[1,'base'][0]).mean(0)
            direction = 8*(pos-neg)/(pos-neg).square().sum()
            offset = -direction.dot((pos+neg)/2)
            self.model.fc1.weight.zero_(); self.model.fc1.bias.zero_()
            self.model.fc1.weight[0].copy_(direction); self.model.fc1.bias[0] = offset
            self.model.fc1.weight[1].copy_(-direction); self.model.fc1.bias[1] = -offset
            self.model.fc2.weight.zero_(); self.model.fc2.bias.fill_(-100.)
            self.model.fc2.bias[6] = 0.; self.model.fc2.bias[8] = -.001
        donor = deepcopy(self.model); donor.unit_ranks['fc2'][8] = 2
        record_local_update(self.model,0,1,samples[0,'base'][1],1,successful_classes={6})
        record_local_update(donor,1,1,samples[1,'base'][1],1,successful_classes={8})
        donor_route = deepcopy(self.detector); donor_route.episode_classes[1] = [8]
        class Panel:
            def __init__(self):
                self.base = {owner:samples[owner,'base'] for owner in (0,1)}
                self.access = []
            def count_upper_bounds(self,owner,role,*,task,per_class):
                if task!=1 or role not in ('fit','calibration'): raise AssertionError('Metadata scope')
                return {int(c):min(per_class,int((samples[owner,role][1]==c).sum())) for c in (6,7,8)}
            def get(self,owner,role,*,task,per_class):
                if task!=1 or role not in ('base','fit','calibration'):
                    raise AssertionError('Historical/evaluation access')
                self.access.append(dict(owner=owner,role=role,task=task))
                return samples[owner,role]
        data = Panel()
        before = {k:v.clone() for k,v in self.model.state_dict().items() if not k.startswith('fc2.')}
        cfg = dict(candle_v3_teaching=asdict(self.config))
        kwargs = dict(config=cfg,task=1,models={0:self.model,1:donor},
            detectors={0:self.detector,1:donor_route},active_ids=[0,1],data=data,
            seen_classes=list(range(12)),preprocessing_sha256=shared.PREPROCESS,
            device='cpu',allowed_edges={0:[1],1:[0]})
        with tempfile.TemporaryDirectory(prefix='candle-v3-head-') as directory:
            result = run_task_teaching(output_dir=directory,**kwargs)
            accepted = [row for row in result['receipts'] if row['status']=='PASS']
            self.assertEqual(len(accepted),1,result['receipts'])
            row = accepted[0]
            self.assertEqual(len(row['head_training']),8)
            self.assertEqual(row['totals']['negative_break'],0)
            self.assertIn('8',self.model.candle_v3_state['peer_acquired'])
            for key,value in before.items(): self.assertTrue(torch.equal(value,self.model.state_dict()[key]),key)
            repeated = run_task_teaching(output_dir=directory,**kwargs)
            self.assertEqual(repeated['receipts'],[])
            self.assertEqual(repeated['wire']['teaching_bytes'],result['wire']['teaching_bytes'])
            type(self).controlled_service_receipt = dict(receipt=row,wire=result['wire'],access=data.access,
                fixture='BASE-engineered synthetic encoder; actual CNN-GRU forward and eight head rounds; no inference/acceptance mocks',
                scientific_transfer_evidence=False)


COMMON = [
    'test_private_runtime_import','test_canonical_features_equal_inference',
    'test_training_mode_features_rejected','test_snapshot_does_not_mutate_live',
    'test_staging_preserves_every_other_tensor','test_commit_and_checkpoint_restore',
    'test_stale_reference_rejected_without_commit','test_imported_row_survives_local_optimizer',
    'test_imported_row_survives_native_aggregation','test_fail_receipt_does_not_install',
    'test_exception_after_write_rolls_back','test_successful_step_ownership_only',
    'test_current_only_role_guard','test_acceptance_and_accounting','test_negative_break_rejects',
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True); args = parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(2); started = time.monotonic()
    names = COMMON+sorted(name for name in ReceiverHeadSmoke.__dict__ if name.startswith('test_c_'))
    suite = unittest.TestSuite(ReceiverHeadSmoke(name) for name in names)
    with (args.out/'tests.log').open('w',encoding='utf-8') as log:
        result = unittest.TextTestRunner(stream=log,verbosity=2).run(suite)
    payload = dict(schema='candle_v3_receiver_head_structural_smoke_v1',variant='receiver_head_learning',
        completed=result.wasSuccessful(),tests=result.testsRun,failures=len(result.failures),
        errors=len(result.errors),skipped=len(result.skipped),elapsed_seconds=time.monotonic()-started,
        synthetic=True,accuracy_benchmark=False,test_dataset_accessed=False,
        source_sha256=shared.source_identity(shared.RUNTIME)['source_sha256'],
        controlled_service=ReceiverHeadSmoke.controlled_service_receipt,
        failures_detail=[dict(test=str(t),traceback=trace) for t,trace in result.failures+result.errors])
    (args.out/'receipt.json').write_bytes((json.dumps(payload,indent=2)+'\n').encode('utf-8'))
    print(json.dumps({k:v for k,v in payload.items() if k not in ('controlled_service','failures_detail')},indent=2))
    if not result.wasSuccessful():
        print((args.out/'tests.log').read_text(encoding='utf-8')); raise SystemExit(1)


if __name__=='__main__': main()
