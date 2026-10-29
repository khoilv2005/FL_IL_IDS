"""Migration regressions: no raw TRAIN/TEST and no backbone/expert training."""
from copy import deepcopy
from pathlib import Path
import sys
import unittest
import io
import json
import hashlib
import tempfile
import zipfile
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'candle_v4_runtime'))
import migrate_candle_v4_task4 as migration
from test_candle_v4_historical_repair import fixture
from fed_learning.training.candle_v4_shadow import digest
from fed_learning.training.candle_v4_original_train import OriginalTrainService
from build_candle_v4_latest_resume_notebook import build


def setup():
    state,snapshots=fixture()
    old=dict(version=migration.OLD,seed=42,arms=['coverage'],fixed_threshold=.5,
             activation_policy='training_only_unverified',input_sha256='x',data_dir='old',output_dir='old',per_expert_bytes=100000)
    v=state['candle_v4_shadow_state']
    v.update(version=migration.OLD,spec=old,bytes={'coverage':123},engines={'coverage':{}},
             certificates={'coverage':{}},entries={},status={'coverage':{}},thresholds={'coverage':{}},
             task=4,boundary=(4,19),gated=[],history=[],acceptances=[],evaluations=[],runtime=0.)
    state.update(meta=dict(completed_task=4,resume_from_task=5,boundary='task'),round_state=None,
                 config={'candle_v4_full_service':old},rng_state={'seed':42})
    spec=dict(old,version=migration.NEW,data_dir='new',output_dir='new',evaluation_mode='v4_only')
    return state,snapshots,spec


class MigrationTests(unittest.TestCase):
    def test_extract_only_task_boundaries_checks_locked_task4(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); archive=root/'results16.zip'
            with zipfile.ZipFile(archive,'w') as outer:
                outer.writestr('run/candle_v4_run_lock.json',json.dumps(dict(source_commit=migration.SOURCE,version=migration.OLD)))
                for task in range(5):
                    data=f'task{task}'.encode();name=f'continuation_state_task_{task}.pt';buf=io.BytesIO()
                    with zipfile.ZipFile(buf,'w') as inner:
                        inner.writestr(name,data);inner.writestr('unused_round_model.pt',b'ignore')
                        inner.writestr('checkpoint_archive_manifest.json',json.dumps(dict(task_id=task,completed_rounds=list(range(20)),continuation_checkpoint=name,checksums={name:hashlib.sha256(data).hexdigest()})))
                    outer.writestr(f'run/checkpoint_task_{task}_all_rounds.zip',buf.getvalue())
            with patch.object(migration,'TASK4_SHA',hashlib.sha256(b'task4').hexdigest()):
                paths,_=migration.extract_snapshots(archive,root/'cache')
            self.assertEqual(set(paths),set(range(5)))
            self.assertFalse(list((root/'cache').glob('input_task_*.zip')))
            self.assertFalse((root/'cache/unused_round_model.pt').exists())
            with self.assertRaises(ValueError):migration.extract_snapshots(archive,root/'cache')

    def test_repair_preserves_native_experts_rng_and_restores_latest_service(self):
        state,snapshots,spec=setup()
        before=digest(state['client_model_states']),digest(state['client_algorithm_states']),digest(state['rng_state'])
        receipt=migration.migrate_state(state,snapshots,spec)
        self.assertEqual(receipt['repaired_clients'],1)
        self.assertEqual(before,(digest(state['client_model_states']),digest(state['client_algorithm_states']),digest(state['rng_state'])))
        service=OriginalTrainService(spec,state['candle_v4_shadow_state'])
        self.assertEqual(set(service.continuous_routers[1].class_to_task.values()),{0,1})
        restored=OriginalTrainService(spec,service.state_dict())
        self.assertEqual(digest(restored.state_dict()),digest(service.state_dict()))
        self.assertEqual(service.bytes['continuous_bootstrap'],0)
        self.assertGreater(service.bytes['historical_migration'],0)

    def test_budget_changes_and_partial_task_are_rejected(self):
        state,snapshots,spec=setup();spec['fixed_threshold']=.6
        with self.assertRaises(ValueError):migration.migrate_state(state,snapshots,spec)
        state,snapshots,spec=setup();state['meta']['boundary']='round'
        with self.assertRaises(ValueError):migration.migrate_state(state,snapshots,spec)

    def test_coordinate_mismatch_refuses_migration(self):
        state,snapshots,spec=setup();snapshots[0]['client_algorithm_states'][0]['context_detector']['binarize_thresholds']['conv1']=99
        before=digest(state)
        with self.assertRaises(ValueError):migration.migrate_state(state,snapshots,spec)
        self.assertEqual(before,digest(state))

    def test_certificate_identity_updated_without_changing_expert_state(self):
        state,snapshots,spec=setup()
        state['candle_v4_shadow_state']['certificates']['coverage'][(1,6)]=dict(version=migration.OLD,sha256='weight-hash')
        migration.migrate_state(state,snapshots,spec)
        self.assertEqual(state['candle_v4_shadow_state']['certificates']['coverage'][(1,6)],dict(version=migration.NEW,sha256='weight-hash'))

    def test_notebook_pinned_import_and_v4_only_terminal_path(self):
        source=build('a'*40);compile(source,'notebook','exec')
        self.assertIn('--import-task4-archive',source)
        self.assertIn('--v4-only',source)
        self.assertIn('--continuous-bootstrap',source)
        self.assertNotIn('RESULTS17',source)
        self.assertNotIn("'--resume'",source)


if __name__=='__main__':unittest.main()
