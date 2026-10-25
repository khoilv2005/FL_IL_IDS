"""Launcher bootstrap tests without importing training or contacting Drive."""
import ast
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import zipfile


def helpers():
    source = Path(__file__).resolve().parents[1]/'train_incremental_kaggle.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name in ('download_resume_zip', 'extract_cluster_resume', 'configure_cluster_resume', 'ensure_resume_torch')]
    scope = dict(Path=Path, json=json, os=os, shutil=shutil, subprocess=subprocess,
                 sys=sys, tempfile=tempfile, zipfile=zipfile)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), scope)
    return scope


class ClusterDriveResumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.h = helpers()
        env = patch.dict(os.environ, {}, clear=True)
        env.start(); self.addCleanup(env.stop)
        self.config = dict(server_only_storage=True, defer_evaluation=True, batch_size=2048,
                           device='cuda:0', output_budget_gib=18, seed=42,
                           worker_devices=['cuda:0', 'cuda:1'])
        (self.root/'protocol_lock.json').write_text(json.dumps({'config': self.config}))
        (self.root/'execution_provenance.json').write_text(json.dumps({'commit': 'original', 'torch': '2.10.0+cu128'}))

    def checkpoint(self, task, round_id, sealed=False):
        suffix = 'all_rounds' if sealed else f'round_{round_id:03d}'
        with zipfile.ZipFile(self.root/f'checkpoint_task_{task}_{suffix}.zip', 'w') as z:
            z.writestr(f'rounds/{round_id:03d}.json', json.dumps(dict(schema=1, task=task, round=round_id)))

    def test_latest_saved_round_ignores_partial_cluster_receipts(self):
        self.checkpoint(1, 20, True); self.checkpoint(2, 0)
        (self.root/'round_task_2_01_cluster.json').write_text('{}')
        self.h['configure_cluster_resume'](self.root)
        self.assertEqual(os.environ['DENICE_CLUSTER_RESUME_TASK'], '2')
        self.assertEqual(os.environ['DENICE_CLUSTER_RESUME_ROUND'], '0')
        self.assertEqual(os.environ['DENICE_CODE_REF'], 'original')
        self.assertEqual(os.environ['DENICE_WORKER_DEVICES'], 'cuda:0,cuda:1')

    def test_sealed_task_boundary_and_explicit_selection(self):
        self.checkpoint(1, 20, True); self.checkpoint(2, 0)
        os.environ['DENICE_CLUSTER_RESUME_TASK'] = '1'
        self.h['configure_cluster_resume'](self.root)
        self.assertEqual(os.environ['DENICE_CLUSTER_RESUME_ROUND'], '20')

    def test_nested_output_zip_skips_cache_and_keeps_checkpoint(self):
        self.checkpoint(2, 0)
        archive = self.root/'outer.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for p in self.root.iterdir():
                if p != archive: z.write(p, 'results (2)/run/'+p.name)
            z.writestr('results (2)/run/current_data_cache/huge.npy', b'cache')
        extracted = self.root/'extracted'
        extracted.mkdir()
        with patch.object(tempfile, 'mkdtemp', return_value=str(extracted)):
            result = self.h['extract_cluster_resume'](archive)
        self.assertTrue((result/'checkpoint_task_2_round_000.zip').exists())
        self.assertFalse((result/'current_data_cache').exists())

    def test_unsafe_zip_rejected(self):
        archive = self.root/'bad.zip'
        with zipfile.ZipFile(archive, 'w') as z: z.writestr('../escape', 'bad')
        with patch.object(tempfile, 'mkdtemp', return_value=str(self.root/'extract')):
            with self.assertRaisesRegex(ValueError, 'Unsafe'):
                self.h['extract_cluster_resume'](archive)

    def test_missing_requested_round_rejected(self):
        self.checkpoint(2, 0)
        os.environ['DENICE_CLUSTER_RESUME_ROUND'] = '1'
        with self.assertRaisesRegex(ValueError, 'No checkpoint'):
            self.h['configure_cluster_resume'](self.root)

    def test_matching_runtime_does_not_install(self):
        result = subprocess.CompletedProcess([], 0, '2.10.0+cu128\n', '')
        with patch.object(subprocess, 'run', return_value=result) as run:
            self.h['ensure_resume_torch']({'torch': '2.10.0+cu128'})
        self.assertEqual(run.call_count, 1)

    def test_runtime_install_reprobes_fresh_interpreter(self):
        old = subprocess.CompletedProcess([], 0, '2.11.0+cu130\n', '')
        new = subprocess.CompletedProcess([], 0, '2.10.0+cu128\n', '')
        with patch.object(subprocess, 'run', side_effect=[old, None, new]) as run:
            self.h['ensure_resume_torch']({'torch': '2.10.0+cu128'})
        self.assertEqual(run.call_count, 3)
        self.assertIn('torch==2.10.0+cu128', run.call_args_list[1].args[0])
        self.assertEqual(run.call_args_list[0].args[0][0], sys.executable)

    def test_unresolved_install_stops_before_training(self):
        old = subprocess.CompletedProcess([], 0, '2.11.0+cu130\n', '')
        with patch.object(subprocess, 'run', side_effect=[old, None, old]):
            with self.assertRaisesRegex(RuntimeError, 'did not resolve'):
                self.h['ensure_resume_torch']({'torch': '2.10.0+cu128'})


if __name__ == '__main__':
    unittest.main()
