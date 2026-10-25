import hashlib
import importlib.util
import json
import os
import contextlib
import io
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch
import zipfile


class EvalDriveLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        spec = importlib.util.spec_from_file_location('eval_launcher', Path(__file__).resolve().parents[1]/'eval_selective.py')
        self.mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(self.mod)
        env = patch.dict(os.environ, {}, clear=True); env.start(); self.addCleanup(env.stop)

    def output(self):
        out = self.root/'run'; out.mkdir()
        (out/'completion.json').write_text(json.dumps(dict(completed=True, tasks_completed=6,
            checkpoint_format='denice_v2_cluster_archive_v1')))
        (out/'checkpoint_task_5_all_rounds.zip').write_bytes(b'checkpoint placeholder')
        (out/'execution_provenance.json').write_text(json.dumps(dict(torch='2.10.0+cu128',
            source_sha256={'module.py':hashlib.sha256(b'# original').hexdigest()})))
        return out

    def test_download_to_fresh_eval_process(self):
        out = self.output()
        source = self.root/'source'; source.mkdir(); (source/'module.py').write_bytes(b'# original')
        data = self.root/'data'; data.mkdir(); (data/'global_test_data.npz').write_bytes(b'test placeholder')
        archive = self.root/'outer.zip'
        with zipfile.ZipFile(archive, 'w') as z:
            for p in out.iterdir(): z.write(p, 'nested/run/'+p.name)
            z.writestr('nested/run/current_data_cache/unneeded.npy', b'cache')
        os.environ['DENICE_CODE_DIR'] = str(source)
        calls = []
        def fake_run(args, **kwargs):
            calls.append(args)
            if 'gdown' in args and '--fuzzy' in args:
                shutil.copyfile(archive, args[-1])
        with patch.object(self.mod, 'ensure_eval_torch') as ensure, patch.object(self.mod.subprocess, 'run', side_effect=fake_run):
            self.mod.main(['--drive-url','https://drive.google.com/file/d/example/view',
                '--data-dir',str(data),'--output',str(self.root/'eval')])
        ensure.assert_called_once()
        command = calls[-1]
        self.assertEqual(command[1:3], ['-u','-c'])
        compile(command[3], '<eval_worker>', 'exec')
        settings = json.loads(command[-1])
        self.assertEqual(settings['expected_torch'], '2.10.0+cu128')
        self.assertEqual(settings['batch_size'], 8192)
        extracted = Path(settings['checkpoint_root'])
        self.assertTrue((extracted/'checkpoint_task_5_all_rounds.zip').is_file())
        self.assertFalse((extracted/'current_data_cache').exists())
        with patch.object(self.mod, 'ensure_eval_torch'), patch.object(self.mod.subprocess, 'run', side_effect=fake_run):
            self.mod.main(['--drive-url', '', '--checkpoint-root', str(out), '--data-dir', str(data),
                '--output', str(self.root/'eval'), '--diagnostic', '--diagnostic-per-class', '64'])
        command = calls[-1]
        compile(command[3], '<diagnostic_worker>', 'exec')
        settings = json.loads(command[-1])
        self.assertEqual(settings['per_class'], 64)
        self.assertEqual(Path(settings['output_dir']).name, 'diagnostics')

    def test_incomplete_train_rejected(self):
        out = self.output(); (out/'completion.json').write_text('{"completed":false}')
        with self.assertRaisesRegex(ValueError, 'Finish TRAIN'):
            self.mod.check_completed(out)

    def test_path_traversal_rejected(self):
        archive = self.root/'bad.zip'
        with zipfile.ZipFile(archive, 'w') as z: z.writestr('../completion.json','{}')
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            self.mod.extract_output(archive,self.root/'extract')

    def test_missing_provenance_rejected(self):
        out = self.output(); (out/'execution_provenance.json').unlink()
        with self.assertRaisesRegex(ValueError, 'provenance'):
            self.mod.check_completed(out)

    def test_notebook_kernel_flags_are_not_eval_arguments(self):
        for launcher in ('colab_kernel_launcher.py', 'ipykernel_launcher.py'):
            with self.subTest(launcher=launcher):
                os.environ['DENICE_CHECKPOINT_ROOT'] = 'configured-output'
                os.environ['DENICE_EVAL_DRIVE_URL'] = ''
                with patch.object(self.mod.sys, 'argv', [launcher, '-f', '/tmp/kernel.json',
                        '--HistoryManager.hist_file=:memory:']), patch.object(
                        self.mod, 'check_completed', side_effect=RuntimeError('reached checkpoint')) as check:
                    with self.assertRaisesRegex(RuntimeError, 'reached checkpoint'):
                        self.mod.main()
                    check.assert_called_once_with('configured-output')

    def test_explicit_arguments_survive_notebook_launcher(self):
        os.environ['DENICE_EVAL_DRIVE_URL'] = ''
        with patch.object(self.mod.sys, 'argv', ['colab_kernel_launcher.py', '-f', 'kernel.json']), patch.object(
                self.mod, 'check_completed', side_effect=RuntimeError('reached checkpoint')) as check:
            with self.assertRaisesRegex(RuntimeError, 'reached checkpoint'):
                self.mod.main(['--checkpoint-root', 'explicit-output'])
            check.assert_called_once_with('explicit-output')

    def test_cli_typos_still_rejected(self):
        with patch.object(self.mod.sys, 'argv', ['eval_selective.py', '--bad-option']), contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                self.mod.main()
            self.assertEqual(error.exception.code, 2)


if __name__ == '__main__': unittest.main()
