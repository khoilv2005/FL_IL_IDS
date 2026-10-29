"""Exercise the real launch configuration without downloading or training."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def download_setup_namespace(monkeypatch, tmp_path):
    import os
    import subprocess
    import tempfile
    from types import SimpleNamespace
    for key in ('DENICE_CODE_DIR', 'DENICE_CODE_REF'):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kw: str(tmp_path))
    calls = []
    def fake_git(command, **kwargs):
        calls.append(command)
        if command[1] == 'clone':
            support = tmp_path / 'fed_learning/training/denice_overlay_resume.py'
            support.parent.mkdir(parents=True, exist_ok=True)
            support.write_bytes((ROOT / 'fed_learning/training/denice_overlay_resume.py').read_bytes())
    monkeypatch.setattr(subprocess, 'run', fake_git)
    namespace = dict(os=os, subprocess=subprocess,
                     sys=SimpleNamespace(path=[], modules={}),
                     TRAIN_PROFILE='row_overlay', target='continuation_state_task_3.pt')
    nodes = []
    for node in ast.parse((ROOT / 'train_incremental_kaggle.py').read_text(encoding='utf-8')).body:
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'DEFAULT_CODE_REF' for t in node.targets)
                or isinstance(node, ast.FunctionDef) and node.name == 'setup_imports'):
            nodes.append(node)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), '<download-setup>', 'exec'), namespace)
    return namespace, calls


def test_small_launcher_downloads_pinned_resume_capable_revision(monkeypatch, tmp_path):
    namespace, calls = download_setup_namespace(monkeypatch, tmp_path)
    namespace['setup_imports']()
    assert namespace['REPO_PATH'] == str(tmp_path)
    assert calls[1][-1] == namespace['DEFAULT_CODE_REF']
    assert 'checkout' in calls[1]
    data = (ROOT / 'train_incremental_kaggle.py').read_bytes()
    assert b'# BEGIN DENICE SUPPORT PAYLOAD' in data
    assert b'EMBEDDED_SUPPORT_B85' not in data
    assert data == (ROOT / 'output/single_file/train_incremental_kaggle.py').read_bytes()


def test_explicit_stale_source_fails_with_actionable_error(monkeypatch, tmp_path):
    import pytest
    namespace, _ = download_setup_namespace(monkeypatch, tmp_path)
    (tmp_path / 'fed_learning').mkdir()
    monkeypatch.setenv('DENICE_CODE_DIR', str(tmp_path))
    with pytest.raises(RuntimeError, match='Selected source lacks overlay resume support'):
        namespace['setup_imports']()


def test_default_resume_enables_recovery_and_runs_tasks4_and5(monkeypatch):
    config = resolve(monkeypatch)
    assert config['task_start'] == 4
    assert config['task_end'] == 5
    assert config['denice_overlay_rebuild_private_state'] is True


def drive_download_prefix(monkeypatch, tmp_path, members):
    import os
    import subprocess
    import tempfile
    import zipfile
    for key in list(os.environ):
        if key.startswith('DENICE_'):
            monkeypatch.delenv(key)
    monkeypatch.setattr(tempfile, 'mkdtemp', lambda **kw: str(tmp_path))
    calls = []
    def fake_run(command, **kwargs):
        calls.append(command)
        if 'gdown' in command and '-O' in command:
            with zipfile.ZipFile(command[-1], 'w') as archive:
                for name in members:
                    archive.writestr(name, b'fixture-checkpoint')
    monkeypatch.setattr(subprocess, 'run', fake_run)
    prefix = []
    for node in ast.parse((ROOT / 'train_incremental_kaggle.py').read_text(encoding='utf-8')).body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'REPO_PATH' for t in node.targets):
            break
        prefix.append(node)
    namespace = {}
    exec(compile(ast.Module(body=prefix, type_ignores=[]), '<download>', 'exec'), namespace)
    return namespace, calls


def test_in_file_drive_url_works_without_environment(monkeypatch, tmp_path):
    namespace, calls = drive_download_prefix(monkeypatch, tmp_path,
        ['nested/continuation_state_task_3.pt', 'nested/checkpoint_task_4_base.pt'])
    assert calls[1][calls[1].index('--fuzzy') + 1] == namespace['resume_drive_url']
    assert '1UMtq48e1z0MnOMWDoscGMVEjsmtAizO0' in namespace['resume_drive_url']
    assert Path(namespace['target']).read_bytes() == b'fixture-checkpoint'
    assert not (tmp_path / 'nested').exists()


def test_wrong_drive_archive_does_not_reuse_stale_checkpoint(monkeypatch, tmp_path):
    import pytest
    (tmp_path / 'continuation_state_task_3.pt').write_bytes(b'stale')
    with pytest.raises(ValueError, match='found 0'):
        drive_download_prefix(monkeypatch, tmp_path, ['continuation_state_task_2.pt'])


def test_ambiguous_drive_archive_is_rejected(monkeypatch, tmp_path):
    import pytest
    with pytest.raises(ValueError, match='found 2'):
        drive_download_prefix(monkeypatch, tmp_path,
            ['run_a/continuation_state_task_3.pt', 'run_b/continuation_state_task_3.pt'])


def resolve(monkeypatch, **environment):
    import os
    for key in list(os.environ):
        if key.startswith('DENICE_'):
            monkeypatch.delenv(key)
    for key, value in environment.items():
        monkeypatch.setenv(key, value)
    tree = ast.parse((ROOT / 'train_incremental_kaggle.py').read_text(encoding='utf-8'))
    prefix, config = [], []
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'resume_drive_url' for t in node.targets):
            break
        prefix.append(node)
    started = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'CONFIG' for t in node.targets):
            started = True
        if started:
            if isinstance(node, ast.ImportFrom) and node.module == 'fed_learning.strategies.decentralized.denice_appliance':
                break
            config.append(node)
    namespace = {}
    exec(compile(ast.Module(body=prefix, type_ignores=[]), '<launch>', 'exec'), namespace)
    namespace.update(REPO_PATH=str(ROOT), target=environment.get('DENICE_RESUME_STATE_PATH'))
    exec(compile(ast.Module(body=config, type_ignores=[]), '<config>', 'exec'), namespace)
    return namespace['CONFIG']


def test_single_uploaded_file_defaults_to_updated_profile(monkeypatch):
    config = resolve(monkeypatch, DENICE_TRAIN_PHASE='5')
    assert config['denice_appliance_enabled'] is True
    assert config['appliance_enabled'] is False
    assert config['denice_expected_method'] == 'row_overlay'
    assert config['denice_router_mode'] == 'multiclass_balanced'
    assert config['denice_amp_enabled'] is False
    assert config['denice_eval_require_full_coverage'] is False
    assert config['resume_state_path'] is None
    assert config['denice_clustering_mode'] == 'adaptive_ap'
    assert 'row_overlay' in config['output_dir']


def test_overlay_resume_keeps_task_range_and_checkpoint(monkeypatch):
    config = resolve(monkeypatch, DENICE_RUN_PROFILE='row_overlay', DENICE_TRAIN_PHASE='4',
                     DENICE_TASK_END='4', DENICE_RESUME_STATE_PATH='/input/continuation_state_task_3.pt',
                     DENICE_OVERLAY_REBUILD_PRIVATE_STATE='1')
    assert config['task_start'] == config['task_end'] == 4
    assert config['resume_state_path'] == '/input/continuation_state_task_3.pt'
    assert config['denice_overlay_rebuild_private_state'] is True


def test_reject_invalid_task_end(monkeypatch):
    import pytest
    with pytest.raises(ValueError, match='DENICE_TASK_END'):
        resolve(monkeypatch, DENICE_TRAIN_PHASE='4', DENICE_TASK_END='2')


def test_explicit_legacy_profile_remains_available(monkeypatch):
    config = resolve(monkeypatch, DENICE_RUN_PROFILE='legacy')
    assert not config.get('denice_appliance_enabled', False)
    assert config['denice_router_mode'] == 'binary_cosine'


def test_existing_ablation_overrides_do_not_inherit_overlay(monkeypatch):
    config = resolve(monkeypatch, DENICE_CONFIG_OVERRIDES='{"denice_amp_enabled": false}')
    assert not config.get('denice_appliance_enabled', False)
    assert config['denice_amp_enabled'] is False
