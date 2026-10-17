"""Exercise the real launch configuration without downloading or training."""
import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


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
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'phase_config' for t in node.targets):
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
    namespace.update(REPO_PATH=str(ROOT), target=None,
                     phase_config=namespace['PHASE_CONFIG'][namespace['TRAIN_PHASE']])
    exec(compile(ast.Module(body=config, type_ignores=[]), '<config>', 'exec'), namespace)
    return namespace['CONFIG']


def test_single_uploaded_file_defaults_to_updated_profile(monkeypatch):
    config = resolve(monkeypatch)
    assert config['denice_appliance_enabled'] is True
    assert config['appliance_enabled'] is False
    assert config['denice_expected_method'] == 'row_overlay'
    assert config['denice_router_mode'] == 'multiclass_balanced'
    assert config['denice_amp_enabled'] is False
    assert config['denice_eval_require_full_coverage'] is False
    assert config['resume_state_path'] is None
    assert config['denice_clustering_mode'] == 'adaptive_ap'
    assert 'row_overlay' in config['output_dir']


def test_explicit_legacy_profile_remains_available(monkeypatch):
    config = resolve(monkeypatch, DENICE_RUN_PROFILE='legacy')
    assert not config.get('denice_appliance_enabled', False)
    assert config['denice_router_mode'] == 'binary_cosine'


def test_existing_ablation_overrides_do_not_inherit_overlay(monkeypatch):
    config = resolve(monkeypatch, DENICE_CONFIG_OVERRIDES='{"denice_amp_enabled": false}')
    assert not config.get('denice_appliance_enabled', False)
    assert config['denice_amp_enabled'] is False
