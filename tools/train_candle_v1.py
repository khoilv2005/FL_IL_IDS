"""Kaggle-only CANDLE V1 run using the audited DeNICE V1 configuration."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys


def resolve_data(root):
    root = Path(root).resolve()
    candidates = [root] if (root / 'metadata.json').is_file() else [
        path.parent for path in root.rglob('metadata.json')]
    candidates = [path for path in candidates
                  if (path / 'global_test_data.npz').is_file()
                  and next(path.glob('client_*_train.npz'), None) is not None]
    if len(candidates) != 1:
        raise ValueError(f'Set --data-dir to one original dataset; found {candidates}')
    return candidates[0]


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--roles', required=True,
                        help='Existing locked V1 roles, or an empty directory to prepare them')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    import torch
    from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
    from tools.prepare_denice_clean_roles import prepare

    if not torch.cuda.is_available():
        raise RuntimeError('Enable a Kaggle GPU. This launcher does not run local CPU training.')
    data = resolve_data(args.data_dir)
    output = Path(args.output).resolve()
    roles_path = Path(args.roles).resolve()
    if output == roles_path or output in roles_path.parents or roles_path in output.parents:
        raise ValueError('Keep roles and training output in separate directories')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use an empty output directory; existing runs are never overwritten')
    inherited_controls = [name for name in ('DENICE_CONFIG_FILE', 'DENICE_CONFIG_OVERRIDES',
                                           'DENICE_RESUME_STATE_PATH', 'DENICE_TASK_END')
                          if os.environ.get(name)]
    if inherited_controls or os.environ.get('DENICE_CONFIG_ONLY') == '1':
        raise ValueError('Use a fresh session without inherited config/resume controls for frozen V1')
    config_path = root / 'configs/candle_v1.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    origin = json.loads((root / 'configs/candle_v1_provenance.json').read_text(encoding='utf-8'))
    if (roles_path / 'role_lock.json').is_file():
        roles = CleanRoleData(roles_path, source_data_dir=data)
        if (roles.manifest['split_seed'] != origin['role_split_seed']
                or roles.manifest['requested_fractions'] !=
                {'base': .8, 'calibration': .02, 'fit': .08, 'validation': .1}):
            raise ValueError('Existing roles do not match the historical V1 split protocol')
    else:
        prepare(str(data), str(roles_path), origin['role_split_seed'])
        roles = CleanRoleData(roles_path, source_data_dir=data)

    config.update(data_dir=str(data), output_dir=str(output), resume_output_dir=str(output),
                  denice_clean_roles_dir=str(roles_path))
    code_ref = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=root, text=True).strip()
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / 'candle_v1_protocol.json', dict(
        experiment='CANDLE V1', supporting_source_commit=code_ref, historical_reference=origin,
        frozen_config_sha256=file_sha256(config_path),
        role_manifest_sha256=file_sha256(roles.root / 'role_manifest.json'),
        runtime=dict(python=sys.version, torch=str(torch.__version__),
                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        primary_policy='BinarySelf', secondary_policy='MulticlassSelf',
        test_protocol='Full original 34-class TEST; disjoint receiver shards; pooled metrics',
        no_exact_numerical_reproduction_claim=True))
    write_json(output / 'candle_v1_requested_config.json', config)
    environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (output / 'environment.txt').write_text(environment, encoding='utf-8')
    os.environ.update(DENICE_RUN_PROFILE='legacy', DENICE_VARIANT='legacy',
                      DENICE_TRAIN_PHASE='5', DENICE_CODE_DIR=str(root), DENICE_SEED='42',
                      DENICE_OUTPUT_DIR=str(output), DENICE_CONFIG_OVERRIDES=json.dumps(config))
    print('CANDLE V1: DeNICE legacy engine; frozen audited V1 configuration.', flush=True)
    print('Fresh T0..T5; 20 rounds/task; seed42; batch2048; xi=.8; binary_cosine; AMP.', flush=True)
    print('Historical BASE split; no replay, APPLIANCE, CGoFed or CME.', flush=True)
    print('One GPU; checkpoint each round; seal one ZIP/task; output budget 12 GiB.', flush=True)
    print('No intermediate TEST; terminal full TEST, BinarySelf primary / MulticlassSelf secondary.', flush=True)
    runpy.run_path(str(root / 'train_incremental_kaggle.py'), run_name='__main__')

    from tools.eval_denice_legacy_self import run_legacy_self
    gc.collect()
    torch.cuda.empty_cache()
    archive = output / 'checkpoint_task_5_all_rounds.zip'
    evaluation = output / 'candle_v1_self_task_5'
    result = run_legacy_self(archive, roles.root, evaluation, data, device='cuda',
                             batch_size=512, expected_xi=.8)
    # Keep the evaluator's original lock untouched; label the baseline externally.
    write_json(output / 'candle_v1_completion.json', dict(
        completed=result['completed'], experiment='CANDLE V1', main_policy='BinarySelf',
        metrics=result['metrics'], test_rows=result['test_rows'],
        evaluation_completion=str(evaluation / 'completion.json'),
        historical_reference=origin, supporting_source_commit=code_ref))
    packaged = shutil.make_archive(str(evaluation), 'zip', root_dir=evaluation)
    print('CANDLE V1 metrics:', json.dumps(result['metrics'], indent=2), flush=True)
    print('Evaluation ZIP:', packaged, flush=True)


if __name__ == '__main__':
    main()
