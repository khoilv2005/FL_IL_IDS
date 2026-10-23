"""Isolated CANDLE V3 launcher; peer teaching is an explicit experimental opt-in."""
import argparse
import gc
import hashlib
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
    parser.add_argument('--_runtime-worker', action='store_true', help=argparse.SUPPRESS)
    variant = parser.add_mutually_exclusive_group()
    variant.add_argument('--peer-teaching', action='store_true',
                        help='Use the experimental V3 teaching implementation (not yet smoke validated)')
    variant.add_argument('--direct-fc2', action='store_true',
                        help='Direct trained FC2 row transfer (no head optimizer; unvalidated variant A)')
    args = parser.parse_args()
    repository = Path(__file__).resolve().parents[1]
    root = repository / 'candle_v3_runtime'
    manifest = json.loads((root / 'runtime_manifest.json').read_text(encoding='utf-8'))
    if manifest['stage'] != 'peer_teaching_implementation':
        raise ValueError('Unsupported CANDLE V3 runtime stage')
    for entry in manifest['files']:
        if hashlib.sha256((root / entry['path']).read_bytes()).hexdigest() != entry['sha256']:
            raise ValueError('Locked V3 source changed: ' + entry['path'])
    baseline = json.loads((root / 'configs/candle_v1.json').read_text(encoding='utf-8'))
    profile = ('candle_v3_direct_fc2.json' if args.direct_fc2 else
               'candle_v3_peer_teaching.json' if args.peer_teaching else 'candle_v3.json')
    requested = json.loads((repository / 'configs' / profile).read_text(encoding='utf-8'))
    path_keys = {'output_dir', 'resume_output_dir', 'denice_clean_roles_dir'}
    if args.peer_teaching or args.direct_fc2:
        path_keys |= {'candle_v3_teaching', 'denice_expected_method'}
    if ({k:v for k,v in baseline.items() if k not in path_keys} !=
            {k:v for k,v in requested.items() if k not in path_keys}):
        raise ValueError('V3 must retain the V1 baseline protocol outside declared teaching changes')
    expected_method = ('candle_v3_direct_fc2' if args.direct_fc2 else
                       'candle_v3_peer_teaching' if args.peer_teaching else 'denice_base')
    if requested.get('denice_expected_method') != expected_method:
        raise ValueError('Profile has an incorrect method identity')
    if not args._runtime_worker:
        env = {**os.environ, 'PYTHONPATH': str(root)}
        command = [sys.executable, '-u', str(Path(__file__).resolve()),
            '--_runtime-worker', '--data-dir', str(Path(args.data_dir).resolve()),
            '--output', str(Path(args.output).resolve()), '--roles', str(Path(args.roles).resolve())]
        if args.peer_teaching:
            command.append('--peer-teaching')
        if args.direct_fc2:
            command.append('--direct-fc2')
        subprocess.check_call(command, cwd=root, env=env)
        return
    # All original absolute imports resolve to this private fork in a fresh process.
    excluded = {repository.resolve(), (repository / 'tools').resolve()}
    sys.path[:] = [p for p in sys.path if Path(p or os.getcwd()).resolve() not in excluded]
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
    config_path = repository / 'configs' / profile
    config = json.loads(config_path.read_text(encoding='utf-8'))
    origin = json.loads((repository / 'configs/candle_v3_provenance.json').read_text(encoding='utf-8'))
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
    write_json(output / 'candle_v3_protocol.json', dict(
        experiment='CANDLE V3', stage='transfer_implementation' if args.peer_teaching or args.direct_fc2 else 'v1_baseline',
        peer_teaching_enabled=args.peer_teaching, direct_fc2_enabled=args.direct_fc2,
        validation_status=manifest['native_quality_status'],
        supporting_source_commit=code_ref, historical_reference=origin,
        baseline_source_commit=manifest['baseline_source_commit'],
        runtime_manifest_sha256=file_sha256(root / 'runtime_manifest.json'),
        frozen_config_sha256=file_sha256(config_path),
        role_manifest_sha256=file_sha256(roles.root / 'role_manifest.json'),
        runtime=dict(python=sys.version, torch=str(torch.__version__),
                     cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0)),
        primary_policy='BinarySelf', secondary_policy='MulticlassSelf',
        test_protocol='Full original 34-class TEST; disjoint receiver shards; pooled metrics',
        no_exact_numerical_reproduction_claim=True))
    write_json(output / 'candle_v3_requested_config.json', config)
    environment = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (output / 'environment.txt').write_text(environment, encoding='utf-8')
    os.environ.update(DENICE_RUN_PROFILE='legacy', DENICE_VARIANT='legacy',
                      DENICE_TRAIN_PHASE='5', DENICE_CODE_DIR=str(root), DENICE_SEED='42',
                      DENICE_OUTPUT_DIR=str(output), DENICE_CONFIG_OVERRIDES=json.dumps(config))
    print(f'CANDLE V3: native V1 aggregation; peer_teaching={args.peer_teaching}; '
          f'direct_fc2={args.direct_fc2}; implementation has not been smoke validated.', flush=True)
    print('Fresh T0..T5; 20 rounds/task; seed42; batch2048; xi=.8; binary_cosine; AMP.', flush=True)
    print('Historical BASE split; no replay, APPLIANCE, CGoFed or CME.', flush=True)
    print('One GPU; checkpoint each round; seal one ZIP/task; output budget 12 GiB.', flush=True)
    print('No intermediate TEST; terminal full TEST, BinarySelf primary / MulticlassSelf secondary.', flush=True)
    runpy.run_path(str(root / 'train_incremental_kaggle.py'), run_name='__main__')

    from tools.eval_denice_legacy_self import run_legacy_self
    gc.collect()
    torch.cuda.empty_cache()
    archive = output / 'checkpoint_task_5_all_rounds.zip'
    evaluation = output / 'candle_v3_self_task_5'
    result = run_legacy_self(archive, roles.root, evaluation, data, device='cuda',
                             batch_size=512, expected_xi=.8)
    # Keep the evaluator's original lock untouched; label the baseline externally.
    write_json(output / 'candle_v3_completion.json', dict(
        completed=result['completed'], experiment='CANDLE V3', main_policy='BinarySelf',
        metrics=result['metrics'], test_rows=result['test_rows'],
        evaluation_completion=str(evaluation / 'completion.json'),
        historical_reference=origin, supporting_source_commit=code_ref))
    packaged = shutil.make_archive(str(evaluation), 'zip', root_dir=evaluation)
    print('CANDLE V3 metrics:', json.dumps(result['metrics'], indent=2), flush=True)
    print('Evaluation ZIP:', packaged, flush=True)


if __name__ == '__main__':
    main()
