"""Fresh/resumable Kaggle CANDLE V2 with training-time prototype integration."""
import argparse
import gc
import json
import os
from pathlib import Path
import runpy
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.train_candle_v1 import resolve_data, write_json


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data-dir', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--roles', required=True)
    p.add_argument('--resume', default='')
    p.add_argument('--batch-size', type=int, default=2048)
    a = p.parse_args()
    import torch
    from fed_learning.data.denice_clean_roles import CleanRoleData, file_sha256
    from fed_learning.strategies.decentralized.candle_v2 import validate
    from tools.prepare_denice_clean_roles import prepare
    if not torch.cuda.is_available():
        raise RuntimeError('Enable GPU in Kaggle settings')
    for name in ('DENICE_CONFIG_FILE', 'DENICE_CONFIG_OVERRIDES', 'DENICE_RESUME_STATE_PATH', 'DENICE_TASK_END'):
        if os.environ.get(name):
            raise ValueError(f'Use a fresh environment without inherited {name}')
    if os.environ.get('DENICE_CONFIG_ONLY') == '1':
        raise ValueError('Unset DENICE_CONFIG_ONLY to run training')
    data = resolve_data(a.data_dir)
    output = Path(a.output).resolve()
    roles_dir = Path(a.roles).resolve()
    if output == roles_dir or output in roles_dir.parents or roles_dir in output.parents:
        raise ValueError('Keep roles and results in separate directories')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Use a new output directory, including when resuming a mounted checkpoint')
    cfg_path = ROOT/'configs/candle_v2.json'
    config = json.loads(cfg_path.read_text(encoding='utf-8'))
    if a.batch_size < 1:
        raise ValueError('Positive batch size required')
    if (roles_dir/'role_lock.json').is_file():
        roles = CleanRoleData(roles_dir, source_data_dir=data)
    else:
        prepare(str(data), str(roles_dir), 20261006)
        roles = CleanRoleData(roles_dir, source_data_dir=data)
    if roles.manifest['split_seed'] != 20261006 or roles.manifest['requested_fractions'] != {
            'base': .8, 'calibration': .02, 'fit': .08, 'validation': .1}:
        raise ValueError('Use the V1-matched locked split for this comparison')
    config.update(data_dir=str(data), output_dir=str(output), resume_output_dir=str(output),
                  denice_clean_roles_dir=str(roles_dir), batch_size=a.batch_size)
    if a.resume:
        from fed_learning.training.decentralized_denice_il import _load_denice_continuation_state
        resume_path = Path(a.resume).resolve()
        if resume_path.is_dir():
            manifests = list(resume_path.rglob('checkpoint_archive_manifest.json'))
            if len(manifests) != 1:
                raise ValueError('Choose one extracted CANDLE V2 task archive directory')
            manifest = json.loads(manifests[0].read_text(encoding='utf-8'))
            member = manifest.get('continuation_checkpoint')
            if not member or Path(member).name != member:
                raise ValueError('Extracted task archive has no continuation checkpoint')
            resume_path = manifests[0].parent/member
            if file_sha256(resume_path) != manifest['checksums'][member]:
                raise ValueError('Continuation checkpoint checksum mismatch')
        saved = _load_denice_continuation_state(str(resume_path))
        if not saved['config'].get('candle_v2_enabled'):
            raise ValueError('Cannot resume a legacy checkpoint as CANDLE V2')
        if int(saved['meta']['resume_from_task']) > 5:
            raise ValueError('This checkpoint already completed Task5; no remaining training')
        config.update(resume_state_path=str(resume_path),
                      task_start=int(saved['meta']['resume_from_task']))
        del saved
    validate(config)
    output.mkdir(parents=True, exist_ok=True)
    print('Archiving exact role indices for checkpoint/resume...', flush=True)
    shutil.make_archive(str(output/'candle_v2_roles'), 'zip', root_dir=roles_dir)
    ref = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip()
    write_json(output/'candle_v2_requested_config.json', config)
    write_json(output/'candle_v2_protocol.json', dict(experiment='CANDLE V2', source_commit=ref,
        primary='TaskFreeCosineSelf', config_sha256=file_sha256(cfg_path),
        module_sha256=file_sha256(ROOT/'fed_learning/strategies/decentralized/candle_v2.py'),
        role_manifest_sha256=file_sha256(roles_dir/'role_manifest.json'),
        changes_from_V1=['training prototype alignment', 'virtual missing-class CE',
                         'current-input frozen-teacher preservation', 'task-free cosine FC2',
                         'no context-specific micro-adapters'],
        data_protocol='Same V1 BASE split and internal current-task holdout; not full original TRAIN',
        evaluation='Only terminal Task5/Round19, one full TEST partition across receiver models',
        effectiveness_unverified=True, GPU=torch.cuda.get_device_name(0)))
    (output/'environment.txt').write_text(subprocess.check_output(
        [sys.executable, '-m', 'pip', 'freeze'], text=True), encoding='utf-8')
    os.environ.update(DENICE_RUN_PROFILE='legacy', DENICE_VARIANT='legacy', DENICE_TRAIN_PHASE='5',
        DENICE_CODE_DIR=str(ROOT), DENICE_SEED='42', DENICE_OUTPUT_DIR=str(output),
        DENICE_CONFIG_OVERRIDES=json.dumps(config))
    print('CANDLE V2: fresh T0..T5, 20 rounds/task, xi=.8, seed42; one GPU.', flush=True)
    print('Real cumulative CE + peer alignment + virtual prototype CE + current-input preservation.', flush=True)
    print('No raw replay/donor inference; task-free cosine FC2; only final TEST.', flush=True)
    runpy.run_path(str(ROOT/'train_incremental_kaggle.py'), run_name='__main__')
    from tools.eval_candle_v2 import run
    gc.collect()
    torch.cuda.empty_cache()
    evaluation = output/'candle_v2_self_task_5'
    result = run(output/'checkpoint_task_5_all_rounds.zip', roles_dir, data, evaluation)
    write_json(output/'candle_v2_completion.json', result)
    shutil.make_archive(str(evaluation), 'zip', root_dir=evaluation)
    print('CANDLE V2 completed:', json.dumps(result['metrics']), flush=True)


if __name__ == '__main__':
    main()
