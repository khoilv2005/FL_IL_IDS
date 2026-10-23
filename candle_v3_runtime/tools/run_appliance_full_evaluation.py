"""Recover the final full evaluation from a completed APPLIANCE training output.

No training, discovery, CAL reads, threshold changes or certificate rebinding.
An outer Kaggle output ZIP is selectively extracted, avoiding all six tasks.
"""
import gc
import hashlib
import json
import shutil
import zipfile
from pathlib import Path, PurePosixPath

from fed_learning.data.denice_clean_roles import file_sha256
from fed_learning.training.denice_delta_checkpoint import load_denice_checkpoint
from tools.eval_denice_legacy_self import resolve_evaluation_device, run_legacy_self


TASK_ARCHIVE = 'checkpoint_task_5_all_rounds.zip'


def verify_terminal(checkpoint):
    """Check the sealed full terminal, never substitute a reconstructed delta."""
    checkpoint = Path(checkpoint)
    if checkpoint.suffix.lower() == '.zip':
        with zipfile.ZipFile(checkpoint) as archive:
            manifest = json.loads(archive.read('checkpoint_archive_manifest.json'))
            terminal = manifest.get('full_terminal_checkpoint')
            if (manifest.get('task_id') != 5 or manifest.get('completed_rounds') != list(range(20))
                    or not terminal or terminal not in manifest.get('checksums', {})):
                raise ValueError('Expected sealed Task5 archive with 20 rounds and an exact full terminal')
            digest = hashlib.sha256()
            with archive.open(terminal) as source:
                for block in iter(lambda: source.read(8 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != manifest['checksums'][terminal]:
                raise ValueError('Full terminal checkpoint checksum differs from the task seal')
    elif (checkpoint.parent / 'checkpoint_archive_manifest.json').is_file():
        manifest = json.loads((checkpoint.parent / 'checkpoint_archive_manifest.json').read_text(encoding='utf-8'))
        if (manifest.get('task_id') != 5 or manifest.get('completed_rounds') != list(range(20)) or
                manifest.get('full_terminal_checkpoint') != checkpoint.name or
                file_sha256(checkpoint) != manifest['checksums'].get(checkpoint.name)):
            raise ValueError('Extracted full terminal differs from the original task archive seal')


def select_inputs(results, workspace, role_dir=None):
    results, workspace = Path(results), Path(workspace)
    if not results.exists():
        raise FileNotFoundError(f'Training output does not exist: {results}')
    role_roots = []
    if results.is_dir():
        archives = list(results.rglob(TASK_ARCHIVE))
        if not archives:
            # Kaggle can unpack the task archive itself. Use its exact full
            # terminal member, not the lossy round-19 delta checkpoint.
            for path in results.rglob('checkpoint_archive_manifest.json'):
                manifest = json.loads(path.read_text(encoding='utf-8'))
                terminal = manifest.get('full_terminal_checkpoint')
                if terminal and 'task_5' in Path(terminal).name:
                    archives.append(path.parent / terminal)
        role_roots = [p.parent for p in results.rglob('role_manifest.json')]
    elif results.suffix.lower() == '.zip':
        with zipfile.ZipFile(results) as source:
            if 'checkpoint_archive_manifest.json' in source.namelist():
                if results.name != TASK_ARCHIVE:
                    raise ValueError(f'Use the completed {TASK_ARCHIVE}, not a round-only archive')
                archives = [results]
            else:
                members = [n for n in source.namelist() if PurePosixPath(n).name == TASK_ARCHIVE]
                if len(members) != 1:
                    raise ValueError(f'Expected one {TASK_ARCHIVE} in {results}; found {members}')
                workspace.mkdir(parents=True, exist_ok=False)
                target = workspace / TASK_ARCHIVE
                with source.open(members[0]) as incoming, target.open('wb') as outgoing:
                    shutil.copyfileobj(incoming, outgoing, length=8 * 1024 * 1024)
                archives = [target]
                # Evaluation uses only the original locked manifest and lock;
                # no raw BASE/CAL or role indices are opened by this evaluator.
                names = set(source.namelist())
                for number, member in enumerate(sorted(names)):
                    if PurePosixPath(member).name != 'role_manifest.json':
                        continue
                    sibling = str(PurePosixPath(member).parent / 'role_lock.json')
                    if sibling not in names:
                        continue
                    root = workspace / f'roles_{number}'
                    root.mkdir()
                    for original in (member, sibling):
                        (root / PurePosixPath(original).name).write_bytes(source.read(original))
                    role_roots.append(root)
    else:
        # Explicit full continuation is also allowed, with task/round checks.
        archives = [results]
    if len(archives) != 1 or not archives[0].is_file():
        raise ValueError(f'Expected one final Task5 checkpoint; found {archives}')
    checkpoint = archives[0]
    verify_terminal(checkpoint)
    ckpt = load_denice_checkpoint(str(checkpoint))
    config = ckpt['config']
    meta = ckpt.get('meta', {})
    task = ckpt.get('task', ckpt.get('task_id', meta.get('completed_task')))
    final_round = ckpt.get('final_round_id', ckpt.get('round_id'))
    if final_round is None and meta.get('boundary') == 'task':
        final_round = int(config.get('rounds_per_task', 0)) - 1
    if (task != 5 or final_round != 19 or
            config.get('denice_cl_method', 'legacy') != 'legacy' or
            not config.get('appliance_enabled') or
            config.get('appliance_scope_mode') != 'appliance_empirical_current_CAL_v2'):
        raise ValueError('Expected completed empirical DeNICE + APPLIANCE Task5/round19; '
                         'do not use results9 or the original legacy-only checkpoint')
    # Fail before fitting routers or reading the test set when GPU/Torch differ.
    device = resolve_evaluation_device(ckpt, 'auto', include_appliance=True)
    expected = config['denice_data_roles_sha256']
    del ckpt
    gc.collect()
    if role_dir:
        role_roots = [Path(role_dir)]
    elif not role_roots:
        role_roots = [p.parent for p in results.parent.rglob('role_manifest.json')]
    matching = [root for root in role_roots if (root / 'role_lock.json').is_file()
                and file_sha256(root / 'role_manifest.json') == expected]
    if not matching:
        raise FileNotFoundError('Add the original denice_clean_roles/role_manifest.json and '
                                'role_lock.json from this training output, or set DENICE_CLEAN_ROLES_DIR. '
                                'Do not regenerate or edit the split.')
    # Identical copies are interchangeable; the evaluator validates the lock.
    return checkpoint, sorted(matching)[0], device


def run(results, data_dir, output_dir, workspace, role_dir=None, batch_size=512):
    checkpoint, roles, device = select_inputs(results, workspace, role_dir)
    print(f'Eval only: checkpoint={checkpoint}; roles={roles}; backend={device}', flush=True)
    return run_legacy_self(checkpoint, roles, output_dir, data_dir, device='auto',
                           batch_size=batch_size, expected_xi=.8, include_appliance=True)
