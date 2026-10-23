"""Run the frozen full audit and native replay, teeing stderr as visible output.

This avoids PowerShell treating a harmless sklearn compatibility warning as a
NativeCommandError while still retaining the warning and actual child exit code.
Inputs must already be staged by prepare_denice_bottleneck_inputs.py.
"""
import argparse
import subprocess
import sys
from pathlib import Path


def execute(python, script, arguments, log, root):
    command = [str(python), '-u', str(root / 'tools' / script), *map(str, arguments)]
    print(f'Running {script}; log={log}', flush=True)
    with log.open('w', encoding='utf-8') as destination:
        with subprocess.Popen(command, cwd=root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, encoding='utf-8', errors='replace', bufsize=1) as process:
            for line in process.stdout:
                destination.write(line)
                destination.flush()
                print(line, end='', flush=True)
            code = process.wait()
    if code:
        raise RuntimeError(f'{script} exited with {code}; inspect {log}')


def main():
    root = Path(__file__).resolve().parents[1]
    legacy_inputs = root / 'audit_denice/appliance_legacy_results11/inputs'
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--python', default=str(root / '.venv-denice-gpu/Scripts/python.exe'))
    parser.add_argument('--checkpoint', default=str(legacy_inputs / 'checkpoint_task_5_all_rounds.zip'))
    parser.add_argument('--legacy', default=str(legacy_inputs / 'legacy_self_task_5'))
    parser.add_argument('--roles', default=str(legacy_inputs / 'roles'))
    parser.add_argument('--inputs', default=str(root / 'audit_denice/denice_bottleneck_20261010/inputs'))
    parser.add_argument('--out', required=True)
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args()
    if not Path(args.python).is_file():
        parser.error('Set --python to a Python environment with CUDA Torch and the audit dependencies')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=True)
    common = ['--checkpoint', args.checkpoint, '--legacy', args.legacy, '--inputs', args.inputs]
    full = out / 'full_cuda'
    execute(args.python, 'audit_denice_bottleneck.py',
            common + ['--roles', args.roles, '--out', full, '--device', args.device, '--batch-size', 512],
            out / 'full_cuda.log', root)
    execute(args.python, 'verify_denice_bottleneck_audit.py',
            ['--audit-dir', full, '--legacy-dir', args.legacy], out / 'verification.log', root)
    execute(args.python, 'audit_denice_class_mask_variant.py',
            common + ['--audit-dir', full, '--out', out / 'class_mask_native_replay.json', '--device', args.device],
            out / 'class_mask_native_replay.log', root)
    print(f'Completed: {full / "completion.json"}', flush=True)


if __name__ == '__main__':
    main()
