"""Run after train_incremental_kaggle.py: FULL TEST for every final temporary server."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import zipfile

# Paste the entire file into ONE Kaggle cell. Enable Internet and add 100-clients Input.
EVAL_DRIVE_URL = "https://drive.google.com/file/d/1ng8AwBvAW2wy7VDRMAWY1QktoQNu21uo/view?usp=sharing"  # Dan link FILE ZIP output train da hoan tat vao day.
DATA_DIR = '/kaggle/input/datasets/khoilv2005/100-clients/100-clients'
OUTPUT_DIR = '/kaggle/working/eval_cluster_servers'
EVAL_BATCH_SIZE = 8192
DIAGNOSTIC = True  # True: compare routing with known-task inference; diagnostic only.
DIAGNOSTIC_PER_CLASS = 64


def extract_output(archive_path, destination):
    root = Path(destination).resolve()
    keep = {'completion.json', 'execution_provenance.json', 'protocol_lock.json',
            'checkpoint_task_5_all_rounds.zip', 'temporary_servers_task_5.json', 'denice_cluster_server_source.zip'}
    with zipfile.ZipFile(archive_path) as archive:
        for entry in archive.infolist():
            relative = Path(entry.filename.replace('\\', '/'))
            target = (root/relative).resolve()
            if not target.is_relative_to(root):
                raise ValueError('Unsafe output ZIP path')
            if entry.is_dir() or relative.name not in keep:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(entry) as src, target.open('wb') as dst:
                shutil.copyfileobj(src, dst)
    candidates = [p.parent for p in root.rglob('completion.json')]
    if len(candidates) != 1:
        raise ValueError(f'ZIP must contain one completed run; found {len(candidates)}')
    return candidates[0]


def check_completed(root):
    path = Path(root)/'completion.json'
    if not path.is_file():
        raise ValueError('Missing completion.json: finish TRAIN Task 0-5 first')
    receipt = json.loads(path.read_text(encoding='utf-8'))
    if not receipt.get('completed') or receipt.get('tasks_completed') != 6:
        raise ValueError('Finish TRAIN Task 0-5 before evaluation')
    if receipt.get('checkpoint_format') != 'denice_v2_cluster_archive_v1':
        raise ValueError('Expected cluster-server output, not legacy checkpoints')
    if not (Path(root)/'checkpoint_task_5_all_rounds.zip').is_file():
        raise ValueError('Missing final checkpoint_task_5_all_rounds.zip')
    provenance = Path(root)/'execution_provenance.json'
    if not provenance.is_file():
        raise ValueError('Include execution_provenance.json in the output ZIP')
    return json.loads(provenance.read_text(encoding='utf-8'))


def ensure_eval_torch(provenance):
    """Probe a fresh interpreter; never rely on torch already imported by Jupyter."""
    expected = provenance['torch']

    def probe():
        result = subprocess.run(
            [sys.executable, '-c', 'import torch; print(torch.__version__)'],
            capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError('Cannot import torch in training interpreter:\n' + result.stderr)
        return result.stdout.strip().splitlines()[-1]

    actual = probe()
    print(f'Eval interpreter: {sys.executable}; torch installed={actual}; required={expected}', flush=True)
    if actual == expected:
        return
    if os.environ.get('DENICE_AUTO_INSTALL_TORCH', '1') != '1':
        raise RuntimeError(f'Install torch {expected} in {sys.executable}; currently {actual}')
    if expected != '2.10.0+cu128':
        raise RuntimeError(f'Automatic install is only configured for 2.10.0+cu128; required {expected}')
    print('Installing original PyTorch CUDA 12.8 packages before starting a fresh evaluation process...', flush=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '--no-cache-dir',
        'torch==2.10.0+cu128', 'torchvision==0.25.0+cu128', 'torchaudio==2.10.0+cu128',
        '--index-url', 'https://download.pytorch.org/whl/cu128'], check=True)
    actual = probe()
    if actual != expected:
        raise RuntimeError(f'PyTorch installation did not resolve mismatch: installed={actual}, required={expected}')
    print(f'Eval runtime verified: torch={actual}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', default=os.environ.get('DENICE_CHECKPOINT_ROOT'))
    parser.add_argument('--drive-url', default=os.environ.get('DENICE_EVAL_DRIVE_URL', EVAL_DRIVE_URL))
    parser.add_argument('--data-dir', default=os.environ.get('DENICE_DATA_DIR', DATA_DIR))
    parser.add_argument('--output', default=os.environ.get('DENICE_EVAL_OUTPUT_DIR', OUTPUT_DIR))
    parser.add_argument('--batch-size', type=int, default=EVAL_BATCH_SIZE)
    parser.add_argument('--device', default=None)
    parser.add_argument('--diagnostic', action='store_true', default=DIAGNOSTIC)
    parser.add_argument('--diagnostic-per-class', type=int, default=DIAGNOSTIC_PER_CLASS)
    # Pasted notebook cells inherit kernel flags, including Kaggle's Colab launcher.
    # Explicit main([...]) / runpy script arguments must still be parsed normally.
    if argv is None and Path(sys.argv[0]).name in ('ipykernel_launcher.py', 'colab_kernel_launcher.py'):
        argv = []
    args = parser.parse_args(argv)
    if args.batch_size < 1:
        parser.error('--batch-size must be positive')
    if args.diagnostic_per_class < 1:
        parser.error('--diagnostic-per-class must be positive')
    if bool(args.drive_url.strip()) == bool(args.checkpoint_root):
        parser.error('Fill EVAL_DRIVE_URL or provide --checkpoint-root; choose exactly one')
    if args.drive_url.strip():
        directory = Path(tempfile.mkdtemp(prefix='denice-eval-download-'))
        archive = directory/'output.zip'
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'gdown'], check=True)
        print('Downloading completed TRAIN output from Google Drive...', flush=True)
        subprocess.run([sys.executable, '-m', 'gdown', '--fuzzy', args.drive_url.strip(), '-O', str(archive)], check=True)
        if not zipfile.is_zipfile(archive):
            raise ValueError('Drive download is not a ZIP. Check link and download access.')
        args.checkpoint_root = str(extract_output(archive, directory/'extracted'))
    provenance = check_completed(args.checkpoint_root)
    if not (Path(args.data_dir)/'global_test_data.npz').is_file():
        raise FileNotFoundError('Set DATA_DIR to the directory containing global_test_data.npz')
    ensure_eval_torch(provenance)
    here = Path(__file__).resolve().parent if '__file__' in globals() else Path.cwd()
    local = os.environ.get('DENICE_CODE_DIR')
    if not local:
        saved_bundle = Path(args.checkpoint_root)/'denice_cluster_server_source.zip'
        bundle = Path(os.environ.get('DENICE_SOURCE_ZIP', str(saved_bundle if saved_bundle.is_file()
            else here/'denice_cluster_server_source.zip')))
        local = tempfile.mkdtemp(prefix='denice-cluster-eval-')
        if bundle.is_file():
            with zipfile.ZipFile(bundle) as archive:
                root = Path(local).resolve()
                if any(not (root/n).resolve().is_relative_to(root) for n in archive.namelist()):
                    raise ValueError('Unsafe source archive path')
                archive.extractall(root)
        else:
            revision = provenance.get('commit')
            if not isinstance(revision, str) or not re.fullmatch(r'[0-9a-fA-F]{40}', revision):
                raise ValueError('No original commit: supply DENICE_SOURCE_ZIP or DENICE_CODE_DIR')
            env = dict(os.environ, GIT_LFS_SKIP_SMUDGE='1')
            subprocess.run(['git', 'clone', 'https://github.com/khoilv2005/FL_IL_IDS.git', local], env=env, check=True)
            subprocess.run(['git', '-C', local, 'checkout', '--detach', revision], env=env, check=True)
    source = Path(local).resolve()
    if not provenance.get('source_sha256'):
        raise ValueError('Original source hashes are missing')
    for name, expected in provenance['source_sha256'].items():
        path = (source/name).resolve()
        if not path.is_relative_to(source) or not path.is_file():
            raise ValueError(f'Missing or invalid source path: {name}')
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f'Original source hash mismatch: {name}')
    print('Original source verified. ' + ('DIAGNOSTIC ONLY: paired stratified sample, not full-test metrics.'
        if args.diagnostic else 'Evaluating ALL final servers on FULL TEST.'), flush=True)
    # Fresh process avoids notebook torch/module caches after installing the original runtime.
    worker = '''import json, sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
import torch
from tools.eval_cluster_servers import evaluate_servers
args = json.loads(sys.argv[2])
assert torch.__version__ == args.pop('expected_torch'), 'Eval runtime mismatch'
args['device'] = args['device'] or ('cuda:0' if torch.cuda.is_available() else 'cpu')
result = evaluate_servers(**args)
print('Servers:', result['K'], 'FULL TEST rows/server:', result['unique_test_rows'], flush=True)
print('Mean server accuracy:', result['mean_cluster_accuracy'], flush=True)
print('Results:', Path(args['output_dir'])/'temporary_server_metrics.csv', flush=True)
'''
    settings = dict(checkpoint_root=str(Path(args.checkpoint_root).resolve()),
        data_dir=str(Path(args.data_dir).resolve()), output_dir=str(Path(args.output).resolve()),
        batch_size=args.batch_size, device=args.device, expected_torch=provenance['torch'])
    if args.diagnostic:
        settings['per_class'] = args.diagnostic_per_class
        settings['output_dir'] = str(Path(settings['output_dir'])/'diagnostics')
        worker = DIAGNOSTIC_SOURCE + DIAGNOSTIC_ENTRY
    subprocess.run([sys.executable, '-u', '-c', worker, str(source), json.dumps(settings)], cwd=source, check=True)


# BEGIN EMBEDDED DIAGNOSTIC
DIAGNOSTIC_SOURCE = 'import sys\nsys.path.insert(0, sys.argv[1])\n"""Read-only paired routing/oracle diagnostic; never an official benchmark."""\nimport json\nfrom pathlib import Path\n\nimport numpy as np\nimport torch\n\nfrom tools.eval_cluster_servers import completed_payload\nfrom fed_learning.strategies.decentralized.denice_v2.state import restore\nfrom fed_learning.strategies.decentralized.denice_v2.router import ClusterRouter\nfrom fed_learning.strategies.decentralized.denice_v2.evaluation import metrics, atomic_json\nfrom fed_learning.strategies.decentralized.denice_v2.contracts import state_hash\n\n\ndef masked_predictions(logits, tasks, registered):\n    """Missing task/classes count as failure, never silently drop those rows."""\n    result = np.full(len(tasks), -1, dtype=np.int64)\n    for task in np.unique(tasks):\n        labels = registered.get(int(task), ())\n        rows = np.flatnonzero(tasks == task)\n        if len(labels):\n            labels = np.asarray(labels, dtype=np.int64)\n            result[rows] = labels[logits[np.ix_(rows, labels)].argmax(axis=1)]\n    return result\n\n\ndef diagnose(checkpoint_root, x, y, output_dir, *, batch_size=1024, device=\'cpu\',\n             per_class=256, seed=42, server_ids=None):\n    if batch_size < 1 or per_class < 1:\n        raise ValueError(\'Positive batch size and per-class sample count required\')\n    payload = completed_payload(checkpoint_root)\n    rng = np.random.default_rng(seed)\n    indices = np.sort(np.concatenate([rng.choice(np.flatnonzero(y == c),\n        min(per_class, int(np.sum(y == c))), replace=False) for c in range(34)]))\n    if not len(indices):\n        raise ValueError(\'Empty diagnostic sample\')\n    truth = np.asarray(y[indices], np.int64)\n    inputs = np.asarray(x[indices], np.float32)\n    if not np.isfinite(inputs).all():\n        raise ValueError(\'Nonfinite diagnostic inputs\')\n    output = Path(output_dir)\n    output.mkdir(parents=True, exist_ok=True)\n    binding = state_hash(dict(indices=indices, x=inputs, y=truth))\n    rows = []\n    for uid in sorted(payload[\'active\']):\n        state = restore(payload[\'states\'][uid])\n        if state.router_state[\'router_mode\'] == \'routerless_v1\':\n            raise ValueError(\'This checkpoint predicts without a task router. Set DIAGNOSTIC=False for full evaluation.\')\n        if server_ids is not None and state.coordinator_id not in server_ids:\n            continue\n        predictor = ClusterRouter(state, device=device)\n        registered = state.router_state[\'episode_classes\']\n        cms = {k: np.zeros((34, 35), np.int64) for k in (\'routed\', \'oracle\')}\n        route = np.zeros((6, 7), np.int64)\n        zero_bits = ties = 0\n        for start in range(0, len(truth), batch_size):\n            target = truth[start:start+batch_size]\n            true_tasks = np.minimum(target // 6, 5)\n            with torch.inference_mode():\n                logits, acts = predictor.model.get_output_and_context_activations(\n                    torch.from_numpy(inputs[start:start+batch_size]).to(device))\n            raw = logits.cpu().numpy()\n            bits = predictor.detector.binarize_layer_activations(\n                {k: v.cpu().numpy() for k, v in acts.items()})\n            tasks, scores = predictor.detector.predict_episodes_with_scores(bits)\n            if not np.isfinite(raw).all() or not np.isfinite(scores).all():\n                raise ValueError(\'Nonfinite predictions\')\n            zero_bits += int(np.sum(~np.any(bits, axis=1)))\n            ties += int(np.sum(np.sum(scores == scores.max(axis=1, keepdims=True), axis=1) > 1))\n            predictions = dict(routed=masked_predictions(raw, tasks, registered),\n                               oracle=masked_predictions(raw, true_tasks, registered))\n            for key, pred in predictions.items():\n                np.add.at(cms[key], (target, np.where(pred < 0, 34, pred)), 1)\n            np.add.at(route, (true_tasks, np.where(tasks < 0, 6, tasks)), 1)\n        memory = {}\n        for task, bank in state.router_state[\'activation_memory\'].items():\n            bank = np.asarray(bank)\n            memory[int(task)] = dict(rows=len(bank), zero_rows=int(np.sum(~np.any(bank, axis=1))),\n                unique_rows=int(len(np.unique(bank, axis=0))), classes=list(registered.get(task, ())))\n        result = dict(cluster_uid=uid, temporary_server=state.coordinator_id, members=list(state.members),\n            sample_binding=binding, checkpoint_state=state.fingerprint(), diagnostic_only=True,\n            oracle_uses_true_task=True, samples=len(truth), memory=memory,\n            zero_query_fraction=zero_bits/len(truth), score_tie_fraction=ties/len(truth),\n            router_accuracy=float(np.trace(route[:, :6])/len(truth)), router_confusion=route.tolist(),\n            routed=metrics(cms[\'routed\'], range(34)), oracle=metrics(cms[\'oracle\'], range(34)))\n        result[\'accuracy_gain\'] = result[\'oracle\'][\'accuracy\'] - result[\'routed\'][\'accuracy\']\n        atomic_json(output/f\'diagnostic_{uid}.json\', result)\n        rows.append(result)\n        print(f"Diagnostic server {state.coordinator_id}: routed={result[\'routed\'][\'accuracy\']:.4f}, "\n              f"oracle={result[\'oracle\'][\'accuracy\']:.4f}, router={result[\'router_accuracy\']:.4f}", flush=True)\n        del predictor\n    if not rows:\n        raise ValueError(\'No selected servers found\')\n    summary = dict(diagnostic_only=True, sampling=\'fixed stratified TEST subset; not full-test benchmark\',\n        runtime=dict(torch=torch.__version__, device=device),\n        seed=seed, per_class=per_class, samples=len(truth), sample_binding=binding,\n        servers=len(rows), protocol_lock=payload[\'lock\'],\n        mean_routed_accuracy=float(np.mean([r[\'routed\'][\'accuracy\'] for r in rows])),\n        mean_oracle_accuracy=float(np.mean([r[\'oracle\'][\'accuracy\'] for r in rows])),\n        mean_router_accuracy=float(np.mean([r[\'router_accuracy\'] for r in rows])), results=rows)\n    atomic_json(output/\'diagnostic_summary.json\', summary)\n    return summary\n'
DIAGNOSTIC_ENTRY = "\nfrom types import SimpleNamespace\nfrom fed_learning.strategies.decentralized.denice_v2.data import TaskData\nargs = json.loads(sys.argv[2])\nassert torch.__version__ == args.pop('expected_torch'), 'Eval runtime mismatch'\nargs['device'] = args['device'] or ('cuda:0' if torch.cuda.is_available() else 'cpu')\npayload = completed_payload(args['checkpoint_root'])\nstate = restore(payload['states'][next(iter(payload['active']))])\nsource = SimpleNamespace(dataset=Path(args.pop('data_dir')), cache=Path(args['output_dir']).parent/'test_cache',\n    input_shape=tuple(state.algorithm_state['native_input_shape']))\nx, y, _ = TaskData.test(source)\ndel payload, state\nresult = diagnose(x=x, y=y, **args)\nprint({k:v for k,v in result.items() if k != 'results'}, flush=True)\n"

if __name__ == '__main__':
    main()
