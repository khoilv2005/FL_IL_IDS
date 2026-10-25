"""Environment adapter for train_incremental_kaggle's cluster-server profile."""
import os
from pathlib import Path
import subprocess
import sys


def main():
    import torch
    root = Path(__file__).resolve().parents[1]
    if os.environ.get('DENICE_CONFIG_FILE') or os.environ.get('DENICE_CONFIG_OVERRIDES') or os.environ.get('DENICE_VARIANT', 'legacy') != 'legacy':
        raise ValueError('Cluster-server uses its own protocol. Remove legacy CONFIG_FILE/CONFIG_OVERRIDES/VARIANT overrides.')
    if os.environ.get('DENICE_RESUME_STATE_PATH'):
        raise ValueError('Legacy per-client .pt checkpoints cannot resume cluster-server training. '
                         'Use DENICE_CLUSTER_RESUME_ROOT/TASK for a cluster archive, or train fresh.')
    if os.environ.get('DENICE_TRAIN_PHASE', '5') != '5' or os.environ.get('DENICE_TASK_END', '5') != '5':
        raise ValueError('Cluster-server training runs Task 0-5. Use cluster resume options to continue.')
    if int(os.environ.get('DENICE_SEED', '42')) != 42:
        raise ValueError('The current cluster-server protocol is locked to seed 42.')
    data = Path(os.environ.get('DENICE_DATA_DIR',
        '/kaggle/input/datasets/khoilv2005/100-clients/100-clients'))
    if not (data/'metadata.json').is_file():
        candidates = [p.parent for p in data.rglob('metadata.json')
                      if list(p.parent.glob('client_*_train.npz'))]
        if len(candidates) != 1:
            raise ValueError('Set DENICE_DATA_DIR to the directory containing metadata.json and client TRAIN shards')
        data = candidates[0]
    device = os.environ.get('DENICE_DEVICE', 'cuda:0' if torch.cuda.is_available() else 'cpu')
    workers = os.environ.get('DENICE_WORKER_DEVICES',
        'cuda:0,cuda:1' if torch.cuda.device_count() >= 2 and device.startswith('cuda') else '')
    output = os.environ.get('DENICE_OUTPUT_DIR', '/kaggle/working/results_denice_cluster_server_seed_42')
    args = [sys.executable, '-u', str(root/'tools/train_denice_v2.py'),
        '--data-dir', str(data), '--output', output, '--device', device,
        '--worker-devices', workers, '--batch-size', os.environ.get('DENICE_BATCH_SIZE', '2048'),
        '--output-budget-gib', os.environ.get('DENICE_OUTPUT_BUDGET_GIB', '18'),
        '--defer-evaluation', '--server-only-storage',
        '--prediction-mode', os.environ.get('DENICE_PREDICTION_MODE', 'cnn_coordinate_router_v1')]
    resume_root = os.environ.get('DENICE_CLUSTER_RESUME_ROOT')
    if os.environ.get('DENICE_SERVER_EXCHANGE') == '1':
        args.append('--server-exchange')
    resume_task = os.environ.get('DENICE_CLUSTER_RESUME_TASK')
    resume_round = os.environ.get('DENICE_CLUSTER_RESUME_ROUND')
    if resume_root and resume_task is None or resume_round and resume_task is None:
        raise ValueError('Cluster resume needs DENICE_CLUSTER_RESUME_TASK')
    if resume_task is not None:
        args += ['--resume-task', resume_task]
        if resume_root: args += ['--resume-root', resume_root]
        if resume_round is not None: args += ['--resume-round', resume_round]
    print('TRAIN ONLY: one model/router per cluster, ephemeral local workers; Task 0-5, 20 rounds/task.', flush=True)
    print('TEST is not read. Run eval_selective after completion.json reports completed=true.', flush=True)
    subprocess.run(args, cwd=root, check=True)


if __name__ == '__main__':
    main()
