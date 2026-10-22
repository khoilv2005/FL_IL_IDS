"""Run after train_incremental_kaggle.py: FULL TEST for every final temporary server."""
import argparse
import os
from pathlib import Path
import sys
import tempfile
import zipfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', default=os.environ.get('DENICE_CHECKPOINT_ROOT'), required=not bool(os.environ.get('DENICE_CHECKPOINT_ROOT')))
    parser.add_argument('--data-dir', default=os.environ.get('DENICE_DATA_DIR', '/kaggle/input/datasets/khoilv2005/100-clients/100-clients'))
    parser.add_argument('--output', default=os.environ.get('DENICE_EVAL_OUTPUT_DIR', '/kaggle/working/eval_cluster_servers'))
    parser.add_argument('--batch-size', type=int, default=8192)
    parser.add_argument('--device', default=None)
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    local = os.environ.get('DENICE_CODE_DIR')
    if not local and (here/'fed_learning').is_dir(): local = str(here)
    if not local:
        bundle = Path(os.environ.get('DENICE_SOURCE_ZIP', str(here/'denice_cluster_server_source.zip')))
        local = tempfile.mkdtemp(prefix='denice-cluster-eval-')
        with zipfile.ZipFile(bundle) as archive:
            root = Path(local).resolve()
            if any(not (root/n).resolve().is_relative_to(root) for n in archive.namelist()):
                raise ValueError('Unsafe source archive path')
            archive.extractall(root)
    sys.path.insert(0, local)
    import torch
    from tools.eval_cluster_servers import evaluate_servers
    result = evaluate_servers(args.checkpoint_root, args.data_dir, args.output,
        batch_size=args.batch_size, device=args.device or ('cuda:0' if torch.cuda.is_available() else 'cpu'))
    print('Servers:', result['K'], 'FULL TEST rows/server:', result['unique_test_rows'])
    print('Mean server accuracy:', result['mean_cluster_accuracy'])
    print('Results:', Path(args.output)/'temporary_server_metrics.csv')


if __name__ == '__main__':
    main()
