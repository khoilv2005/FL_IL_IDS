"""Exploratory local run on cached test; leaves original artifacts untouched."""
import importlib.util
import sys
import zipfile
from pathlib import Path

root = Path(__file__).resolve().parents[1]
out = root/'output/router_diagnostic'
source = out/'original_source'
source.mkdir(parents=True, exist_ok=True)
with zipfile.ZipFile(root/'output/kaggle_v59_resume/denice_v59_exact_source.zip') as z:
    z.extractall(source)
sys.path.insert(0, str(source))
import numpy as np
import torch
spec = importlib.util.spec_from_file_location('diagnostic', root/'tools/diagnose_cluster_servers.py')
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
train = Path(r'E:\Study\DACN_KLTN\Results\100-clients\DENICE\Train\Task45\results (3)\results_denice_cluster_server_seed_42')
cache = Path(r'E:\Study\DACN_KLTN\Results\100-clients\DENICE\Eval\results (2)\eval_cluster_servers\test_cache\test')
import json, hashlib
provenance = json.loads((train/'execution_provenance.json').read_text())
assert all(hashlib.sha256((source/p).read_bytes()).hexdigest()==h for p,h in provenance['source_sha256'].items())
(out/'runtime.json').write_text(json.dumps(dict(torch=torch.__version__, expected=provenance['torch'],
    exploratory=True, source_hashes_verified=True, device=torch.cuda.get_device_name(0))))
torch.set_num_threads(4)
mod.diagnose(train, np.load(cache/'X_test.npy', mmap_mode='r'), np.load(cache/'y_test.npy', mmap_mode='r'),
    out/'results', batch_size=512, device='cuda:0', per_class=64)
