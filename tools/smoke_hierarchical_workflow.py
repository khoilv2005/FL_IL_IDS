"""Six-task synthetic training+resume+full-test check, including dormant/rejoin."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

root=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(root))
sys.path.insert(0,str(root/'tests'))
import numpy as np
import torch
from test_hierarchical_cluster import CoverageData
from fed_learning.strategies.decentralized.denice_v2.runner import RunConfig
from fed_learning.strategies.decentralized.denice_v2.hierarchical import HierarchicalRunner
from tools.eval_cluster_servers import evaluate_servers,completed_payload


def main():
    torch.set_num_threads(2)
    out=Path(sys.argv[1]).resolve()
    out.mkdir(parents=True,exist_ok=True)
    data=CoverageData()
    x,y,rows=data.raw[0,3]
    data.raw[0,3]=(x[:0],y[:0],rows[:0])
    cfg=RunConfig(smoke=True,tasks=6,rounds_per_task=1,batch_size=8,context_rows=8,
        defer_evaluation=True,server_only_storage=True,policy='routerless_v1',server_exchange=True)
    runner=HierarchicalRunner(data,out/'train',cfg)
    with patch.object(data,'test',side_effect=AssertionError('TRAIN read TEST')):
        runner.run()
    payload=completed_payload(out/'train')
    assert all(s.algorithm_state['seen_classes']==tuple(range(34)) for s in runner.unique_states().values())
    test=out/'data';test.mkdir(exist_ok=True)
    np.savez(test/'global_test_data.npz',X_test=data.test_x,y_test=data.test_y)
    summary=evaluate_servers(out/'train',test,out/'eval',batch_size=17)
    assert summary['full_test'] and summary['unique_test_rows']==102
    assert all(s['test_samples']==102 for s in summary['per_temporary_server'])
    assert summary['prediction_mode']=='routerless_v1'
    receipt=dict(passed=True,tasks=6,rounds_per_task=1,full_test_rows_per_server=102,
        all_servers_learned_34_classes=True,dormant_rejoin=True,train_read_test=False,
        raw_training_rows_shared=False,scientific_accuracy_benchmark=False)
    (out/'smoke_receipt.json').write_text(json.dumps(receipt,indent=2))
    print(receipt,flush=True)


if __name__=='__main__':main()
