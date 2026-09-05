"""Fresh-process checks for trusted checkpoints created by the local probe."""
import argparse
import json
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from eval_checkpoint import _make_denice_client_model
from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--observed', type=Path, required=True)
    parser.add_argument('--control', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    first = json.loads((args.observed/'transitions.json').read_text())
    second = json.loads((args.control/'transitions.json').read_text())
    assert first['train_source_indices'] == second['train_source_indices']
    assert first['test_source_indices'] == second['test_source_indices']
    cid = first['client_id']
    a = torch.load(args.observed/'training/checkpoint_task_1.pt', map_location='cpu', weights_only=False)
    b = torch.load(args.control/'training/checkpoint_task_1.pt', map_location='cpu', weights_only=False)
    state_a, state_b = a['client_model_states'][cid], b['client_model_states'][cid]
    assert state_a.keys() == state_b.keys()
    max_difference = max(float((state_a[k].double()-state_b[k].double()).abs().max()) for k in state_a)
    assert max_difference == 0
    ma, da = _make_denice_client_model(a, cid, 'cpu')
    mb, db = _make_denice_client_model(b, cid, 'cpu')
    with np.load(args.control/'sampled_panel.npz', allow_pickle=False) as panel:
        x, y = torch.from_numpy(panel['X_test']), torch.from_numpy(panel['y_test'])
    results = {}
    for policy in ['backbone_nomask','oracle_hard','pred_hard']:
        la, _ = _denice_routed_logits_with_episodes(ma,x,da,list(range(12)),'cpu',
                   inference_policy=policy, oracle_episodes=y.numpy()//6)
        lb, _ = _denice_routed_logits_with_episodes(mb,x,db,list(range(12)),'cpu',
                   inference_policy=policy, oracle_episodes=y.numpy()//6)
        error = float((la-lb).abs().max())
        assert error == 0
        pred = la.argmax(1).numpy()
        acc = {}
        for task in [0,1]:
            mask = (y.numpy()//6 == task) & np.isin(y.numpy(), first['train_support'])
            value = float(np.mean(pred[mask] == y.numpy()[mask]))
            expected = first['records'][-1]['metrics'][policy][f'task{task}_local_support']['accuracy']
            assert value == expected
            acc[str(task)] = value
        results[policy] = {'observed_control_logit_max_error':error, 'fresh_process_supported_accuracy':acc}
        if policy == 'backbone_nomask':
            old = y < 6
            gap = la[old,6:12].max(1).values-la[old,:6].max(1).values
            results[policy]['old_samples_predicted_as_new_task_fraction'] = float(np.mean(pred[old.numpy()] >= 6))
            results[policy]['new_minus_old_max_logit_median_on_old_samples'] = float(gap.median())
    report = {'same_sample_indices': True, 'observed_control_tensor_max_error': max_difference,
              'policies': results, 'live_accuracy_matches_fresh_process_checkpoint': True}
    (args.observed/'verification.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__ == '__main__':
    main()
