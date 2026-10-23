"""Cheap causal probes; no training data, production model changes or tuning."""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
import numpy as np
from fed_learning.models.denice_model import DeNICEModel
from fed_learning.servers.nice_server import ContextDetector


def recurrent_isolation_probe():
    """Fixed mature gate rows need not imply fixed mature GRU features."""
    torch.manual_seed(73)
    model = DeNICEModel((39, 1), 4).eval()
    dense = deepcopy(model.gru)
    isolated = deepcopy(dense)
    mature = torch.arange(100) < 50
    gate_mature = mature.repeat(3)
    # Diagnostic intervention only, NOT a proposed drop-in production repair.
    with torch.no_grad():
        for name, p in isolated.named_parameters():
            if name.startswith('weight_hh') or name == 'weight_ih_l1':
                p[gate_mature[:, None] & (~mature)[None, :]] = 0
    x = torch.randn(8, 39, 1)
    output = {}
    for label, gru in [('dense', dense), ('isolated_control', isolated)]:
        before_params = deepcopy(gru.state_dict())
        with torch.no_grad():
            before = gru(x)[0][:, -1, mature].clone()
            for p in gru.parameters():
                p[~gate_mature] += 0.2
            after = gru(x)[0][:, -1, mature]
        output[label] = {
            'mature_gate_parameters_unchanged': all(
                torch.equal(p[gate_mature], before_params[name][gate_mature])
                for name, p in gru.named_parameters()),
            'mature_feature_max_abs_change': (after-before).abs().max().item(),
        }
    return output


def stale_router_probe():
    """Holdout is only scored, never used by push/refresh/fit."""
    torch.manual_seed(73)
    model = DeNICEModel((39, 1), 4).eval()
    detector = ContextDetector(router_mode='multiclass')
    detector.episode_classes = {0: [0, 1], 1: [2, 3]}
    refs = [torch.randn(40, 39, 1)+ep*2 for ep in range(2)]
    holdout = torch.cat([torch.randn(40, 39, 1)+ep*2 for ep in range(2)])
    targets = np.repeat([0, 1], 40)
    for ep, bank in enumerate(refs):
        detector.push_activations(model, bank, ep, reference_data=bank)
    detector.train_models(1)
    def accuracy():
        features = detector._binarize_per_sample(model, holdout)
        return float(np.mean(detector.predict_episodes_batch(features) == targets))
    result = {'before_perturbation': accuracy()}
    old_features = detector.activation_memory[0].copy()
    with torch.no_grad():
        model.conv1.weight.add_(0.5)
    detector.mark_router_stale('controlled_backbone_perturbation')
    result['after_perturbation_before_refresh'] = accuracy()
    detector.refresh_activation_memory(model, task_id=1, round_id=0)
    result['after_train_reference_refresh'] = accuracy()
    result['old_episode_sketch_fraction_changed'] = float(np.mean(old_features != detector.activation_memory[0]))
    result['reference_samples'] = sum(len(a) for a in detector.reference_input_memory.values())
    result['holdout_samples'] = len(holdout)
    return result


def checkpoint_metadata_probe(archive):
    """Only for a trusted local checkpoint: torch pickle loading executes code."""
    from zipfile import ZipFile
    with ZipFile(archive) as z:
        member = 'denice_full_d2_seed_42/terminal_task_5/checkpoint_task_5_round_19.pt'
        with z.open(member) as stream:
            ckpt = torch.load(stream, map_location='cpu', weights_only=False)
    clients = ckpt['client_algorithm_states']
    states = {cid: a.get('denice', a) for cid,a in clients.items()}
    first = next(iter(states.values()))
    return {
        'archive': str(archive), 'member': member,
        'client_count': len(states),
        'clients_with_connection_masks': sum('connection_masks' in a for a in states.values()),
        'clients_with_bn_frozen_state': sum('bn_frozen_state' in a for a in states.values()),
        'first_client_state_keys': list(first),
        'first_client_raw_reference_samples': sum(len(v) for v in first['context_detector'].get('reference_input_memory', {}).values()),
        'first_client_adapter_count': len(first.get('adapter_registry', {})),
        'task_id': ckpt['task_id'], 'round_id': ckpt['round_id'],
        'training_seed': ckpt['config'].get('random_seed'),
    }


def two_task_client_probe():
    """Local lifecycle integration, not a full CANC/cluster benchmark."""
    from fed_learning.clients.denice_client import DeNICEClient
    from fed_learning.strategies.incremental.denice import DeNICETrainer
    from fed_learning.strategies.incremental.nice import increase_unit_ranks, update_freeze_masks
    from fed_learning.training.decentralized_denice_il import _enforce_minimum_free_capacity
    from fed_learning.training.denice_eval import _denice_routed_logits_with_episodes
    torch.manual_seed(73)
    model = DeNICEModel((39, 1), 4)
    detector = ContextDetector(router_mode='multiclass')
    train = [torch.randn(16, 39, 1)*0.3 + (2*c-3) for c in range(4)]
    holdout = [torch.randn(16, 39, 1)*0.3 + (2*c-3) for c in range(4)]
    trainer = DeNICETrainer(max_phases=2, phase_epochs=1, tau=0.95)
    records = []
    for task in range(2):
        classes = [2*task, 2*task+1]
        model.clear_active_adapters()
        # Fixed adapter fixture exercises lifecycle; does not simulate CANC.
        model.add_adapter(task, 'fc1')
        for cls in classes:
            model.unit_ranks['fc2'][cls] = 1
        detector.episode_classes[task] = classes
        start = model.get_neuron_ages_state()
        x = torch.cat([train[c] for c in classes])
        y = torch.tensor([c for c in classes for _ in range(16)])
        client = DeNICEClient(17, x, y, max_phases=2, phase_epochs=1)
        client.setup_for_gpu(model, 'cpu')
        losses = []
        for round_id in range(2):
            update_freeze_masks(model)
            result = client.train(trainer=trainer, epochs=1, batch_size=8, lr=0.001,
                                  max_phases_override=1, phase_offset=round_id,
                                  is_last_task=task == 1)
            losses.append(float(result['loss']))
            # No task-boundary aging is allowed inside a communication round.
            assert all(model.unit_ranks['fc2'][c] == 1 for c in classes)
            if task == 0:
                _enforce_minimum_free_capacity(model, start, 0.25)
        detector.push_activations(model, x, task, reference_data=x)
        detector.refresh_activation_memory(model, task_id=task, round_id=1)
        increase_unit_ranks(model)
        update_freeze_masks(model)
        model.freeze_bn_for_mature()
        assert all(model.unit_ranks['fc2'][c] == 2 for c in classes)
        matrix_row = []
        for old_task in range(task+1):
            old_classes = [2*old_task, 2*old_task+1]
            hx = torch.cat([holdout[c] for c in old_classes])
            hy = torch.tensor([c for c in old_classes for _ in range(16)])
            logits, _ = _denice_routed_logits_with_episodes(
                model, hx, detector, list(range(2*task+2)), 'cpu',
                inference_policy='oracle_hard', oracle_episodes=np.full(len(hx), old_task))
            matrix_row.append(float((logits.argmax(1) == hy).float().mean()))
        records.append({'task': task, 'losses': losses, 'oracle_accuracy_row': matrix_row,
                        'output_ages': model.unit_ranks['fc2'].tolist()})
    return {'client_id': 17, 'train_samples_per_class': 16, 'holdout_samples_per_class': 16,
            'rounds_per_task': 2, 'capacity_reserve_task0': 0.25, 'records': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=Path('audit_denice/invariant_evidence.json'))
    parser.add_argument('--trusted-seed42-archive', type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    result = {'seed': 73, 'torch': torch.__version__, 'gru_isolation': recurrent_isolation_probe(),
              'stale_router': stale_router_probe(), 'two_task_local_lifecycle': two_task_client_probe()}
    if args.trusted_seed42_archive:
        result['historical_checkpoint_state'] = checkpoint_metadata_probe(args.trusted_seed42_archive)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
