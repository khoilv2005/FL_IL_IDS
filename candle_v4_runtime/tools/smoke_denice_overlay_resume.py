"""Exercise real runner saves and both overlay continuation paths on synthetic data."""
import contextlib
import json
from pathlib import Path
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from tools.benchmark_denice_incremental import synthetic_data
from fed_learning.training.decentralized_denice_il import run_decentralized_denice_il


def main():
    root = Path(__file__).resolve().parents[1]
    output = Path(tempfile.mkdtemp(prefix='overlay_resume_', dir=root / 'output'))
    torch.set_num_threads(1)
    config = synthetic_data(output / 'data', 42)
    config.update(json.loads((root / 'configs/denice_appliance.json').read_text()))
    config.update(seed=42, random_seed=42, task_end=1, output_dir=str(output / 'initial'),
                  denice_eval_report_nomask=False, denice_eval_representative_ensemble=False,
                  denice_router_replay_enabled=True, denice_router_replay_per_class=8,
                  denice_amp_enabled=False)
    with (output / 'training.log').open('w', encoding='utf-8') as log, contextlib.redirect_stdout(log):
        result = run_decentralized_denice_il(dict(config))
        checkpoint = Path(result['output_dir']) / 'continuation_state_task_0.pt'
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        assert state['denice_appliance_state']['version'] == 1
        legacy = output / 'legacy_task_0.pt'
        del state['denice_appliance_state']
        torch.save(state, legacy)
        for name, source in [('saved', checkpoint), ('rebuilt', legacy)]:
            run = {**config, 'task_start': 1, 'resume_state_path': str(source),
                   'output_dir': str(output / name), 'resume_output_dir': str(output / name),
                   'denice_overlay_rebuild_private_state': name == 'rebuilt'}
            result = run_decentralized_denice_il(run)
            final = torch.load(Path(result['output_dir']) / 'continuation_state_task_1.pt',
                               map_location='cpu', weights_only=False)
            assert final['meta']['completed_task'] == 1
            assert final['denice_appliance_state']['banks']
            rounds = [json.loads(line) for line in (Path(result['output_dir']) / 'appliance_rounds.jsonl').read_text().splitlines()]
            assert {r['task'] for r in rounds} == {1}, 'Completed task must not retrain'
    summary = {'status': 'passed', 'data': 'synthetic_non_IDS',
               'paths': ['serialized_private_state', 'legacy_private_state_reconstruction'],
               'output': str(output)}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
