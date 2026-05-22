import pytest
import torch

from fed_learning.training.task_loop import _run_tracked_rounds, run_incremental_training


class _FakeServer:
    def __init__(self):
        self.evaluate_calls = 0

    def evaluate_global(self, **kwargs):
        self.evaluate_calls += 1
        return {
            "loss": 0.5,
            "accuracy": 0.25,
            "precision_macro": 0.2,
            "recall_macro": 0.3,
            "f1_macro": 0.24,
            "f1_weighted": 0.26,
        }

    def get_global_params(self):
        return {"weight": torch.tensor([1.0])}


def test_run_tracked_rounds_respects_eval_every(tmp_path):
    server = _FakeServer()
    history = {"round_metrics": []}

    _run_tracked_rounds(
        server=server,
        train_round_fn=lambda local_round: {
            "train_loss": 1.0 + local_round,
            "round_time": 2.0,
        },
        total_rounds=3,
        task_id=0,
        output_dir=str(tmp_path),
        history=history,
        all_test_data={},
        best_acc_per_task={},
        trainer=object(),
        config={"eval_every": 5, "round_checkpoint_every": None},
        seen_classes=[0, 1],
        is_last_task=False,
    )

    assert server.evaluate_calls == 0
    assert [record["evaluated"] for record in history["round_metrics"]] == [
        False,
        False,
        False,
    ]
    assert history["round_metrics"][0]["accuracy"] is None
    assert history["round_metrics"][-1]["accuracy"] is None

def test_plexus_rejected_outside_decentralized_mode():
    with pytest.raises(ValueError, match="mode='decentralized'"):
        run_incremental_training({"mode": "fed_il", "algorithm": "plexus"})
