import json
import numpy as np
from pathlib import Path

from fed_learning.training.nice_neuron_usage import append_nice_neuron_usage


class DummyNICEModel:
    LAYER_NAMES = ["conv1", "fc2"]

    def __init__(self):
        self.unit_ranks = {
            "conv1": np.array([0, 1, 2, 3], dtype=np.int32),
            "fc2": np.array([0, 1, 0], dtype=np.int32),
        }


def test_nice_neuron_usage_summary_tracks_total_free_and_new_used(tmp_path):
    model = DummyNICEModel()
    path = append_nice_neuron_usage(str(tmp_path), 0, model)

    payload = json.loads(Path(path).read_text())
    task0 = payload["tasks"][0]
    assert task0["totals"]["total"] == 7
    assert task0["totals"]["used"] == 4
    assert task0["totals"]["free"] == 3
    assert task0["totals"]["new_used_this_task"] == 4

    previous = {k: v.copy() for k, v in model.unit_ranks.items()}
    model.unit_ranks["conv1"][0] = 1
    model.unit_ranks["fc2"][2] = 1
    append_nice_neuron_usage(str(tmp_path), 1, model, previous_state=previous)

    payload = json.loads(Path(path).read_text())
    task1 = payload["tasks"][1]
    assert task1["totals"]["used"] == 6
    assert task1["totals"]["free"] == 1
    assert task1["totals"]["new_used_this_task"] == 2
