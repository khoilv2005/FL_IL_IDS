"""Validate sampling used by the real-data transition harness."""
import io
import numpy as np
import torch
from tools.diagnose_denice_real_client import sample_npz


def test_streamed_sample_preserves_source_alignment_and_is_deterministic():
    y = np.repeat([0,1,2,13], 10)
    x = np.arange(40*39, dtype=np.float32).reshape(40,39,1)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, X_train=x, y_train=y)
    features, labels, indices = sample_npz(buffer.getvalue(), 'train', 4, 2026)
    again = sample_npz(buffer.getvalue(), 'train', 4, 2026)
    assert len(indices) == 12 and len(set(indices)) == 12
    assert indices == again[2]
    torch.testing.assert_close(features, torch.from_numpy(x[indices]), rtol=0, atol=0)
    torch.testing.assert_close(labels, torch.from_numpy(y[indices]), rtol=0, atol=0)
    assert set(labels.tolist()) == {0,1,2}


def test_sampler_uses_available_support_without_replacement():
    buffer = io.BytesIO()
    np.savez_compressed(buffer, X_test=np.ones((3,39,1),dtype=np.float32), y_test=np.array([0,0,1]))
    _, labels, indices = sample_npz(buffer.getvalue(), 'test', 64, 1)
    assert indices == [0,1,2]
    assert labels.tolist() == [0,0,1]


def test_three_task_sampler_keeps_task2_and_excludes_future_labels():
    buffer=io.BytesIO()
    np.savez_compressed(buffer,X_test=np.ones((4,39,1),dtype=np.float32),y_test=np.array([0,12,17,18]))
    _,labels,indices=sample_npz(buffer.getvalue(),'test',16,2027,num_classes=18)
    assert labels.tolist()==[0,12,17]
    assert indices==[0,1,2]


def test_group_summary_weights_by_evaluated_support_not_client_count():
    from tools.analyze_denice_group_probe import weighted_metric
    rows=[{'metrics':{'oracle_hard':{'0_supported':{'n':16,'accuracy':1.0}}}},
          {'metrics':{'oracle_hard':{'0_supported':{'n':32,'accuracy':0.0}}}}]
    assert weighted_metric(rows,'oracle_hard',0,'supported')==1/3


def test_peer_maturity_then_pruning_is_not_a_functional_freeze_guarantee():
    from copy import deepcopy
    from fed_learning.models.denice_model import DeNICEModel
    from fed_learning.strategies.decentralized.denice_aggregation import merge_neuron_ages
    from fed_learning.strategies.incremental.nice import drop_young_to_learner
    model=DeNICEModel((39,1),34)
    model.unit_ranks['conv1'][0]=1
    model.unit_ranks['conv2'][0]=1
    model.conv2.weight.data[0,0,:]=0.4
    control=deepcopy(model)
    peer={'conv2':model.unit_ranks['conv2'].copy()}
    peer['conv2'][0]=2
    model.set_neuron_ages_state(merge_neuron_ages(model.unit_ranks,[peer],[0.6],policy='consensus'))
    control.set_neuron_ages_state(merge_neuron_ages(control.unit_ranks,[peer],[0.6],policy='none'))
    # Later local selection returns the input unit to young. Pruning changes
    # the now-mature receiver's effective connection; Adam is not involved.
    assert model.unit_ranks['conv2'][0]==2
    assert control.unit_ranks['conv2'][0]==1
    model.unit_ranks['conv1'][0]=0
    before=(model.conv2.weight*model.weight_masks['conv2'])[0,0].detach().clone()
    drop_young_to_learner(model)
    after=(model.conv2.weight*model.weight_masks['conv2'])[0,0]
    assert torch.all(before==0.4) and torch.all(after==0)
