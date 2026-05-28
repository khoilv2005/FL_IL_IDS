import torch

from fed_learning.clients.rne_client import RNEClient
from fed_learning.factories.client_factory import create_client
from fed_learning.models.rne_model import RNEModel
from fed_learning.strategies import get_strategy, STRATEGIES
from fed_learning.strategies.fed_incremental.rne import RNEAggregator
from fed_learning.strategies.incremental import get_incremental_strategy
from fed_learning.strategies.incremental.rne import RNETrainer


def test_rne_model_expands_with_recurrent_decoupled_heads():
    model = RNEModel(input_shape=(40,), num_classes=4)
    model.add_task([0, 1], s_max=15.0)

    x = torch.randn(3, 40)
    logits_t0 = model(x)
    assert logits_t0.shape == (3, 4)
    assert torch.isfinite(logits_t0[:, :2]).all()
    assert (logits_t0[:, 2:] < -1e8).all()

    model.add_task([2, 3], s_max=15.0)
    logits_t1 = model(x)
    assert logits_t1.shape == (3, 4)
    assert torch.isfinite(logits_t1).all()
    assert len(model.classifier_heads) == 2
    assert model.classifier_heads[0].in_features == model.feat_dim
    assert model.classifier_heads[1].in_features == model.feat_dim * 2
    assert model.recurrent_adapter is not None


def test_rne_freezes_old_expert_after_expansion():
    model = RNEModel(input_shape=(40,), num_classes=4)
    model.add_task([0, 1], s_max=15.0)
    model.add_task([2, 3], s_max=15.0)

    assert all(not p.requires_grad for p in model.extractors[0].parameters())
    assert any(p.requires_grad for p in model.extractors[1].parameters())


def test_rne_strategy_and_client_registration():
    assert "rne" in STRATEGIES

    trainer, aggregator = get_strategy("rne")
    assert isinstance(trainer, RNETrainer)
    assert isinstance(aggregator, RNEAggregator)

    local_trainer = get_incremental_strategy("rne")
    assert isinstance(local_trainer, RNETrainer)

    client = create_client(
        0,
        torch.randn(8, 40),
        torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]),
        {"algorithm": "rne", "buffer_size": 20},
    )
    assert isinstance(client, RNEClient)
