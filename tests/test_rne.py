import torch

from fed_learning.clients.rne_client import RNEClient
from fed_learning.factories.client_factory import create_client
from fed_learning.models.rne_model import RNECompressModel, RNEModel
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
    assert model.recurrent_mapper is not None
    features = model.get_feature_sequence(x)
    old_head_logits = model.classifier_heads[0](features[0])
    new_head_logits = model.classifier_heads[1](torch.cat(features, dim=1))
    assert torch.allclose(logits_t1[:, :2], old_head_logits, atol=1e-6)
    assert torch.allclose(logits_t1[:, 2:], new_head_logits, atol=1e-6)


def test_rne_freezes_old_expert_after_expansion():
    model = RNEModel(input_shape=(40,), num_classes=4)
    model.add_task([0, 1], s_max=15.0)
    model.add_task([2, 3], s_max=15.0)

    assert all(not p.requires_grad for p in model.extractors[0].parameters())
    assert any(p.requires_grad for p in model.extractors[1].parameters())
    assert not any(k.startswith("frozen_mask_") for k in model.state_dict())

def test_rne_has_no_der_mask_parameters_or_sparsity():
    model = RNEModel(input_shape=(40,), num_classes=4)
    model.add_task([0, 1], s_max=15.0)

    assert not any("mask_embeds" in name for name, _ in model.named_parameters())
    assert model.get_sparsity_loss(100.0).item() == 0.0

    compress = RNECompressModel(input_shape=(40,), num_classes=4)
    compress.add_task([0, 1], s_max=15.0)
    assert not any("mask_embeds" in name for name, _ in compress.named_parameters())
    assert compress.get_sparsity_loss(100.0).item() == 0.0


def test_rne_strategy_and_client_registration():
    assert "rne" in STRATEGIES
    assert "rne_compress" in STRATEGIES

    trainer, aggregator = get_strategy("rne")
    assert isinstance(trainer, RNETrainer)
    assert isinstance(aggregator, RNEAggregator)
    assert trainer.lambda_sparsity == 0.0
    compress_trainer, compress_aggregator = get_strategy("rne_compress")
    assert isinstance(compress_trainer, RNETrainer)
    assert isinstance(compress_aggregator, RNEAggregator)
    assert compress_trainer.lambda_sparsity == 0.0

    local_trainer = get_incremental_strategy("rne")
    assert isinstance(local_trainer, RNETrainer)
    compress_local_trainer = get_incremental_strategy("rne_compress")
    assert isinstance(compress_local_trainer, RNETrainer)

    client = create_client(
        0,
        torch.randn(8, 40),
        torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]),
        {"algorithm": "rne", "buffer_size": 20},
    )
    assert isinstance(client, RNEClient)
    compress_client = create_client(
        0,
        torch.randn(8, 40),
        torch.tensor([0, 1, 0, 1, 2, 3, 2, 3]),
        {"algorithm": "rne_compress", "buffer_size": 20},
    )
    assert isinstance(compress_client, RNEClient)

def test_rne_compress_uses_shared_backbone_and_smaller_experts():
    full = RNEModel(input_shape=(40,), num_classes=4)
    full.add_task([0, 1], s_max=15.0)

    model = RNECompressModel(input_shape=(40,), num_classes=4)
    model.add_task([0, 1], s_max=15.0)
    model.add_task([2, 3], s_max=15.0)

    x = torch.randn(3, 40)
    logits = model(x)
    assert logits.shape == (3, 4)
    assert torch.isfinite(logits).all()
    assert hasattr(model, "shared_backbone")
    assert hasattr(model, "backbone_mapper")
    assert model.feat_dim < full.feat_dim
    assert model.compressed_channels == (16, 32, 64, 25)
    assert all(not p.requires_grad for p in model.extractors[0].parameters())
    assert all(not p.requires_grad for p in model.shared_backbone.parameters())
    assert any(p.requires_grad for p in model.backbone_mapper.parameters())


def test_rne_client_stage2_uses_pseudo_features():
    model = RNEModel(input_shape=(40,), num_classes=4)
    model.add_task([0, 1], s_max=15.0)

    client = RNEClient(
        0,
        torch.randn(12, 40),
        torch.tensor([0, 1] * 6),
        buffer_size=20,
    )
    trainer, _ = get_strategy("rne")
    trainer.set_task(0, [0, 1])
    client.set_task_data(client.X_train, client.y_train, 0, [0, 1])
    client.setup_for_gpu(model, "cpu")
    client.train(trainer, epochs=1, batch_size=4, lr=1e-3, stage=1)
    client.update_exemplars(model, batch_size=4)
    assert client.old_model_state is not None

    model.add_task([2, 3], s_max=15.0)
    client.set_task_data(
        torch.randn(12, 40),
        torch.tensor([2, 3] * 6),
        1,
        [2, 3],
    )
    trainer.set_task(1, [2, 3])
    client.setup_for_gpu(model, "cpu")
    assert client._load_old_model("cpu") is not None
    result = client.train(
        trainer,
        epochs=1,
        batch_size=4,
        lr=1e-3,
        stage=2,
        rne_pseudo_per_class=4,
        rne_pseudo_new_per_class=4,
    )

    assert result["pseudo_features"] == 16
    assert result["loss"] > 0

def test_rne_stage1_concats_replay_once():
    client = RNEClient(
        0,
        torch.arange(24, dtype=torch.float32).view(6, 4),
        torch.tensor([2, 2, 3, 3, 4, 4]),
        buffer_size=20,
    )
    client.current_task = 1
    client.current_classes = {2, 3, 4}
    client.seen_classes = {0, 1, 2, 3, 4}
    client.replay_buffer.add_samples(
        torch.arange(8, dtype=torch.float32).view(2, 4) + 100,
        torch.tensor([0, 1]),
    )
    client.device = "cpu"

    batches = list(client._create_combined_batches(batch_size=4, replay_ratio=0.5))
    seen_labels = torch.cat([y for _, y in batches])

    assert len(seen_labels) == 8
    assert int((seen_labels < 2).sum().item()) == 2
    assert int((seen_labels >= 2).sum().item()) == 6

def test_rne_client_feature_mean_cache_extends_old_classes():
    model = RNEModel(input_shape=(40,), num_classes=4)
    model.add_task([0, 1], s_max=15.0)
    client = RNEClient(
        0,
        torch.randn(12, 40),
        torch.tensor([0, 1] * 6),
        buffer_size=20,
    )
    trainer, _ = get_strategy("rne")
    trainer.set_task(0, [0, 1])
    client.set_task_data(client.X_train, client.y_train, 0, [0, 1])
    client.setup_for_gpu(model, "cpu")
    client.train(trainer, epochs=1, batch_size=4, lr=1e-3, stage=1)
    client.update_exemplars(model, batch_size=4)
    old_dim = client.rne_feature_means[0].numel()

    model.add_task([2, 3], s_max=15.0)
    trainer.set_task(1, [2, 3])
    client.set_task_data(
        torch.randn(12, 40),
        torch.tensor([2, 3] * 6),
        1,
        [2, 3],
    )
    client.setup_for_gpu(model, "cpu")
    client.train(trainer, epochs=1, batch_size=4, lr=1e-3, stage=1)

    assert client.rne_feature_means[0].numel() == old_dim + model.feat_dim
    assert client.rne_feature_means[2].numel() == model.feat_dim * 2

def test_rne_compress_old_snapshot_reloads_compress_model():
    model = RNECompressModel(input_shape=(40,), num_classes=4)
    model.add_task([0, 1], s_max=15.0)
    client = RNEClient(
        0,
        torch.randn(8, 40),
        torch.tensor([0, 1, 0, 1, 0, 1, 0, 1]),
        buffer_size=20,
    )
    client.setup_for_gpu(model, "cpu")
    client.save_model_snapshot(model)
    old_model = client._load_old_model("cpu")
    assert isinstance(old_model, RNECompressModel)
    assert old_model.feat_dim == model.feat_dim
