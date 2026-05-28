"""RNE client with pseudo-feature bias correction."""

import gc
from collections import OrderedDict
from typing import Any, Dict, Optional

import torch

try:
    from torch.amp import GradScaler
except ImportError:
    from torch.cuda.amp import GradScaler

from .der_client import DERClient
from ..core import BaseTrainer


class RNEClient(DERClient):
    """
    Client for Recurrent Network Expansion.

    Stage 1 reuses DERClient mechanics. Stage 2 follows RNE bias correction:
    freeze experts, generate balanced pseudo-feature vectors, then retrain only
    classifier heads.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rne_feature_means = {}

    def train(
        self,
        trainer: BaseTrainer,
        epochs: int,
        batch_size: int,
        lr: float,
        global_params: Optional[OrderedDict] = None,
        stage: int = 1,
        replay_ratio: float = 0.5,
        **kwargs,
    ) -> Dict[str, Any]:
        if stage == 1:
            old_model = self._load_old_model(self.device)
            if old_model is not None:
                kwargs["old_model"] = old_model

        if stage != 2 or not hasattr(self.model, "classify_features"):
            result = super().train(
                trainer=trainer,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                global_params=global_params,
                stage=stage,
                replay_ratio=replay_ratio,
                **kwargs,
            )
            if stage == 1 and hasattr(self.model, "extract_vector"):
                self._refresh_feature_stats(batch_size=batch_size)
            return result
        return self._train_pseudo_feature_stage(
            trainer=trainer,
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
            global_params=global_params,
            pseudo_old_per_class=int(
                kwargs.get(
                    "rne_pseudo_old_per_class",
                    kwargs.get("rne_pseudo_per_class", 400),
                )
            ),
            pseudo_new_per_class=int(kwargs.get("rne_pseudo_new_per_class", 100)),
        )

    def save_model_snapshot(self, model) -> None:
        self.old_model_state = OrderedDict(
            (k, v.detach().cpu().clone()) for k, v in model.state_dict().items()
        )
        self.old_model_task_classes_history = [
            list(classes) for classes in getattr(model, "task_classes_history", [])
        ]
        self.old_model_current_task = int(getattr(model, "current_task", -1))
        self.old_model = None

    def update_exemplars(self, model, batch_size: int = 128):
        super().update_exemplars(model, batch_size=batch_size)
        self.save_model_snapshot(model)

    def _load_old_model(self, device: str):
        state = getattr(self, "old_model_state", None)
        if state is None:
            return None
        cached = getattr(self, "old_model", None)
        if cached is not None and next(cached.parameters()).device == torch.device(device):
            return cached

        old_model = self.model.clone_empty_like().to(device)
        for classes in getattr(self, "old_model_task_classes_history", []):
            old_model.add_task(classes, s_max=getattr(self.model, "_s_max", 15.0))
        old_model.load_state_dict({k: v.to(device) for k, v in state.items()})
        old_model.eval()
        for param in old_model.parameters():
            param.requires_grad = False
        self.old_model = old_model
        return old_model

    def _train_pseudo_feature_stage(
        self,
        trainer: BaseTrainer,
        epochs: int,
        batch_size: int,
        lr: float,
        global_params: Optional[OrderedDict],
        pseudo_old_per_class: int,
        pseudo_new_per_class: int,
    ) -> Dict[str, Any]:
        self.model.freeze_all_extractors()
        opt_params = self.model.get_classifier_params()
        if not opt_params:
            return self._empty_result()

        features, labels = self._build_pseudo_feature_set(
            batch_size=batch_size,
            pseudo_old_per_class=pseudo_old_per_class,
            pseudo_new_per_class=pseudo_new_per_class,
        )
        if features is None or len(labels) == 0:
            return super().train(
                trainer=trainer,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                global_params=global_params,
                stage=2,
            )

        optimizer = trainer.get_optimizer_class()(opt_params, lr=lr)
        scaler = GradScaler(enabled=self.use_amp)
        if hasattr(trainer, "set_stage"):
            trainer.set_stage(2)
        trainer.pre_train(self.model, global_params, lr=lr)

        total_loss = 0.0
        total_samples = 0
        self.model.train()
        self.model.freeze_all_extractors()

        for epoch in range(epochs):
            trainer.current_epoch = epoch
            trainer.total_epochs = epochs
            indices = torch.randperm(len(labels))
            for start in range(0, len(labels), batch_size):
                idx = indices[start : start + batch_size]
                X_batch = features[idx].to(self.device, non_blocking=True)
                y_batch = labels[idx].to(self.device, non_blocking=True)
                optimizer.zero_grad()

                with self._amp_ctx():
                    output = self.model(X_batch, mode="fc")
                    loss = trainer.compute_loss(
                        self.model,
                        output,
                        y_batch,
                        global_params,
                        inputs=None,
                        s=None,
                    )

                if self.use_amp:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                    trainer.pre_step(self.model, global_params)
                    scaler.step(optimizer)
                    scaler.update()
                    trainer.post_step(self.model, global_params)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(opt_params, max_norm=1.0)
                    trainer.pre_step(self.model, global_params)
                    optimizer.step()
                    trainer.post_step(self.model, global_params)

                bs = len(y_batch)
                total_loss += loss.item() * bs
                total_samples += bs

        trainer.post_train(self.model, global_params)
        self.model.unfreeze_current_extractor()
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": total_loss / max(1, total_samples),
            "params": OrderedDict(
                (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
            ),
            "replay_samples": self.replay_buffer.total_samples,
            "replay_classes": self.replay_buffer.num_classes,
            "pseudo_features": len(labels),
        }

    def _empty_result(self) -> Dict[str, Any]:
        return {
            "client_id": self.client_id,
            "num_samples": self.num_samples,
            "loss": 0.0,
            "params": OrderedDict(
                (k, v.cpu().clone()) for k, v in self.model.state_dict().items()
            ),
            "replay_samples": self.replay_buffer.total_samples,
            "replay_classes": self.replay_buffer.num_classes,
        }

    def _available_training_data(self):
        all_X = [self.X_train]
        all_y = [self.y_train]
        X_replay, y_replay = self.replay_buffer.get_all_samples()
        if X_replay is not None and len(y_replay) > 0:
            all_X.append(X_replay)
            all_y.append(y_replay)
        return torch.cat(all_X, dim=0), torch.cat(all_y, dim=0)

    def _extract_features_by_class(self, batch_size: int):
        X_all, y_all = self._available_training_data()
        device = self.device
        self.model.eval()
        by_class = {}
        with torch.no_grad():
            for cls_id in sorted(int(c) for c in self.seen_classes):
                mask = y_all == cls_id
                if not mask.any():
                    continue
                X_cls = X_all[mask]
                chunks = []
                for start in range(0, len(X_cls), batch_size):
                    X_batch = X_cls[start : start + batch_size].to(device, non_blocking=True)
                    chunks.append(self.model.extract_vector(X_batch).detach().cpu())
                if chunks:
                    by_class[cls_id] = torch.cat(chunks, dim=0)
        return by_class

    def _refresh_feature_stats(self, batch_size: int):
        by_class = self._extract_features_by_class(batch_size)
        if not by_class:
            return

        current_means = {c: feats.mean(dim=0) for c, feats in by_class.items()}
        feat_dim = int(getattr(self.model, "feat_dim", 0) or 0)
        refreshed = {}

        old_classes = sorted(int(c) for c in self.seen_classes - self.current_classes)
        for cls_id in old_classes:
            cur_mean = current_means.get(cls_id)
            prev_mean = self.rne_feature_means.get(cls_id)
            if cur_mean is None and prev_mean is None:
                continue
            if (
                cur_mean is not None
                and prev_mean is not None
                and feat_dim > 0
                and prev_mean.numel() + feat_dim == cur_mean.numel()
            ):
                refreshed[cls_id] = torch.cat(
                    [prev_mean.detach().cpu(), cur_mean[-feat_dim:].detach().cpu()],
                    dim=0,
                )
            elif cur_mean is not None:
                refreshed[cls_id] = cur_mean.detach().cpu()
            else:
                refreshed[cls_id] = prev_mean.detach().cpu()

        for cls_id in sorted(int(c) for c in self.current_classes):
            if cls_id in current_means:
                refreshed[cls_id] = current_means[cls_id].detach().cpu()

        self.rne_feature_means = refreshed

    def _build_pseudo_feature_set(
        self,
        batch_size: int,
        pseudo_old_per_class: int,
        pseudo_new_per_class: int,
    ):
        if self.current_task <= 0 or self.replay_buffer.total_samples == 0:
            return None, None

        by_class = self._extract_features_by_class(batch_size)
        old_classes = sorted(int(c) for c in self.seen_classes - self.current_classes)
        new_classes = sorted(int(c) for c in self.current_classes)
        old_classes = [c for c in old_classes if c in by_class]
        new_classes = [c for c in new_classes if c in by_class]
        if not old_classes or not new_classes:
            return None, None

        raw_means = {c: by_class[c].mean(dim=0) for c in by_class}
        means = {}
        for cls_id in set(old_classes + new_classes):
            cached = self.rne_feature_means.get(cls_id)
            if cached is not None and cached.numel() == raw_means[cls_id].numel():
                means[cls_id] = cached.to(raw_means[cls_id].device)
            else:
                means[cls_id] = raw_means[cls_id]
        old_mean_stack = torch.stack([means[c] for c in old_classes], dim=0)
        generator = max(
            new_classes,
            key=lambda c: float(((means[c].unsqueeze(0) - old_mean_stack) ** 2).sum(dim=1).mean()),
        )
        gen_features = by_class[generator]
        gen_mean = means[generator]

        feature_chunks = []
        label_chunks = []

        for cls_id in old_classes:
            source = gen_features.clone()
            pseudo = source - gen_mean + means[cls_id]
            pseudo = self._herding_rows(pseudo, pseudo_old_per_class)
            feature_chunks.append(pseudo)
            label_chunks.append(torch.full((len(pseudo),), cls_id, dtype=torch.long))

        for cls_id in new_classes:
            source = self._herding_rows(by_class[cls_id], pseudo_new_per_class)
            feature_chunks.append(source)
            label_chunks.append(torch.full((len(source),), cls_id, dtype=torch.long))

        features = torch.cat(feature_chunks, dim=0)
        labels = torch.cat(label_chunks, dim=0)
        gc.collect()
        return features, labels

    @staticmethod
    def _sample_rows(features: torch.Tensor, n_rows: int) -> torch.Tensor:
        if len(features) <= n_rows:
            return features.clone()
        indices = torch.randperm(len(features))[:n_rows]
        return features[indices].clone()

    @staticmethod
    def _herding_rows(features: torch.Tensor, n_rows: int) -> torch.Tensor:
        if len(features) <= n_rows:
            return features.clone()
        feats = features.detach().cpu()
        columns = feats.t().clone()
        columns = columns / (columns.norm(dim=0, keepdim=True) + 1e-8)
        mu = columns.mean(dim=1)
        selected = []
        selected_mask = torch.zeros(columns.shape[1], dtype=torch.bool)
        w_t = mu.clone()
        attempts = 0
        while len(selected) < min(n_rows, columns.shape[1]) and attempts < 1000:
            scores = torch.mv(columns.t(), w_t)
            scores[selected_mask] = -float("inf")
            idx = int(torch.argmax(scores).item())
            if not selected_mask[idx]:
                selected.append(idx)
                selected_mask[idx] = True
            w_t = w_t + mu - columns[:, idx]
            attempts += 1
        if len(selected) < min(n_rows, columns.shape[1]):
            rest = torch.nonzero(~selected_mask, as_tuple=False).flatten().tolist()
            selected.extend(rest[: min(n_rows, columns.shape[1]) - len(selected)])
        return features[selected].clone()
