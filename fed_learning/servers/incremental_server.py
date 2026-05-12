"""
CGoFed Server - Standard Incremental Server for CGoFed and EWC.
"""

from typing import Any, Dict, Optional
import numpy as np
import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from .server import FederatedServer


class IncrementalServer(FederatedServer):
    """
    Server chung cho bài toán federated class-incremental learning.

    So với `FederatedServer`, class này thêm:
    - theo dõi `current_task`, `task_classes`, `seen_classes`
    - đánh giá chỉ trên các lớp đã thấy nếu cần
    """

    def __init__(self, clients, test_data: Dict, config: Dict):
        """Khởi tạo server incremental và thêm state theo dõi task/class."""
        super().__init__(clients, test_data, config)

        self.seen_classes = []
        self.task_classes: Dict[int, list] = {}
        self.current_task = 0

    def set_task(
        self, task_id: int, task_classes: list, seen_classes: Optional[list] = None
    ):
        """
        Cập nhật server khi chuyển sang task mới.

        Hàm này không train gì cả; nó chỉ đồng bộ:
        - task hiện tại
        - class mới của task
        - danh sách tất cả class đã thấy tới thời điểm hiện tại
        """
        self.current_task = task_id
        self.task_classes[task_id] = task_classes

        # CRITICAL: Use explicit seen_classes list if provided (prevents reset bug)
        if seen_classes is not None:
            self.seen_classes = list(seen_classes)  # Use the full cumulative list
        else:
            # Fallback: extend (only works if Server persists across tasks)
            self.seen_classes.extend(task_classes)

        print(f"\n📌 Task {task_id}: classes {task_classes}")
        print(f"   Total seen classes: {len(self.seen_classes)}")

        # Some standard FL trainers (FedAvg/FedProx) do not implement set_task(),
        # but when they are run through the incremental loop they still need
        # class-incremental output masking during local CE training.
        if hasattr(self, "trainer"):
            setattr(self.trainer, "current_task", task_id)
            setattr(self.trainer, "seen_classes", set(self.seen_classes))
            setattr(self.trainer, "new_classes", list(task_classes))

    def evaluate_global(
        self,
        batch_size: int = 1024,
        compute_auc: bool = False,
        seen_classes_only: bool = True,
    ) -> Dict:
        """
        Đánh giá global model theo setting incremental.

        Điểm khác biệt chính so với server cơ bản là có thể lọc test set
        chỉ giữ các lớp đã thấy, nhờ đó metric phản ánh đúng tiến trình IL.
        """
        self.global_model.eval()
        criterion = nn.CrossEntropyLoss()

        X_test = self.test_data["X_test"]
        y_test = self.test_data["y_test"]

        # print(f"   📊 Eval Debug: Global Test Set size = {len(y_test)}")

        # Filter to seen classes if requested
        if seen_classes_only and self.seen_classes:
            seen_set = set(self.seen_classes)
            # Use list comprehension (user reverted optimization, keeping as is)
            mask = torch.tensor([y.item() in seen_set for y in y_test])
            X_test = X_test[mask]
            y_test = y_test[mask]
            # print(f"   📊 Eval Debug: After filtering (seen_classes), n_test = {len(y_test)}")

        n_test = len(y_test)
        if n_test == 0:
            print(
                f"   ⚠️ CRITICAL: Test set is EMPTY after filtering for classes {self.seen_classes}!"
            )
            return {"loss": 0.0, "accuracy": 0.0, "f1_macro": 0.0, "f1_weighted": 0.0}

        all_preds = []
        all_targets = []
        total_loss = 0.0

        with torch.no_grad():
            for i in range(0, n_test, batch_size):
                X_batch = X_test[i : i + batch_size].to(self.primary_device)
                y_batch = y_test[i : i + batch_size].to(self.primary_device)

                out = self.global_model(X_batch)
                if seen_classes_only and self.seen_classes:
                    seen_mask = torch.ones(out.shape[1], dtype=torch.bool, device=out.device)
                    for cls_id in self.seen_classes:
                        if 0 <= int(cls_id) < out.shape[1]:
                            seen_mask[int(cls_id)] = False
                    out = out.clone()
                    out[:, seen_mask] = float("-inf")

                loss = criterion(out, y_batch)
                total_loss += loss.item() * len(y_batch)

                preds = out.argmax(dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())


        y_true = np.array(all_targets)
        y_pred = np.array(all_preds)
        zero_division: Any = 0

        # DEBUG: Per-class accuracy and prediction distribution
        print(f"  DEBUG[Eval]: n_test={n_test}, seen_classes={self.seen_classes}")
        print(f"  DEBUG[Eval]: Predicted classes: {np.unique(y_pred)}")
        print(f"  DEBUG[Eval]: True classes: {np.unique(y_true)}")

        # Per-class accuracy breakdown
        unique_classes = np.unique(np.concatenate([y_true, y_pred]))
        per_class_accs = {}
        for c in unique_classes:
            mask = y_true == c
            if mask.sum() > 0:
                acc = (y_pred[mask] == c).mean()
                per_class_accs[int(c)] = acc
                print(f"  DEBUG[Eval]: Class {int(c):2d}: acc={acc:.3f} (n={mask.sum()})")

        # Prediction bias: how many predictions per true class
        print(f"  DEBUG[Eval]: Prediction distribution per true class:")
        for c in sorted(unique_classes)[:6]:  # Show first 6 classes
            mask = y_true == c
            if mask.sum() > 0:
                pred_counts = np.bincount(y_pred[mask], minlength=34)[:12]  # First 12 classes
                print(f"    Class {int(c):2d} true → predictions: {pred_counts}")

        metrics = {
            "loss": total_loss / n_test,
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "recall_macro": recall_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "f1_macro": f1_score(
                y_true, y_pred, average="macro", zero_division=zero_division
            ),
            "f1_weighted": f1_score(
                y_true, y_pred, average="weighted", zero_division=zero_division
            ),
        }

        return metrics


# Alias giữ tương thích ngược với tên cũ trong một số chỗ của project.
CGoFedServer = IncrementalServer
