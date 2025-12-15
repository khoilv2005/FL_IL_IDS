"""
Non-IID Visualization Script
----------------------------

Sinh biểu đồ bubble (giống step2-version3) dựa trên dữ liệu federated đã có sẵn.

- Đọc dữ liệu client từ thư mục data (định dạng step2-version3):
    client_0_train.npz, client_1_train.npz, ...
- Nếu số client <= 50: vẽ toàn bộ.
- Nếu số client > 50:
    - Lấy 25 client đầu
    - Lấy 25 client cuối
    - Bỏ qua phần giữa (không vẽ)

Chạy:
    # Cách mới (tiện lợi): tự động tìm thư mục từ số client
    python step3_visualize.py --10clients
    
    # Cách cũ (vẫn hỗ trợ)
    python step3_visualize.py --num_clients 10 --data_dir ./data/federated_splits/10-clients

Hoặc import vào notebook khác và gọi:
    from step3_visualize import run_visualization
    run_visualization(data_dir="...", num_clients=..., output_dir="...")
"""

import os
import sys
import argparse
import re
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Style sáng, dễ nhìn (gần giống seaborn default)
plt.style.use("seaborn-v0_8-whitegrid")


# =========================
# Data loading utilities
# =========================

def load_client_label_distributions(
    data_dir: str,
    num_clients: int,
) -> Tuple[List[Dict], int]:
    """
    Đọc dữ liệu client_*_train.npz và trả về:
        - danh sách stats cho từng client
        - num_classes (số class tổng)
    """
    print("\n" + "=" * 80)
    print("LOADING CLIENT DATA FOR NON-IID VISUALIZATION")
    print("=" * 80)

    client_stats: List[Dict] = []
    all_labels = []

    for cid in range(num_clients):
        path = os.path.join(data_dir, f"client_{cid}_train.npz")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing: {path}")

        data = np.load(path)
        if "y_train" not in data:
            raise KeyError(f"'y_train' not found in {path}")

        y_train = data["y_train"].astype(np.int64)
        unique, counts = np.unique(y_train, return_counts=True)

        all_labels.append(y_train)

        # full distribution vector (size = num_classes, tạm thời chưa biết num_classes)
        # nên lưu tạm unique & counts, lát nữa sau khi biết num_classes sẽ map lại.
        client_stats.append(
            {
                "client_id": cid,
                "y": y_train,
                "unique": unique,
                "counts": counts,
                "total_samples": len(y_train),
            }
        )

        print(
            f"  Client {cid}: {len(y_train):,} samples, "
            f"{len(unique)} classes, "
            f"dominant class {unique[np.argmax(counts)]} "
            f"({counts.max() / len(y_train) * 100:.1f}%)"
        )

    # Xác định num_classes toàn cục
    all_labels_np = np.concatenate(all_labels)
    classes = np.unique(all_labels_np)
    num_classes = int(len(classes))

    print(f"\n  Detected num_classes: {num_classes}")
    print(f"  Total train samples: {len(all_labels_np):,}")
    print("=" * 80)

    # Chuẩn hoá: tạo vector phân bố full length num_classes cho mỗi client
    stats_out: List[Dict] = []
    for s in client_stats:
        full_dist = np.zeros(num_classes, dtype=np.int64)
        for u, c in zip(s["unique"], s["counts"]):
            full_dist[int(u)] = int(c)

        stats_out.append(
            {
                "client_id": s["client_id"],
                "total_samples": int(s["total_samples"]),
                "class_distribution": full_dist,
            }
        )

    return stats_out, num_classes


# =========================
# Grouping & visualization
# =========================

def prepare_client_groups_for_large(
    client_stats: List[Dict],
    num_classes: int,
    max_clients: int = 50,
) -> Tuple[List[str], np.ndarray]:
    """
    Chuẩn hoá client list khi số lượng client quá lớn.

    - Nếu số client <= 50: trả về đầy đủ.
    - Nếu > 50:
        + Chỉ giữ một số client đầu
        + Và một số client cuối (mặc định 50 mỗi đầu)
        + Phần giữa bỏ qua hoàn toàn (không vẽ, không gộp)
    """
    n = len(client_stats)
    dist_matrix = np.zeros((n, num_classes), dtype=np.float64)
    client_ids = []

    for idx, stat in enumerate(client_stats):
        client_ids.append(stat["client_id"])
        dist_matrix[idx] = stat["class_distribution"]

    # Quy tắc cố định: nếu số client <= 50 thì vẽ full, > 50 thì group,
    # không phụ thuộc tham số max_clients_full bên ngoài.
    if n <= 50:
        # Nhãn trục X: C0, C1, ..., cho gọn
        labels = [f"C{cid}" for cid in client_ids]
        return labels, dist_matrix

    # Giữ tối đa 25 client đầu và 25 client cuối.
    keep_each_side = min(25, n // 2)
    front_k = keep_each_side
    back_k = keep_each_side

    # Nếu tổng số client nhỏ hơn hoặc bằng số giữ lại, trả về đầy đủ để tránh mất dữ liệu
    if front_k + back_k >= n:
        labels = [f"C{cid}" for cid in client_ids]
        return labels, dist_matrix

    front_indices = list(range(front_k))
    back_indices = list(range(n - back_k, n))

    # Ma trận cho 25 client đầu
    front_mat = dist_matrix[front_indices]
    front_labels = [f"C{client_ids[i]}" for i in front_indices]

    # Một hàng rỗng (không có sample) để hiển thị cột "..."
    middle_mat = np.zeros((1, num_classes), dtype=np.float64)
    middle_label = "..."

    # Ma trận cho 25 client cuối
    back_mat = dist_matrix[back_indices]
    back_labels = [f"C{client_ids[i]}" for i in back_indices]

    labels: List[str] = front_labels + [middle_label] + back_labels
    combined_mat = np.vstack([front_mat, middle_mat, back_mat])

    return labels, combined_mat


def bubble_and_heatmap(
    client_stats: List[Dict],
    num_classes: int,
    save_dir: str,
    max_clients_full: int = 50,
    class_names: Optional[List[str]] = None,
):
    """
    Vẽ bubble chart + heatmap nâng cao cho phân bố Non-IID.

    - Nếu số client <= max_clients_full: vẽ toàn bộ client.
    - Nếu > max_clients_full:
        + 25 client đầu
        + 25 client cuối
        + phần giữa gộp thành 1 nhóm
    """
    labels, mat_counts = prepare_client_groups_for_large(
        client_stats, num_classes, max_clients=max_clients_full
    )
    num_clients_vis = len(labels)

    # Đảo ma trận để: trục X = client/group, trục Y = class
    # mat_counts: (num_clients_vis, num_classes) -> (num_classes, num_clients_vis)
    mat_counts_T = mat_counts.T
    if class_names is not None and len(class_names) == num_classes:
        class_labels = class_names
    else:
        class_labels = [f"C{i}" for i in range(num_classes)]
    # Lật theo chiều dọc để class 0 nằm ở hàng dưới cùng
    mat_counts_plot = np.flipud(mat_counts_T)
    class_labels_plot = class_labels[::-1]

    width_scale = max(12, num_clients_vis * 0.5)

    # Bubble chart (percentage per client/group)
    # Tính % theo mỗi client/group (theo cột của mat_counts_T)
    col_sums = mat_counts_T.sum(axis=0, keepdims=True)  # shape: (1, num_clients_vis)
    col_sums_safe = np.where(col_sums == 0, 1, col_sums)
    pct_T = mat_counts_T / col_sums_safe * 100.0

    # Cải thiện: sử dụng sqrt scale để làm phẳng sự khác biệt, giúp bubble nhỏ dễ nhìn hơn
    # và thêm minimum size để đảm bảo bubble nhỏ nhất vẫn nhìn thấy được
    min_bubble_size = 10  # Kích thước tối thiểu để nhìn thấy rõ
    max_bubble_size = 800  # Kích thước tối đa
    base_scale = 25.0      # Hệ số nhân cơ bản
    
    # Tính kích thước bubble với sqrt scale để làm phẳng sự khác biệt
    # sqrt scale giúp bubble nhỏ lớn hơn tương đối, bubble lớn nhỏ hơn tương đối
    pct_flat = np.sqrt(pct_T)
    # Normalize về [0, 1] rồi scale về [min_bubble_size, max_bubble_size]
    pct_min = pct_flat.min()
    pct_max = pct_flat.max()
    if pct_max - pct_min > 1e-10:
        pct_normalized = (pct_flat - pct_min) / (pct_max - pct_min)
    else:
        pct_normalized = np.ones_like(pct_flat) * 0.5  # Nếu tất cả bằng nhau
    sizes_scaled = min_bubble_size + pct_normalized * (max_bubble_size - min_bubble_size)

    xs, ys, sizes = [], [], []
    for j in range(num_clients_vis):         # X: client/group
        for i in range(num_classes):         # Y: class
            if mat_counts_T[i, j] > 0:      # chỉ vẽ bubble nếu có mẫu
                xs.append(j)
                ys.append(i)  # class index, C0 sẽ ở dưới (y=0)
                # Sử dụng kích thước đã scale với sqrt để bubble nhỏ dễ nhìn hơn
                sizes.append(sizes_scaled[i, j])

    fig_bb, ax_bb = plt.subplots(figsize=(width_scale, max(8, num_classes * 0.4)))

    # Bubble: dùng 1 màu cố định, kích thước đã được cải thiện với min size và sqrt scale
    sc = ax_bb.scatter(
        xs,
        ys,
        s=sizes,
        c="#1f77b4",      # một màu xanh dương cố định
        alpha=0.8,
        edgecolors="white",
        linewidths=1.0,
    )
    ax_bb.set_xticks(range(num_clients_vis))
    ax_bb.set_xticklabels(labels)
    # Nhãn client dựng đứng
    plt.setp(ax_bb.get_xticklabels(), rotation=90, ha="center")
    ax_bb.set_yticks(range(num_classes))
    ax_bb.set_yticklabels(class_labels)
    ax_bb.set_ylim(-0.5, num_classes - 0.5)
    ax_bb.set_xlabel("Client")
    ax_bb.set_ylabel("Class")
    ax_bb.set_title("Label Distribution - Bubble", fontsize=14)

    os.makedirs(save_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    fname_bb = f"client_distribution_bubble_{ts}.png"
    path_bb = os.path.join(save_dir, fname_bb)
    fig_bb.savefig(path_bb, dpi=150, bbox_inches="tight")
    print(f"  💾 Saved bubble chart: {path_bb}")

    fig_bb.show()
    return fig_bb


# =========================
# Main runner
# =========================

def run_visualization(
    data_dir: str,
    num_clients: int,
    output_dir: str | None,
    max_clients_full: int = 50,
    class_names: Optional[List[str]] = None,
):
    """
    Hàm tiện ích để chạy full pipeline từ code khác / notebook.
    """
    # Nếu không cung cấp output_dir, tự động dùng chính thư mục data_dir
    target_output = output_dir or data_dir

    stats, num_classes = load_client_label_distributions(
        data_dir=data_dir,
        num_clients=num_clients,
    )

    adjusted_names = None
    if class_names:
        adjusted_names = class_names[:num_classes]
        if len(adjusted_names) < num_classes:
            adjusted_names += [f"C{i}" for i in range(len(adjusted_names), num_classes)]

    bubble_and_heatmap(
        stats,
        num_classes,
        save_dir=target_output,
        max_clients_full=max_clients_full,
        class_names=adjusted_names,
    )


def _load_class_names(path: str) -> List[str]:
    """
    Đọc tên class từ file (mỗi dòng một tên).
    """
    with open(path, "r", encoding="utf-8") as f:
        names = [line.strip() for line in f if line.strip()]
    return names


def _find_class_names_file(data_dir: str) -> Optional[str]:
    """
    Tự động tìm file class_names.txt theo thứ tự ưu tiên:
    1. Trong thư mục data_dir
    2. Ở root của project (thư mục chứa script)
    """
    # Thử tìm trong data_dir trước
    path_in_data = os.path.join(data_dir, "class_names.txt")
    if os.path.exists(path_in_data):
        return path_in_data
    
    # Thử tìm ở root của project (thư mục chứa script)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path_in_root = os.path.join(script_dir, "class_names.txt")
    if os.path.exists(path_in_root):
        return path_in_root
    
    return None


def parse_args():
    # Kiểm tra xem có argument dạng --{số}clients không (ví dụ: --10clients)
    num_clients_from_flag = None
    for arg in sys.argv[1:]:
        match = re.match(r'--(\d+)clients$', arg)
        if match:
            num_clients_from_flag = int(match.group(1))
            # Loại bỏ argument này khỏi sys.argv để argparse không báo lỗi
            sys.argv.remove(arg)
            break
    
    parser = argparse.ArgumentParser(
        description="Non-IID Bubble + Heatmap Visualization"
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
        help="Thư mục chứa các file client_*_train.npz (tự động nếu dùng --{số}clients)",
    )
    parser.add_argument(
        "--num_clients",
        type=int,
        default=None,
        help="Số client cần đọc (client_0 ... client_{num_clients-1})",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Thư mục lưu hình (mặc định = chính data_dir)",
    )
    parser.add_argument(
        "--class_names_file",
        type=str,
        default=None,
        help="File chứa tên class (mỗi dòng một tên). Nếu không truyền dùng C0..Cn.",
    )
    parser.add_argument(
        "--max_clients_full",
        type=int,
        default=50,
        help="Ngưỡng số client tối đa vẽ đầy đủ; lớn hơn thì group lại",
    )
    
    args = parser.parse_args()
    
    # Nếu có --{số}clients, tự động set num_clients và data_dir
    if num_clients_from_flag is not None:
        args.num_clients = num_clients_from_flag
        if args.data_dir is None:
            args.data_dir = f"./data/federated_splits/{num_clients_from_flag}-clients"
    
    # Nếu không có --{số}clients nhưng có --num_clients, tự động set data_dir nếu chưa có
    if args.num_clients is not None and args.data_dir is None:
        args.data_dir = f"./data/federated_splits/{args.num_clients}-clients"
    
    # Mặc định nếu cả hai đều None
    if args.num_clients is None:
        args.num_clients = 5
    if args.data_dir is None:
        args.data_dir = "./data/federated_splits/5-clients"
    
    return args


if __name__ == "__main__":
    args = parse_args()
    print("\n=== Non-IID Visualization (Bubble + Heatmap) ===")
    print(f"  data_dir      : {args.data_dir}")
    print(f"  num_clients   : {args.num_clients}")
    print(f"  output_dir    : {args.output_dir}")
    print(f"  class_names   : {args.class_names_file}")
    print(f"  max_clients   : {args.max_clients_full}")

    class_names = None
    class_names_file = args.class_names_file
    
    # Nếu không chỉ định class_names_file, tự động tìm
    if not class_names_file:
        class_names_file = _find_class_names_file(args.data_dir)
        if class_names_file:
            print(f"  ✅ Tự động tìm thấy class_names.txt: {class_names_file}")
    
    if class_names_file:
        try:
            class_names = _load_class_names(class_names_file)
            print(f"  ✅ Đã load {len(class_names)} class names từ file")
        except Exception as exc:
            print(f"⚠️  Không đọc được class_names_file: {exc}. Sẽ dùng C0..C{args.num_clients-1}.")
    else:
        print(f"  ℹ️  Không tìm thấy class_names.txt, sẽ dùng C0..C{args.num_clients-1}")

    run_visualization(
        data_dir=args.data_dir,
        num_clients=args.num_clients,
        output_dir=args.output_dir,
        max_clients_full=args.max_clients_full,
        class_names=class_names,
    )


