
# ============================================================ #
# Download resume state from Google Drive
# ============================================================ #
import os
import json
import subprocess
import sys
import zipfile

# Environment overrides make P6 multi-seed runs reproducible without editing
# this file between Kaggle sessions, e.g. DENICE_SEED=43.
TRAIN_PHASE = int(os.environ.get("DENICE_TRAIN_PHASE", "5"))  # 1..5
TRAIN_SEED = int(os.environ.get("DENICE_SEED", "42"))
TRAIN_OUTPUT_DIR = os.environ.get(
    "DENICE_OUTPUT_DIR", f"/kaggle/working/results_denice_seed_{TRAIN_SEED}"
)

PHASE_CONFIG = {
    1: {
        "task_start": 0,
        "task_end": 1,
        "save_resume_after_task": 1,
        "resume_file": None,
    },
    2: {
        "task_start": 2,
        "task_end": 2,
        "save_resume_after_task": 2,
        "resume_file": "continuation_state_task_1.pt",
    },
    3: {
        "task_start": 3,
        "task_end": 3,
        "save_resume_after_task": 3,
        "resume_file": "continuation_state_task_2.pt",
    },
    4: {
        "task_start": 4,
        "task_end": 5,
        "save_resume_after_task": None,
        "resume_file": "continuation_state_task_3.pt",
    },
    5:{
        "task_start": 0,
        "task_end": 5,
        "save_resume_after_task": None,
        "resume_file": None,
    }
}

phase_config = PHASE_CONFIG[TRAIN_PHASE]
desired_resume_file = phase_config["resume_file"]
target = None

if desired_resume_file is None:
    print(f"Phase {TRAIN_PHASE}: no resume state needed.")
else:
    os.makedirs("/tmp/next/continue", exist_ok=True)

    archive_path = "/tmp/next/continue/continue.zip"
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "gdown"], check=True)
    subprocess.run(
        [
            sys.executable,
            "-m",
            "gdown",
            "--fuzzy",
            "https://drive.google.com/file/d/15Ki3iBYinEmGFk5SRsSno46WI44RVtvk/view?usp=drive_link",
            "-O",
            archive_path,
        ],
        check=True,
    )

    with zipfile.ZipFile(archive_path, "r") as zf:
        zf.extractall("/tmp/next/")

    resume_candidates = []
    for root, _, files in os.walk("/tmp/next"):
        for f in files:
            if f.endswith(".pt"):
                resume_candidates.append(os.path.join(root, f))

    print("PT files found:")
    for p in resume_candidates:
        print(p)

    for p in resume_candidates:
        name = os.path.basename(p).lower()
        if name == desired_resume_file:
            target = p
            break

    if target is None:
        raise FileNotFoundError(
            f"Phase {TRAIN_PHASE} requires {desired_resume_file}, but it was not found in /tmp/next."
        )

print("Selected resume path:", target)

if target is not None:
    print("exists:", os.path.exists(target))
    print("size:", os.path.getsize(target))


# =============================================================================#


"""
Federated Class Incremental Learning - Training Entry Point
============================================================
CONFIG-only entry point. All training logic lives in fed_learning.training.task_loop.

Usage:
    Chọn mode trong CONFIG["mode"]:
    - "fed_il": federated incremental learning
    - "il": local incremental learning
    - "decentralized": decentralized Plexus-IL or DeNICE-IL (no server)

    Sau đó chọn thuật toán qua CONFIG["algorithm"]:
    - fed_il: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf",
              "fedprox_lwf", "fedcbdr", "der", "nice", "glfc", "refed",
              "dfca_il"
    - decentralized: "plexus", "denice"
    - il: "ewc", "lwf", "der", "nice", "denice"

    Upload fed_learning folder to Kaggle dataset, then run this script.
"""

import os
import sys

# =============================================================================
# KAGGLE SETUP - Clone from GitHub
# =============================================================================
REPO_PATH = "/tmp/FL_IL_IDS"


def setup_imports():
    """Clone repo from GitHub to get proper nested structure."""
    import shutil

    # Force fresh clone to avoid any stale bytecode issues
    if os.path.exists(REPO_PATH):
        print(f"Removing stale clone at {REPO_PATH}...")
        shutil.rmtree(REPO_PATH)

    print(f"Cloning from GitHub...")
    os.system(f"git clone https://github.com/khoilv2005/FL_IL_IDS.git {REPO_PATH}")

    # Remove any Kaggle dataset paths that might override our import
    kaggle_prefix = "/kaggle/input"
    new_sys_path = []
    for p in sys.path:
        if not p.startswith(kaggle_prefix):
            new_sys_path.append(p)

    # Put REPO_PATH at the front
    sys.path = [REPO_PATH] + new_sys_path

    # Clear any cached fed_learning modules
    to_remove = [k for k in sys.modules.keys() if 'fed_learning' in k]
    for k in to_remove:
        del sys.modules[k]

    print(f"sys.path[0]: {sys.path[0]}")


setup_imports()


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Data
    "data_dir": "/kaggle/input/datasets/khoilv2005/100-clients/100-clients",
    # Reproducibility
    "random_seed": TRAIN_SEED,
    # Training Mode
    # Options:
    #   - "fed_il": federated incremental learning
    #   - "il": local incremental learning
    #   - "decentralized": Plexus or DeNICE decentralized IL (no server)
    "mode": "decentralized",
    # Algorithm Selection
    # fed_il: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf",
    #         "fedprox_lwf", "fedcbdr", "der", "nice", "glfc", "refed",
    #         "dfca_il"
    # il:     "ewc", "lwf", "der", "nice", "denice"
    # decentralized: "plexus", "denice"
    "algorithm": "denice",
    # Output - Use Kaggle's output directory for persistent storage
    # On Kaggle: /kaggle/working/ persists after training (can download from Output tab)
    # On local: ./results_incremental
    "output_dir": TRAIN_OUTPUT_DIR,

    # Split-run / continuation state
    # Set TRAIN_PHASE at the top of this file:
    #   1 -> train tasks 0-1, save continuation_state_task_1.pt
    #   2 -> load task_1 state, train task 2, save continuation_state_task_2.pt
    #   3 -> load task_2 state, train task 3, save continuation_state_task_3.pt
    #   4 -> load task_3 state, train tasks 4-5 (done)
    # These keys are used by IL/non-DFCA modes; pure DFCA ignores task filtering.
    "task_start": phase_config["task_start"],
    "task_end": phase_config["task_end"],
    "save_resume_after_task": phase_config["save_resume_after_task"],
    "resume_state_path": target,
    # Always output IL resume artifacts to /kaggle/working/ for persistence
    # (downloadable from the Kaggle Output tab).
    "resume_output_dir": TRAIN_OUTPUT_DIR,
    # Save periodic mid-task checkpoint every N rounds (for recovery on timeout/Kaggle crash)
    # None = disable; set to 5 for safe recovery without bloating disk.
    #
    # If resume_state_path is set and resume_output_dir is omitted,
    # training continues in the same output directory as the saved state.
    #"task_start": 0,
    #"task_end": 5,
    #"save_resume_after_task": None,
    #"resume_state_path": None,
    #"resume_output_dir": None,
    # Incremental Learning - 6 Tasks Distribution
    # Task 0-4: 6 classes each, Task 5: 4 classes (total 34)
    "num_clients": 100,
    "total_classes": 34,
    "base_classes": 6,       # 6 classes per task (first 5 tasks)
    "classes_per_task": 6,
    # Common Parameters
    # IoT CIC 2023: non-IID Dirichlet α=5.0 (moderate heterogeneity)
    # CGoFed Paper Eq. 14: NO proximal term! Only cross-task regularization A(Θ)
    "mu_fedprox": 0.0,  # 0.0 for CGoFed (paper doesn't have proximal term)
    "rounds_per_task": 20,  # 20 rounds/task: đủ để model hội tụ, sync thường xuyên giảm client drift
    "local_epochs": 1,  # 1 epoch/round: tránh client drift trên non-IID data
    # Giảm batch size + LR tương ứng để gradient updates nhiều hơn
    "learning_rate": 0.001,  # Giảm từ 0.001: stable gradient với EWC regularization
    "batch_size": 2048,  # Giảm từ 512: nhiều gradient steps/epoch hơn, tốt cho client ít data
    # eval_every > rounds_per_task -> chỉ eval ở post-task (round cuối mỗi task).
    # Đặt = rounds_per_task nếu muốn bật mid-task eval lại.
    "eval_every": 9999,
    "round_checkpoint_every": 5,
    # --- Algorithm Specific Params ---
    # CGoFed - RE-TUNED dựa trên training log analysis
    "mu_cgofed": 1.0,  # Paper Eq. 9: full gradient projection
    "lambda_decay": 0.8,
    "theta_threshold": 0.35,  # Tăng từ 0.20: ổn định hơn, ít reset
    "cross_task_weight": 0.3,  # Tăng từ 0.08: regularization mạnh hơn
    "lambda_cross_task": 0.3,  # Paper Eq. 14: cross-task regularization
    "energy_threshold": 0.99,
    "num_samples_rep": 1000,
    "top_k": 2,
    # EWC
    "ewc_lambda": 1000.0,  # Theo mốc scaling factor EWC được nêu trong paper cho Atari
    "fisher_samples": 200,
    "online_ewc": False,
    # LwF (FedLwF)
    "lwf_alpha": 1.0,
    "temperature": 2.0,
    "lwf_alpha_scale": 1.0,
    "distill_old_classes_only": False,
    # FedCBDR
    "tau_old": 0.9,
    "tau_new": 1.1,
    "omega_old": 1.1,
    "omega_new": 0.9,
    "buffer_size": 500,
    "replay_ratio": 0.5,
    "seed": TRAIN_SEED,
    # DER (Dynamically Expandable Representation)
    "lambda_aux": 1.0,
    "lambda_sparsity": 0.1,
    "s_max": 15.0,
    "der_temperature": 2.0,
    "der_stage1_rounds": 12,  # 60% of 20 rounds: representation learning
    "der_stage2_rounds": 8,   # 40% of 20 rounds: classifier finetuning
    # NICE (Neurogenesis Inspired Contextual Encoding)
    "tau": 0.95,
    "nice_max_phases": 20,
    "nice_phase_epochs": 1,
    "nice_context_eval": True,
    "nice_debug_context_detector": True,
    "memo_per_class": 50,
    # DeNICE Phase 2 adapter expansion.
    # Options:
    #   ["fc1"]                 -> Phase 1 MVP
    #   ["fc1", "gru"]          -> Phase 2a
    #   ["fc1", "gru", "conv3"] -> Phase 2b
    "denice_adapter_layers": ["fc1", "gru", "conv3"],
    "denice_debug": True,
    "denice_debug_store_client_details": False,
    "denice_save_round_artifacts": False,
    "denice_checkpoint_format": "delta",
    # Quick diagnostic only at the final checkpoint (task 5, round 19).
    # A stratified, full evaluation matrix still runs offline in P6 afterward.
    "denice_post_task_eval_tasks": [5],
    "denice_eval_max_clients": 3,
    "denice_eval_max_samples": 50000,
    "denice_eval_progress_every_clients": 10,  # in progress mỗi 10 clients
    "denice_eval_progress_every_batches": 0,
    # Store routed, classifier-ceiling (nomask), route confusion and a
    # representative-ensemble metric on exactly the same evaluation samples.
    "denice_eval_route_mode": "hard",
    "denice_eval_route_topk": 1,
    "denice_eval_report_nomask": True,
    "denice_eval_representative_ensemble": True,
    # DeNICE context routing bank. scope="cluster" follows the proposal:
    # share context capsule/sketches only inside the decentralized collaboration group.
    # Use scope="global" only for ablation/debug.
    "denice_shared_context_eval": True,
    "denice_shared_context_scope": "cluster",
    "denice_shared_context_max_per_episode": 512,
    # Pool binary context sketches only when all contributors share a verified
    # calibration/provenance; otherwise evaluation safely falls back to local.
    "denice_shared_context_require_compatible_calibration": True,
    "denice_router_mode": "multiclass",
    # Re-encode each small client-local context reference bank after aggregation
    # so the router never compares final-model activations with stale sketches.
    "denice_refresh_router_memory_after_aggregation": True,
    "denice_router_update_schedule": "task_end",
    "denice_router_reference_per_class": 20,
    "denice_router_refresh_batch_size": 2048,

    # DeNICE decentralized aggregation (Đề xuất §6-§7). Mặc định giữ hành vi cũ.
    # denice_aggregation_method: "weighted_mean" | "coordinate_median" | "trimmed_mean"
    "denice_aggregation_method": "weighted_mean",
    "denice_aggregation_trim_ratio": 0.1,
    "denice_gamma": 0.15,
    # G_i = {j | cùng cluster AND s_ij > delta} (§6). True = lọc theo đồ thị context.
    "denice_collab_use_context_edges": True,
    "denice_require_label_overlap": True,
    "denice_centroid_gate_threshold": 0.75,
    "denice_cluster_delta_sim": 0.0,
    # results.zip: 114/114 raw AP rounds were in [0.2365, 0.4221]; 0.5
    # rejected all of them and silently reduced training to local-only.
    "denice_cluster_theta_s": 0.20,
    "denice_cluster_edge_top_k": 40,
    "denice_cluster_edge_quantile": 0.25,
    "denice_cluster_min_signal_std": 0.02,
    # Invalid AP output must never drive neighbor aggregation.  A compatible
    # prior valid assignment is reused; otherwise every client is self-only.
    "denice_cluster_invalid_policy": "previous_valid_or_self_only",
    "denice_collaboration_guard_mode": "error",
    "denice_max_consecutive_self_only_rounds": 2,
    "denice_min_mean_peer_alpha": 0.05,
    # Prevent capacity collapse caused by union/max propagation of peer ages.
    "denice_age_merge_policy": "consensus",
    "denice_age_merge_consensus_threshold": 0.5,
    "denice_min_free_capacity_ratio": 0.10,

    # DeNICE Phase 4 graceful recycling. Keep disabled for main runs unless
    # ablation explicitly tests retired-neuron reuse.
    "denice_enable_recycling": False,
    "denice_recycle_ratio": 0.02,
    "denice_recycle_min": 1,
    "denice_recycle_max_per_layer": 8,
    "denice_recycle_grace_tasks": 1,
    # GLFC (Global-Local Forgetting Compensation)
    "glfc_memory_size": 2000,
    "glfc_entropy_threshold": 1.2,
    "glfc_distill_weight": 0.5,
    "glfc_recon_iters": 250,
    "glfc_num_recon_images": 20,
    # Re-Fed (Retrieval-Enhanced Federated Incremental Learning)
    "refed_memory_size": 2000,
    "refed_lambda_pim": 0.5,
    "refed_pim_iterations": 5,
    # Plexus (Decentralized FL without a Server - EuroMLSys 2025)
    "plexus_sample_size": 10,        # Number of training participants per round
    "plexus_num_aggregators": 1,     # Number of aggregators per round
    "plexus_success_fraction": 0.8,  # Fraction of sample needed before aggregation
    "plexus_inactivity_threshold": 50,  # Rounds before peer considered offline
    # Dynamic client scaling (simulates increasing network participation over tasks)
    "plexus_scale_clients": True,    # Enable dynamic client scaling per task
    "plexus_initial_client_ratio": 0.5,  # Initial: 50% of clients participate (task 0)
    "plexus_final_client_ratio": 1.0,   # Final: 100% of clients participate (last task)
    # DFCA-IL (unused by pure DFCA)
    "dfca_num_clusters": 10,
    "dfca_init": "global",
    "dfca_graph": "erdos_renyi",
    "dfca_connectivity": 0.15,
    "dfca_client_ratios": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
    "dfca_round_participation": 1.0,
    "dfca_aggregation": "sequential_running_average",
    "dfca_debug_messages": False,
    "dfca_debug_message_limit": 25,
    # Pure DFCA
    "dfca_participation_rate": 1.0,
    "dfca_debug_assignments": False,
    "dfca_debug_cluster_models": True,
    # Checkpoint / resume (per-round checkpoint inside each task)
    "round_checkpoint_every": 5,
}

# A JSON object supplied by a launcher can override only the fields under
# investigation without editing this production configuration.  It keeps D1
# peer/self-only smoke runs exactly comparable and records the effective config
# in the normal DeNICE artifacts.
_config_overrides_raw = os.environ.get("DENICE_CONFIG_OVERRIDES")
if _config_overrides_raw:
    try:
        _config_overrides = json.loads(_config_overrides_raw)
    except json.JSONDecodeError as exc:
        raise ValueError("DENICE_CONFIG_OVERRIDES must be a JSON object") from exc
    if not isinstance(_config_overrides, dict):
        raise ValueError("DENICE_CONFIG_OVERRIDES must decode to a JSON object")
    CONFIG.update(_config_overrides)
    print("Applied DENICE_CONFIG_OVERRIDES:", sorted(_config_overrides))


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    mode = CONFIG.get("mode", "fed_il").lower()
    algo = CONFIG.get("algorithm", "dfca_il").lower()

    if mode == "decentralized" and algo == "dfca":
        # ---- Pure DFCA ----
        import csv
        import glob as glob_module
        import json
        import random as rnd_module
        import numpy as np
        import torch
        import collections
        from datetime import datetime
        from fed_learning.data.incremental_loader import IncrementalDataLoader
        from fed_learning.models import CNN_GRU_Model
        from fed_learning.dfca import (
            run_dfca_training,
            build_dfca_checkpoint,
            find_latest_checkpoint,
        )

        if CONFIG.get("num_gpus", 0) == 0 and torch.cuda.is_available():
            CONFIG["num_gpus"] = torch.cuda.device_count()

        print("\n" + "=" * 60)
        print("DFCA - Pure Decentralized Federated Clustering Algorithm")
        print("=" * 60)

        # ---- Resolve output directory and checkpoint ----
        base_output_dir = CONFIG.get("output_dir", "/kaggle/working/results_dfca")
        explicit_ckpt = CONFIG.get("checkpoint_path")
        auto_resume = CONFIG.get("auto_resume_latest", False)
        resume_state = None
        resume_round = 0
        output_dir = None

        if explicit_ckpt and os.path.exists(explicit_ckpt):
            resume_state = torch.load(explicit_ckpt, map_location="cpu", weights_only=False)
            resume_round = resume_state.get("current_round", 0) + 1
            output_dir = base_output_dir
            print(f"\n  [Checkpoint Resume] Loading from: {explicit_ckpt}")
            print(f"  [Checkpoint Resume] Resuming from round {resume_round}")
            print(f"  [Checkpoint Resume] Writing output to: {output_dir}")
        elif auto_resume:
            ckpt_path = find_latest_checkpoint(base_output_dir)
            if ckpt_path:
                resume_state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                resume_round = resume_state.get("current_round", 0) + 1
                output_dir = os.path.dirname(ckpt_path)
                print(f"\n  [Auto-Resume] Found checkpoint: {ckpt_path}")
                print(f"  [Auto-Resume] Resuming from round {resume_round}")
                print(f"  [Auto-Resume] Writing output to: {output_dir}")
            else:
                print(f"\n  [Auto-Resume] No checkpoint found in {base_output_dir} or subdirectories.")
        else:
            print("\n  Starting fresh (no checkpoint loaded)")

        # Fresh output dir if no resume
        if output_dir is None:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"{base_output_dir}_{ts}"
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            json.dump(CONFIG, f, indent=2)

        # ---- Prepare data ----
        data_loader = IncrementalDataLoader(data_dir=CONFIG["data_dir"])
        print(f"\n{data_loader}")

        CONFIG["input_shape"] = data_loader.input_shape
        CONFIG["num_classes"] = CONFIG["total_classes"]

        node_data = {}
        all_client_ids = data_loader.get_all_client_ids()
        actual_classes = set()
        for cid in all_client_ids:
            X, y = data_loader.get_client_full_data(cid)
            if len(y) > 0:
                node_data[cid] = (X, y)
                actual_classes.update(y.unique().tolist())
        actual_classes_sorted = sorted(actual_classes)

        print(f"  Total clients in metadata: {len(all_client_ids)}")
        print(f"  Clients with data: {len(node_data)}")
        print(f"  Actual classes: {len(actual_classes_sorted)} classes -> {actual_classes_sorted}")
        print(f"  CONFIG.num_classes: {CONFIG['num_classes']}")

        model_template = CNN_GRU_Model(
            input_shape=CONFIG["input_shape"],
            num_classes=CONFIG["num_classes"],
        )

        test_X, test_y = data_loader.get_full_test_data()
        test_data = {"X_test": test_X, "y_test": test_y}
        print(f"  Full test samples: {len(test_y)}")

        # ---- Rounds ----
        checkpoint_every_raw = CONFIG.get("round_checkpoint_every", 1)
        checkpoint_every = 0 if checkpoint_every_raw is None else int(checkpoint_every_raw)
        total_rounds = int(CONFIG.get("num_rounds", 150))
        remaining_rounds = total_rounds - resume_round

        if remaining_rounds <= 0:
            print(f"\n  All {total_rounds} rounds already completed (checkpoint at round {resume_round - 1}).")
            print(f"  Writing existing data to {output_dir}...")
            if resume_state:
                hist = resume_state.get("history", {})
                if hist and hist.get("round"):
                    rms = []
                    for ri in range(len(hist["round"])):
                        rm = {
                            "round": hist["round"][ri],
                            "train_loss": hist["train_loss"][ri] if ri < len(hist["train_loss"]) else None,
                            "train_loss_std": hist["train_loss_std"][ri] if ri < len(hist["train_loss_std"]) else None,
                            "assignment_changes": hist["assignment_changes"][ri] if ri < len(hist["assignment_changes"]) else None,
                            "assignment_margin_avg": hist["assignment_margin_avg"][ri] if ri < len(hist["assignment_margin_avg"]) else None,
                            "num_messages": hist["num_messages"][ri] if ri < len(hist["num_messages"]) else None,
                            "participating_nodes": hist["participating_nodes"][ri] if ri < len(hist["participating_nodes"]) else None,
                            "cluster_distribution": hist["cluster_distribution"][ri] if ri < len(hist["cluster_distribution"]) else None,
                            "per_cluster_updates": hist["per_cluster_updates"][ri] if ri < len(hist["per_cluster_updates"]) else None,
                            "round_time": hist["round_time"][ri] if ri < len(hist["round_time"]) else None,
                            "test_loss": hist["test_loss"][ri] if ri < len(hist["test_loss"]) else None,
                            "test_accuracy": hist["test_accuracy"][ri] if ri < len(hist["test_accuracy"]) else None,
                            "test_precision_macro": hist["test_precision_macro"][ri] if ri < len(hist["test_precision_macro"]) else None,
                            "test_recall_macro": hist["test_recall_macro"][ri] if ri < len(hist["test_recall_macro"]) else None,
                            "test_precision_weighted": hist.get("test_precision_weighted", [])[ri] if ri < len(hist.get("test_precision_weighted", [])) else None,
                            "test_recall_weighted": hist.get("test_recall_weighted", [])[ri] if ri < len(hist.get("test_recall_weighted", [])) else None,
                            "test_f1_macro": hist["test_f1_macro"][ri] if ri < len(hist["test_f1_macro"]) else None,
                            "test_f1_weighted": hist["test_f1_weighted"][ri] if ri < len(hist["test_f1_weighted"]) else None,
                        }
                        rms.append(rm)
                    with open(os.path.join(output_dir, "round_metrics.json"), "w") as f:
                        json.dump(rms, f, indent=2, default=str)
                with open(os.path.join(output_dir, "message_history.json"), "w") as f:
                    json.dump(resume_state.get("message_history", []), f, indent=2, default=str)
                with open(os.path.join(output_dir, "rep_params_history.json"), "w") as f:
                    json.dump(resume_state.get("rep_params_history", []), f, indent=2, default=str)
                with open(os.path.join(output_dir, "cluster_history.json"), "w") as f:
                    json.dump(resume_state.get("cluster_history", []), f, indent=2, default=str)
                final_assign = resume_state.get("node_assignments", {})
                with open(os.path.join(output_dir, "final_cluster_assignments.json"), "w") as f:
                    json.dump(final_assign, f, indent=2)
                final_dist = collections.Counter(final_assign.values())
                with open(os.path.join(output_dir, "results.json"), "w") as f:
                    json.dump({
                        "algorithm": "dfca",
                        "total_rounds": total_rounds,
                        "already_completed": True,
                        "final_cluster_distribution": dict(sorted(final_dist.items())),
                    }, f, indent=2)
            print(f"\n{'=' * 60}")
            print(f"No training needed — all {total_rounds} rounds completed.")
            print(f"Output: {output_dir}")
            print(f"{'=' * 60}")
        else:
            print(f"\nStarting DFCA: rounds {resume_round}..{total_rounds - 1} "
                  f"({remaining_rounds} rounds), "
                  f"k={CONFIG.get('dfca_num_clusters', 10)}, "
                  f"lr={CONFIG.get('learning_rate', 0.1)}, "
                  f"local_epochs={CONFIG.get('local_epochs', 5)}, "
                  f"init={CONFIG.get('dfca_init', 'global')}, "
                  f"graph={CONFIG.get('dfca_graph', 'erdos_renyi')}(p={CONFIG.get('dfca_connectivity', 0.15)}), "
                  f"checkpoint_every={checkpoint_every}")

            message_history = list(resume_state.get("message_history", [])) if resume_state else []
            rep_params_history = list(resume_state.get("rep_params_history", [])) if resume_state else []

            # ---- Callbacks ----
            def _round_metrics_from_history(hist):
                rows = []
                for ri in range(len(hist.get("round", []))):
                    test_loss = hist["test_loss"][ri] if ri < len(hist["test_loss"]) else None
                    accuracy = hist["test_accuracy"][ri] if ri < len(hist["test_accuracy"]) else None
                    precision_macro = hist["test_precision_macro"][ri] if ri < len(hist["test_precision_macro"]) else None
                    recall_macro = hist["test_recall_macro"][ri] if ri < len(hist["test_recall_macro"]) else None
                    precision_weighted = hist.get("test_precision_weighted", [])[ri] if ri < len(hist.get("test_precision_weighted", [])) else None
                    recall_weighted = hist.get("test_recall_weighted", [])[ri] if ri < len(hist.get("test_recall_weighted", [])) else None
                    f1_macro = hist["test_f1_macro"][ri] if ri < len(hist["test_f1_macro"]) else None
                    f1_weighted = hist["test_f1_weighted"][ri] if ri < len(hist["test_f1_weighted"]) else None
                    rows.append({
                        # Spreadsheet-compatible columns. Pure DFCA is FL-only,
                        # so task is fixed to 0 and forgetting is not applicable.
                        "task": 0,
                        "round": hist["round"][ri],
                        "train_loss": hist["train_loss"][ri] if ri < len(hist["train_loss"]) else None,
                        "test_loss": test_loss,
                        "accuracy": accuracy,
                        "precision_macro": precision_macro,
                        "recall_macro": recall_macro,
                        "precision_weighted": precision_weighted,
                        "recall_weighted": recall_weighted,
                        "f1_macro": f1_macro,
                        "avg_forgetting": None,
                        "f1_weighted": f1_weighted,
                        # DFCA debug/diagnostic columns.
                        "train_loss_std": hist["train_loss_std"][ri] if ri < len(hist["train_loss_std"]) else None,
                        "assignment_changes": hist["assignment_changes"][ri] if ri < len(hist["assignment_changes"]) else None,
                        "assignment_margin_avg": hist["assignment_margin_avg"][ri] if ri < len(hist["assignment_margin_avg"]) else None,
                        "num_messages": hist["num_messages"][ri] if ri < len(hist["num_messages"]) else None,
                        "participating_nodes": hist["participating_nodes"][ri] if ri < len(hist["participating_nodes"]) else None,
                        "cluster_distribution": hist["cluster_distribution"][ri] if ri < len(hist["cluster_distribution"]) else None,
                        "per_cluster_updates": hist["per_cluster_updates"][ri] if ri < len(hist["per_cluster_updates"]) else None,
                        "round_time": hist["round_time"][ri] if ri < len(hist["round_time"]) else None,
                        "test_accuracy": accuracy,
                        "test_precision_macro": precision_macro,
                        "test_recall_macro": recall_macro,
                        "test_precision_weighted": precision_weighted,
                        "test_recall_weighted": recall_weighted,
                        "test_f1_macro": f1_macro,
                        "test_f1_weighted": f1_weighted,
                    })
                return rows

            def _write_round_metrics_artifacts(hist, current_round=None):
                rows = _round_metrics_from_history(hist)
                json_path = os.path.join(output_dir, "round_metrics.json")
                csv_path = os.path.join(output_dir, "round_metrics.csv")
                with open(json_path, "w") as f:
                    json.dump(rows, f, indent=2, default=str)
                if rows:
                    with open(csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                        writer.writeheader()
                        writer.writerows(rows)
                with open(os.path.join(output_dir, "message_history.json"), "w") as f:
                    json.dump(message_history, f, indent=2, default=str)
                with open(os.path.join(output_dir, "rep_params_history.json"), "w") as f:
                    json.dump(rep_params_history, f, indent=2, default=str)
                if current_round is not None:
                    with open(os.path.join(output_dir, "live_status.json"), "w") as f:
                        json.dump({
                            "status": "running",
                            "latest_completed_round": current_round,
                            "num_round_metric_rows": len(rows),
                            "note": "Updated during training; final files are rewritten when training completes.",
                        }, f, indent=2)
                return rows

            def round_callback(round_r, history, record):
                message_history.append({
                    "round": round_r,
                    "delivery_log": record.get("msg_log", {}),
                })
                rep_params_history.append({
                    "round": round_r,
                    "clusters": record.get("rep_params_summary", {}),
                })
                _write_round_metrics_artifacts(history, current_round=round_r)

            def checkpoint_callback(
                nodes_dict, graph_neighbors_dict, prev_assign_dict,
                history_dict, cluster_history_list, rep_params_dict,
                cfg, current_round_int, num_rounds_int,
            ):
                should_save = (
                    checkpoint_every > 0
                    and (
                        (current_round_int + 1) % checkpoint_every == 0
                        or (current_round_int + 1) >= num_rounds_int
                    )
                )
                if not should_save:
                    return
                ckpt = build_dfca_checkpoint(
                    nodes=nodes_dict,
                    graph_neighbors=graph_neighbors_dict,
                    prev_assignments=prev_assign_dict,
                    history=history_dict,
                    cluster_history=cluster_history_list,
                    representative_params=rep_params_dict,
                    config=cfg,
                    current_round=current_round_int,
                    num_rounds=num_rounds_int,
                )
                ckpt["message_history"] = list(message_history)
                ckpt["rep_params_history"] = list(rep_params_history)
                ckpt_path = os.path.join(output_dir, f"checkpoint_round_{current_round_int}.pt")
                torch.save(ckpt, ckpt_path)
                _write_round_metrics_artifacts(history_dict, current_round=current_round_int)
                print(f"  [Checkpoint] saved to {ckpt_path}")

            result = run_dfca_training(
                node_ids=sorted(node_data.keys()),
                node_data=node_data,
                model_template=model_template,
                config=CONFIG,
                test_data=test_data,
                verbose=True,
                round_callback=round_callback,
                checkpoint_callback=checkpoint_callback,
                resume_state=resume_state,
            )

            # ---- Write output files ----
            hist = result["history"]
            round_metrics = _write_round_metrics_artifacts(hist, current_round=hist["round"][-1] if hist["round"] else None)
            with open(os.path.join(output_dir, "final_cluster_assignments.json"), "w") as f:
                json.dump(result["final_assignments"], f, indent=2)
            with open(os.path.join(output_dir, "cluster_history.json"), "w") as f:
                json.dump(result["cluster_history"], f, indent=2, default=str)
            with open(os.path.join(output_dir, "graph_summary.json"), "w") as f:
                json.dump(result["graph_summary"], f, indent=2)

            final_dist = collections.Counter(result["final_assignments"].values())
            last_metrics = round_metrics[-1] if round_metrics else {}
            results_summary = {
                "algorithm": "dfca",
                "total_rounds": len(result["history"]["round"]),
                "num_clusters": CONFIG.get("dfca_num_clusters", 10),
                "final_cluster_distribution": dict(sorted(final_dist.items())),
                "graph_summary": result["graph_summary"],
                "final_metrics": {
                    "test_accuracy": last_metrics.get("test_accuracy"),
                    "test_f1_macro": last_metrics.get("test_f1_macro"),
                    "test_loss": last_metrics.get("test_loss"),
                },
            }
            with open(os.path.join(output_dir, "results.json"), "w") as f:
                json.dump(results_summary, f, indent=2)

            # Save representative models
            rep_state = {
                "config": CONFIG,
                "representative_params": result["representative_params"],
            }
            torch.save(rep_state, os.path.join(output_dir, "representative_models.pt"))

            # Save final full checkpoint
            final_ckpt = dict(result["checkpoint_state"])
            final_ckpt["message_history"] = list(message_history)
            final_ckpt["rep_params_history"] = list(rep_params_history)
            torch.save(final_ckpt, os.path.join(output_dir, "final_dfca_state.pt"))

            print(f"\n{'=' * 60}")
            print(f"DFCA Complete — Output: {output_dir}")
            print(f"  Rounds completed: {len(result['history']['round'])}")
            print(f"  Message history entries: {len(message_history)}")
            print(f"  Rep params history entries: {len(rep_params_history)}")
            print(f"  Last test accuracy: {last_metrics.get('test_accuracy')}")
            print(f"{'=' * 60}")

    else:
        from fed_learning.training.task_loop import run_incremental_training
        run_incremental_training(CONFIG)
