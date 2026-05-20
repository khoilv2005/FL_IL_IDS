
# ============================================================ #
# Download resume state from Google Drive
# ============================================================ #
import os
import subprocess
import sys
import zipfile

TRAIN_PHASE = 1  # 1: task 0-1, 2: task 2, 3: task 3, 4: task 4-5

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
    - "decentralized": decentralized FL (Plexus or pure DFCA)

    Sau đó chọn thuật toán qua CONFIG["algorithm"]:
    - fed_il: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf",
              "fedprox_lwf", "fedcbdr", "der", "nice", "glfc", "refed",
              "plexus", "plexus_der", "plexus_nice", "dfca_il"
    - decentralized: "plexus" (Plexus), "dfca" (pure DFCA paper)
    - il: "ewc", "lwf", "der", "nice"

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
    "random_seed": 42,  # Set to None for random behavior
    # Training Mode
    # Options:
    #   - "fed_il": federated incremental learning
    #   - "il": local incremental learning
    #   - "decentralized": Plexus decentralized FL (no server),DFCA
    "mode": "decentralized",
    # Algorithm Selection
    # fed_il: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf",
    #         "fedprox_lwf", "fedcbdr", "der", "nice", "glfc", "refed",
    #         "plexus", "plexus_der", "plexus_nice"
    # il:     "ewc", "lwf", "der", "nice"
    "algorithm": "dfca",
    # Output - Use Kaggle's output directory for persistent storage
    # On Kaggle: /kaggle/working/ persists after training (can download from Output tab)
    # On local: ./results_incremental
    "output_dir": "/kaggle/working/results_incremental",

    # Split-run / continuation state
    # Set TRAIN_PHASE at the top of this file:
    #   1 -> train tasks 0-1, save continuation_state_task_1.pt
    #   2 -> load task_1 state, train task 2, save continuation_state_task_2.pt
    #   3 -> load task_2 state, train task 3, save continuation_state_task_3.pt
    #   4 -> load task_3 state, train tasks 4-5 (done)
    "task_start": phase_config["task_start"],
    "task_end": phase_config["task_end"],
    "save_resume_after_task": phase_config["save_resume_after_task"],
    "resume_state_path": target,
    # Always output to /kaggle/working/ for persistence (downloadable from Output tab)
    "resume_output_dir": "/kaggle/working/results_incremental",
    # Save periodic mid-task checkpoint every N rounds (for recovery on timeout/Kaggle crash)
    # None = disable; set to 5 for safe recovery without bloating disk

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
    "eval_every": 1,
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
    "seed": 42,
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
    "nice_context_eval": False,
    "nice_debug_context_detector": False,
    "memo_per_class": 50,
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
    # DFCA-IL (Decentralized Federated Clustering Algorithm with Incremental Learning)
    "dfca_num_clusters": 10,          # Number of clusters k (fixed)
    "dfca_init": "global",           # Initialization: "global" (DFCA-GI) or "local"
    "dfca_graph": "erdos_renyi",      # Graph type: Erdos-Renyi random graph
    "dfca_connectivity": 0.15,       # Edge probability for Erdos-Renyi graph
    "dfca_client_ratios": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],  # Active clients per task
    "dfca_round_participation": 1.0,  # Fraction of active clients participating per round
    "dfca_aggregation": "sequential_running_average",  # Aggregation method
    "dfca_debug_messages": True,          # Enable detailed message passing debug logs
    "dfca_debug_message_limit": 25,        # Max debug log lines per round (0=unlimited)
    # ---- Pure DFCA params (used when mode="decentralized" AND algorithm="dfca") ----
    # Overrides the dfca_il params above when running pure DFCA
    "dfca_participation_rate": 1.0,   # Fraction of nodes participating per round (pure DFCA)
    "dfca_debug_assignments": True,   # Log per-node assignment details
    "dfca_debug_cluster_models": True, # Log cluster collapse / zero-update warnings
}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    mode = CONFIG.get("mode", "fed_il").lower()
    algo = CONFIG.get("algorithm", "").lower()

    if mode == "decentralized" and algo == "dfca":
        # ---- Pure DFCA ----
        import torch
        from datetime import datetime
        from fed_learning.data.incremental_loader import IncrementalDataLoader
        from fed_learning.models import CNN_GRU_Model
        from fed_learning.dfca import run_dfca_training

        if CONFIG.get("num_gpus", 0) == 0 and torch.cuda.is_available():
            CONFIG["num_gpus"] = torch.cuda.device_count()

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"{CONFIG.get('output_dir', '/kaggle/working/results_dfca')}_{ts}"
        import os
        os.makedirs(output_dir, exist_ok=True)

        with open(os.path.join(output_dir, "config.json"), "w") as f:
            import json
            json.dump(CONFIG, f, indent=2)

        print("\n" + "=" * 60)
        print("DFCA - Pure Decentralized Federated Clustering Algorithm")
        print("=" * 60)

        data_loader = IncrementalDataLoader(data_dir=CONFIG["data_dir"])
        print(f"\n{data_loader}")

        CONFIG["input_shape"] = data_loader.input_shape
        CONFIG["num_classes"] = CONFIG["total_classes"]

        node_data = {}
        for cid in data_loader.get_all_client_ids():
            X, y = data_loader.get_client_data(cid, task_id=0)
            if len(y) > 0:
                node_data[cid] = (X, y)

        print(f"  Nodes with data: {len(node_data)}")

        model_template = CNN_GRU_Model(
            input_shape=CONFIG["input_shape"],
            num_classes=CONFIG["num_classes"],
        )

        test_X, test_y = data_loader.get_test_data(task_id=0, cumulative=True)
        test_data = {"X_test": test_X, "y_test": test_y}
        print(f"  Test samples: {len(test_y)}")

        # Collectors for detailed round info
        round_records_collector = []
        message_history = []
        rep_params_history = []

        def round_callback(round_r, history, record):
            round_records_collector.append(record)
            if "msg_log" in record:
                message_history.append({
                    "round": round_r,
                    "nodes": record["msg_log"],
                })
            if "rep_params_summary" in record:
                rep_params_history.append({
                    "round": round_r,
                    "clusters": record["rep_params_summary"],
                })
            if checkpoint_every and (round_r + 1) % checkpoint_every == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_round_{round_r}.pt")
                torch.save({
                    "round": round_r,
                    "config": CONFIG,
                    "history": history,
                    "message_history": list(message_history),
                    "rep_params_history": list(rep_params_history),
                }, ckpt_path)
                print(f"  [Checkpoint] saved to {ckpt_path}")

        print(f"\nStarting DFCA: {CONFIG.get('num_rounds', 150)} rounds, "
              f"k={CONFIG.get('dfca_num_clusters', 2)}, "
              f"lr={CONFIG.get('learning_rate', 0.1)}, "
              f"local_epochs={CONFIG.get('local_epochs', 5)}, "
              f"init={CONFIG.get('dfca_init', 'global')}, "
              f"graph={CONFIG.get('dfca_graph', 'erdos_renyi')}(p={CONFIG.get('dfca_connectivity', 0.15)})")

        result = run_dfca_training(
            node_ids=sorted(node_data.keys()),
            node_data=node_data,
            model_template=model_template,
            config=CONFIG,
            test_data=test_data,
            verbose=True,
            round_callback=round_callback,
        )

        # Build output files
        round_metrics = []
        message_history = []
        rep_params_history = []
        for r in range(len(result["history"]["round"])):
            rm = {
                "round": result["history"]["round"][r],
                "train_loss": result["history"]["train_loss"][r],
                "train_loss_std": result["history"]["train_loss_std"][r],
                "assignment_changes": result["history"]["assignment_changes"][r],
                "assignment_margin_avg": result["history"]["assignment_margin_avg"][r],
                "num_messages": result["history"]["num_messages"][r],
                "participating_nodes": result["history"]["participating_nodes"][r],
                "cluster_distribution": result["history"]["cluster_distribution"][r],
                "per_cluster_updates": result["history"]["per_cluster_updates"][r],
                "round_time": result["history"]["round_time"][r],
                "test_loss": result["history"]["test_loss"][r],
                "test_accuracy": result["history"]["test_accuracy"][r],
                "test_precision_macro": result["history"]["test_precision_macro"][r],
                "test_recall_macro": result["history"]["test_recall_macro"][r],
                "test_f1_macro": result["history"]["test_f1_macro"][r],
                "test_f1_weighted": result["history"]["test_f1_weighted"][r],
            }
            round_metrics.append(rm)

        # Use collected data from round_callback
        with open(os.path.join(output_dir, "round_metrics.json"), "w") as f:
            json.dump(round_metrics, f, indent=2, default=str)

        with open(os.path.join(output_dir, "message_history.json"), "w") as f:
            json.dump(message_history, f, indent=2, default=str)

        with open(os.path.join(output_dir, "rep_params_history.json"), "w") as f:
            json.dump(rep_params_history, f, indent=2, default=str)

        with open(os.path.join(output_dir, "final_cluster_assignments.json"), "w") as f:
            json.dump(result["final_assignments"], f, indent=2)

        with open(os.path.join(output_dir, "cluster_history.json"), "w") as f:
            json.dump(result["cluster_history"], f, indent=2, default=str)

        with open(os.path.join(output_dir, "graph_summary.json"), "w") as f:
            json.dump(result["graph_summary"], f, indent=2)

        import collections
        final_dist = collections.Counter(result["final_assignments"].values())
        results_summary = {
            "algorithm": "dfca",
            "total_rounds": len(result["history"]["round"]),
            "num_clusters": CONFIG.get("dfca_num_clusters", 10),
            "final_cluster_distribution": dict(sorted(final_dist.items())),
            "graph_summary": result["graph_summary"],
            "final_metrics": {
                "test_accuracy": round_metrics[-1]["test_accuracy"] if round_metrics else None,
                "test_f1_macro": round_metrics[-1]["test_f1_macro"] if round_metrics else None,
                "test_loss": round_metrics[-1]["test_loss"] if round_metrics else None,
            },
        }
        with open(os.path.join(output_dir, "results.json"), "w") as f:
            json.dump(results_summary, f, indent=2)

        # ---- Final representative model params (full tensors) ----
        rep_state = {
            "config": CONFIG,
            "representative_params": result["representative_params"],
        }
        torch.save(rep_state, os.path.join(output_dir, "representative_models.pt"))

        torch.save({
            "config": CONFIG,
            "final_assignments": result["final_assignments"],
            "cluster_history": result["cluster_history"],
            "graph_summary": result["graph_summary"],
            "round_metrics": round_metrics,
            "message_history": message_history,
            "rep_params_history": rep_params_history,
        }, os.path.join(output_dir, "final_dfca_state.pt"))

        print(f"\n{'=' * 60}")
        print(f"DFCA Complete — Output: {output_dir}")
        print(f"{'=' * 60}")

    else:
        from fed_learning.training.task_loop import run_incremental_training
        run_incremental_training(CONFIG)
