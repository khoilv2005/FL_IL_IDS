
# ============================================================ #
# Download resume state from Google Drive
# ============================================================ #
import os
import subprocess
import sys
import zipfile

TRAIN_PHASE = 2  # 1: task 0-1, 2: task 2, 3: task 3, 4: task 4-5

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
    - "decentralized": decentralized Plexus FL (no server)

    Sau đó chọn thuật toán qua CONFIG["algorithm"]:
    - fed_il: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf",
              "fedprox_lwf", "fedcbdr", "der", "nice", "glfc", "refed",
              "plexus"
    - decentralized: "plexus" (uses PlexusTrainer/Aggregator)
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
    #   - "decentralized": Plexus decentralized FL (no server)
    "mode": "fed_il",
    # Algorithm Selection
    # fed_il: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf",
    #         "fedprox_lwf", "fedcbdr", "der", "nice", "glfc", "refed",
    #         "plexus", "plexus_der", "plexus_nice"
    # il:     "ewc", "lwf", "der", "nice"
    "algorithm": "fedcbdr",
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
    # "task_start": phase_config["task_start"],
    # "task_end": phase_config["task_end"],
    # "save_resume_after_task": phase_config["save_resume_after_task"],
    # "resume_state_path": target,
    # # Always output to /kaggle/working/ for persistence (downloadable from Output tab)
    # "resume_output_dir": "/kaggle/working/results_incremental",
    # # Save periodic mid-task checkpoint every N rounds (for recovery on timeout/Kaggle crash)
    # # None = disable; set to 5 for safe recovery without bloating disk

    # If resume_state_path is set and resume_output_dir is omitted,
    # training continues in the same output directory as the saved state.
    "task_start": 0,
    "task_end": 5,
    "save_resume_after_task": None,
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
    "ewc_lambda": 400.0,  # Theo mốc scaling factor EWC được nêu trong paper cho Atari
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
    "plexus_sample_size": 4,        # Number of training participants per round
    "plexus_num_aggregators": 1,     # Number of aggregators per round
    "plexus_success_fraction": 0.8,  # Fraction of sample needed before aggregation
    "plexus_inactivity_threshold": 50,  # Rounds before peer considered offline
}


# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    from fed_learning.training.task_loop import run_incremental_training

    run_incremental_training(CONFIG)
