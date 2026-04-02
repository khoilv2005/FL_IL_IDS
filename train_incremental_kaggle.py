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
# KAGGLE SETUP
# =============================================================================
MODULE_PATH = "/kaggle/input/ai4fids-fedlearning-modules"


def setup_imports():
    """Setup imports for both nested and flattened dataset structures."""
    if not os.path.exists(MODULE_PATH):
        print(f"Warning: Module path {MODULE_PATH} not found!")
        return

    # Case 1: Standard structure
    pkg_path = os.path.join(MODULE_PATH, "fed_learning")
    if os.path.exists(pkg_path):
        print(f"Found standard package structure at {pkg_path}")
        if MODULE_PATH not in sys.path:
            sys.path.insert(0, MODULE_PATH)
        return

    # Case 2: Flattened structure - create symlink
    init_path = os.path.join(MODULE_PATH, "__init__.py")
    if os.path.exists(init_path):
        print(f"Found flattened package structure at {MODULE_PATH}")
        try:
            tmp_dir = "/tmp/fed_pkg_fix"
            os.makedirs(tmp_dir, exist_ok=True)
            symlink_path = os.path.join(tmp_dir, "fed_learning")

            if os.path.exists(symlink_path):
                os.remove(symlink_path)

            os.symlink(MODULE_PATH, symlink_path)

            if tmp_dir not in sys.path:
                sys.path.insert(0, tmp_dir)

            print(f"Created symlink {symlink_path} -> {MODULE_PATH}")
        except Exception as e:
            print(f"Failed to create symlink: {e}")


setup_imports()


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # Data
    "data_dir": "/kaggle/input/data-10clients",
    # Reproducibility
    "random_seed": 42,  # Set to None for random behavior
    # Training Mode
    # Options:
    #   - "fed_il": federated incremental learning
    #   - "il": local incremental learning
    #   - "decentralized": Plexus decentralized FL (no server)
    "mode": "decentralized",
    # Algorithm Selection
    # fed_il: "cgofed", "fedavg_ewc", "fedprox_ewc", "fedavg_lwf",
    #         "fedprox_lwf", "fedcbdr", "der", "nice", "glfc", "refed",
    #         "plexus"
    # decentralized: "plexus"
    # il:     "ewc", "lwf", "der", "nice"
    "algorithm": "plexus_nice",
    # Output
    "output_dir": "./results_incremental",
    # Incremental Learning - 5 Tasks Distribution
    "num_clients": 10,
    "total_classes": 34,
    "base_classes": 10,
    "classes_per_task": 6,
    # Common Parameters
    # IoT CIC 2023: non-IID Dirichlet α=5.0 (moderate heterogeneity)
    # CGoFed Paper Eq. 14: NO proximal term! Only cross-task regularization A(Θ)
    "mu_fedprox": 0.0,  # 0.0 for CGoFed (paper doesn't have proximal term)
    "rounds_per_task": 5,  # Giữ 5 rounds: sync thường xuyên giảm client drift trên non-IID data
    "local_epochs": 2,  # Tăng từ 2: nhiều gradient updates hơn nhưng không quá cao gây drift
    # Giảm batch size + LR tương ứng để gradient updates nhiều hơn
    "learning_rate": 0.001,  # Giảm từ 0.002: scale theo batch_size nhỏ hơn
    "batch_size": 512,  # Giảm từ 512: nhiều gradient steps/epoch hơn, tốt cho client ít data
    "eval_every": 1,
    # --- Algorithm Specific Params ---
    # CGoFed - RE-TUNED dựa trên training log analysis
    "mu_cgofed": 1.0,
    "lambda_decay": 0.8,
    "theta_threshold": 0.20,
    "cross_task_weight": 0.08,
    "lambda_cross_task": 0.08,
    "energy_threshold": 0.99,
    "num_samples_rep": 2000,
    "top_k": 2,
    # EWC
    "ewc_lambda": 1000.0,
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
    "buffer_size": 1000,
    "replay_ratio": 0.5,
    "seed": 42,
    # DER (Dynamically Expandable Representation)
    "lambda_aux": 1.0,
    "lambda_sparsity": 0.1,
    "s_max": 15.0,
    "der_temperature": 2.0,
    "der_stage1_rounds": 5,
    "der_stage2_rounds": 3,
    # NICE (Neurogenesis Inspired Contextual Encoding)
    "tau": 0.95,
    "nice_max_phases": 5,
    "nice_phase_epochs": 5,
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
