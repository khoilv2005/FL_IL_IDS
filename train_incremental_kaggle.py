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
import shutil

# =============================================================================
# GITHUB UPDATE
# =============================================================================
GITHUB_REPO_URL = "https://github.com/khoilv2005/FL_IL_IDS.git"
GITHUB_BRANCH = "main"


def update_from_github(github_url: str = GITHUB_REPO_URL, branch: str = GITHUB_BRANCH, force: bool = False):
    """
    Clone or pull the latest version of the project from GitHub.

    This function will:
    1. Check if the project directory exists
    2. If exists and force=True, remove and re-clone
    3. If exists and force=False, pull latest changes
    4. If not exists, clone the repository

    Args:
        github_url: GitHub repository URL
        branch: Branch to clone/pull
        force: If True, remove existing directory and re-clone

    Returns:
        bool: True if update was successful, False otherwise
    """
    import subprocess

    project_dir = "/kaggle/working/FL_IL_IDS"
    fed_learning_src = os.path.join(project_dir, "fed_learning")
    fed_learning_dest = "/kaggle/working/fed_learning"

    def run_cmd(cmd, cwd=None):
        """Run shell command and return success status."""
        try:
            result = subprocess.run(
                cmd,
                shell=True,
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=300
            )
            if result.returncode != 0:
                print(f"⚠️ Command failed: {cmd}")
                print(f"   stdout: {result.stdout[:500] if result.stdout else ''}")
                print(f"   stderr: {result.stderr[:500] if result.stderr else ''}")
                return False
            return True
        except Exception as e:
            print(f"⚠️ Exception running command: {e}")
            return False

    print("\n" + "=" * 60)
    print("🔄 GITHUB UPDATE CHECK")
    print("=" * 60)

    # Check git availability
    if not run_cmd("which git"):
        print("⚠️ Git not available, skipping update")
        return False

    if os.path.exists(project_dir):
        if force:
            print(f"🗑️ Removing existing project directory (force=True)...")
            shutil.rmtree(project_dir)
            print(f"📥 Cloning {github_url} (branch: {branch})...")
            if run_cmd(f"git clone -b {branch} {github_url} {project_dir}"):
                print(f"✅ Successfully cloned {github_url}")
                return _copy_fed_learning(fed_learning_src, fed_learning_dest)
        else:
            print(f"📦 Project exists, pulling latest changes...")
            if run_cmd("git fetch origin", cwd=project_dir):
                if run_cmd(f"git reset --hard origin/{branch}", cwd=project_dir):
                    print(f"✅ Successfully pulled latest changes")
                    return _copy_fed_learning(fed_learning_src, fed_learning_dest)
    else:
        print(f"📥 Cloning {github_url} (branch: {branch})...")
        if run_cmd(f"git clone -b {branch} {github_url} {project_dir}"):
            print(f"✅ Successfully cloned {github_url}")
            return _copy_fed_learning(fed_learning_src, fed_learning_dest)

    return False


def _copy_fed_learning(src: str, dest: str) -> bool:
    """Copy fed_learning from source to destination."""
    try:
        if os.path.exists(src):
            if os.path.exists(dest):
                shutil.rmtree(dest)
            shutil.copytree(src, dest)
            print(f"📁 Copied {src} -> {dest}")
            return True
    except Exception as e:
        print(f"⚠️ Failed to copy fed_learning: {e}")
    return False


# =============================================================================
# KAGGLE SETUP
# =============================================================================
# Priority order for module paths:
# 1. /kaggle/working/fed_learning (from GitHub clone - freshest code)
# 2. /kaggle/input/ai4fids-fedlearning-modules/fed_learning (Kaggle dataset)
# 3. /kaggle/input/ai4fids-fedlearning-modules (flattened structure)
MODULE_PATHS = [
    "/kaggle/working/fed_learning",  # GitHub clone (priority)
    "/kaggle/input/ai4fids-fedlearning-modules",  # Kaggle dataset
]


def setup_imports(use_github_fresh: bool = False):
    """
    Setup imports for Kaggle environment.

    Args:
        use_github_fresh: If True, force use of /kaggle/working/fed_learning
                         (from GitHub clone) regardless of other paths.
    """
    # If GitHub fresh code is requested, use it directly
    github_path = "/kaggle/working/fed_learning"
    if use_github_fresh and os.path.exists(github_path):
        print(f"📦 Using fresh code from GitHub: {github_path}")
        if github_path not in sys.path:
            sys.path.insert(0, github_path)
        return

    # Try each module path in priority order
    for base_path in MODULE_PATHS:
        if not os.path.exists(base_path):
            continue

        # Case 1: Standard structure (nested fed_learning folder)
        pkg_path = os.path.join(base_path, "fed_learning")
        if os.path.exists(pkg_path) and os.path.exists(os.path.join(pkg_path, "__init__.py")):
            print(f"📦 Found standard package structure at {pkg_path}")
            if pkg_path not in sys.path:
                sys.path.insert(0, pkg_path)
            return

        # Case 2: Flattened structure (fed_learning/__init__.py at base_path)
        init_path = os.path.join(base_path, "__init__.py")
        if os.path.exists(init_path):
            print(f"📦 Found flattened package structure at {base_path}")
            try:
                tmp_dir = "/tmp/fed_pkg_fix"
                os.makedirs(tmp_dir, exist_ok=True)
                symlink_path = os.path.join(tmp_dir, "fed_learning")

                if os.path.exists(symlink_path):
                    os.remove(symlink_path)

                os.symlink(base_path, symlink_path)

                if tmp_dir not in sys.path:
                    sys.path.insert(0, tmp_dir)

                print(f"🔗 Created symlink {symlink_path} -> {base_path}")
                return
            except Exception as e:
                print(f"⚠️ Failed to create symlink: {e}")
                continue

    print(f"⚠️ Warning: No fed_learning module found in any path!")
    print(f"   Searched paths: {MODULE_PATHS}")


# =============================================================================
# CONFIGURATION
# =============================================================================
CONFIG = {
    # GitHub Update
    "update_from_github": True,  # Set True to pull latest code from GitHub
    "github_force_clone": False,  # If True, remove existing and re-clone
    "github_url": "https://github.com/khoilv2005/FL_IL_IDS.git",
    "github_branch": "main",
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
    "algorithm": "plexus",
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
    # GitHub Update (optional)
    if CONFIG.get("update_from_github", False):
        github_url = CONFIG.get("github_url", GITHUB_REPO_URL)
        github_branch = CONFIG.get("github_branch", GITHUB_BRANCH)
        force_clone = CONFIG.get("github_force_clone", False)

        print(f"\n🔍 Checking for GitHub updates...")
        print(f"   URL: {github_url}")
        print(f"   Branch: {github_branch}")
        print(f"   Force clone: {force_clone}")

        updated = update_from_github(
            github_url=github_url,
            branch=github_branch,
            force=force_clone
        )

        if updated:
            print("✅ GitHub update completed successfully")
            # Re-setup imports after update - use fresh GitHub code
            setup_imports(use_github_fresh=True)
        else:
            print("⚠️ GitHub update failed or skipped, using existing modules")
            setup_imports()  # Fallback to Kaggle dataset modules

    from fed_learning.training.task_loop import run_incremental_training

    print("\n" + "=" * 60)
    print(f"🚀 STARTING TRAINING")
    print(f"   Mode: {CONFIG['mode']}")
    print(f"   Algorithm: {CONFIG['algorithm']}")
    print("=" * 60 + "\n")

    run_incremental_training(CONFIG)
