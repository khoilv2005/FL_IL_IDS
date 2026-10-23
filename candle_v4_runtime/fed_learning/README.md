# AI4FIDS - Federated Class Incremental Learning

A modular framework for **Federated Class Incremental Learning (FCIL)** applied to network intrusion detection using CNN-GRU models.

## Project Structure

```
ai4fids_project/
├── train_incremental_kaggle.py     # Entry point: CONFIG only + Kaggle setup
├── requirements.txt                # Python dependencies
├── .gitignore                      # Git ignore rules
│
├── fed_learning/                   # Core library package
│   ├── __init__.py                 # Package exports
│   │
│   ├── core/                       # Abstract base classes
│   │   ├── __init__.py
│   │   ├── trainer.py              # BaseTrainer (ABC with hooks)
│   │   └── aggregator.py           # BaseAggregator (ABC with weighted average)
│   │
│   ├── models/                     # Neural network architectures
│   │   ├── __init__.py
│   │   ├── cnn_gru.py              # CNN_GRU_Model (main architecture)
│   │   ├── der_model.py            # DERModel (dynamic multi-extractor)
│   │   └── nice_model.py           # NICEModel (neurogenesis-inspired)
│   │
│   ├── clients/                    # Federated client implementations
│   │   ├── __init__.py
│   │   ├── client.py               # FederatedClient (base class)
│   │   ├── cgofed_client.py        # CGoFedClient
│   │   ├── der_client.py           # DERClient
│   │   ├── fedcbdr_client.py       # FedCBDRClient
│   │   ├── fedlwf_client.py        # FedLwFClient
│   │   ├── glfc_client.py          # GLFCClient
│   │   ├── nice_client.py          # NICEClient
│   │   └── refed_client.py         # ReFedClient
│   │
│   ├── servers/                    # Federated server implementations
│   │   ├── __init__.py
│   │   ├── server.py               # FederatedServer (base, multi-GPU)
│   │   ├── incremental_server.py   # IncrementalServer (task tracking)
│   │   ├── cgofed_server.py        # CGoFedServer
│   │   ├── der_server.py           # DERServer
│   │   ├── fedcbdr_server.py       # FedCBDRServer
│   │   ├── fedlwf_server.py        # FedLwFServer
│   │   ├── glfc_server.py          # GLFCServer
│   │   ├── nice_server.py          # NICEServer
│   │   └── refed_server.py         # ReFedServer
│   │
│   ├── strategies/                 # Training strategies (Trainer + Aggregator pairs)
│   │   ├── __init__.py             # Strategy factory: get_strategy(), STRATEGIES registry
│   │   ├── federated/              # Standard FL strategies
│   │   │   ├── __init__.py
│   │   │   ├── fedavg.py           # FedAvg (baseline)
│   │   │   ├── fedavgm.py          # FedAvgM (server momentum)
│   │   │   ├── fedprox.py          # FedProx (proximal term)
│   │   │   └── fedplus.py          # Fed+ (dynamic regularization)
│   │   └── incremental/            # FCIL strategies
│   │       ├── __init__.py
│   │       ├── cgofed.py           # CGoFed (constrained gradient optimization)
│   │       ├── der.py              # DER (dynamically expandable representation)
│   │       ├── ewc.py              # EWC (elastic weight consolidation)
│   │       ├── fedcbdr.py          # FedCBDR (class-balancing data replay)
│   │       ├── fedlwf.py           # FedLwF (learning without forgetting)
│   │       ├── glfc.py             # GLFC (global-local forgetting compensation)
│   │       ├── nice.py             # NICE (neurogenesis inspired contextual encoding)
│   │       └── refed.py            # Re-Fed (retrieval-enhanced)
│   │
│   ├── training/                   # Multi-GPU training orchestration
│   │   ├── __init__.py
│   │   ├── base_worker.py          # BaseGPUWorker (template method pattern)
│   │   ├── runner.py               # train_federated_multigpu() entry point
│   │   ├── task_loop.py            # run_incremental_training() - main FCIL pipeline
│   │   ├── post_task.py            # post_task_processing() - end-of-task hooks
│   │   ├── worker.py               # StandardWorker (FedAvg/FedProx/EWC)
│   │   ├── cgofed_worker.py        # CGoFedWorker
│   │   ├── der_worker.py           # DERWorker
│   │   ├── fedcbdr_worker.py       # FedCBDRWorker
│   │   ├── fedlwf_worker.py        # FedLwFWorker
│   │   ├── glfc_worker.py          # GLFCWorker
│   │   ├── nice_worker.py          # NICEWorker
│   │   └── refed_worker.py         # ReFedWorker
│   │
│   ├── factories/                  # Client and server creation factories
│   │   ├── __init__.py
│   │   ├── client_factory.py       # create_clients(), registry-based client creation
│   │   └── server_factory.py       # create_server(), registry-based server creation
│   │
│   ├── data/                       # Data loading utilities
│   │   ├── __init__.py
│   │   ├── loader.py               # Basic DataLoader wrapper
│   │   └── incremental_loader.py   # IncrementalDataLoader (task-based)
│   │
│   ├── visualization/              # Plotting and metrics visualization
│   │   ├── __init__.py
│   │   ├── metrics.py              # Confusion matrix, per-class metrics
│   │   ├── plots.py                # Training plots (loss, accuracy curves)
│   │   ├── fcil_plots.py           # FCIL-specific plots (heatmaps, forgetting)
│   │   └── style.py                # IEEE-style matplotlib configuration
│   │
│   ├── utils/                      # Shared utility functions
│   │   ├── __init__.py
│   │   ├── seed.py                 # set_seed() for reproducibility
│   │   └── cleanup.py              # cleanup_temp_folders()
│   │
│   ├── dataset-metadata.json       # Kaggle dataset metadata
│   ├── README.md                   # Package documentation
│   └── README-kaggle.md            # Kaggle-specific setup guide
│
├── prepare_data/                   # Data preprocessing pipeline
│   ├── __init__.py                 # Pipeline documentation
│   ├── step1_prepare_chunks.py     # Raw CSV → chunked NPZ
│   ├── step2_federated_splits.py   # Chunks → federated client splits
│   ├── step3_visualize.py          # Visualize data distributions
│   ├── check_class_data.py         # Verify class distribution
│   ├── check_participation.py      # Verify client participation
│   ├── detect_label.py             # Quick label detection
│   └── class_names.txt             # Human-readable class labels
│
├── paper/                          # Research paper notes
│   ├── CGoFed.md
│   ├── FEDCBDR.md
│   └── FedLwF.md
│
└── tests/                          # Unit tests
    ├── conftest.py                 # Pytest configuration (sys.path setup)
    ├── helpers.py                  # Shared test utilities
    ├── test_strategy_factory.py    # Strategy registry tests
    ├── test_fedavg_fedprox.py      # FedAvg/FedProx tests
    ├── test_cgofed.py              # CGoFed + CGoFedServer tests
    ├── test_incremental_strategies.py  # EWC, FedLwF, FedCBDR tests
    ├── test_nice.py                # NICE algorithm tests
    └── test_core.py                # Model, hooks, edge cases, config tests
```

## Architecture

### Inheritance Hierarchies

#### Strategies (Trainer + Aggregator)

```
BaseTrainer (ABC)
├── FedAvgTrainer                   # CE loss only
│   ├── FedProxTrainer              # + proximal term (μ/2 ||w - w_g||²)
│   │   ├── FedProxEWCTrainer       # + EWC penalty (Fisher)
│   │   └── FedPlusTrainer          # + dynamic regularization
│   └── FedAvgEWCTrainer            # + EWC penalty (no proximal)
├── CGoFedTrainer                   # + gradient projection + cross-task reg (Eq.14)
├── FedLwFTrainer                   # + knowledge distillation
├── FedCBDRTrainer                  # + temperature-scaled CE
├── DERTrainer                      # + HAT mask annealing + aux classifier
├── NICETrainer                     # + gradient masking for mature neurons
├── GLFCTrainer                     # + distillation from global/local exemplars
└── ReFedTrainer                    # + PIM-weighted importance sampling

BaseAggregator (ABC)
├── FedAvgAggregator                # Weighted average by sample count
│   ├── FedAvgMAggregator           # + server momentum
│   ├── FedLwFAggregator            # (same as FedAvg)
│   ├── FedCBDRAggregator           # (same as FedAvg)
│   ├── GLFCAggregator              # (same as FedAvg)
│   └── ReFedAggregator             # (same as FedAvg)
├── CGoFedAggregator                # + cross-task blending (Eq.11) + personalization (Eq.12)
├── DERAggregator                   # + mask-aware parameter merging
└── NICEAggregator                  # + frozen mature neuron protection
```

#### Clients

```
FederatedClient
├── CGoFedClient                    # + representation extraction
├── FedLwFClient                    # + set_task_data(), teacher snapshot
├── FedCBDRClient                   # + replay buffer, GDR, leverage sampling
├── DERClient                       # + exemplar buffer, herding selection
├── NICEClient                      # + phase-based training, age/mask transfer
├── GLFCClient                      # + exemplar management, distillation
└── ReFedClient                     # + PIM cache, importance-weighted replay
```

#### Servers

```
FederatedServer                     # Multi-GPU dispatch, train_round()
└── IncrementalServer               # + task tracking, seen_classes
    ├── CGoFedServer                # + SVD space, Eq.12/14 per-client reg
    ├── DERServer                   # + two-stage training, exemplar coordination
    ├── FedCBDRServer               # + GDR coordination
    ├── FedLwFServer                # + model snapshots
    ├── NICEServer                  # + age transition, mask coordination
    ├── GLFCServer                  # + exemplar coordination, proxy reconstruction
    └── ReFedServer                 # + PIM caching coordination
```

#### Training Workers (GPU parallelism)

```
BaseGPUWorker                       # Template method: run() with hooks
├── StandardWorker                  # Default for FedAvg/FedProx/EWC
├── CGoFedWorker                    # + init_params from per-client reg info
├── DERWorker                       # + DERModel, two-stage, annealing
├── FedCBDRWorker                   # + replay kwargs
├── FedLwFWorker                    # (empty subclass)
├── NICEWorker                      # + age/mask transfer, NICEModel
├── GLFCWorker                      # + exemplar set update
└── ReFedWorker                     # (empty subclass)
```

### Design Patterns

| Pattern | Usage |
|---------|-------|
| **Strategy** | `get_strategy()` factory returns (Trainer, Aggregator) pair |
| **Template Method** | `BaseGPUWorker.run()` defines skeleton; subclasses override hooks |
| **Factory Method** | `create_clients()`, `create_server()` in `factories/` module |
| **Observer** | Trainer hooks: `pre_train()`, `post_train()`, `pre_step()`, `post_step()` |
| **Registry** | `STRATEGIES` dict, `_CLIENT_REGISTRY`, `_SERVER_REGISTRY` map names to classes |

### Adding a New Algorithm

1. **Strategy**: Create `strategies/fed_incremental/your_algo.py` with `YourTrainer(BaseTrainer)` and `YourAggregator(BaseAggregator)`
2. **Register**: Add to `STRATEGIES` dict in `strategies/__init__.py`
3. **Client**: If needed, create `clients/your_client.py` extending `FederatedClient`
4. **Server**: If needed, create `servers/your_server.py` extending `IncrementalServer`
5. **Worker**: If GPU training needs customization, create `training/your_worker.py` extending `BaseGPUWorker`
6. **Factories**: Register in `factories/client_factory.py` and `factories/server_factory.py`
7. **Config**: Add algorithm-specific params to `CONFIG` in `train_incremental_kaggle.py`
8. **Tests**: Add `tests/test_your_algo.py`

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run tests
python -m pytest tests/ -v

# Train (on Kaggle or locally with data)
python train_incremental_kaggle.py
```

## Supported Algorithms

| Algorithm | Type | Anti-Forgetting Method |
|-----------|------|----------------------|
| FedAvg | Federated | None (baseline) |
| FedAvgM | Federated | Server momentum |
| FedProx | Federated | Proximal regularization |
| Fed+ | Federated | Dynamic regularization |
| CGoFed | FCIL | Constrained gradient + cross-task regularization |
| EWC | FCIL | Fisher information penalty |
| FedLwF | FCIL | Knowledge distillation |
| FedCBDR | FCIL | Class-balanced data replay |
| DER | FCIL | Dynamic expandable representation + HAT masks |
| NICE | FCIL | Neurogenesis-inspired neuron aging |
| GLFC | FCIL | Global-local forgetting compensation |
| Re-Fed | FCIL | Retrieval-enhanced PIM caching |
