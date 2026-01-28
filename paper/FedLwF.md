# FedLwF: Federated Learning without Forgetting

## Overview

FedLwF (Federated Learning without Forgetting) is a regularization-based approach for **Federated Class-Incremental Learning (FCIL)**. It combines **FedAvg** for federated aggregation with **Knowledge Distillation (KD)** to prevent catastrophic forgetting when learning new classes.

## Key Concept

Unlike replay-based methods (like FedCBDR) that store and replay old data, FedLwF uses the **old model's predictions** as soft labels to preserve knowledge about previous tasks. This approach:

1. **Preserves Privacy**: No need to store or transmit old data samples
2. **Memory Efficient**: Only stores model snapshots, not data
3. **Regularization-based**: Constrains learning to prevent forgetting

## Algorithm Details

### Loss Function

The FedLwF loss combines cross-entropy loss for new task learning with knowledge distillation loss:

$$\mathcal{L} = \mathcal{L}_{CE}(y, \hat{y}) + \alpha \cdot T^2 \cdot D_{KL}(\sigma(z_{old}/T) \| \sigma(z_{new}/T))$$

Where:

- $\mathcal{L}_{CE}$: Cross-entropy loss for classification
- $D_{KL}$: Kullback-Leibler divergence
- $\alpha$: Distillation weight (controls old vs new knowledge balance)
- $T$: Temperature parameter (softens probability distributions)
- $z_{old}$: Logits from the old (frozen) model
- $z_{new}$: Logits from the current model
- $\sigma$: Softmax function

### Training Process

```
For each task t:
    1. Save model snapshot M_t-1 (teacher)
    2. For each federated round:
        a. Distribute global model to clients
        b. Each client:
            - Forward pass through current model (student)
            - Forward pass through old model (teacher) for old classes
            - Compute combined loss (CE + KD)
            - Backpropagate and update
        c. Server aggregates using FedAvg
    3. Save final task model as next teacher
```

### Distillation Strategy

- **Old Class Outputs Only**: KD is applied only to the output units corresponding to previously learned classes
- **Temperature Scaling**: Higher temperature (T > 1) produces softer probability distributions, making knowledge transfer smoother
- **Adaptive Alpha**: α typically increases with task number since more old knowledge needs to be preserved

## Implementation Structure

```
fed_learning/
├── strategies/incremental/
│   └── fedlwf.py              # FedLwFTrainer, FedLwFAggregator
├── clients/
│   └── fedlwf_client.py       # FedLwFClient with KD support
├── servers/
│   └── fedlwf_server.py       # FedLwFServer with task management
├── training/
│   └── fedlwf_worker.py       # Multi-GPU training worker
└── train_fedlwf_kaggle.py     # Main training script
```

## Hyperparameters

| Parameter                  | Default | Description                    |
| -------------------------- | ------- | ------------------------------ |
| `distillation_alpha`       | 1.0     | Base weight for KD loss        |
| `distillation_alpha_scale` | 1.5     | Scale factor per task          |
| `temperature`              | 2.0     | Softmax temperature            |
| `proximal_mu`              | 0.0     | FedProx coefficient (optional) |

## Usage

### In Kaggle Notebook

```python
from train_fedlwf_kaggle import main

# Run training
results = main()
```

### Custom Configuration

```python
from train_fedlwf_kaggle import get_config, main

# Get default config
config = get_config()

# Customize
config["distillation_alpha"] = 2.0
config["temperature"] = 3.0
config["num_rounds"] = 20

# Run with custom config
# ... modify main() to accept config parameter
```

## Comparison with Other Methods

| Method  | Approach                | Memory             | Privacy | Complexity |
| ------- | ----------------------- | ------------------ | ------- | ---------- |
| FedLwF  | Regularization (KD)     | Low (model only)   | High    | Medium     |
| FedCBDR | Replay + Distillation   | Medium (exemplars) | Medium  | High       |
| FedEWC  | Regularization (Fisher) | Low                | High    | Low        |
| CGoFed  | Generator + Replay      | High (generator)   | High    | High       |

## Advantages

1. **No Data Storage**: Does not require storing old samples
2. **Privacy Preserving**: Old data never leaves the client
3. **Simple Implementation**: Easy to integrate with existing FedAvg
4. **Scalable**: Memory cost does not grow with number of classes

## Limitations

1. **Model Capacity**: Requires sufficient model capacity for all tasks
2. **Cold Start**: First task has no regularization
3. **Accumulating Constraints**: May limit plasticity for new tasks
4. **No True Replay**: Soft labels may not capture all data characteristics

## References

1. Li, Z., & Hoiem, D. (2017). Learning without Forgetting. IEEE TPAMI.
2. McMahan, B., et al. (2017). Communication-Efficient Learning of Deep Networks from Decentralized Data. AISTATS.

## Citation

```bibtex
@article{li2017learning,
  title={Learning without forgetting},
  author={Li, Zhizhong and Hoiem, Derek},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  year={2017}
}
```
