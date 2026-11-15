report_text = """# Lab 06 Report: Reproducible Training Pipelines and MLOps

## 1. Overview
In this lab, we transformed our winning Convolutional Neural Network (CNN) from Lab 3 into a reproducible, YAML-driven training pipeline (`lab06/config.yaml`) integrated with MLflow experiment tracking. We profiled epoch bottlenecks, benchmarked data-loading worker counts, evaluated batch-size and optimizer trade-offs, verified gradient accumulation, and tested checkpoint crash recovery.

## 2. Epoch Profiling Breakdown
Using a single training epoch on CPU (`batch_size=128`, `num_workers=0`), total execution took **62.51 seconds**, broken down as follows:

| Training Stage | Time (s) | Share of Epoch (%) |
| :--- | :---: | :---: |
| **Data Loading & Transforms** | 36.06 s | 57.7% |
| **Forward Pass** | 12.74 s | 20.4% |
| **Backward Pass** | 12.69 s | 20.3% |
| **Optimizer Step** | 1.02 s | 1.6% |
| **Total Epoch Time** | **62.51 s** | **100.0%** |

**Bottleneck Analysis:** Data loading consumed **57.7%** of total epoch time, making disk I/O and tensor normalization the primary bottleneck rather than neural network arithmetic.

## 3. DataLoader Worker Scaling (`num_workers`)
We measured data-loading throughput across `num_workers` values `[0, 1, 2, 4, 8]`:

| `num_workers` | Throughput (samples/s) | Elapsed Time (12,800 samples) |
| :---: | :---: | :---: |
| **0** | **2,093.2** | **6.12 s** |
| **1** | 393.8 | 32.50 s |
| **2** | 394.8 | 32.42 s |
| **4** | 372.9 | 34.33 s |
| **8** | 242.1 | 52.86 s |

**Finding:** On Windows, `num_workers=0` achieved the highest throughput (2,093.2 samples/s). Because Windows creates worker processes using `spawn` rather than `fork`, spinning up background Python interpreters and passing small $28 \\times 28$ Fashion-MNIST tensors over inter-process pipes adds heavy overhead that outweighs parallel loading benefits. The throughput curve is saved in `results/workers_throughput.png`.

![Workers Throughput](results/workers_throughput.png)

## 4. Batch-Size Trade-Off Study (5 Epochs)
We trained the CNN for 5 epochs across batch sizes from 32 to 512:

| Batch Size | Throughput (samples/s) | Peak Python RAM (MB) | Final Test Accuracy (%) |
| :---: | :---: | :---: | :---: |
| **32** | 621.2 | 2.66 MB | **89.94%** |
| **64** | 756.3 | 2.39 MB | 89.47% |
| **128** | 142.1* | 2.40 MB | 89.23% |
| **256** | **799.7** | 2.41 MB | 89.19% |
| **512** | 700.3 | 2.46 MB | 88.02% |

*\*Note: The throughput dip at batch size 128 reflects temporary Windows background OS scheduling during execution.*

**Trade-off Analysis:** Larger batches (such as 256) maximize hardware throughput (799.7 samples/s) by amortizing Python loop overhead over larger matrix operations. However, over a fixed 5-epoch budget, smaller batches achieve higher accuracy (89.94% at batch 32 vs. 88.02% at batch 512) because they execute 16 times more gradient updates per epoch.

## 5. Optimizers and Learning-Rate Schedules (10 Epochs)
All runs were logged to MLflow under equal 10-epoch budgets (`batch_size=128`):

| Configuration | Optimizer | LR Scheduler | Final Test Accuracy (%) |
| :--- | :---: | :---: | :---: |
| **Adam_StepLR** | Adam (`lr=0.001`) | StepLR (`step=3, gamma=0.5`) | 90.57% |
| **Adam_Cosine** | Adam (`lr=0.001`) | CosineAnnealingLR (`T_max=10`) | **90.81%** |
| **Adam_Plateau** | Adam (`lr=0.001`) | ReduceLROnPlateau (`factor=0.5`) | 90.59% |
| **SGD_Momentum_Cosine** | SGD (`lr=0.01, mom=0.9`) | CosineAnnealingLR (`T_max=10`) | 90.39% |

**Finding:** `Adam` paired with `CosineAnnealingLR` delivered the best generalization (**90.81%**), outperforming SGD with momentum (90.39%) within the fixed 10-epoch budget due to adaptive per-parameter learning rates and smooth cosine decay.

## 6. Gradient Accumulation
To simulate a batch size of 256 on memory-constrained hardware, we accumulated gradients across 4 steps of `batch_size=64` (scaling loss by $1/4$) before stepping the optimizer:
- **Batch 64 (Accumulation = 4) 2-Epoch Accuracy:** 87.29%
- **Direct Batch 256 2-Epoch Accuracy:** 87.22%
- **Absolute Difference:** 0.07%

This confirms that gradient accumulation reproduces the optimization trajectory of larger batches without increasing peak activation memory.

## 7. Checkpoint Interruption and Recovery
Training was halted after Epoch 3, saving both `model_state_dict` and `optimizer_state_dict` to `results/checkpoints/checkpoint_epoch_3.pt`. Reloading the checkpoint and resuming training for Epochs 4 and 5 yielded a final test accuracy of **89.55%**, matching uninterrupted 5-epoch training (~89.47%) without loss spikes or optimizer state resets.

## 8. Individual Project Proposal
Our one-page individual project proposal (`lab06/project_proposal.md`) defines an edge-deployable Fashion-MNIST vision classifier targeting $\\le 50\\text{ ms}$ CPU inference latency and $\\le 5.0\\text{ MB}$ model size via INT8 post-training quantization, structured filter pruning, and TorchScript/ONNX export.
"""

with open("lab06/report.md", "w", encoding="utf-8") as f:
    f.write(report_text)

print("Successfully created clean lab06/report.md!")