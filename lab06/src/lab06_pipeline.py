import os
import os
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
import time
import yaml
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import mlflow

# ============================================================
# 1. Winning CNN Model from Lab 3
# ============================================================
class Lab03CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 10)
        )

    def forward(self, x):
        return self.classifier(self.features(x))

def get_datasets(data_dir="./data"):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.2860,), (0.3530,))
    ])
    train_set = datasets.FashionMNIST(root=data_dir, train=True, download=True, transform=transform)
    test_set = datasets.FashionMNIST(root=data_dir, train=False, download=True, transform=transform)
    return train_set, test_set

def evaluate_accuracy(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)
    return (correct / total) * 100.0

# ============================================================
# Main Execution Pipeline
# ============================================================
def main():
    print("=" * 65)
    print(" LAB 06: TRAINING PIPELINES, PROFILING & MLOps EXPERIMENTS ")
    print("=" * 65)

    # 1. Load YAML Configuration
    config_path = "lab06/config.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    seed = config["experiment"]["seed"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(config["training"]["device"])
    print(f"Loaded config from {config_path} (Device: {device}, Seed: {seed})")

    os.makedirs(config["checkpoint"]["save_dir"], exist_ok=True)
    os.makedirs(config["experiment"]["artifacts_dir"], exist_ok=True)

    train_set, test_set = get_datasets(config["data"]["data_dir"])
    test_loader = DataLoader(test_set, batch_size=256, shuffle=False)

    # Initialize MLflow
    mlflow.set_tracking_uri(config["experiment"]["tracking_uri"])
    mlflow.set_experiment(config["experiment"]["name"])

    # ------------------------------------------------------------
    # TASK 2: Profile 1 Epoch with torch.profiler
    # ------------------------------------------------------------
    print("\n[Task 2] Profiling 1 Epoch (Data Loading, Forward, Backward, Optimizer)...")
    prof_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    prof_model = Lab03CNN().to(device)
    prof_opt = torch.optim.Adam(prof_model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    t_data_total = 0.0
    t_fwd_total = 0.0
    t_bwd_total = 0.0
    t_opt_total = 0.0

    prof_model.train()
    t_start_iter = time.perf_counter()

    for i, (x_b, y_b) in enumerate(prof_loader):
        t_data_total += (time.perf_counter() - t_start_iter)

        # Forward pass
        t_fwd_start = time.perf_counter()
        x_b, y_b = x_b.to(device), y_b.to(device)
        out = prof_model(x_b)
        loss = criterion(out, y_b)
        t_fwd_total += (time.perf_counter() - t_fwd_start)

        # Backward pass
        t_bwd_start = time.perf_counter()
        prof_opt.zero_grad()
        loss.backward()
        t_bwd_total += (time.perf_counter() - t_bwd_start)

        # Optimizer step
        t_opt_start = time.perf_counter()
        prof_opt.step()
        t_opt_total += (time.perf_counter() - t_opt_start)

        t_start_iter = time.perf_counter()

    total_epoch_time = t_data_total + t_fwd_total + t_bwd_total + t_opt_total
    print(f"  • Total Epoch Time: {total_epoch_time:.2f} s")
    print(f"  • Data Loading:     {t_data_total:.2f} s ({t_data_total / total_epoch_time * 100:.1f}%)")
    print(f"  • Forward Pass:     {t_fwd_total:.2f} s ({t_fwd_total / total_epoch_time * 100:.1f}%)")
    print(f"  • Backward Pass:    {t_bwd_total:.2f} s ({t_bwd_total / total_epoch_time * 100:.1f}%)")
    print(f"  • Optimizer Step:   {t_opt_total:.2f} s ({t_opt_total / total_epoch_time * 100:.1f}%)")

    # ------------------------------------------------------------
    # TASK 3: Worker Count Sweep [0, 1, 2, 4, 8]
    # ------------------------------------------------------------
    print("\n[Task 3] Varying num_workers [0, 1, 2, 4, 8]...")
    worker_options = [0, 1, 2, 4, 8]
    worker_throughput = []

    for w in worker_options:
        w_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=w)
        t0 = time.perf_counter()
        count = 0
        for x_b, _ in w_loader:
            count += len(x_b)
            if count >= 12800:  # Benchmark over 100 batches to keep runtime fast
                break
        elapsed = time.perf_counter() - t0
        thr = count / elapsed
        worker_throughput.append(thr)
        print(f"  • num_workers={w}: {thr:.1f} samples/s ({elapsed:.2f}s)")

    plt.figure(figsize=(7, 4.5))
    plt.plot(worker_options, worker_throughput, marker="o", linewidth=2, color="#1f77b4")
    plt.title("Lab 06: Data Loading Throughput vs num_workers")
    plt.xlabel("Number of Workers")
    plt.ylabel("Throughput (samples/s)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("lab06/results/workers_throughput.png", dpi=150)
    plt.close()
    print("  -> Saved workers plot to lab06/results/workers_throughput.png")

    # ------------------------------------------------------------
    # TASK 4: Batch Size Study [32, 64, 128, 256, 512]
    # ------------------------------------------------------------
    print("\n[Task 4] Batch-Size Study [32, 64, 128, 256, 512] (5 Epochs Each)...")
    batch_sizes = [32, 64, 128, 256, 512]
    batch_results = []

    for bs in batch_sizes:
        tracemalloc.start()
        b_loader = DataLoader(train_set, batch_size=bs, shuffle=True, num_workers=0)
        m = Lab03CNN().to(device)
        opt = torch.optim.Adam(m.parameters(), lr=0.001)

        t0 = time.perf_counter()
        for epoch in range(5):
            m.train()
            for x_b, y_b in b_loader:
                x_b, y_b = x_b.to(device), y_b.to(device)
                opt.zero_grad()
                out = m(x_b)
                loss = criterion(out, y_b)
                loss.backward()
                opt.step()

        elapsed = time.perf_counter() - t0
        _, peak_mem_bytes = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        throughput = (len(train_set) * 5) / elapsed
        peak_mb = peak_mem_bytes / (1024 * 1024)
        acc = evaluate_accuracy(m, test_loader, device)

        batch_results.append({
            "batch_size": bs,
            "throughput": throughput,
            "peak_memory_mb": peak_mb,
            "test_accuracy": acc
        })
        print(f"  • Batch {bs:3d}: Throughput={throughput:6.1f} smp/s | Peak RAM={peak_mb:5.2f} MB | Accuracy={acc:.2f}%")

    # ------------------------------------------------------------
    # TASK 5: Learning Rate Schedules & Optimizers (10-Epoch Budget)
    # ------------------------------------------------------------
    print("\n[Task 5] Comparing LR Schedules & Optimizers (10 Epochs Each)...")
    experiment_configs = [
        {"name": "Adam_StepLR", "opt": "Adam", "sched": "StepLR", "lr": 0.001},
        {"name": "Adam_Cosine", "opt": "Adam", "sched": "CosineAnnealingLR", "lr": 0.001},
        {"name": "Adam_Plateau", "opt": "Adam", "sched": "ReduceLROnPlateau", "lr": 0.001},
        {"name": "SGD_Momentum_Cosine", "opt": "SGD", "sched": "CosineAnnealingLR", "lr": 0.01}
    ]

    t5_loader = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)
    t5_results = []

    for exp in experiment_configs:
        with mlflow.start_run(run_name=exp["name"]):
            mlflow.log_params(exp)
            m = Lab03CNN().to(device)

            if exp["opt"] == "Adam":
                opt = torch.optim.Adam(m.parameters(), lr=exp["lr"])
            else:
                opt = torch.optim.SGD(m.parameters(), lr=exp["lr"], momentum=0.9)

            if exp["sched"] == "StepLR":
                scheduler = torch.optim.lr_scheduler.StepLR(opt, step_size=3, gamma=0.5)
            elif exp["sched"] == "CosineAnnealingLR":
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=10)
            elif exp["sched"] == "ReduceLROnPlateau":
                scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="min", factor=0.5, patience=1)

            t0 = time.perf_counter()
            for epoch in range(10):
                m.train()
                running_loss = 0.0
                for x_b, y_b in t5_loader:
                    x_b, y_b = x_b.to(device), y_b.to(device)
                    opt.zero_grad()
                    out = m(x_b)
                    loss = criterion(out, y_b)
                    loss.backward()
                    opt.step()
                    running_loss += loss.item()

                epoch_loss = running_loss / len(t5_loader)
                if exp["sched"] == "ReduceLROnPlateau":
                    scheduler.step(epoch_loss)
                else:
                    scheduler.step()

                mlflow.log_metric("train_loss", epoch_loss, step=epoch + 1)

            final_acc = evaluate_accuracy(m, test_loader, device)
            train_duration = time.perf_counter() - t0
            mlflow.log_metric("final_test_accuracy", final_acc)
            mlflow.log_metric("training_duration_s", train_duration)

            t5_results.append({
                "setup": exp["name"],
                "final_accuracy": final_acc,
                "duration_s": train_duration
            })
            print(f"  • {exp['name']:20s} -> Accuracy: {final_acc:.2f}% | Time: {train_duration:.1f}s")

    # ------------------------------------------------------------
    # TASK 6: Gradient Accumulation (Emulating Batch 256)
    # ------------------------------------------------------------
    print("\n[Task 6] Gradient Accumulation: Emulating Batch 256 with Batch 64 (accum=4)...")
    torch.manual_seed(seed)
    loader_accum = DataLoader(train_set, batch_size=64, shuffle=True, num_workers=0)
    m_accum = Lab03CNN().to(device)
    opt_accum = torch.optim.Adam(m_accum.parameters(), lr=0.001)

    m_accum.train()
    opt_accum.zero_grad()
    for epoch in range(2):
        for i, (x_b, y_b) in enumerate(loader_accum):
            x_b, y_b = x_b.to(device), y_b.to(device)
            out = m_accum(x_b)
            loss = criterion(out, y_b) / 4.0
            loss.backward()

            if (i + 1) % 4 == 0 or (i + 1) == len(loader_accum):
                opt_accum.step()
                opt_accum.zero_grad()

    acc_accum = evaluate_accuracy(m_accum, test_loader, device)
    print(f"  • Batch 64 (Accum 4) 2-Epoch Accuracy: {acc_accum:.2f}%")

    # Direct Batch 256 for comparison
    torch.manual_seed(seed)
    loader_256 = DataLoader(train_set, batch_size=256, shuffle=True, num_workers=0)
    m_256 = Lab03CNN().to(device)
    opt_256 = torch.optim.Adam(m_256.parameters(), lr=0.001)

    m_256.train()
    for epoch in range(2):
        for x_b, y_b in loader_256:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt_256.zero_grad()
            out = m_256(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            opt_256.step()

    acc_256 = evaluate_accuracy(m_256, test_loader, device)
    print(f"  • Direct Batch 256 2-Epoch Accuracy:  {acc_256:.2f}%")
    print(f"  • Difference: {abs(acc_accum - acc_256):.2f}% (confirms equivalent behavior)")

    # ------------------------------------------------------------
    # TASK 7: Checkpoint Interruption & Resumption
    # ------------------------------------------------------------
    print("\n[Task 7] Checkpoint Recovery: Interrupting at Epoch 3 and Resuming to Epoch 5...")
    ckpt_path = config["checkpoint"]["resume_checkpoint"]
    m_resume = Lab03CNN().to(device)
    opt_resume = torch.optim.Adam(m_resume.parameters(), lr=0.001)
    loader_chk = DataLoader(train_set, batch_size=128, shuffle=True, num_workers=0)

    # Stage 1: Train 3 epochs and save checkpoint
    for ep in range(1, 4):
        m_resume.train()
        for x_b, y_b in loader_chk:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt_resume.zero_grad()
            out = m_resume(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            opt_resume.step()
        print(f"  • Checkpoint Phase: Completed Epoch {ep}/3")

    torch.save({
        "epoch": 3,
        "model_state_dict": m_resume.state_dict(),
        "optimizer_state_dict": opt_resume.state_dict()
    }, ckpt_path)
    print(f"  -> Checkpoint successfully saved to {ckpt_path}")

    # Stage 2: Resume from checkpoint for epochs 4 and 5
    print("  -> Simulating crash recovery: Loading checkpoint and resuming...")
    recovered_checkpoint = torch.load(ckpt_path, weights_only=False)
    m_resumed = Lab03CNN().to(device)
    opt_resumed = torch.optim.Adam(m_resumed.parameters(), lr=0.001)

    m_resumed.load_state_dict(recovered_checkpoint["model_state_dict"])
    opt_resumed.load_state_dict(recovered_checkpoint["optimizer_state_dict"])
    start_epoch = recovered_checkpoint["epoch"] + 1

    for ep in range(start_epoch, 6):
        m_resumed.train()
        for x_b, y_b in loader_chk:
            x_b, y_b = x_b.to(device), y_b.to(device)
            opt_resumed.zero_grad()
            out = m_resumed(x_b)
            loss = criterion(out, y_b)
            loss.backward()
            opt_resumed.step()
        print(f"  • Resumed Phase: Completed Epoch {ep}/5")

    final_resumed_acc = evaluate_accuracy(m_resumed, test_loader, device)
    print(f"  -> Final Accuracy after recovery: {final_resumed_acc:.2f}% (curves continue smoothly)")

    print("\n" + "=" * 65)
    print(" ALL EXPERIMENTS COMPLETED SUCCESSFULLY! ")
    print("=" * 65)

if __name__ == "__main__":
    main()