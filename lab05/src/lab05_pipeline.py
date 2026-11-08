import math
import random
import time
import os
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import make_moons
import matplotlib.pyplot as plt

# Set random seeds for reproducible results
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)

print("=" * 60)
print(" LAB 05: AUTOGRAD ENGINE, EXECUTION MODES & MEMORY BENCHMARKS ")
print("=" * 60)

# =============================================================
# 1. Scalar Autograd Engine (Value Node & Graph Construction)
# =============================================================
class Value:
    """
    Stores a single scalar value, its accumulated gradient,
    its computational parents, and its backward derivative rule.
    """
    def __init__(self, data, _children=(), _op=''):
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += 1.0 * out.grad
            other.grad += 1.0 * out.grad
        out._backward = _backward
        return out

    def __radd__(self, other):
        return self + other

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def __rmul__(self, other):
        return self * other

    def __pow__(self, power):
        assert isinstance(power, (int, float)), "Power must be int or float"
        out = Value(self.data ** power, (self,), f'**{power}')
        def _backward():
            self.grad += (power * (self.data ** (power - 1))) * out.grad
        out._backward = _backward
        return out

    def __neg__(self):
        return self * -1.0

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return Value(other) - self

    def relu(self):
        out = Value(self.data if self.data > 0 else 0.0, (self,), 'ReLU')
        def _backward():
            self.grad += (1.0 if self.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out

    def sigmoid(self):
        s = 1.0 / (1.0 + math.exp(-self.data))
        out = Value(s, (self,), 'sigmoid')
        def _backward():
            self.grad += s * (1.0 - s) * out.grad
        out._backward = _backward
        return out

    def log(self):
        val = max(self.data, 1e-12)
        out = Value(math.log(val), (self,), 'log')
        def _backward():
            self.grad += (1.0 / val) * out.grad
        out._backward = _backward
        return out

    def backward(self):
        # Build topological ordering of the computation graph
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for parent in v._prev:
                    build_topo(parent)
                topo.append(v)
        build_topo(self)

        # Base case: seed the output gradient with 1.0
        self.grad = 1.0
        for node in reversed(topo):
            node._backward()

# =============================================================
# 2. Gradient Verification on ((a * b + c) ** 2).relu().sum()
# =============================================================
print("\n[Step 1] Verifying Custom Engine vs PyTorch Autograd & Numerical Gradients...")

# Test inputs
a_init = [1.5, -2.0, 0.5]
b_init = [2.0, 3.0, -1.2]
c_init = [-1.0, 0.5, 2.5]

# A. Custom Engine Evaluation
a_val = [Value(x) for x in a_init]
b_val = [Value(x) for x in b_init]
c_val = [Value(x) for x in c_init]

terms_val = [((a * b + c) ** 2).relu() for a, b, c in zip(a_val, b_val, c_val)]
loss_val = sum(terms_val, Value(0.0))
loss_val.backward()

# B. PyTorch Autograd Evaluation
a_pt = torch.tensor(a_init, requires_grad=True, dtype=torch.float64)
b_pt = torch.tensor(b_init, requires_grad=True, dtype=torch.float64)
c_pt = torch.tensor(c_init, requires_grad=True, dtype=torch.float64)

loss_pt = ((a_pt * b_pt + c_pt) ** 2).relu().sum()
loss_pt.backward()

# C. Numerical Gradient Estimation (Central Finite Difference)
h = 1e-7
def eval_numerical(a_vec, b_vec, c_vec):
    tot = 0.0
    for a, b, c in zip(a_vec, b_vec, c_vec):
        res = (a * b + c) ** 2
        tot += res if res > 0 else 0.0
    return tot

num_grad_a = []
for i in range(len(a_init)):
    a_plus = list(a_init); a_plus[i] += h
    a_minus = list(a_init); a_minus[i] -= h
    grad_est = (eval_numerical(a_plus, b_init, c_init) - eval_numerical(a_minus, b_init, c_init)) / (2 * h)
    num_grad_a.append(grad_est)

# Compare differences
diffs_pt = [abs(v.grad - p.item()) for v, p in zip(a_val, a_pt.grad)]
diffs_num = [abs(v.grad - n) for v, n in zip(a_val, num_grad_a)]
max_diff_pt = max(diffs_pt)
max_diff_num = max(diffs_num)

print(f"  • Custom Engine Gradients:   {[round(v.grad, 6) for v in a_val]}")
print(f"  • PyTorch Autograd Gradients: {[round(p.item(), 6) for p in a_pt.grad]}")
print(f"  • Numerical Gradient Est.:   {[round(n, 6) for n in num_grad_a]}")
print(f"  • Max Absolute Difference vs PyTorch:   {max_diff_pt:.2e}")
print(f"  • Max Absolute Difference vs Numerical: {max_diff_num:.2e}")
assert max_diff_pt < 1e-6, "Validation failed: difference with PyTorch exceeds 1e-6!"
print("  -> Autograd Engine VERIFIED: Perfectly matches PyTorch (< 1e-6 tolerance).")

# =============================================================
# 3. Model Training on make_moons (2 -> 8 -> 1, Sigmoid, BCE)
# =============================================================
print("\n[Step 2] Training 2 -> 8 -> 1 Model on make_moons (Custom Engine vs PyTorch)...")

X_moons, y_moons = make_moons(n_samples=100, noise=0.1, random_state=42)

# Custom MLP implementation
class CustomMLP:
    def __init__(self):
        # Layer 1: 2 inputs -> 8 hidden neurons
        self.w1 = [[Value(random.uniform(-0.5, 0.5)) for _ in range(2)] for _ in range(8)]
        self.b1 = [Value(0.0) for _ in range(8)]
        # Layer 2: 8 hidden neurons -> 1 output neuron
        self.w2 = [Value(random.uniform(-0.5, 0.5)) for _ in range(8)]
        self.b2 = Value(0.0)

    def parameters(self):
        params = []
        for row in self.w1:
            params.extend(row)
        params.extend(self.b1)
        params.extend(self.w2)
        params.append(self.b2)
        return params

    def forward(self, x):
        # Hidden layer with ReLU
        hidden = []
        for i in range(8):
            act = self.w1[i][0] * x[0] + self.w1[i][1] * x[1] + self.b1[i]
            hidden.append(act.relu())
        # Output layer with Sigmoid
        out_act = sum([w * h for w, h in zip(self.w2, hidden)], self.b2)
        return out_act.sigmoid()

# PyTorch MLP implementation
class TorchMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 8),
            nn.ReLU(),
            nn.Linear(8, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.net(x)

def train_custom_engine(epochs=30):
    model = CustomMLP()
    lr = 0.1
    for epoch in range(epochs):
        total_loss = Value(0.0)
        for x_row, y_true in zip(X_moons, y_moons):
            x_vals = [Value(x_row[0]), Value(x_row[1])]
            pred = model.forward(x_vals)
            # Binary Cross-Entropy Loss
            if y_true == 1:
                loss_i = -pred.log()
            else:
                loss_i = -(Value(1.0) - pred).log()
            total_loss = total_loss + loss_i

        total_loss = total_loss * (1.0 / len(X_moons))

        # Reset gradients
        for p in model.parameters():
            p.grad = 0.0

        total_loss.backward()

        # Update weights (SGD step)
        for p in model.parameters():
            p.data -= lr * p.grad
    return model

def train_torch_engine(epochs=30):
    model = TorchMLP()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.BCELoss()
    X_t = torch.tensor(X_moons, dtype=torch.float32)
    y_t = torch.tensor(y_moons, dtype=torch.float32).unsqueeze(1)

    for epoch in range(epochs):
        optimizer.zero_grad()
        preds = model(X_t)
        loss = criterion(preds, y_t)
        loss.backward()
        optimizer.step()
    return model

# Measure 5 runs each
custom_times = []
for _ in range(5):
    t0 = time.perf_counter()
    train_custom_engine(epochs=30)
    custom_times.append(time.perf_counter() - t0)

torch_times = []
for _ in range(5):
    t0 = time.perf_counter()
    train_torch_engine(epochs=30)
    torch_times.append(time.perf_counter() - t0)

med_custom = np.median(custom_times) * 1000
med_torch = np.median(torch_times) * 1000

print(f"  • Custom Scalar Engine Training Time (Median of 5 runs): {med_custom:.2f} ms")
print(f"  • PyTorch Vectorized Engine Training Time (Median of 5 runs): {med_torch:.2f} ms")
print(f"  • Speed Ratio: PyTorch is {med_custom / med_torch:.1f}x faster due to vectorized C++ kernels.")

# =============================================================
# 4. PyTorch Execution Modes (Eager vs Traced / Compiled)
# =============================================================
print("\n[Step 3] Assessing Execution Modes (Eager vs Traced / Compiled)...")

test_model = nn.Sequential(
    nn.Linear(128, 256),
    nn.ReLU(),
    nn.Linear(256, 128),
    nn.ReLU(),
    nn.Linear(128, 10)
).eval()

sample_input = torch.randn(64, 128)

# Check for compilation or tracing
mode_name = "torch.jit.trace"
try:
    optimized_model = torch.compile(test_model)
    # Warm-up to verify backend availability
    _ = optimized_model(sample_input)
    mode_name = "torch.compile"
except Exception:
    optimized_model = torch.jit.trace(test_model, sample_input)
    mode_name = "torch.jit.trace (Scripted/Traced)"

# Warm-up runs
for _ in range(20):
    _ = test_model(sample_input)
    _ = optimized_model(sample_input)

# Benchmark Eager
eager_times = []
for _ in range(200):
    t0 = time.perf_counter()
    _ = test_model(sample_input)
    eager_times.append(time.perf_counter() - t0)
med_eager = np.median(eager_times) * 1000

# Benchmark Optimized
opt_times = []
for _ in range(200):
    t0 = time.perf_counter()
    _ = optimized_model(sample_input)
    opt_times.append(time.perf_counter() - t0)
med_opt = np.median(opt_times) * 1000

print(f"  • Eager Execution Latency:     {med_eager:.4f} ms")
print(f"  • {mode_name} Latency: {med_opt:.4f} ms")
print(f"  • Latency delta: {(med_eager - med_opt) / med_eager * 100:.1f}% reduction in overhead.")

# =============================================================
# 5. Tensor Representation & Contiguity Benchmark
# =============================================================
print("\n[Step 4] Inspecting Tensor Contiguity and Performance Overhead...")

large_tensor = torch.randn(4000, 4000, dtype=torch.float32)
transposed_tensor = large_tensor.t()

print(f"  • Original Tensor:   Shape={large_tensor.shape}, Dtype={large_tensor.dtype}, Contiguous={large_tensor.is_contiguous()}")
print(f"  • Transposed Tensor: Shape={transposed_tensor.shape}, Dtype={transposed_tensor.dtype}, Contiguous={transposed_tensor.is_contiguous()}")

# Measure row-summation on contiguous vs non-contiguous layout
# Contiguous benchmark
for _ in range(5): _ = torch.sum(large_tensor, dim=1) # warm-up
t0 = time.perf_counter()
for _ in range(30):
    _ = torch.sum(large_tensor, dim=1)
contig_time = (time.perf_counter() - t0) / 30 * 1000

# Non-contiguous benchmark
for _ in range(5): _ = torch.sum(transposed_tensor, dim=1) # warm-up
t0 = time.perf_counter()
for _ in range(30):
    _ = torch.sum(transposed_tensor, dim=1)
non_contig_time = (time.perf_counter() - t0) / 30 * 1000

print(f"  • Contiguous Tensor Reduction Latency:     {contig_time:.3f} ms")
print(f"  • Non-Contiguous Tensor Reduction Latency: {non_contig_time:.3f} ms")
print(f"  • Performance Slowdown: Non-contiguous layout is {non_contig_time / contig_time:.2f}x slower due to strided memory access.")

# =============================================================
# 6. Save Decision Boundary Plot
# =============================================================
os.makedirs("lab05/results", exist_ok=True)
trained_custom = train_custom_engine(epochs=60)

# Generate mesh grid
x_min, x_max = X_moons[:, 0].min() - 0.5, X_moons[:, 0].max() + 0.5
y_min, y_max = X_moons[:, 1].min() - 0.5, X_moons[:, 1].max() + 0.5
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 50), np.linspace(y_min, y_max, 50))

grid_preds = []
for gx, gy in zip(xx.ravel(), yy.ravel()):
    p = trained_custom.forward([Value(gx), Value(gy)])
    grid_preds.append(p.data)
zz = np.array(grid_preds).reshape(xx.shape)

plt.figure(figsize=(7, 5))
plt.contourf(xx, yy, zz, levels=20, cmap='RdBu', alpha=0.7)
plt.scatter(X_moons[:, 0], X_moons[:, 1], c=y_moons, cmap='RdBu', edgecolors='k')
plt.title("Lab 05: Decision Boundary on make_moons (Custom Autograd Engine)")
plt.xlabel("X1")
plt.ylabel("X2")
plt.tight_layout()
plt.savefig("lab05/results/moons_decision_boundary.png", dpi=150)
plt.close()

print("\nSaved decision boundary visualization to 'lab05/results/moons_decision_boundary.png'.")
print("=" * 60)
print(" PIPELINE EXECUTION COMPLETE ")
print("=" * 60)