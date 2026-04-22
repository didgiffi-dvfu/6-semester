#!/usr/bin/env python
"""
Backend-flexible empirical NTK demo for a small 1D regression problem.

The original version of this script required PyTorch at import time. In this
workspace the main environment provides TensorFlow but not necessarily torch,
so the demo now supports:

    - `--backend auto`       : prefer PyTorch, otherwise fall back to TensorFlow
    - `--backend torch`      : require PyTorch
    - `--backend tensorflow` : require TensorFlow

The generated files keep the same names as before so they can still be reused
in reports and downstream LaTeX documents.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Dict, Tuple

_CACHE_ROOT = Path(tempfile.gettempdir()) / "ntk_demo_cache"
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_ROOT / "xdg"))
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_ROOT / "mpl"))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
for _cache_dir in (
    Path(os.environ["XDG_CACHE_HOME"]),
    Path(os.environ["MPLCONFIGDIR"]),
    Path(os.environ["XDG_CACHE_HOME"]) / "fontconfig",
):
    _cache_dir.mkdir(parents=True, exist_ok=True)

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
    import torch.nn as nn

    HAS_TORCH = True
    TORCH_IMPORT_ERROR = None
except Exception as exc:
    torch = None
    nn = None
    HAS_TORCH = False
    TORCH_IMPORT_ERROR = exc

try:
    import tensorflow as tf

    HAS_TENSORFLOW = True
    TENSORFLOW_IMPORT_ERROR = None
except Exception as exc:
    tf = None
    HAS_TENSORFLOW = False
    TENSORFLOW_IMPORT_ERROR = exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate before/after NTK plots for a small 1D regression demo."
    )
    parser.add_argument(
        "--backend",
        choices=["auto", "torch", "tensorflow"],
        default="auto",
        help="Autodiff backend used for training and NTK computation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="ntk_report_assets",
        help="Directory where plots and metrics are saved.",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=128,
        help="Hidden width of the MLP.",
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=32,
        help="Number of training samples in the 1D toy dataset.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=1500,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Adam learning rate.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    return parser.parse_args()


def choose_backend(requested: str) -> str:
    if requested == "torch":
        if not HAS_TORCH:
            raise RuntimeError(
                "PyTorch backend requested, but `torch` is not installed in the current environment."
            ) from TORCH_IMPORT_ERROR
        return "torch"

    if requested == "tensorflow":
        if not HAS_TENSORFLOW:
            raise RuntimeError(
                "TensorFlow backend requested, but `tensorflow` is not installed in the current environment."
            ) from TENSORFLOW_IMPORT_ERROR
        return "tensorflow"

    if HAS_TORCH:
        return "torch"
    if HAS_TENSORFLOW:
        return "tensorflow"

    raise RuntimeError(
        "Neither PyTorch nor TensorFlow is available, so the NTK demo cannot run."
    )


def set_seed(seed: int, backend: str) -> None:
    np.random.seed(seed)
    if backend == "torch":
        torch.manual_seed(seed)
    elif backend == "tensorflow":
        tf.random.set_seed(seed)
        tf.keras.backend.set_floatx("float64")


def analyze_ntk(ntk: np.ndarray) -> Dict[str, object]:
    eigenvalues = np.linalg.eigvalsh(ntk)
    eigenvalues_desc = np.sort(eigenvalues)[::-1]
    positive = eigenvalues_desc[eigenvalues_desc > 1e-12]
    cond = float(positive[0] / positive[-1]) if positive.size > 0 else float("inf")
    probs = positive / positive.sum() if positive.size > 0 else np.array([])
    eff_rank = float(np.exp(-(probs * np.log(probs + 1e-12)).sum())) if probs.size > 0 else 0.0
    return {
        "trace": float(np.trace(ntk)),
        "rank": int(np.linalg.matrix_rank(ntk)),
        "effective_rank": eff_rank,
        "condition_number": cond,
        "min_eigenvalue": float(np.min(eigenvalues_desc)),
        "max_eigenvalue": float(np.max(eigenvalues_desc)),
        "eigenvalues_desc": eigenvalues_desc.tolist(),
    }


def plot_heatmap(matrix: np.ndarray, path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(6.2, 5.2), dpi=160)
    image = ax.imshow(matrix, cmap="viridis", aspect="auto")
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("NTK value")
    ax.set_title(title)
    ax.set_xlabel("Sample index")
    ax.set_ylabel("Sample index")
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_heatmap_pair(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8), dpi=160)
    for ax, matrix, title in zip(
        axes,
        [before, after],
        ["NTK before training", "NTK after training"],
    ):
        image = ax.imshow(matrix, cmap="viridis", aspect="auto")
        ax.set_title(title)
        ax.set_xlabel("Sample index")
        ax.set_ylabel("Sample index")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def plot_spectrum(before: np.ndarray, after: np.ndarray, path: Path) -> None:
    eig_before = np.sort(np.linalg.eigvalsh(before))[::-1]
    eig_after = np.sort(np.linalg.eigvalsh(after))[::-1]
    idx = np.arange(1, len(eig_before) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), dpi=160)

    axes[0].plot(idx, eig_before, marker="o", linewidth=1.8, label="before training")
    axes[0].plot(idx, eig_after, marker="s", linewidth=1.8, label="after training")
    axes[0].set_xlabel("Eigenvalue index")
    axes[0].set_ylabel("Eigenvalue")
    axes[0].set_title("Eigenvalue spectrum (linear scale)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    axes[1].semilogy(idx, np.clip(eig_before, 1e-12, None), marker="o", linewidth=1.8, label="before training")
    axes[1].semilogy(idx, np.clip(eig_after, 1e-12, None), marker="s", linewidth=1.8, label="after training")
    axes[1].set_xlabel("Eigenvalue index")
    axes[1].set_ylabel("Eigenvalue")
    axes[1].set_title("Eigenvalue spectrum (log scale)")
    axes[1].grid(True, which="both", alpha=0.25)
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


if HAS_TORCH:
    class TorchMLP(nn.Module):
        def __init__(self, width: int = 128) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(1, width),
                nn.Tanh(),
                nn.Linear(width, width),
                nn.Tanh(),
                nn.Linear(width, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.net(x)


    def empirical_ntk_torch(model: nn.Module, x: torch.Tensor) -> np.ndarray:
        params = [p for p in model.parameters() if p.requires_grad]
        grad_rows = []

        for i in range(x.shape[0]):
            model.zero_grad(set_to_none=True)
            y = model(x[i : i + 1]).squeeze()
            grads = torch.autograd.grad(y, params, retain_graph=False, create_graph=False)
            flat_grad = torch.cat([g.reshape(-1) for g in grads]).detach()
            grad_rows.append(flat_grad)

        jacobian = torch.stack(grad_rows, dim=0)
        ntk = jacobian @ jacobian.T
        ntk = 0.5 * (ntk + ntk.T)
        return ntk.cpu().numpy()


    def run_torch_demo(width: int, num_points: int, steps: int, lr: float) -> Tuple[np.ndarray, np.ndarray]:
        x = torch.linspace(-2.0, 2.0, num_points, dtype=torch.float64).unsqueeze(1)
        y = torch.sin(3.0 * x) + 0.2 * x

        model = TorchMLP(width=width).to(dtype=torch.float64)
        ntk_before = empirical_ntk_torch(model, x)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        loss_fn = nn.MSELoss()
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            prediction = model(x)
            loss = loss_fn(prediction, y)
            loss.backward()
            optimizer.step()

        ntk_after = empirical_ntk_torch(model, x)
        return ntk_before, ntk_after
else:
    class TorchMLP:  # pragma: no cover - convenience placeholder
        pass


    def empirical_ntk_torch(model, x) -> np.ndarray:  # pragma: no cover - convenience placeholder
        del model, x
        raise RuntimeError("PyTorch is not available in the current environment.") from TORCH_IMPORT_ERROR


    def run_torch_demo(width: int, num_points: int, steps: int, lr: float) -> Tuple[np.ndarray, np.ndarray]:
        del width, num_points, steps, lr
        raise RuntimeError("PyTorch is not available in the current environment.") from TORCH_IMPORT_ERROR


if HAS_TENSORFLOW:
    class TensorFlowMLP(tf.keras.Model):
        def __init__(self, width: int = 128) -> None:
            super().__init__(name="TensorFlowMLP")
            self.net = tf.keras.Sequential(
                [
                    tf.keras.layers.Dense(width, activation="tanh", dtype="float64"),
                    tf.keras.layers.Dense(width, activation="tanh", dtype="float64"),
                    tf.keras.layers.Dense(1, activation="linear", dtype="float64"),
                ]
            )

        def call(self, x: tf.Tensor, training: bool = False) -> tf.Tensor:
            del training
            return self.net(x)


    def empirical_ntk_tensorflow(model: tf.keras.Model, x: tf.Tensor) -> np.ndarray:
        with tf.GradientTape(persistent=True, watch_accessed_variables=True) as tape:
            outputs = model(x, training=False)
            outputs = tf.reshape(outputs, (tf.shape(outputs)[0], -1))

        jacobian_blocks = []
        for variable in model.trainable_variables:
            jac = tape.jacobian(outputs, variable, experimental_use_pfor=False)
            jacobian_blocks.append(tf.reshape(jac, (tf.shape(outputs)[0], tf.shape(outputs)[1], -1)))
        del tape

        full_jacobian = tf.concat(jacobian_blocks, axis=-1)
        ntk = tf.einsum("bop,cop->bc", full_jacobian, full_jacobian)
        ntk = 0.5 * (ntk + tf.transpose(ntk))
        return ntk.numpy()


    def run_tensorflow_demo(width: int, num_points: int, steps: int, lr: float) -> Tuple[np.ndarray, np.ndarray]:
        x = tf.linspace(tf.constant(-2.0, dtype=tf.float64), tf.constant(2.0, dtype=tf.float64), num_points)
        x = tf.reshape(x, (-1, 1))
        y = tf.sin(3.0 * x) + 0.2 * x

        model = TensorFlowMLP(width=width)
        _ = model(tf.zeros((1, 1), dtype=tf.float64), training=False)
        ntk_before = empirical_ntk_tensorflow(model, x)

        optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        loss_fn = tf.keras.losses.MeanSquaredError()
        for _ in range(steps):
            with tf.GradientTape() as tape:
                prediction = model(x, training=True)
                loss = loss_fn(y, prediction)
            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        ntk_after = empirical_ntk_tensorflow(model, x)
        return ntk_before, ntk_after
else:
    class TensorFlowMLP:  # pragma: no cover - convenience placeholder
        pass


    def empirical_ntk_tensorflow(model, x) -> np.ndarray:  # pragma: no cover - convenience placeholder
        del model, x
        raise RuntimeError("TensorFlow is not available in the current environment.") from TENSORFLOW_IMPORT_ERROR


    def run_tensorflow_demo(width: int, num_points: int, steps: int, lr: float) -> Tuple[np.ndarray, np.ndarray]:
        del width, num_points, steps, lr
        raise RuntimeError("TensorFlow is not available in the current environment.") from TENSORFLOW_IMPORT_ERROR


def main() -> None:
    args = parse_args()
    backend = choose_backend(args.backend)
    set_seed(args.seed, backend)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    if backend == "torch":
        ntk_before, ntk_after = run_torch_demo(
            width=args.width,
            num_points=args.num_points,
            steps=args.steps,
            lr=args.lr,
        )
    else:
        ntk_before, ntk_after = run_tensorflow_demo(
            width=args.width,
            num_points=args.num_points,
            steps=args.steps,
            lr=args.lr,
        )

    plot_heatmap(ntk_before, output_dir / "ntk_heatmap_before.png", "Empirical NTK before training")
    plot_heatmap(ntk_after, output_dir / "ntk_heatmap_after.png", "Empirical NTK after training")
    plot_heatmap_pair(ntk_before, ntk_after, output_dir / "ntk_heatmaps_before_after.png")
    plot_spectrum(ntk_before, ntk_after, output_dir / "ntk_eigenspectrum_before_after.png")

    metrics = {
        "backend": backend,
        "before": analyze_ntk(ntk_before),
        "after": analyze_ntk(ntk_after),
    }
    (output_dir / "ntk_metrics_before_after.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Backend: {backend}")
    print(f"Saved demo figures to: {output_dir}")
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
