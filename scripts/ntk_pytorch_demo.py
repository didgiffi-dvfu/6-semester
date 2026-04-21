#!/usr/bin/env python
"""
Small PyTorch demo for empirical NTK analysis on a toy 1D regression problem.

It produces:
    - NTK heatmap before training
    - NTK heatmap after training
    - side-by-side comparison
    - eigenvalue spectrum before/after training

The script is intentionally compact so it can be cited in a course report.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


class MLP(nn.Module):
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


def empirical_ntk(model: nn.Module, x: torch.Tensor) -> np.ndarray:
    """
    Compute the empirical NTK for scalar outputs:
        K_ij = <grad_theta f(x_i), grad_theta f(x_j)>
    """
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


def analyze_ntk(ntk: np.ndarray) -> dict:
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


def main() -> None:
    set_seed(42)
    output_dir = Path("ntk_report_assets")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Toy one-dimensional regression dataset.
    x = torch.linspace(-2.0, 2.0, 32).unsqueeze(1)
    y = torch.sin(3.0 * x) + 0.2 * x

    model = MLP(width=128)
    ntk_before = empirical_ntk(model, x)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.MSELoss()
    for _ in range(1500):
        optimizer.zero_grad(set_to_none=True)
        prediction = model(x)
        loss = loss_fn(prediction, y)
        loss.backward()
        optimizer.step()

    ntk_after = empirical_ntk(model, x)

    plot_heatmap(ntk_before, output_dir / "ntk_heatmap_before.png", "Empirical NTK before training")
    plot_heatmap(ntk_after, output_dir / "ntk_heatmap_after.png", "Empirical NTK after training")
    plot_heatmap_pair(ntk_before, ntk_after, output_dir / "ntk_heatmaps_before_after.png")
    plot_spectrum(ntk_before, ntk_after, output_dir / "ntk_eigenspectrum_before_after.png")

    metrics = {
        "before": analyze_ntk(ntk_before),
        "after": analyze_ntk(ntk_after),
    }
    (output_dir / "ntk_metrics_before_after.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=True),
        encoding="utf-8",
    )

    print(f"Saved demo figures to: {output_dir.resolve()}")
    print(json.dumps(metrics, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
