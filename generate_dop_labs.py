from __future__ import annotations

import json
import math
import statistics as st
from pathlib import Path
from textwrap import dedent

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "доп_лабы"
FIG_DIR = OUT_DIR / "figures"


def ensure_dirs() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    FIG_DIR.mkdir(exist_ok=True)


def setup_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.unicode_minus": False,
            "figure.dpi": 160,
            "savefig.dpi": 160,
        }
    )


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": dedent(text).strip() + "\n"}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": dedent(text).strip() + "\n"}


def fmt(x, digits: int = 4) -> str:
    if isinstance(x, (int, np.integer)):
        return str(int(x))
    if isinstance(x, (float, np.floating)):
        if math.isnan(float(x)):
            return "nan"
        return f"{float(x):.{digits}f}"
    if isinstance(x, (list, tuple, np.ndarray)):
        return ", ".join(fmt(v, digits) for v in list(x))
    return str(x)


def md_table(headers, rows) -> str:
    rows = [[str(cell) for cell in row] for row in rows]
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(widths[i], len(row[i])) for i in range(len(headers))]

    def row_line(row):
        return "| " + " | ".join(row[i].ljust(widths[i]) for i in range(len(headers))) + " |"

    lines = [row_line(headers), "| " + " | ".join("-" * w for w in widths) + " |"]
    lines.extend(row_line(row) for row in rows)
    return "\n".join(lines)


def discrete_series(sample):
    arr = np.asarray(sample)
    values, counts = np.unique(arr, return_counts=True)
    rel = counts / counts.sum()
    cum = np.cumsum(rel)
    return values, counts, rel, cum


def interval_series(sample, bins=None):
    arr = np.asarray(sample, dtype=float)
    n = arr.size
    if bins is None:
        bins = max(1, int(round(math.sqrt(n))))
    edges = np.linspace(arr.min(), arr.max(), bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    rel = counts / counts.sum()
    cum = np.cumsum(rel)
    mids = (edges[:-1] + edges[1:]) / 2
    return edges, mids, counts, rel, cum


def discrete_bins(sample):
    arr = np.asarray(sample)
    lo = int(arr.min())
    hi = int(arr.max())
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    counts, _ = np.histogram(arr, bins=edges)
    rel = counts / counts.sum()
    cum = np.cumsum(rel)
    mids = np.arange(lo, hi + 1)
    return edges, mids, counts, rel, cum


def save_ecdf_fig_discrete(sample, title, filename):
    values, counts, rel, cum = discrete_series(sample)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.step(values, cum, where="post", color="#2F6BFF", linewidth=2.2)
    ax.scatter(values, cum, color="#2F6BFF", s=28, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("F_n(x)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_ecdf_fig_interval(sample, title, filename):
    edges, mids, counts, rel, cum = interval_series(sample)
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.step(edges[1:], cum, where="post", color="#D94F70", linewidth=2.2)
    ax.scatter(edges[1:], cum, color="#D94F70", s=28, zorder=3)
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("F_n(x)")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_graphs_lab2(sample, title, filename, discrete=False):
    if discrete:
        edges, mids, counts, rel, cum = discrete_bins(sample)
    else:
        edges, mids, counts, rel, cum = interval_series(sample)
    width = np.diff(edges)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))

    ax = axes[0]
    ax.bar(mids, counts, width=width, align="center", color="#5B8FF9", edgecolor="black")
    ax.set_title("Гистограмма частот")
    ax.set_xlabel("Интервалы / варианты")
    ax.set_ylabel("Частота")
    ax.grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.plot(mids, counts, marker="o", color="#D94F70", linewidth=2.0, label="Частоты")
    ax2 = ax.twinx()
    ax2.plot(mids, rel, marker="s", color="#2F6BFF", linewidth=2.0, label="Относительные частоты")
    ax.set_title("Полигон частот")
    ax.set_xlabel("Интервалы / варианты")
    ax.set_ylabel("Частота")
    ax2.set_ylabel("Отн. частота")
    ax.grid(alpha=0.25)
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc="upper right", fontsize=8)

    ax = axes[2]
    ax.step(edges[1:], cum, where="post", color="#3A9D5D", linewidth=2.2)
    ax.scatter(edges[1:], cum, color="#3A9D5D", s=24, zorder=3)
    ax.set_title("Огива")
    ax.set_xlabel("x")
    ax.set_ylabel("Накопл. относительная частота")
    ax.set_ylim(-0.03, 1.05)
    ax.grid(alpha=0.25)

    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def save_histogram(sample, title, filename):
    arr = np.asarray(sample, dtype=float)
    n = arr.size
    bins = max(1, int(round(math.sqrt(n))))
    edges = np.linspace(arr.min(), arr.max(), bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    rel_density = counts / (n * np.diff(edges))

    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(mids, rel_density, width=np.diff(edges), align="center", color="#5B8FF9", edgecolor="black", alpha=0.85)
    ax.axvline(arr.mean(), color="#D94F70", linewidth=2, label="Среднее")
    ax.axvline(np.median(arr), color="#3A9D5D", linewidth=2, linestyle="--", label="Медиана")
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("Плотность относительной частоты")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / filename, bbox_inches="tight")
    plt.close(fig)


def sample_stats(sample):
    arr = np.asarray(sample, dtype=float)
    n = arr.size
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    modes = sorted(st.multimode(arr.tolist()))
    variance = float(np.var(arr, ddof=1))
    std_dev = float(np.sqrt(variance))
    min_v = float(np.min(arr))
    max_v = float(np.max(arr))
    data_range = max_v - min_v
    total = float(np.sum(arr))
    count = int(n)
    stderr = std_dev / math.sqrt(n)

    if n > 2 and std_dev > 0:
        skew = float(
            n
            / ((n - 1) * (n - 2))
            * np.sum(((arr - mean) / std_dev) ** 3)
        )
    else:
        skew = float("nan")

    if n > 3 and std_dev > 0:
        excess = float(
            n
            * (n + 1)
            / ((n - 1) * (n - 2) * (n - 3))
            * np.sum(((arr - mean) / std_dev) ** 4)
            - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )
    else:
        excess = float("nan")

    ci_half = 1.96 * stderr
    ci = (mean - ci_half, mean + ci_half)

    return {
        "n": count,
        "mean": mean,
        "median": median,
        "modes": modes,
        "std_dev": std_dev,
        "variance": variance,
        "skew": skew,
        "excess": excess,
        "min": min_v,
        "max": max_v,
        "range": data_range,
        "sum": total,
        "stderr": stderr,
        "ci": ci,
        "ci_half": ci_half,
    }


def write_notebook(path: Path, cells) -> None:
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=2), encoding="utf-8")


def build_lab1():
    discrete = [2, 3, 2, 1, 2, 2, 4, 0, 2, 4, 4, 0, 1, 5, 1, 1, 3, 2, 2, 1, 4, 2, 2, 0, 1, 2, 4, 2, 0, 2]
    continuous = [10.2, 9.23, 8.77, 10.4, 9.44, 9.09, 6.30, 9.42, 6.12, 9.69, 8.59, 8.68, 7.97, 8.64, 6.45, 5.29, 5.00, 8.42, 8.84, 8.26, 6.66, 6.96, 6.51, 6.72, 6.00, 5.36]

    values, counts, rel, cum = discrete_series(discrete)
    edges, mids, icounts, irel, icum = interval_series(continuous)

    save_ecdf_fig_discrete(
        discrete,
        "Лабораторная работа 1: выборочная функция распределения, дискретный вариант",
        "lab1_discrete_ecdf.png",
    )
    save_ecdf_fig_interval(
        continuous,
        "Лабораторная работа 1: выборочная функция распределения, интервальный вариант",
        "lab1_interval_ecdf.png",
    )

    discrete_rows = [
        [fmt(v, 0), fmt(c, 0), fmt(r), fmt(k)] for v, c, r, k in zip(values, counts, rel, cum)
    ]
    interval_rows = []
    for i in range(len(icounts)):
        left = edges[i]
        right = edges[i + 1]
        interval_rows.append(
            [f"[{fmt(left)}; {fmt(right)}]", fmt(mids[i]), fmt(icounts[i], 0), fmt(irel[i]), fmt(icum[i])]
        )

    discrete_table = md_table(
        ["Варианта", "Частота", "Отн. частота", "Накопл. отн. частота"],
        discrete_rows,
    )
    interval_table = md_table(
        ["Интервал", "Середина", "Частота", "Отн. частота", "Накопл. отн. частота"],
        interval_rows,
    )

    code_cell = code(
        """
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

def discrete_series(sample):
    values, counts = np.unique(np.asarray(sample), return_counts=True)
    rel = counts / counts.sum()
    cum = np.cumsum(rel)
    return values, counts, rel, cum

def interval_series(sample, bins=None):
    arr = np.asarray(sample, dtype=float)
    if bins is None:
        bins = max(1, int(round(math.sqrt(arr.size))))
    edges = np.linspace(arr.min(), arr.max(), bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    rel = counts / counts.sum()
    cum = np.cumsum(rel)
    mids = (edges[:-1] + edges[1:]) / 2
    return edges, mids, counts, rel, cum

discrete = [2, 3, 2, 1, 2, 2, 4, 0, 2, 4, 4, 0, 1, 5, 1, 1, 3, 2, 2, 1, 4, 2, 2, 0, 1, 2, 4, 2, 0, 2]
continuous = [10.2, 9.23, 8.77, 10.4, 9.44, 9.09, 6.30, 9.42, 6.12, 9.69, 8.59, 8.68, 7.97, 8.64, 6.45, 5.29, 5.00, 8.42, 8.84, 8.26, 6.66, 6.96, 6.51, 6.72, 6.00, 5.36]

values, counts, rel, cum = discrete_series(discrete)
edges, mids, icounts, irel, icum = interval_series(continuous)

print("Дискретный вариационный ряд:")
for v, c, r, k in zip(values, counts, rel, cum):
    print(f"x={v:>2}  n={c:>2}  w={r:.4f}  F_n={k:.4f}")

print()
print("Интервальный вариационный ряд:")
for i in range(len(icounts)):
    print(f"[{edges[i]:.2f}; {edges[i+1]:.2f}]  n={icounts[i]:>2}  w={irel[i]:.4f}  F_n={icum[i]:.4f}")
"""
    )

    cells = [
        md(
            """
# Лабораторная работа №1
## Вариационные ряды и выборочная функция распределения

**Вариант:** `16 % 10 = 6`

Цель работы: научиться строить дискретный и интервальный вариационные ряды, а также выборочную (эмпирическую) функцию распределения.
"""
        ),
        md(
            """
## Краткая теория

Пусть наблюдается выборка $x_1, x_2, \\dots, x_n$.

- **Дискретный вариационный ряд**: упорядоченные различные значения выборки $x_{(1)} < x_{(2)} < \\dots < x_{(N)}$ и их частоты $n_i$.
- **Относительная частота**: $\\omega_i = \\frac{n_i}{n}$.
- **Интервальный вариационный ряд**: разбиение диапазона значений на интервалы одинаковой длины $\\Delta$.
- **Эмпирическая функция распределения**:

$$
F_n(x) = \\frac{1}{n} \\sum_{i=1}^{n} \\mathbf{1}\\{x_i \\le x\\}.
$$

Она является ступенчатой неубывающей функцией, а величина скачка в точке варианта равна его относительной частоте.
"""
        ),
        code_cell,
        md(
            f"""
## Дискретная выборка

Ниже показана выборка варианта 6 и полученный дискретный вариационный ряд.

{discrete_table}

![Эмпирическая функция распределения для дискретной выборки](figures/lab1_discrete_ecdf.png)
"""
        ),
        md(
            f"""
## Интервальная выборка

Для непрерывной выборки использован разбиение на `round(sqrt(n)) = {int(round(math.sqrt(len(continuous))))}` интервалов одинаковой длины.

{interval_table}

![Эмпирическая функция распределения для интервальной выборки](figures/lab1_interval_ecdf.png)
"""
        ),
        md(
            """
## Вывод

Для варианта 6 построены оба вида вариационных рядов и соответствующие эмпирические функции распределения. Для дискретной выборки скачки функции совпадают с относительными частотами, а для непрерывной выборки функция строится по серединам интервалов группировки.
"""
        ),
    ]
    write_notebook(OUT_DIR / "lab1.ipynb", cells)


def build_lab2():
    continuous = [6.52, 9.27, 7.91, 5.77, 8.02, 3.07, 2.22, 5.76, 11.6, 6.62, 7.07, 12.5, 1.65, 10.5, 3.67, 7.62, 4.94, 5.39, 3.64, 4.62, 8.88, 6.75, 5.77, 6.38, 10.3, 5.74]
    discrete = [2, 3, 2, 1, 2, 2, 4, 0, 2, 4, 4, 0, 1, 5, 1, 1, 3, 2, 2, 1, 4, 2, 2, 0, 1, 2, 4, 2, 0, 2]

    c_edges, c_mids, c_counts, c_rel, c_cum = interval_series(continuous)
    d_edges, d_mids, d_counts, d_rel, d_cum = discrete_bins(discrete)

    save_graphs_lab2(
        continuous,
        "Лабораторная работа 2: графическое представление непрерывной выборки",
        "lab2_continuous_graphs.png",
        discrete=False,
    )
    save_graphs_lab2(
        discrete,
        "Лабораторная работа 2: графическое представление дискретной выборки",
        "lab2_discrete_graphs.png",
        discrete=True,
    )

    cont_rows = []
    for i in range(len(c_counts)):
        cont_rows.append(
            [f"[{fmt(c_edges[i])}; {fmt(c_edges[i+1])}]", fmt(c_mids[i]), fmt(c_counts[i], 0), fmt(c_rel[i]), fmt(c_cum[i])]
        )
    disc_rows = []
    for v, c, r, k in zip(d_mids, d_counts, d_rel, d_cum):
        disc_rows.append([fmt(v, 0), fmt(c, 0), fmt(r), fmt(k)])

    cont_table = md_table(
        ["Интервал", "Середина", "Частота", "Отн. частота", "Накопл. отн. частота"],
        cont_rows,
    )
    disc_table = md_table(
        ["Варианта", "Частота", "Отн. частота", "Накопл. отн. частота"],
        disc_rows,
    )

    code_cell = code(
        """
import math
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

def interval_series(sample, bins=None):
    arr = np.asarray(sample, dtype=float)
    if bins is None:
        bins = max(1, int(round(math.sqrt(arr.size))))
    edges = np.linspace(arr.min(), arr.max(), bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    rel = counts / counts.sum()
    cum = np.cumsum(rel)
    mids = (edges[:-1] + edges[1:]) / 2
    return edges, mids, counts, rel, cum

def discrete_bins(sample):
    arr = np.asarray(sample)
    lo = int(arr.min())
    hi = int(arr.max())
    edges = np.arange(lo - 0.5, hi + 1.5, 1.0)
    counts, _ = np.histogram(arr, bins=edges)
    rel = counts / counts.sum()
    cum = np.cumsum(rel)
    mids = np.arange(lo, hi + 1)
    return edges, mids, counts, rel, cum

def plot_graphs(sample, title, discrete=False, filename="graph.png"):
    if discrete:
        edges, mids, counts, rel, cum = discrete_bins(sample)
    else:
        edges, mids, counts, rel, cum = interval_series(sample)
    width = np.diff(edges)

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.6))
    axes[0].bar(mids, counts, width=width, color="#5B8FF9", edgecolor="black")
    axes[0].set_title("Гистограмма частот")
    axes[0].set_xlabel("Интервалы / варианты")
    axes[0].set_ylabel("Частота")
    axes[0].grid(axis="y", alpha=0.25)

    ax = axes[1]
    ax.plot(mids, counts, marker="o", color="#D94F70", linewidth=2.0, label="Частоты")
    ax2 = ax.twinx()
    ax2.plot(mids, rel, marker="s", color="#2F6BFF", linewidth=2.0, label="Отн. частоты")
    ax.set_title("Полигон частот")
    ax.set_xlabel("Интервалы / варианты")
    ax.set_ylabel("Частота")
    ax2.set_ylabel("Отн. частота")
    ax.grid(alpha=0.25)
    l1, lab1 = ax.get_legend_handles_labels()
    l2, lab2 = ax2.get_legend_handles_labels()
    ax2.legend(l1 + l2, lab1 + lab2, fontsize=8)

    axes[2].step(edges[1:], cum, where="post", color="#3A9D5D", linewidth=2.2)
    axes[2].scatter(edges[1:], cum, color="#3A9D5D", s=24, zorder=3)
    axes[2].set_title("Огива")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("Накопл. относительная частота")
    axes[2].set_ylim(-0.03, 1.05)
    axes[2].grid(alpha=0.25)

    fig.suptitle(title, y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

continuous = [6.52, 9.27, 7.91, 5.77, 8.02, 3.07, 2.22, 5.76, 11.6, 6.62, 7.07, 12.5, 1.65, 10.5, 3.67, 7.62, 4.94, 5.39, 3.64, 4.62, 8.88, 6.75, 5.77, 6.38, 10.3, 5.74]
discrete = [2, 3, 2, 1, 2, 2, 4, 0, 2, 4, 4, 0, 1, 5, 1, 1, 3, 2, 2, 1, 4, 2, 2, 0, 1, 2, 4, 2, 0, 2]

plot_graphs(continuous, "Непрерывная выборка варианта 6", filename="figures/lab2_continuous_graphs.png")
plot_graphs(discrete, "Дискретная выборка варианта 6", discrete=True, filename="figures/lab2_discrete_graphs.png")
"""
    )

    cells = [
        md(
            """
# Лабораторная работа №2
## Графическое представление выборки

**Вариант:** `16 % 10 = 6`

Цель работы: научиться представлять выборку в виде гистограммы, полигона частот и огивы.
"""
        ),
        md(
            """
## Краткая теория

Основные графические представления выборки:

- **Гистограмма** показывает распределение частот по интервалам.
- **Полигон частот** соединяет точки $(x_i, n_i)$.
- **Полигон относительных частот** соединяет точки $(x_i, \\omega_i)$.
- **Огива** строится по накопленным относительным частотам.

Для непрерывной выборки используются интервалы группировки, а для дискретной выборки удобнее работать с отдельными значениями признака.
"""
        ),
        code_cell,
        md(
            f"""
## Непрерывная выборка

{cont_table}

![Графики для непрерывной выборки](figures/lab2_continuous_graphs.png)
"""
        ),
        md(
            f"""
## Дискретная выборка

{disc_table}

![Графики для дискретной выборки](figures/lab2_discrete_graphs.png)
"""
        ),
        md(
            """
## Вывод

Для непрерывной и дискретной выборок построены основные графические представления: гистограмма, полигон частот и огива. Наблюдается одинаковая логика построения для обоих типов данных, меняется только способ группировки значений.
"""
        ),
    ]
    write_notebook(OUT_DIR / "lab2.ipynb", cells)


def build_lab4():
    sample = [16.3, 20.6, 19.4, 18.7, 16.3, 18.7, 19.3, 18.8, 21.8, 23.2, 22.7, 17.4, 21.8, 18.8, 20.2, 19.3, 19.4, 18.4, 19.3, 18.1, 19.4, 19.7, 21.8, 18.8]
    stats = sample_stats(sample)
    save_histogram(sample, "Лабораторная работа 4: гистограмма выборки", "lab4_hist.png")

    rows = [
        ["Объём выборки", fmt(stats["n"], 0)],
        ["Среднее", fmt(stats["mean"])],
        ["Медиана", fmt(stats["median"])],
        ["Моды", fmt(stats["modes"])],
        ["Стандартное отклонение", fmt(stats["std_dev"])],
        ["Дисперсия", fmt(stats["variance"])],
        ["Асимметрия", fmt(stats["skew"])],
        ["Эксцесс", fmt(stats["excess"])],
        ["Минимум", fmt(stats["min"])],
        ["Максимум", fmt(stats["max"])],
        ["Размах", fmt(stats["range"])],
        ["Сумма", fmt(stats["sum"])],
        ["Стандартная ошибка", fmt(stats["stderr"])],
        ["95% ДИ для среднего", f"[{fmt(stats['ci'][0])}; {fmt(stats['ci'][1])}]"],
    ]
    summary_table = md_table(["Показатель", "Значение"], rows)

    code_cell = code(
        """
import math
import statistics as st
from collections import Counter
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

def sample_stats(sample):
    arr = np.asarray(sample, dtype=float)
    n = arr.size
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    modes = sorted(st.multimode(arr.tolist()))
    variance = float(np.var(arr, ddof=1))
    std_dev = float(np.sqrt(variance))
    stderr = std_dev / math.sqrt(n)
    if n > 2 and std_dev > 0:
        skew = float(n / ((n - 1) * (n - 2)) * np.sum(((arr - mean) / std_dev) ** 3))
    else:
        skew = float("nan")
    if n > 3 and std_dev > 0:
        excess = float(
            n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((arr - mean) / std_dev) ** 4)
            - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )
    else:
        excess = float("nan")
    ci_half = 1.96 * stderr
    return {
        "mean": mean,
        "median": median,
        "modes": modes,
        "std_dev": std_dev,
        "variance": variance,
        "skew": skew,
        "excess": excess,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "sum": float(np.sum(arr)),
        "stderr": stderr,
        "ci": (mean - ci_half, mean + ci_half),
    }

def draw_hist(sample, filename):
    arr = np.asarray(sample, dtype=float)
    bins = max(1, int(round(math.sqrt(arr.size))))
    edges = np.linspace(arr.min(), arr.max(), bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    densities = counts / (arr.size * np.diff(edges))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(mids, densities, width=np.diff(edges), align="center", color="#5B8FF9", edgecolor="black")
    ax.axvline(arr.mean(), color="#D94F70", linewidth=2, label="Среднее")
    ax.axvline(np.median(arr), color="#3A9D5D", linewidth=2, linestyle="--", label="Медиана")
    ax.set_xlabel("x")
    ax.set_ylabel("Плотность относительной частоты")
    ax.set_title("Гистограмма выборки")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

sample = [16.3, 20.6, 19.4, 18.7, 16.3, 18.7, 19.3, 18.8, 21.8, 23.2, 22.7, 17.4, 21.8, 18.8, 20.2, 19.3, 19.4, 18.4, 19.3, 18.1, 19.4, 19.7, 21.8, 18.8]
stats = sample_stats(sample)
draw_hist(sample, "figures/lab4_hist.png")
print(stats)
"""
    )

    cells = [
        md(
            """
# Лабораторная работа №4
## Числовые характеристики выборки с помощью встроенных функций

**Вариант:** `16 % 10 = 6`

Цель работы: научиться вычислять числовые характеристики выборки по формулам, которые обычно реализуются встроенными функциями Excel.
"""
        ),
        md(
            """
## Краткая теория

Для выборки $x_1, \\dots, x_n$ используются следующие характеристики:

- среднее $\\bar x$;
- медиана;
- мода;
- стандартное отклонение $s$;
- дисперсия $s^2$;
- асимметрия;
- эксцесс;
- минимум и максимум;
- размах $R = x_{\\max} - x_{\\min}$;
- сумма и объём выборки;
- стандартная ошибка среднего;
- 95% доверительный интервал для среднего.

В Python те же величины удобно вычислять напрямую по формулам, а не через Excel-функции.
"""
        ),
        code_cell,
        md(
            f"""
## Результаты вычислений

{summary_table}

![Гистограмма выборки](figures/lab4_hist.png)
"""
        ),
        md(
            """
## Вывод

Для варианта 6 получен полный набор основных описательных характеристик выборки. При наличии нескольких мод удобнее явно перечислять все значения, которые встречаются с максимальной частотой.
"""
        ),
    ]
    write_notebook(OUT_DIR / "lab4.ipynb", cells)


def build_lab5():
    sample = [18.7, 16.3, 18.4, 19.3, 18.8, 19.4, 18.7, 18.5, 20.6, 20.6, 19.4, 20.7, 16.3, 18.4, 19.3, 18.8, 18.4, 19.3, 19.3, 19.9, 23.1, 18.8, 17.4, 21.6, 19.1, 18.4, 19.3]
    stats = sample_stats(sample)
    save_histogram(sample, "Лабораторная работа 5: описательная статистика выборки", "lab5_hist.png")

    rows = [
        ["Среднее", fmt(stats["mean"])],
        ["Стандартная ошибка", fmt(stats["stderr"])],
        ["Медиана", fmt(stats["median"])],
        ["Мода", fmt(stats["modes"])],
        ["Стандартное отклонение", fmt(stats["std_dev"])],
        ["Дисперсия", fmt(stats["variance"])],
        ["Эксцесс", fmt(stats["excess"])],
        ["Асимметрия", fmt(stats["skew"])],
        ["Интервал", fmt(stats["range"])],
        ["Минимум", fmt(stats["min"])],
        ["Максимум", fmt(stats["max"])],
        ["Сумма", fmt(stats["sum"])],
        ["Счёт", fmt(stats["n"], 0)],
        ["95% ДИ для среднего", f"[{fmt(stats['ci'][0])}; {fmt(stats['ci'][1])}]"],
    ]
    summary_table = md_table(["Показатель", "Значение"], rows)

    code_cell = code(
        """
import math
import statistics as st
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.unicode_minus": False,
})

def sample_stats(sample):
    arr = np.asarray(sample, dtype=float)
    n = arr.size
    mean = float(np.mean(arr))
    median = float(np.median(arr))
    modes = sorted(st.multimode(arr.tolist()))
    variance = float(np.var(arr, ddof=1))
    std_dev = float(np.sqrt(variance))
    stderr = std_dev / math.sqrt(n)
    if n > 2 and std_dev > 0:
        skew = float(n / ((n - 1) * (n - 2)) * np.sum(((arr - mean) / std_dev) ** 3))
    else:
        skew = float("nan")
    if n > 3 and std_dev > 0:
        excess = float(
            n * (n + 1) / ((n - 1) * (n - 2) * (n - 3)) * np.sum(((arr - mean) / std_dev) ** 4)
            - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )
    else:
        excess = float("nan")
    ci_half = 1.96 * stderr
    return {
        "mean": mean,
        "median": median,
        "modes": modes,
        "std_dev": std_dev,
        "variance": variance,
        "skew": skew,
        "excess": excess,
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "range": float(np.max(arr) - np.min(arr)),
        "sum": float(np.sum(arr)),
        "stderr": stderr,
        "ci": (mean - ci_half, mean + ci_half),
    }

def draw_hist(sample, filename):
    arr = np.asarray(sample, dtype=float)
    bins = max(1, int(round(math.sqrt(arr.size))))
    edges = np.linspace(arr.min(), arr.max(), bins + 1)
    counts, _ = np.histogram(arr, bins=edges)
    mids = (edges[:-1] + edges[1:]) / 2
    densities = counts / (arr.size * np.diff(edges))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    ax.bar(mids, densities, width=np.diff(edges), align="center", color="#5B8FF9", edgecolor="black")
    ax.axvline(arr.mean(), color="#D94F70", linewidth=2, label="Среднее")
    ax.axvline(np.median(arr), color="#3A9D5D", linewidth=2, linestyle="--", label="Медиана")
    ax.set_xlabel("x")
    ax.set_ylabel("Плотность относительной частоты")
    ax.set_title("Описательная статистика выборки")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(filename, bbox_inches="tight")
    plt.close(fig)

sample = [18.7, 16.3, 18.4, 19.3, 18.8, 19.4, 18.7, 18.5, 20.6, 20.6, 19.4, 20.7, 16.3, 18.4, 19.3, 18.8, 18.4, 19.3, 19.3, 19.9, 23.1, 18.8, 17.4, 21.6, 19.1, 18.4, 19.3]
stats = sample_stats(sample)
draw_hist(sample, "figures/lab5_hist.png")
print(stats)
"""
    )

    cells = [
        md(
            """
# Лабораторная работа №5
## Описательная статистика выборки стандартными средствами

**Вариант:** `16 % 10 = 6`

Цель работы: собрать краткий отчёт по выборке в стиле Excel "Описательная статистика" и интерпретировать основные характеристики.
"""
        ),
        md(
            """
## Краткая теория

В описательной статистике обычно рассматривают:

- среднее и стандартную ошибку;
- медиану и моду;
- стандартное отклонение и дисперсию;
- эксцесс и асимметрию;
- минимум, максимум, размах;
- сумму и объём выборки;
- доверительный интервал для среднего.

Такой отчёт удобен для быстрого анализа распределения и предварительной проверки гипотез о форме выборки.
"""
        ),
        code_cell,
        md(
            f"""
## Итоговый отчёт

{summary_table}

![Гистограмма и опорные линии](figures/lab5_hist.png)
"""
        ),
        md(
            """
## Вывод

Для варианта 6 построен компактный статистический отчёт. По нему можно быстро оценить центральную тенденцию, разброс, асимметрию и возможный доверительный интервал для среднего.
"""
        ),
    ]
    write_notebook(OUT_DIR / "lab5.ipynb", cells)


def main():
    ensure_dirs()
    setup_matplotlib()
    build_lab1()
    build_lab2()
    build_lab4()
    build_lab5()
    print(f"Created notebooks in: {OUT_DIR}")


if __name__ == "__main__":
    main()
