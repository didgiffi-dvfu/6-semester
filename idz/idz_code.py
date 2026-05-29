from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


ROOT = Path(__file__).resolve().parent
FIGURES = ROOT / "figures"
TABLES = ROOT / "tables"


def ensure_dirs() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    TABLES.mkdir(parents=True, exist_ok=True)


def fmt(x: float, ndigits: int = 4) -> str:
    return f"{x:.{ndigits}f}".replace(".", ",")


def fmt_int(x: int) -> str:
    return str(int(x))


def fmt_interval(a: float, b: float, ndigits: int = 4) -> str:
    return f"[{fmt(a, ndigits)}; {fmt(b, ndigits)}]"


def write_table(path: Path, caption: str, columns: list[str], rows: list[list[str]]) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write(f"\\caption{{{caption}}}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{%s}\n" % ("c" * len(columns)))
        f.write("\\toprule\n")
        f.write(" & ".join(columns) + " \\\\\n")
        f.write("\\midrule\n")
        for row in rows:
            f.write(" & ".join(row) + " \\\\\n")
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")


TASK1_SAMPLE = np.array(
    [
        0.11, -1.53, -0.94, 0.21, 0.77, 1.10, 0.23, -0.15, 0.79, -0.71,
        1.17, 0.01, 0.45, 1.55, 1.48, -0.09, 0.01, 1.00, 1.25, 1.35,
        0.52, -1.61, 2.16, 0.64, 0.19, 0.02, 0.20, 1.43, 0.74, -0.21,
        0.41, 0.80, -0.41, 0.31, -1.26, 0.75, 1.05, 2.04, -0.42, -1.06,
        0.33, -0.30, -0.34, -0.10, -1.54, 0.67, -0.40, -0.15, 0.98, -1.04,
        1.55, -1.58, 1.78, -0.71, 0.75, 0.48, -0.18, 0.49, -0.07, 0.90,
        1.04, 2.75, 1.03, 0.76, -2.53, 0.27, 0.92, -1.17, -0.85, -1.83,
        -0.35, -1.07, -0.02, 1.64, 0.35, -0.86, -0.06, 0.69, 2.16, -0.54,
        1.20, -0.57, 1.57, -0.05, 0.34, 0.83, -0.28, 0.48, 1.85, 0.93,
        0.91, -1.50, -1.08, 0.53, -0.53, 0.29, 0.77, -1.13, -0.76, 2.30,
    ],
    dtype=float,
)


TASK3_SAMPLE = np.array(
    [
        3.02, 0.19, 10.23, 1.42, 20.05, 8.85, 30.27, 10.71, 19.97, 4.25,
        16.37, 20.77, 8.81, 17.37, 13.78, 22.46, 7.26, 20.45, 17.57, 7.82,
        16.86, -2.40, 16.99, 13.49, -0.45, 12.39, 8.68, 5.75, 14.54, 16.34,
        6.14, 12.24, 2.39, 7.93, 0.12, 26.79,
    ],
    dtype=float,
)


TASK7_PAIRS = np.array(
    [
        [8.50, 13.74], [3.61, 38.50], [11.22, 48.30], [16.38, 30.34],
        [15.99, 55.70], [18.67, 44.78], [-0.92, 52.11], [8.83, 41.22],
        [15.48, 52.45], [4.57, 35.70], [6.55, 25.66], [1.55, 35.63],
        [0.77, 16.51], [5.11, 58.01], [6.13, 22.70], [-0.59, 42.35],
        [7.16, 48.75], [7.98, 47.47], [10.67, 43.74], [8.17, 16.78],
        [8.37, 39.54], [8.15, 15.87], [16.71, 54.16], [9.57, 21.23],
        [9.07, 20.42], [7.43, 55.34], [19.86, 13.29], [14.33, 56.27],
        [21.88, 23.90], [6.73, 9.70], [18.31, 20.82], [1.94, 54.89],
        [12.69, 24.21], [14.51, 54.01], [19.59, 37.85], [9.58, 19.14],
        [7.38, 53.79], [13.38, 42.25], [8.09, 45.05], [13.79, 18.17],
    ],
    dtype=float,
)


def task1() -> dict[str, float]:
    x = TASK1_SAMPLE
    n = x.size
    x_mean = float(np.mean(x))
    x_var = float(np.mean((x - x_mean) ** 2))

    k = max(1, math.ceil(1 + 3.322 * math.log10(n)))
    edges = np.linspace(float(x.min()), float(x.max()), k + 1)
    counts, _ = np.histogram(x, bins=edges)
    widths = np.diff(edges)
    rel = counts / n
    dens = rel / widths
    mid = (edges[:-1] + edges[1:]) / 2

    rows = []
    for i in range(k):
        rows.append(
            [
                f"$[{fmt(edges[i], 2)}; {fmt(edges[i + 1], 2)})$" if i < k - 1 else f"$[{fmt(edges[i], 2)}; {fmt(edges[i + 1], 2)}]$",
                fmt_int(int(counts[i])),
                fmt(float(rel[i]), 4),
                fmt(float(dens[i]), 4),
            ]
        )
    write_table(
        TABLES / "task1_interval_table.tex",
        "Интервальный ряд для задачи 1",
        ["Интервал", "$n_i$", "$\\omega_i$", "$h_i$"],
        rows,
    )

    sorted_x = np.sort(x)
    ecdf = np.arange(1, n + 1) / n

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("Задача 1: интервальный ряд, полигон частот, гистограмма и ЭФР", fontsize=15)

    axes[0].bar(mid, rel, width=0.9 * widths, color="skyblue", edgecolor="black", alpha=0.8)
    axes[0].plot(mid, rel, color="crimson", marker="o", linewidth=2)
    axes[0].set_title("Полигон частот")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("ωi")
    axes[0].grid(alpha=0.3)

    axes[1].hist(x, bins=edges, density=True, color="lightsteelblue", edgecolor="black", alpha=0.85)
    axes[1].set_title("Гистограмма плотности")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("h_i")
    axes[1].grid(alpha=0.3)

    axes[2].step(sorted_x, ecdf, where="post", color="darkgreen", linewidth=2)
    axes[2].set_title("Эмпирическая функция распределения")
    axes[2].set_xlabel("x")
    axes[2].set_ylabel("F_n(x)")
    axes[2].set_ylim(-0.05, 1.05)
    axes[2].grid(alpha=0.3)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(FIGURES / "task1_plots.png", dpi=200)
    plt.close(fig)

    return {
        "n": n,
        "mean": x_mean,
        "var": x_var,
        "bins": k,
        "min": float(x.min()),
        "max": float(x.max()),
    }


def task3() -> dict[str, float]:
    x = TASK3_SAMPLE
    n = x.size
    beta = 0.89
    alpha = 1.0 - beta
    x_mean = float(np.mean(x))
    s2 = float(np.var(x, ddof=1))
    s = math.sqrt(s2)
    t_quant = float(st.t.ppf(1 - alpha / 2, df=n - 1))
    chi_low = float(st.chi2.ppf(alpha / 2, df=n - 1))
    chi_high = float(st.chi2.ppf(1 - alpha / 2, df=n - 1))
    mean_ci = (x_mean - t_quant * s / math.sqrt(n), x_mean + t_quant * s / math.sqrt(n))
    var_ci = ((n - 1) * s2 / chi_high, (n - 1) * s2 / chi_low)
    std_ci = (math.sqrt(var_ci[0]), math.sqrt(var_ci[1]))

    rows = [
        ["$n$", fmt_int(n)],
        ["$\\overline{x}$", fmt(x_mean)],
        ["$S^2$", fmt(s2)],
        ["$t_{1-\\alpha/2;\\,n-1}$", fmt(t_quant)],
        ["$\\chi^2_{\\alpha/2;\\,n-1}$", fmt(chi_low)],
        ["$\\chi^2_{1-\\alpha/2;\\,n-1}$", fmt(chi_high)],
        ["ДИ для $m$", fmt_interval(*mean_ci)],
        ["ДИ для $\\sigma^2$", fmt_interval(*var_ci)],
        ["ДИ для $\\sigma$", fmt_interval(*std_ci)],
    ]
    write_table(
        TABLES / "task3_results_table.tex",
        "Точные доверительные интервалы для задачи 3",
        ["Параметр", "Значение"],
        rows,
    )

    return {
        "n": n,
        "beta": beta,
        "alpha": alpha,
        "mean": x_mean,
        "var": s2,
        "t_quant": t_quant,
        "chi_low": chi_low,
        "chi_high": chi_high,
        "mean_ci_low": mean_ci[0],
        "mean_ci_high": mean_ci[1],
        "var_ci_low": var_ci[0],
        "var_ci_high": var_ci[1],
        "std_ci_low": std_ci[0],
        "std_ci_high": std_ci[1],
    }


def task4() -> dict[str, float]:
    n = 20
    nu = n - 1
    eps = 0.3
    p = float(st.chi2.cdf(nu * (1 + eps), df=nu) - st.chi2.cdf(nu * (1 - eps), df=nu))
    rows = [
        ["$n$", fmt_int(n)],
        ["$\\nu = n-1$", fmt_int(nu)],
        ["$\\varepsilon$", fmt(eps)],
        ["$P\\{|s^2-\\sigma^2|/\\sigma^2 \\le 0{,}3\\}$", fmt(p)],
    ]
    write_table(
        TABLES / "task4_results_table.tex",
        "Ответ к задаче 4",
        ["Параметр", "Значение"],
        rows,
    )
    return {"n": n, "nu": nu, "eps": eps, "p": p}


def task5() -> dict[str, float]:
    obs = np.array([6, 11, 17, 13, 14, 9], dtype=float)
    intervals = [(-1, 0), (0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    n = float(obs.sum())
    mu = 2.0
    sigma = math.sqrt(2.0)
    probs = np.array(
        [
            st.norm.cdf(b, loc=mu, scale=sigma) - st.norm.cdf(a, loc=mu, scale=sigma)
            for a, b in intervals
        ],
        dtype=float,
    )
    expected = n * probs
    contrib = (obs - expected) ** 2 / expected
    chi2 = float(contrib.sum())
    p_value = float(1 - st.chi2.cdf(chi2, df=len(obs) - 1))
    crit = float(st.chi2.ppf(0.99, df=len(obs) - 1))

    rows = []
    for (a, b), o, p, e, c in zip(intervals, obs, probs, expected, contrib):
        rows.append(
            [
                f"$({fmt(a, 0)}; {fmt(b, 0)})$".replace(",0", ""),
                fmt_int(int(o)),
                fmt(p),
                fmt(e),
                fmt(c),
            ]
        )
    rows.append(["Итого", fmt_int(int(n)), fmt(probs.sum()), fmt(expected.sum()), fmt(chi2)])
    write_table(
        TABLES / "task5_results_table.tex",
        "Проверка гипотезы по критерию Пирсона для задачи 5",
        ["Интервал", "$n_k$", "$p_k$", "$n p_k$", "$\\frac{(n_k-np_k)^2}{np_k}$"],
        rows,
    )

    return {
        "n": n,
        "mu": mu,
        "sigma": sigma,
        "chi2": chi2,
        "p_value": p_value,
        "crit": crit,
        "probs_sum": float(probs.sum()),
    }


def task6() -> dict[str, float]:
    nx, ny = 15, 13
    xbar, ybar = 3.8, 3.5
    sx2, sy2 = 0.15, 0.1
    alpha = 0.025
    sp2 = ((nx - 1) * sx2 + (ny - 1) * sy2) / (nx + ny - 2)
    sp = math.sqrt(sp2)
    t_obs = (xbar - ybar) / (sp * math.sqrt(1 / nx + 1 / ny))
    t_crit = float(st.t.ppf(1 - alpha, df=nx + ny - 2))
    p_value = float(1 - st.t.cdf(t_obs, df=nx + ny - 2))

    rows = [
        ["$n_x$", fmt_int(nx)],
        ["$n_y$", fmt_int(ny)],
        ["$\\overline{x}$", fmt(xbar)],
        ["$\\overline{y}$", fmt(ybar)],
        ["$s_x^2$", fmt(sx2)],
        ["$s_y^2$", fmt(sy2)],
        ["$s_p^2$", fmt(sp2)],
        ["$T_{\\text{набл}}$", fmt(t_obs)],
        ["$t_{1-\\alpha;\\,n_x+n_y-2}$", fmt(t_crit)],
        ["$p$-value", fmt(p_value)],
    ]
    write_table(
        TABLES / "task6_results_table.tex",
        "Проверка гипотезы о равенстве средних для задачи 6",
        ["Параметр", "Значение"],
        rows,
    )

    return {
        "nx": nx,
        "ny": ny,
        "xbar": xbar,
        "ybar": ybar,
        "sx2": sx2,
        "sy2": sy2,
        "sp2": sp2,
        "t_obs": t_obs,
        "t_crit": t_crit,
        "p_value": p_value,
        "alpha": alpha,
    }


def task7() -> dict[str, float]:
    data = TASK7_PAIRS
    x = data[:, 0]
    y = data[:, 1]
    n = x.size
    x_mean = float(np.mean(x))
    y_mean = float(np.mean(y))

    x_centered = x - x_mean
    y_centered = y - y_mean
    s_xx = float(np.sum(x_centered**2))
    s_xy = float(np.sum(x_centered * y_centered))

    beta1 = s_xy / s_xx
    beta0 = y_mean - beta1 * x_mean

    A1 = np.column_stack([np.ones_like(x), x])
    coef1, *_ = np.linalg.lstsq(A1, y, rcond=None)
    pred1 = A1 @ coef1
    sse1 = float(np.sum((y - pred1) ** 2))
    s2_1 = sse1 / (n - 2)
    r2_1 = 1 - sse1 / float(np.sum((y - y_mean) ** 2))
    cov1 = s2_1 * np.linalg.inv(A1.T @ A1)
    se1 = np.sqrt(np.diag(cov1))
    t1 = coef1 / se1
    tcrit1 = float(st.t.ppf(1 - 0.075 / 2, df=n - 2))

    A2 = np.column_stack([np.ones_like(x), x, x**2])
    coef2, *_ = np.linalg.lstsq(A2, y, rcond=None)
    pred2 = A2 @ coef2
    sse2 = float(np.sum((y - pred2) ** 2))
    s2_2 = sse2 / (n - 3)
    r2_2 = 1 - sse2 / float(np.sum((y - y_mean) ** 2))
    cov2 = s2_2 * np.linalg.inv(A2.T @ A2)
    se2 = np.sqrt(np.diag(cov2))
    t2 = coef2 / se2
    tcrit2 = float(st.t.ppf(1 - 0.075 / 2, df=n - 3))

    rows_coef = [
        ["Линейная", "$\\beta_0$", fmt(coef1[0]), fmt(se1[0]), fmt(t1[0]), "значим" if abs(t1[0]) > tcrit1 else "не значим"],
        ["Линейная", "$\\beta_1$", fmt(coef1[1]), fmt(se1[1]), fmt(t1[1]), "значим" if abs(t1[1]) > tcrit1 else "не значим"],
        ["Квадратичная", "$\\beta_0$", fmt(coef2[0]), fmt(se2[0]), fmt(t2[0]), "значим" if abs(t2[0]) > tcrit2 else "не значим"],
        ["Квадратичная", "$\\beta_1$", fmt(coef2[1]), fmt(se2[1]), fmt(t2[1]), "значим" if abs(t2[1]) > tcrit2 else "не значим"],
        ["Квадратичная", "$\\beta_2$", fmt(coef2[2]), fmt(se2[2]), fmt(t2[2]), "значим" if abs(t2[2]) > tcrit2 else "не значим"],
    ]
    write_table(
        TABLES / "task7_coefficients_table.tex",
        "Оценки коэффициентов регрессии для задачи 7",
        ["Модель", "Коэффициент", "Оценка", "Std. error", "$t_{\\text{набл}}$", "Вывод"],
        rows_coef,
    )

    rows_model = [
        ["Линейная", fmt(s2_1), fmt(r2_1)],
        ["Квадратичная", fmt(s2_2), fmt(r2_2)],
    ]
    write_table(
        TABLES / "task7_model_table.tex",
        "Сравнение регрессионных моделей для задачи 7",
        ["Модель", "$\\hat\\sigma^2_{\\text{ост}}$", "$R^2$"],
        rows_model,
    )

    x_grid = np.linspace(float(x.min()) - 1, float(x.max()) + 1, 500)
    y1_grid = coef1[0] + coef1[1] * x_grid
    y2_grid = coef2[0] + coef2[1] * x_grid + coef2[2] * x_grid**2

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.scatter(x, y, s=35, color="steelblue", alpha=0.85, label="Наблюдения")
    ax.plot(x_grid, y1_grid, color="crimson", linewidth=2.2, label="Линейная регрессия")
    ax.plot(x_grid, y2_grid, color="darkgreen", linewidth=2.2, linestyle="--", label="Квадратичная регрессия")
    ax.set_title("Задача 7: корреляционное поле и регрессионные зависимости")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGURES / "task7_regression.png", dpi=200)
    plt.close(fig)

    return {
        "n": n,
        "x_mean": x_mean,
        "y_mean": y_mean,
        "beta0_lin": float(coef1[0]),
        "beta1_lin": float(coef1[1]),
        "beta0_quad": float(coef2[0]),
        "beta1_quad": float(coef2[1]),
        "beta2_quad": float(coef2[2]),
        "s2_lin": s2_1,
        "r2_lin": r2_1,
        "s2_quad": s2_2,
        "r2_quad": r2_2,
        "tcrit_lin": tcrit1,
        "tcrit_quad": tcrit2,
    }


def gamma_mle_formula() -> str:
    return r"\hat{\alpha}_{\text{МП}}=\frac{\lambda}{\overline{X}}=\frac{n\lambda}{\sum_{i=1}^n X_i}"


def main() -> None:
    ensure_dirs()
    results = {
        "task1": task1(),
        "task3": task3(),
        "task4": task4(),
        "task5": task5(),
        "task6": task6(),
        "task7": task7(),
    }
    summary_path = ROOT / "results_summary.txt"
    lines = []
    for name, data in results.items():
        lines.append(name)
        for key, value in data.items():
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.10f}")
            else:
                lines.append(f"  {key}: {value}")
    summary_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
