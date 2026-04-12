import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats as st


matplotlib.use("Agg")


def mean_my(values):
    total = 0.0
    count = 0
    for x in values:
        total += float(x)
        count += 1
    return total / count if count else float("nan")


def var_my(values):
    n = len(values)
    if n == 0:
        return float("nan")
    m = mean_my(values)
    sq_sum = 0.0
    for x in values:
        sq_sum += (float(x) - m) ** 2
    return sq_sum / n


def std_my(values):
    return math.sqrt(var_my(values))


def build_samples(random_state=16):
    samples = {}
    samples.update(
        {
            "uniform_100": st.uniform.rvs(
                loc=2, scale=5, size=100, random_state=random_state
            ),
            "uniform_1000": st.uniform.rvs(
                loc=2, scale=5, size=1000, random_state=random_state
            ),
        }
    )
    samples.update(
        {
            "bernoulli_100": st.bernoulli.rvs(
                p=0.27, size=100, random_state=random_state
            ),
            "bernoulli_1000": st.bernoulli.rvs(
                p=0.27, size=1000, random_state=random_state
            ),
        }
    )
    samples.update(
        {
            "binom_100": st.binom.rvs(n=12, p=0.35, size=100, random_state=random_state),
            "binom_1000": st.binom.rvs(
                n=12, p=0.35, size=1000, random_state=random_state
            ),
        }
    )
    samples.update(
        {
            "norm_100": st.norm.rvs(loc=4.5, scale=1.8, size=100, random_state=random_state),
            "norm_1000": st.norm.rvs(
                loc=4.5, scale=1.8, size=1000, random_state=random_state
            ),
        }
    )
    return samples


def save_results_table(samples, out_path):
    order = [
        "uniform_100",
        "uniform_1000",
        "bernoulli_100",
        "bernoulli_1000",
        "binom_100",
        "binom_1000",
        "norm_100",
        "norm_1000",
    ]

    rows = []
    for i, name in enumerate(order, start=1):
        arr = samples[name]
        rows.append(
            (
                i,
                mean_my(arr),
                var_my(arr),
                std_my(arr),
                float(np.mean(arr)),
                float(np.var(arr)),
                float(np.std(arr)),
            )
        )

    with out_path.open("w", encoding="utf-8") as f:
        f.write("\\begin{table}[H]\n")
        f.write("\\centering\n")
        f.write("\\caption{Сравнение выборочных характеристик (свои функции и numpy)}\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{rcccccc}\n")
        f.write("\\toprule\n")
        f.write(
            "№ & Среднее\\_свое & Дисперсия\\_своя & Std\\_свое & Среднее\\_numpy & Дисперсия\\_numpy & Std\\_numpy \\\\\n"
        )
        f.write("\\midrule\n")
        for r in rows:
            f.write(
                f"{r[0]} & {r[1]:.6f} & {r[2]:.6f} & {r[3]:.6f} & {r[4]:.6f} & {r[5]:.6f} & {r[6]:.6f} \\\\\n"
            )
        f.write("\\bottomrule\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")

    return rows


def save_figures(samples, out_dir):
    out_dir.mkdir(exist_ok=True)
    plot_config = [
        (1, "uniform_100", "Равномерное U(2, 7), n=100", "continuous", "uniform"),
        (2, "uniform_1000", "Равномерное U(2, 7), n=1000", "continuous", "uniform"),
        (3, "bernoulli_100", "Бернулли B(0.27), n=100", "discrete", "bernoulli"),
        (4, "bernoulli_1000", "Бернулли B(0.27), n=1000", "discrete", "bernoulli"),
        (5, "binom_100", "Биномиальное Bin(12, 0.35), n=100", "discrete", "binom"),
        (6, "binom_1000", "Биномиальное Bin(12, 0.35), n=1000", "discrete", "binom"),
        (7, "norm_100", "Нормальное N(4.5, 1.8^2), n=100", "continuous", "norm"),
        (8, "norm_1000", "Нормальное N(4.5, 1.8^2), n=1000", "continuous", "norm"),
    ]

    for num, sample_name, title, dist_kind, dist_name in plot_config:
        sample = samples[sample_name]
        plt.figure(figsize=(8, 5))

        if dist_kind == "continuous":
            # Формула Стерджесса для числа столбцов:
            # k = ceil(1 + 3.322 * log10(n))
            # Ширина столбца:
            # Δ = (x_max - x_min) / k
            n_obs = len(sample)
            k_sturges = max(1, math.ceil(1 + 3.322 * math.log10(n_obs)))
            x_min = float(np.min(sample))
            x_max = float(np.max(sample))
            delta = (x_max - x_min) / k_sturges if x_max > x_min else 1.0
            bins = (
                np.linspace(x_min, x_max, k_sturges + 1)
                if x_max > x_min
                else np.array([x_min - 0.5, x_max + 0.5])
            )
            plt.hist(
                sample,
                bins=bins,
                density=True,
                alpha=0.6,
                color="skyblue",
                edgecolor="black",
                label="Гистограмма плотностей относительных частот",
            )
            x = np.linspace(x_min, x_max, 400) if x_max > x_min else np.array([x_min])
            if dist_name == "uniform":
                y = st.uniform.pdf(x, loc=2, scale=5)
            else:
                y = st.norm.pdf(x, loc=4.5, scale=1.8)
            plt.plot(x, y, color="crimson", linewidth=2, label="Плотность вероятности")
        else:
            x_min = int(np.min(sample))
            x_max = int(np.max(sample))
            values = np.arange(x_min, x_max + 1)
            # Формула Стерджесса для числа столбцов:
            # k = ceil(1 + 3.322 * log10(n))
            # Ширина столбца:
            # Δ = (x_max_edge - x_min_edge) / k
            n_obs = len(sample)
            k_sturges = max(1, math.ceil(1 + 3.322 * math.log10(n_obs)))
            x_min_edge = x_min - 0.5
            x_max_edge = x_max + 0.5
            delta = (
                (x_max_edge - x_min_edge) / k_sturges
                if x_max_edge > x_min_edge
                else 1.0
            )
            bins = (
                np.linspace(x_min_edge, x_max_edge, k_sturges + 1)
                if x_max_edge > x_min_edge
                else np.array([x_min_edge, x_max_edge + 1.0])
            )
            plt.hist(
                sample,
                bins=bins,
                density=True,
                alpha=0.6,
                color="skyblue",
                edgecolor="black",
                label="Гистограмма плотностей относительных частот",
            )
            if dist_name == "bernoulli":
                p_theory = st.bernoulli.pmf(values, p=0.27)
            else:
                p_theory = st.binom.pmf(values, n=12, p=0.35)
            plt.plot(
                values,
                p_theory,
                "o-",
                color="crimson",
                linewidth=2,
                label="Функция вероятности",
            )

        plt.title(f"Выборка №{num}: {title}")
        plt.xlabel("x")
        plt.ylabel("Плотность / вероятность")
        plt.grid(alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"sample_{num}.png", dpi=170)
        plt.close()


def save_terminal_output(rows, out_path):
    lines = []
    lines.append("python generate_report_assets.py")
    lines.append("Done. Rows in results table: 8")
    lines.append("Saved figures: 8")
    lines.append("")
    lines.append("Preview of computed rows:")
    for row in rows:
        lines.append(
            f"{row[0]}: mean_my={row[1]:.6f}, var_my={row[2]:.6f}, std_my={row[3]:.6f}, "
            f"mean_np={row[4]:.6f}, var_np={row[5]:.6f}, std_np={row[6]:.6f}"
        )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    root = Path(__file__).resolve().parent
    figures_dir = root / "figures"
    table_path = root / "results_table.tex"
    terminal_path = root / "results_terminal.txt"

    samples = build_samples(random_state=16)
    rows = save_results_table(samples, table_path)
    save_figures(samples, figures_dir)
    save_terminal_output(rows, terminal_path)

    print(f"Saved: {table_path}")
    print(f"Saved: {terminal_path}")
    print(f"Saved figures in: {figures_dir}")


if __name__ == "__main__":
    main()
