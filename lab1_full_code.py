# Полный код, автоматически извлечённый из lab1.ipynb

# ==== Ячейка 2 ====
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st

# ==== Ячейка 5 ====
random_state = 16

# ==== Ячейка 7 ====
samples = {}

samples.update({
        "uniform_100": st.uniform.rvs(loc=2, scale=5, size=100, random_state=random_state),
        "uniform_1000": st.uniform.rvs(loc=2, scale=5, size=1000, random_state=random_state),
    })
samples.update({
        "bernoulli_100": st.bernoulli.rvs(p=0.27, size=100, random_state=random_state),
        "bernoulli_1000": st.bernoulli.rvs(p=0.27, size=1000, random_state=random_state),
    })
samples.update( {
        "binom_100": st.binom.rvs(n=12, p=0.35, size=100, random_state=random_state),
        "binom_1000": st.binom.rvs(n=12, p=0.35, size=1000, random_state=random_state),
    }

)
samples.update({
        "norm_100": st.norm.rvs(loc=4.5, scale=1.8, size=100, random_state=random_state),
        "norm_1000": st.norm.rvs(loc=4.5, scale=1.8, size=1000, random_state=random_state),
    }

)

# ==== Ячейка 9 ====
def mean(X):
    total = 0.0
    count = 0
    for xi in X:
        total += float(xi)
        count += 1
    return total / count if count else float("nan")


def var(X):
    count = len(X)
    if count == 0:
        return float("nan")

    Xr = mean(X)
    sq_sum = 0.0
    for value in X:
        sq_sum +=  (float(value) - Xr)**2
    return sq_sum / count


def std(X):
    return math.sqrt(var(X))

# ==== Ячейка 10 ====
mean_np = {name: np.mean(sample,) for name, sample in samples.items()}
var_np = {name: np.var(sample, ) for name, sample in samples.items()}
std_np = {name: np.std(sample,) for name, sample in samples.items()}

mean_my = {name: mean(sample,) for name, sample in samples.items()}
var_my = {name: var(sample, ) for name, sample in samples.items()}
std_my = {name: std(sample,) for name, sample in samples.items()}

# ==== Ячейка 11 ====
sample_order = [
    "uniform_100",
    "uniform_1000",
    "bernoulli_100",
    "bernoulli_1000",
    "binom_100",
    "binom_1000",
    "norm_100",
    "norm_1000",
]

table_rows = []
for i, name in enumerate(sample_order, start=1):
    table_rows.append({
        "№ выборки": i,
        "среднее своё": mean_my[name],
        "дисперсия своя": var_my[name],
        "стандартное отклонение своё": std_my[name],
        "среднее numpy": mean_np[name],
        "дисперсия numpy": var_np[name],
        "стандартное отклонение numpy": std_np[name],
    })

result_table = pd.DataFrame(table_rows)
result_table

# ==== Ячейка 12 ====
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

for sample_number, sample_name, title, dist_kind, dist_name in plot_config:
    sample = samples[sample_name]
    plt.figure(figsize=(8, 5))

    if dist_kind == "continuous":
        # Плотность относительных частот (высота столбца гистограммы):
        # h_i = n_i / (n * Δ_i), где n_i — число наблюдений в i-м интервале,
        # n — объем выборки, Δ_i — ширина интервала; при этом ω_i = n_i / n
        # и h_i = ω_i / Δ_i. Параметр density=True в plt.hist делает эту
        # нормировку автоматически.
        # Формула Стерджесса для числа столбцов:
        # k = ceil(1 + 3.322 * log10(n))
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

        # Для дискретных выборок тоже берём число столбцов по Стерджессу:
        # k = ceil(1 + 3.322 * log10(n))
        # Δ = (x_max_edge - x_min_edge) / k, где x_min_edge = x_min - 0.5,
        # x_max_edge = x_max + 0.5
        n_obs = len(sample)
        k_sturges = max(1, math.ceil(1 + 3.322 * math.log10(n_obs)))
        x_min_edge = x_min - 0.5
        x_max_edge = x_max + 0.5
        delta = (x_max_edge - x_min_edge) / k_sturges if x_max_edge > x_min_edge else 1.0
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

        plt.plot(values, p_theory, "o-", color="crimson", linewidth=2, label="Функция вероятности")

    plt.title(f"Выборка №{sample_number}: {title}")
    plt.xlabel("x")
    plt.ylabel("Плотность / вероятность")
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()
