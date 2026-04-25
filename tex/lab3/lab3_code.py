import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as st


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (16, 10)
plt.rcParams["axes.titlesize"] = 11
plt.rcParams["axes.labelsize"] = 10


random_state = 16


samples = {
    "uniform_100": st.uniform.rvs(loc=2, scale=5, size=100, random_state=random_state),
    "uniform_1000": st.uniform.rvs(
        loc=2, scale=5, size=1000, random_state=random_state
    ),
    "bernoulli_100": st.bernoulli.rvs(p=0.27, size=100, random_state=random_state),
    "bernoulli_1000": st.bernoulli.rvs(
        p=0.27, size=1000, random_state=random_state
    ),
    "binom_100": st.binom.rvs(n=12, p=0.35, size=100, random_state=random_state),
    "binom_1000": st.binom.rvs(
        n=12, p=0.35, size=1000, random_state=random_state
    ),
    "norm_100": st.norm.rvs(loc=4.5, scale=1.8, size=100, random_state=random_state),
    "norm_1000": st.norm.rvs(
        loc=4.5, scale=1.8, size=1000, random_state=random_state
    ),
}


plot_config = [
    {
        "sample_name": "uniform_100",
        "title": "Равномерное U(2, 7), n=100",
        "kind": "continuous",
        "dist": st.uniform(loc=2, scale=5),
    },
    {
        "sample_name": "uniform_1000",
        "title": "Равномерное U(2, 7), n=1000",
        "kind": "continuous",
        "dist": st.uniform(loc=2, scale=5),
    },
    {
        "sample_name": "bernoulli_100",
        "title": "Бернулли B(0.27), n=100",
        "kind": "discrete",
        "dist": st.bernoulli(p=0.27),
    },
    {
        "sample_name": "bernoulli_1000",
        "title": "Бернулли B(0.27), n=1000",
        "kind": "discrete",
        "dist": st.bernoulli(p=0.27),
    },
    {
        "sample_name": "binom_100",
        "title": "Биномиальное Bin(12, 0.35), n=100",
        "kind": "discrete",
        "dist": st.binom(n=12, p=0.35),
    },
    {
        "sample_name": "binom_1000",
        "title": "Биномиальное Bin(12, 0.35), n=1000",
        "kind": "discrete",
        "dist": st.binom(n=12, p=0.35),
    },
    {
        "sample_name": "norm_100",
        "title": "Нормальное N(4.5, 1.8^2), n=100",
        "kind": "continuous",
        "dist": st.norm(loc=4.5, scale=1.8),
    },
    {
        "sample_name": "norm_1000",
        "title": "Нормальное N(4.5, 1.8^2), n=1000",
        "kind": "continuous",
        "dist": st.norm(loc=4.5, scale=1.8),
    },
]


def empirical_cdf(sample, x, alpha=0.05):
    sample = np.sort(np.asarray(sample))
    n = sample.size
    x_array = np.atleast_1d(x)

    fn = np.searchsorted(sample, x_array, side="right") / n
    epsilon = np.sqrt(np.log(2 / alpha) / (2 * n))
    lower = np.clip(fn - epsilon, 0.0, 1.0)
    upper = np.clip(fn + epsilon, 0.0, 1.0)

    if np.ndim(x) == 0:
        return float(fn[0]), float(lower[0]), float(upper[0])
    return fn, lower, upper


def make_x_grid(sample, kind, frozen_dist):
    sample = np.asarray(sample)
    left, right = frozen_dist.support()

    if kind == "continuous":
        if np.isfinite(left) and np.isfinite(right):
            span = right - left
            pad = 0.05 * span if span > 0 else 1.0
            return np.linspace(left - pad, right + pad, 600)

        q_left = frozen_dist.ppf(0.001)
        q_right = frozen_dist.ppf(0.999)
        left = min(sample.min(), q_left)
        right = max(sample.max(), q_right)
        pad = 0.05 * (right - left) if right > left else 1.0
        return np.linspace(left - pad, right + pad, 600)

    left = int(np.floor(left)) - 1
    right = int(np.ceil(right)) + 1
    return np.arange(left, right + 1)


def build_demo_table(config):
    rows = []
    for item in config:
        sample = samples[item["sample_name"]]
        x0 = float(np.median(sample))
        fn, low, high = empirical_cdf(sample, x0)
        rows.append(
            {
                "Выборка": item["sample_name"],
                "x": round(x0, 4),
                "F_n(x)": round(fn, 4),
                "Нижняя граница 95% ДИ": round(low, 4),
                "Верхняя граница 95% ДИ": round(high, 4),
            }
        )
    return pd.DataFrame(rows)


def plot_with_custom_ecdf(ax, sample, title, kind, frozen_dist):
    x_grid = make_x_grid(sample, kind, frozen_dist)
    true_cdf = frozen_dist.cdf(x_grid)
    emp_cdf, low, high = empirical_cdf(sample, x_grid)

    ax.plot(x_grid, true_cdf, color="crimson", linewidth=2, label="Истинная CDF")
    ax.step(x_grid, emp_cdf, where="post", color="royalblue", linewidth=2, label="ЭФР")
    ax.step(
        x_grid,
        low,
        where="post",
        color="forestgreen",
        linestyle="--",
        linewidth=1.5,
        label="95% ДИ",
    )
    ax.step(
        x_grid,
        high,
        where="post",
        color="forestgreen",
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_ylim(-0.05, 1.05)


def plot_with_scipy_ecdf(ax, sample, title, kind, frozen_dist):
    x_grid = make_x_grid(sample, kind, frozen_dist)
    true_cdf = frozen_dist.cdf(x_grid)

    ecdf_result = st.ecdf(sample)
    ecdf_values = ecdf_result.cdf.evaluate(x_grid)
    ci = ecdf_result.cdf.confidence_interval(confidence_level=0.95)
    low = ci.low.evaluate(x_grid)
    high = ci.high.evaluate(x_grid)

    ax.plot(x_grid, true_cdf, color="crimson", linewidth=2, label="Истинная CDF")
    ax.step(
        x_grid,
        ecdf_values,
        where="post",
        color="royalblue",
        linewidth=2,
        label="ЭФР SciPy",
    )
    ax.step(
        x_grid,
        low,
        where="post",
        color="forestgreen",
        linestyle="--",
        linewidth=1.5,
        label="95% ДИ",
    )
    ax.step(
        x_grid,
        high,
        where="post",
        color="forestgreen",
        linestyle="--",
        linewidth=1.5,
    )
    ax.set_title(title)
    ax.set_xlabel("x")
    ax.set_ylabel("F(x)")
    ax.set_ylim(-0.05, 1.05)


print(build_demo_table(plot_config))


fig, axes = plt.subplots(4, 2, figsize=(16, 18))
axes = axes.ravel()

for ax, item in zip(axes, plot_config):
    sample = samples[item["sample_name"]]
    plot_with_custom_ecdf(ax, sample, item["title"], item["kind"], item["dist"])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
fig.suptitle("Собственная реализация эмпирической функции распределения", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()


fig, axes = plt.subplots(4, 2, figsize=(16, 18))
axes = axes.ravel()

for ax, item in zip(axes, plot_config):
    sample = samples[item["sample_name"]]
    plot_with_scipy_ecdf(ax, sample, item["title"], item["kind"], item["dist"])

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
fig.suptitle("Встроенная эмпирическая функция распределения из scipy.stats", fontsize=16)
fig.tight_layout(rect=[0, 0, 1, 0.97])
plt.show()
