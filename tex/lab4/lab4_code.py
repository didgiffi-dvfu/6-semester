# Exported from lab4.ipynb for inclusion in the LaTeX report.

import math
import numpy as np
import pandas as pd
import scipy.stats as st

RANDOM_STATE = 16
BOOTSTRAP_RANDOM_STATE = 2026
CONFIDENCE_LEVEL = 0.95
N_RESAMPLES = 10_000

pd.set_option("display.max_colwidth", None)


def build_samples(random_state=RANDOM_STATE):
    return {
        "uniform_100": st.uniform.rvs(loc=2, scale=5, size=100, random_state=random_state),
        "uniform_1000": st.uniform.rvs(loc=2, scale=5, size=1000, random_state=random_state),
        "bernoulli_100": st.bernoulli.rvs(p=0.27, size=100, random_state=random_state),
        "bernoulli_1000": st.bernoulli.rvs(p=0.27, size=1000, random_state=random_state),
        "binom_100": st.binom.rvs(n=12, p=0.35, size=100, random_state=random_state),
        "binom_1000": st.binom.rvs(n=12, p=0.35, size=1000, random_state=random_state),
        "norm_100": st.norm.rvs(loc=4.5, scale=1.8, size=100, random_state=random_state),
        "norm_1000": st.norm.rvs(loc=4.5, scale=1.8, size=1000, random_state=random_state),
    }


samples = build_samples()


def sample_mean(sample):
    sample = np.asarray(sample, dtype=float)
    return float(np.mean(sample))


def sample_var(sample):
    sample = np.asarray(sample, dtype=float)
    centered = sample - np.mean(sample)
    return float(np.mean(centered ** 2))


def estimate_uniform_params(sample):
    mu = sample_mean(sample)
    var = sample_var(sample)
    half_width = math.sqrt(max(3.0 * var, 0.0))
    return mu - half_width, mu + half_width


def estimate_uniform_a(sample):
    return estimate_uniform_params(sample)[0]


def estimate_uniform_b(sample):
    return estimate_uniform_params(sample)[1]


def estimate_bernoulli_p(sample):
    return sample_mean(sample)


def estimate_binom_params(sample):
    x = np.asarray(sample, dtype=float)
    mu = sample_mean(x)
    var = sample_var(x)
    eps = 1e-12

    if mu <= eps:
        return 1.0, 0.0

    denominator = mu - var
    if denominator <= eps:
        m_hat = max(float(np.max(x)), 1.0)
    else:
        m_hat = max(mu * mu / denominator, float(np.max(x)), 1.0)

    p_hat = min(max(mu / m_hat, eps), 1 - eps)
    return float(m_hat), float(p_hat)


def estimate_binom_m(sample):
    return estimate_binom_params(sample)[0]


def estimate_binom_p(sample):
    return estimate_binom_params(sample)[1]


def estimate_normal_mu(sample):
    return sample_mean(sample)


def estimate_normal_sigma(sample):
    return math.sqrt(max(sample_var(sample), 0.0))


def bootstrap_percentile(
    sample,
    estimator,
    confidence_level=0.95,
    n_resamples=9999,
    random_state=0,
):
    sample = np.asarray(sample)
    n = sample.size
    rng = np.random.default_rng(random_state)

    theta_hat = float(estimator(sample))
    bootstrap_estimates = np.empty(n_resamples, dtype=float)

    for i in range(n_resamples):
        bootstrap_sample = rng.choice(sample, size=n, replace=True)
        bootstrap_estimates[i] = float(estimator(bootstrap_sample))

    bootstrap_estimates.sort()
    alpha = 1.0 - confidence_level
    lower_rank = max(1, int(np.floor(alpha / 2.0 * n_resamples)))
    upper_rank = min(n_resamples, int(np.ceil((1.0 - alpha / 2.0) * n_resamples)))

    return {
        "estimate": theta_hat,
        "confidence_interval": (
            float(bootstrap_estimates[lower_rank - 1]),
            float(bootstrap_estimates[upper_rank - 1]),
        ),
        "bootstrap_distribution": bootstrap_estimates,
    }


def scipy_percentile_bootstrap(
    sample,
    estimator,
    confidence_level=0.95,
    n_resamples=9999,
    random_state=0,
):
    result = st.bootstrap(
        (np.asarray(sample),),
        estimator,
        n_resamples=n_resamples,
        confidence_level=confidence_level,
        method="percentile",
        vectorized=False,
        random_state=random_state,
    )
    return float(result.confidence_interval.low), float(result.confidence_interval.high)


EXPERIMENTS = [
    {
        "sample_key": "uniform_100",
        "label": "U(2,7), n=100",
        "true_parameters": {"a": 2.0, "b": 7.0},
        "estimators": {"a": estimate_uniform_a, "b": estimate_uniform_b},
    },
    {
        "sample_key": "uniform_1000",
        "label": "U(2,7), n=1000",
        "true_parameters": {"a": 2.0, "b": 7.0},
        "estimators": {"a": estimate_uniform_a, "b": estimate_uniform_b},
    },
    {
        "sample_key": "bernoulli_100",
        "label": "Bernoulli(0.27), n=100",
        "true_parameters": {"p": 0.27},
        "estimators": {"p": estimate_bernoulli_p},
    },
    {
        "sample_key": "bernoulli_1000",
        "label": "Bernoulli(0.27), n=1000",
        "true_parameters": {"p": 0.27},
        "estimators": {"p": estimate_bernoulli_p},
    },
    {
        "sample_key": "binom_100",
        "label": "Bin(12,0.35), n=100",
        "true_parameters": {"m": 12.0, "p": 0.35},
        "estimators": {"m": estimate_binom_m, "p": estimate_binom_p},
    },
    {
        "sample_key": "binom_1000",
        "label": "Bin(12,0.35), n=1000",
        "true_parameters": {"m": 12.0, "p": 0.35},
        "estimators": {"m": estimate_binom_m, "p": estimate_binom_p},
    },
    {
        "sample_key": "norm_100",
        "label": "N(4.5,1.8^2), n=100",
        "true_parameters": {"mu": 4.5, "sigma": 1.8},
        "estimators": {"mu": estimate_normal_mu, "sigma": estimate_normal_sigma},
    },
    {
        "sample_key": "norm_1000",
        "label": "N(4.5,1.8^2), n=1000",
        "true_parameters": {"mu": 4.5, "sigma": 1.8},
        "estimators": {"mu": estimate_normal_mu, "sigma": estimate_normal_sigma},
    },
]


def format_interval(interval):
    return f"[{interval[0]:.4f}; {interval[1]:.4f}]"


summary_rows = []
detail_rows = []

for experiment in EXPERIMENTS:
    sample = samples[experiment["sample_key"]]
    estimate_chunks = []
    custom_chunks = []
    scipy_chunks = []
    max_difference = 0.0

    for offset, (parameter_name, estimator) in enumerate(experiment["estimators"].items()):
        custom_result = bootstrap_percentile(
            sample,
            estimator,
            confidence_level=CONFIDENCE_LEVEL,
            n_resamples=N_RESAMPLES,
            random_state=BOOTSTRAP_RANDOM_STATE + offset,
        )
        scipy_interval = scipy_percentile_bootstrap(
            sample,
            estimator,
            confidence_level=CONFIDENCE_LEVEL,
            n_resamples=N_RESAMPLES,
            random_state=BOOTSTRAP_RANDOM_STATE + offset,
        )

        custom_interval = custom_result["confidence_interval"]
        difference = max(
            abs(custom_interval[0] - scipy_interval[0]),
            abs(custom_interval[1] - scipy_interval[1]),
        )
        max_difference = max(max_difference, difference)

        estimate_chunks.append(f"{parameter_name}={custom_result['estimate']:.4f}")
        custom_chunks.append(f"{parameter_name}: {format_interval(custom_interval)}")
        scipy_chunks.append(f"{parameter_name}: {format_interval(scipy_interval)}")

        detail_rows.append(
            {
                "sample": experiment["label"],
                "parameter": parameter_name,
                "true_value": experiment["true_parameters"][parameter_name],
                "point_estimate": custom_result["estimate"],
                "custom_bootstrap": format_interval(custom_interval),
                "scipy_bootstrap": format_interval(scipy_interval),
                "max_abs_diff": difference,
            }
        )

    summary_rows.append(
        {
            "sample": experiment["label"],
            "true_parameters": ", ".join(
                f"{name}={value:g}" for name, value in experiment["true_parameters"].items()
            ),
            "point_estimates": "; ".join(estimate_chunks),
            "custom_bootstrap_95": "; ".join(custom_chunks),
            "scipy_bootstrap_95": "; ".join(scipy_chunks),
            "max_abs_diff": max_difference,
        }
    )

summary_df = pd.DataFrame(summary_rows)
detail_df = pd.DataFrame(detail_rows)
detail_display_df = detail_df.copy()

for column in ["true_value", "point_estimate", "max_abs_diff"]:
    detail_display_df[column] = detail_display_df[column].astype(float).round(4)

summary_df["max_abs_diff"] = summary_df["max_abs_diff"].astype(float).round(4)

print("Summary table")
print(summary_df.to_string(index=False))
print()

print("Detailed table")
print(detail_display_df.to_string(index=False))
print()

largest_gap_row = detail_display_df.loc[detail_display_df["max_abs_diff"].idxmax()]
print(
    "Conclusion: intervals become narrower when n grows from 100 to 1000. "
    "The custom percentile bootstrap and scipy.stats.bootstrap(..., method='percentile') "
    "give close results. "
    f"The largest gap is for sample {largest_gap_row['sample']} and parameter "
    f"{largest_gap_row['parameter']}; the maximum absolute difference is "
    f"{largest_gap_row['max_abs_diff']:.4f}."
)

summary_df
