from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.stats as st


ALTERNATIVES = {"less", "greater", "two-sided"}


@dataclass
class MeanTestResult:
    test_name: str
    statistic_name: str
    statistic: float
    p_value: float
    sample_mean: float
    sample_std: float | None
    known_std: float | None
    n: int
    df: int | None
    alternative: str
    mu0: float


def _validate_alternative(alternative: str) -> None:
    if alternative not in ALTERNATIVES:
        raise ValueError(
            "alternative must be one of: 'less', 'greater', 'two-sided'"
        )


def _normal_p_value(statistic: float, alternative: str) -> float:
    # For a right-tailed z-test the correct formula is p = 1 - Phi(z).
    # A common mistake is to add 0.5 or to use a two-sided formula here.
    if alternative == "greater":
        return float(1.0 - st.norm.cdf(statistic))
    if alternative == "less":
        return float(st.norm.cdf(statistic))
    return float(2.0 * st.norm.cdf(-abs(statistic)))


def _student_p_value(statistic: float, df: int, alternative: str) -> float:
    if alternative == "greater":
        return float(1.0 - st.t.cdf(statistic, df=df))
    if alternative == "less":
        return float(st.t.cdf(statistic, df=df))
    return float(2.0 * st.t.cdf(-abs(statistic), df=df))


def _prepare_statistics(
    sample=None,
    *,
    sample_mean: float | None = None,
    n: int | None = None,
    std: float | None = None,
    sample_std: float | None = None,
):
    if sample is not None:
        sample_array = np.asarray(sample, dtype=float).ravel()
        if sample_array.size == 0:
            raise ValueError("sample must contain at least one observation")
        n = int(sample_array.size)
        sample_mean = float(np.mean(sample_array))
        if std is None:
            if sample_array.size < 2:
                raise ValueError("at least two observations are required for the t-test")
            sample_std = float(np.std(sample_array, ddof=1))
        return sample_mean, n, sample_std

    if sample_mean is None or n is None:
        raise ValueError(
            "either sample or both sample_mean and n must be provided"
        )

    if n <= 0:
        raise ValueError("n must be positive")

    if std is None and sample_std is None:
        raise ValueError(
            "for the t-test you must provide either sample or sample_std"
        )

    return float(sample_mean), int(n), None if sample_std is None else float(sample_std)


def mean_test(
    sample=None,
    *,
    mu0: float,
    alternative: str = "two-sided",
    std: float | None = None,
    sample_mean: float | None = None,
    n: int | None = None,
    sample_std: float | None = None,
) -> MeanTestResult:
    _validate_alternative(alternative)
    sample_mean, n, sample_std = _prepare_statistics(
        sample,
        sample_mean=sample_mean,
        n=n,
        std=std,
        sample_std=sample_std,
    )

    if std is not None:
        if std <= 0:
            raise ValueError("std must be positive")
        statistic = (sample_mean - mu0) * math.sqrt(n) / std
        p_value = _normal_p_value(statistic, alternative)
        return MeanTestResult(
            test_name="Z-test",
            statistic_name="Z",
            statistic=float(statistic),
            p_value=float(p_value),
            sample_mean=float(sample_mean),
            sample_std=None,
            known_std=float(std),
            n=int(n),
            df=None,
            alternative=alternative,
            mu0=float(mu0),
        )

    if sample_std is None or sample_std <= 0:
        raise ValueError("sample_std must be positive for the t-test")

    statistic = (sample_mean - mu0) * math.sqrt(n) / sample_std
    df = n - 1
    p_value = _student_p_value(statistic, df, alternative)
    return MeanTestResult(
        test_name="Student t-test",
        statistic_name="T",
        statistic=float(statistic),
        p_value=float(p_value),
        sample_mean=float(sample_mean),
        sample_std=float(sample_std),
        known_std=None,
        n=int(n),
        df=int(df),
        alternative=alternative,
        mu0=float(mu0),
    )


def mean_test_pvalue(
    sample=None,
    *,
    mu0: float,
    alternative: str = "two-sided",
    std: float | None = None,
    sample_mean: float | None = None,
    n: int | None = None,
    sample_std: float | None = None,
) -> float:
    return mean_test(
        sample,
        mu0=mu0,
        alternative=alternative,
        std=std,
        sample_mean=sample_mean,
        n=n,
        sample_std=sample_std,
    ).p_value


def expand_frequency_sample(values, frequencies):
    values = np.asarray(values, dtype=float)
    frequencies = np.asarray(frequencies, dtype=int)
    if values.size != frequencies.size:
        raise ValueError("values and frequencies must have the same length")
    if np.any(frequencies < 0):
        raise ValueError("frequencies must be non-negative")
    return np.repeat(values, frequencies)


def decision_text(p_value: float, alpha: float) -> str:
    if p_value < alpha:
        return f"p-value = {p_value:.6f} < alpha = {alpha:.2f}, therefore H0 is rejected."
    return f"p-value = {p_value:.6f} >= alpha = {alpha:.2f}, therefore there is no reason to reject H0."


def solve_task_1(alpha: float = 0.01):
    mu0 = 0.50
    sample_mean = 0.53
    n = 121
    sigma = 0.11

    result = mean_test(
        mu0=mu0,
        alternative="greater",
        std=sigma,
        sample_mean=sample_mean,
        n=n,
    )
    critical_value = float(st.norm.ppf(1.0 - alpha))

    return {
        "alpha": alpha,
        "mu0": mu0,
        "sample_mean": sample_mean,
        "n": n,
        "sigma": sigma,
        "result": result,
        "critical_value": critical_value,
        "conclusion": (
            "At the 1% significance level the average tablet weight is significantly "
            "greater than 0.50 mg."
            if result.p_value < alpha
            else "At the 1% significance level there is no evidence that the average "
            "tablet weight is greater than 0.50 mg."
        ),
    }


def solve_task_2(alpha: float = 0.05):
    values = np.array([34.8, 34.9, 35.0, 35.1, 35.3], dtype=float)
    frequencies = np.array([2, 3, 4, 6, 5], dtype=int)
    sample = expand_frequency_sample(values, frequencies)
    mu0 = 35.0

    custom_result = mean_test(sample, mu0=mu0, alternative="two-sided")
    scipy_result = st.ttest_1samp(sample, popmean=mu0, alternative="two-sided")
    critical_value = float(st.t.ppf(1.0 - alpha / 2.0, df=custom_result.df))

    return {
        "alpha": alpha,
        "mu0": mu0,
        "values": values,
        "frequencies": frequencies,
        "sample": sample,
        "custom_result": custom_result,
        "scipy_statistic": float(scipy_result.statistic),
        "scipy_p_value": float(scipy_result.pvalue),
        "critical_value": critical_value,
        "sample_variance": float(np.var(sample, ddof=1)),
        "conclusion": (
            "At the 5% significance level the mean size differs from 35 mm."
            if custom_result.p_value < alpha
            else "At the 5% significance level there is no evidence that the mean size differs from 35 mm."
        ),
    }


def build_frequency_table_tex(values, frequencies) -> str:
    value_row = " & ".join(f"{value:.1f}" for value in values)
    frequency_row = " & ".join(str(int(freq)) for freq in frequencies)
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Данные задачи 2}",
            r"\begin{tabular}{|c|c|c|c|c|c|}",
            r"\hline",
            rf"$x_i$ & {value_row} \\",
            r"\hline",
            rf"$n_i$ & {frequency_row} \\",
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def build_results_table_tex(task_1_data, task_2_data) -> str:
    return "\n".join(
        [
            r"\begin{table}[H]",
            r"\centering",
            r"\caption{Итоги проверки гипотез}",
            r"\begin{tabular}{|p{3.0cm}|p{1.8cm}|p{2.2cm}|p{2.2cm}|p{5.2cm}|}",
            r"\hline",
            r"Задача & Критерий & Статистика & p-value & Решение \\",
            r"\hline",
            (
                "Задача 1"
                f" & {task_1_data['result'].test_name}"
                f" & $Z={task_1_data['result'].statistic:.4f}$"
                f" & {task_1_data['result'].p_value:.6f}"
                r" & Отклоняем $H_0$, так как $p < 0.01$. \\"
            ),
            r"\hline",
            (
                "Задача 2"
                f" & {task_2_data['custom_result'].test_name}"
                f" & $T={task_2_data['custom_result'].statistic:.4f}$"
                f" & {task_2_data['custom_result'].p_value:.6f}"
                r" & Не отвергаем $H_0$, так как $p > 0.05$. \\"
            ),
            r"\hline",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )


def print_task_1_report(task_1_data) -> None:
    result = task_1_data["result"]
    print("TASK 1")
    print("Null hypothesis:      H0: a = 0.50")
    print("Alternative:          H1: a > 0.50")
    print(f"Sample mean:          x_bar = {task_1_data['sample_mean']:.2f}")
    print(f"Sample size:          n = {task_1_data['n']}")
    print(f"Known std deviation:  sigma = {task_1_data['sigma']:.2f}")
    print(
        f"Test statistic:       {result.statistic_name} = "
        f"((x_bar - mu0) * sqrt(n)) / sigma = {result.statistic:.6f}"
    )
    print(
        f"Critical value:       z_(1-alpha) = "
        f"{task_1_data['critical_value']:.6f}"
    )
    print(f"p-value:              {result.p_value:.6f}")
    print(f"Decision:             {decision_text(result.p_value, task_1_data['alpha'])}")
    print(f"Conclusion:           {task_1_data['conclusion']}")
    print()


def print_task_2_report(task_2_data) -> None:
    result = task_2_data["custom_result"]
    print("TASK 2")
    print("Null hypothesis:      H0: a = 35")
    print("Alternative:          H1: a != 35")
    print(f"Expanded sample size: n = {result.n}")
    print(f"Sample mean:          x_bar = {result.sample_mean:.6f}")
    print(f"Sample variance:      s^2 = {task_2_data['sample_variance']:.6f}")
    print(f"Sample std deviation: s = {result.sample_std:.6f}")
    print(
        f"Test statistic:       {result.statistic_name} = "
        f"((x_bar - mu0) * sqrt(n)) / s = {result.statistic:.6f}"
    )
    print(
        f"Critical value:       t_(1-alpha/2; {result.df}) = "
        f"{task_2_data['critical_value']:.6f}"
    )
    print(f"Custom p-value:       {result.p_value:.6f}")
    print(
        "SciPy ttest_1samp:    "
        f"statistic = {task_2_data['scipy_statistic']:.6f}, "
        f"p-value = {task_2_data['scipy_p_value']:.6f}"
    )
    print(f"Decision:             {decision_text(result.p_value, task_2_data['alpha'])}")
    print(f"Conclusion:           {task_2_data['conclusion']}")
    print()


def write_report_assets(base_dir: Path, task_1_data, task_2_data) -> None:
    (base_dir / "task2_frequency_table.tex").write_text(
        build_frequency_table_tex(task_2_data["values"], task_2_data["frequencies"]),
        encoding="utf-8",
    )
    (base_dir / "results_table.tex").write_text(
        build_results_table_tex(task_1_data, task_2_data),
        encoding="utf-8",
    )


def run_self_checks(task_1_data, task_2_data) -> None:
    task_1_expected = 0.0013498980316301035
    if not math.isclose(
        task_1_data["result"].p_value, task_1_expected, rel_tol=0.0, abs_tol=1e-12
    ):
        raise AssertionError(
            "Task 1 p-value is incorrect. The correct right-tailed z-test value is "
            "0.0013498980316301035."
        )

    if not math.isclose(
        task_2_data["custom_result"].p_value,
        task_2_data["scipy_p_value"],
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise AssertionError("Task 2 custom p-value must match scipy.stats.ttest_1samp.")


def main() -> None:
    task_1_data = solve_task_1()
    task_2_data = solve_task_2()
    run_self_checks(task_1_data, task_2_data)

    print("CUSTOM FUNCTION DEMO")
    print(
        "mean_test_pvalue(...) returns only the p-value; "
        "mean_test(...) returns detailed statistics."
    )
    print(
        "Task 1 p-value from the custom function: "
        f"{mean_test_pvalue(mu0=0.50, alternative='greater', std=0.11, sample_mean=0.53, n=121):.6f}"
    )
    print(
        "Important: for alternative='greater' we use p = 1 - Phi(z), "
        "so the correct Task 1 p-value is 0.001349898..., not 0.501349898..."
    )
    print(
        "Task 2 p-value from the custom function: "
        f"{mean_test_pvalue(task_2_data['sample'], mu0=35.0, alternative='two-sided'):.6f}"
    )
    print()

    print_task_1_report(task_1_data)
    print_task_2_report(task_2_data)

    base_dir = Path(__file__).resolve().parent
    write_report_assets(base_dir, task_1_data, task_2_data)


if __name__ == "__main__":
    main()
