from __future__ import annotations

from pathlib import Path

import numpy as np

import generate_dop_labs as g


ROOT = Path(__file__).resolve().parent
OUT_DIR = ROOT / "доп_лабы"
REPORT_PATH = OUT_DIR / "report_labs_1_2_4_5.md"


def block(code: str, lang: str = "text") -> str:
    return f"```{lang}\n{code.strip()}\n```\n"


def fmt(x, digits: int = 4) -> str:
    return g.fmt(x, digits)


def make_interval_table(sample, intervals=5):
    arr = np.asarray(sample, dtype=float)
    xmin = float(arr.min())
    xmax = float(arr.max())
    n = int(arr.size)
    r = xmax - xmin
    delta = r / intervals
    edges = [xmin + i * delta for i in range(intervals + 1)]

    rows = []
    counts = []
    rel = []
    cum = []
    total = 0
    for i in range(intervals):
        left = edges[i]
        right = edges[i + 1]
        if i < intervals - 1:
            count = int(((arr >= left) & (arr < right)).sum())
        else:
            count = int(((arr >= left) & (arr <= right)).sum())
        total += count
        counts.append(count)
        rel.append(count / n)
        cum.append(total / n)
        rows.append(
            [
                f"[{fmt(left)}; {fmt(right)}{')' if i < intervals - 1 else ']'}",
                fmt((left + right) / 2),
                str(count),
                fmt(count / n),
                fmt(total / n),
            ]
        )

    table = g.md_table(
        ["Интервал", "Середина", "Частота", "Отн. частота", "Накопл. отн. частота"],
        rows,
    )
    return {
        "xmin": xmin,
        "xmax": xmax,
        "r": r,
        "delta": delta,
        "edges": edges,
        "counts": counts,
        "rel": rel,
        "cum": cum,
        "table": table,
    }


def explain_counts(counts):
    return ", ".join(str(c) for c in counts)


def lab1_section() -> str:
    sample = [
        10.2, 9.23, 8.77, 10.4, 9.44, 9.09, 6.30, 9.42, 6.12, 9.69,
        8.59, 8.68, 7.97, 8.64, 6.45, 5.29, 5.00, 8.42, 8.84, 8.26,
        6.66, 6.96, 6.51, 6.72, 6.00, 5.36,
    ]
    series = make_interval_table(sample, intervals=5)
    code_text = f"""
import numpy as np
import matplotlib.pyplot as plt

sample = {sample}

def interval_series(values, intervals=5):
    values = np.asarray(values, dtype=float)
    xmin = values.min()
    xmax = values.max()
    delta = (xmax - xmin) / intervals
    edges = [xmin + i * delta for i in range(intervals + 1)]
    counts = []
    cumulative = []
    rel = []
    total = 0
    for i in range(intervals):
        left, right = edges[i], edges[i + 1]
        if i < intervals - 1:
            count = int(((values >= left) & (values < right)).sum())
        else:
            count = int(((values >= left) & (values <= right)).sum())
        total += count
        counts.append(count)
        rel.append(count / len(values))
        cumulative.append(total / len(values))
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(intervals)]
    return edges, mids, counts, rel, cumulative

def ecdf_from_intervals(mids, cumulative):
    plt.step(mids, cumulative, where='mid')
    plt.scatter(mids, cumulative)
    plt.xlabel('x')
    plt.ylabel('F_n*(x)')
    plt.grid(alpha=0.25)

edges, mids, counts, rel, cumulative = interval_series(sample, intervals=5)
print('Границы интервалов:', edges)
print('Частоты:', counts)
print('Относительные частоты:', rel)
print('Накопленные частоты:', cumulative)
"""

    manual = f"""
Дано:

$$
n = 26, \\quad x_{{min}} = {fmt(series['xmin'])}, \\quad x_{{max}} = {fmt(series['xmax'])}
$$

Размах:

$$
R = x_{{max}} - x_{{min}} = {fmt(series['xmax'])} - {fmt(series['xmin'])} = {fmt(series['r'])}
$$

Число интервалов по формуле Стерджеса:

$$
N = \\lfloor 1 + 3.322\\lg(n) \\rfloor = \\lfloor 1 + 3.322\\lg(26) \\rfloor = 5
$$

Длина интервала:

$$
\\Delta = \\frac{{R}}{{N}} = \\frac{{{fmt(series['r'])}}}{{5}} = {fmt(series['delta'])}
$$

Границы интервалов и частоты:

{series['table']}

Накопленные относительные частоты:

$$
0.1538,\\ 0.4231,\\ 0.4615,\\ 0.8077,\\ 1.0000
$$
"""

    return f"""
# Лабораторная работа №1
## Интервальный вариационный ряд и эмпирическая функция распределения

**Вариант:** `16 % 10 = 6`

**Данные выборки:**

{block(", ".join(fmt(x, 2) for x in sample), "text")}

### 1. Теоретическая часть

Для непрерывной выборки строят интервальный вариационный ряд. Если выборка состоит из наблюдений
$x_1, x_2, \\dots, x_n$, то:

$$
R = x_{{max}} - x_{{min}},
\\qquad
\\Delta = \\frac{{R}}{{N}}
$$

где:

* $R$ - размах варьирования;
* $N$ - число интервалов группировки;
* $\\Delta$ - длина одного интервала.

Эмпирическая функция распределения:

$$
F_n^*(x) = \\frac{{n_x}}{{n}}
$$

где $n_x$ - число элементов выборки, строго меньших $x$.

### 2. Алгоритм решения

1. Найти $x_{{min}}$ и $x_{{max}}$.
2. Вычислить размах $R$.
3. Определить число интервалов $N$ по формуле Стерджеса.
4. Найти длину интервала $\\Delta$.
5. Разбить отрезок варьирования на интервалы.
6. Подсчитать частоты, относительные частоты и накопленные частоты.
7. Построить график эмпирической функции распределения.

### 3. Реализация на Python

{block(code_text, "python")}

### 4. Разбор кода

* `sample` хранит исходную выборку.
* `interval_series(...)` строит интервальный ряд.
* `xmin` и `xmax` - границы диапазона данных.
* `delta` - ширина интервала.
* `counts` - частоты по интервалам.
* `rel` - относительные частоты $\\omega_i = n_i / n$.
* `cumulative` - накопленные частоты для кумуляты.
* `plt.step(...)` строит график накопленной функции.

### 5. Пример выполнения

Ввод:

{block("sample = [10.2, 9.23, 8.77, 10.4, 9.44, 9.09, 6.30, 9.42, 6.12, 9.69, 8.59, 8.68, 7.97, 8.64, 6.45, 5.29, 5.00, 8.42, 8.84, 8.26, 6.66, 6.96, 6.51, 6.72, 6.00, 5.36]", "text")}

Вывод:

{series['table']}

### 6. Пошаговый разбор вычислений

{manual}

### 7. Итог

Для варианта 6 построен интервальный вариационный ряд, вычислены относительные и накопленные частоты, а также получена эмпирическая функция распределения в виде кумуляты.
"""


def lab2_section() -> str:
    cont = [
        6.52, 9.27, 7.91, 5.77, 8.02, 3.07, 2.22, 5.76, 11.6, 6.62, 7.07, 12.5,
        1.65, 10.5, 3.67, 7.62, 4.94, 5.39, 3.64, 4.62, 8.88, 6.75, 5.77, 6.38,
        10.3, 5.74,
    ]
    disc = [4, 2, 1, 5, 1, 2, 4, 5, 3, 4, 4, 0, 1, 5, 1, 3, 3, 4, 0, 2, 3, 1, 3, 2, 1, 2, 4, 2, 0, 2]
    cont_series = make_interval_table(cont, intervals=5)
    values, counts, rel, cum = g.discrete_series(disc)
    disc_rows = [[str(v), str(c), fmt(r), fmt(k)] for v, c, r, k in zip(values, counts, rel, cum)]
    disc_table = g.md_table(
        ["Варианта", "Частота", "Отн. частота", "Накопл. отн. частота"],
        disc_rows,
    )

    code_text = f"""
import numpy as np
import matplotlib.pyplot as plt

continuous = {cont}
discrete = {disc}

def discrete_series(values):
    values = np.asarray(values)
    uniq, counts = np.unique(values, return_counts=True)
    rel = counts / counts.sum()
    cum = np.cumsum(rel)
    return uniq, counts, rel, cum

def interval_series(values, intervals=5):
    values = np.asarray(values, dtype=float)
    xmin = values.min()
    xmax = values.max()
    delta = (xmax - xmin) / intervals
    edges = [xmin + i * delta for i in range(intervals + 1)]
    counts = []
    rel = []
    cum = []
    total = 0
    for i in range(intervals):
        left, right = edges[i], edges[i + 1]
        if i < intervals - 1:
            count = int(((values >= left) & (values < right)).sum())
        else:
            count = int(((values >= left) & (values <= right)).sum())
        total += count
        counts.append(count)
        rel.append(count / len(values))
        cum.append(total / len(values))
    mids = [(edges[i] + edges[i + 1]) / 2 for i in range(intervals)]
    return edges, mids, counts, rel, cum

def plot_hist_polygon_ogive(edges, mids, counts, rel, cum, title):
    # Гистограмма частот
    plt.figure()
    plt.bar(mids, counts, width=np.diff(edges), edgecolor='black')
    plt.title(title + ' - гистограмма')
    plt.xlabel('x')
    plt.ylabel('Частота')
    plt.grid(axis='y', alpha=0.25)
    plt.show()

    # Полигон частот
    plt.figure()
    plt.plot(mids, counts, marker='o')
    plt.title(title + ' - полигон')
    plt.xlabel('x')
    plt.ylabel('Частота')
    plt.grid(alpha=0.25)
    plt.show()

    # Огива
    plt.figure()
    plt.step(edges[1:], cum, where='post')
    plt.scatter(edges[1:], cum)
    plt.title(title + ' - огива')
    plt.xlabel('x')
    plt.ylabel('Накопленная относительная частота')
    plt.grid(alpha=0.25)
    plt.show()

edges_c, mids_c, counts_c, rel_c, cum_c = interval_series(continuous, intervals=5)
values_d, counts_d, rel_d, cum_d = discrete_series(discrete)
print('Непрерывная выборка:', counts_c)
print('Дискретная выборка:', counts_d.tolist())
"""

    manual_cont = f"""
Для непрерывной выборки:

$$
n = 26, \\quad x_{{min}} = {fmt(cont_series['xmin'])}, \\quad x_{{max}} = {fmt(cont_series['xmax'])}
$$

$$
R = {fmt(cont_series['r'])}, \\qquad \\Delta = {fmt(cont_series['delta'])}
$$

{cont_series['table']}
"""

    manual_disc = f"""
Для дискретной выборки:

{disc_table}

$$
\\omega_i = \\frac{{n_i}}{{n}}
$$

Накопленные частоты:

$$
0.1333,\\ 0.3333,\\ 0.7333,\\ 0.8000,\\ 0.9667,\\ 1.0000
$$
"""

    return f"""
# Лабораторная работа №2
## Графическое представление выборки

**Вариант:** `16 % 10 = 6`

### 1. Теоретическая часть

Для визуального анализа выборки используют:

* **гистограмму**;
* **полигон частот**;
* **огиву**.

Для непрерывной выборки данные группируют по интервалам. Для дискретной - считают частоты отдельных значений.

### 2. Алгоритм решения

1. Считать исходные данные.
2. Для непрерывной выборки разбить диапазон на интервалы и найти частоты.
3. Для дискретной выборки подсчитать частоты значений.
4. Найти относительные и накопленные частоты.
5. Построить гистограмму, полигон и огиву.

### 3. Реализация на Python

{block(code_text, "python")}

### 4. Разбор кода

* `discrete_series(...)` - подсчёт частот для дискретной выборки.
* `interval_series(...)` - построение интервального ряда для непрерывной выборки.
* `counts` - абсолютные частоты.
* `rel` - относительные частоты.
* `cum` - накопленные частоты.
* `plt.bar(...)`, `plt.plot(...)`, `plt.step(...)` - построение графиков.

### 5. Пример выполнения

#### 5.1 Непрерывная выборка

{manual_cont}

#### 5.2 Дискретная выборка

{manual_disc}

### 6. Итог

В лабораторной работе построены все основные графические представления выборки для непрерывного и дискретного случая.
"""


def lab4_section() -> str:
    sample = [16.3, 20.6, 19.4, 18.7, 16.3, 18.7, 19.3, 18.8, 21.8, 23.2, 22.7, 17.4, 21.8, 18.8, 20.2, 19.3, 19.4, 18.4, 19.3, 18.1, 19.4, 19.7, 21.8, 18.8]
    stats = g.sample_stats(sample)
    ci_half_approx = 2 * stats["stderr"]
    ci_approx = (stats["mean"] - ci_half_approx, stats["mean"] + ci_half_approx)

    table = g.md_table(
        ["Показатель", "Значение"],
        [
            ["Объём выборки", str(stats["n"])],
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
            ["95% ДИ (прибл.)", f"[{fmt(ci_approx[0])}; {fmt(ci_approx[1])}]"],
        ],
    )

    code_text = f"""
import numpy as np
import statistics as st

sample = {sample}

data = np.asarray(sample, dtype=float)
n = data.size
mean = data.mean()                       # Среднее
median = float(np.median(data))          # Медиана
modes = st.multimode(data.tolist())       # Все моды
variance = float(data.var(ddof=1))        # Исправленная дисперсия
std_dev = float(data.std(ddof=1))         # Исправленное СКО
stderr = std_dev / np.sqrt(n)             # Стандартная ошибка
data_min = float(data.min())              # Минимум
data_max = float(data.max())              # Максимум
data_range = data_max - data_min          # Размах
data_sum = float(data.sum())              # Сумма

print('Среднее:', mean)
print('Медиана:', median)
print('Моды:', modes)
print('Стандартное отклонение:', std_dev)
print('Дисперсия:', variance)
print('Асимметрия и эксцесс можно вычислить отдельно по формулам')
print('Минимум:', data_min)
print('Максимум:', data_max)
print('Размах:', data_range)
print('Сумма:', data_sum)
print('Стандартная ошибка:', stderr)
"""

    manual = f"""
Дано:

$$
n = 24, \\qquad \\sum x_i = {fmt(stats['sum'])}
$$

Среднее:

$$
\\bar x = \\frac{{\\sum x_i}}{{n}} = \\frac{{{fmt(stats['sum'])}}}{{24}} = {fmt(stats['mean'])}
$$

Исправленная дисперсия:

$$
S^2 = \\frac{{1}}{{n-1}} \\sum (x_i - \\bar x)^2 = {fmt(stats['variance'])}
$$

Сумма квадратов отклонений:

$$
\\sum (x_i - \\bar x)^2 = {fmt(np.sum((np.asarray(sample) - stats['mean'])**2))}
$$

Стандартное отклонение:

$$
S = \\sqrt{{S^2}} = {fmt(stats['std_dev'])}
$$

Стандартная ошибка:

$$
m = \\frac{{S}}{{\\sqrt{{n}}}} = {fmt(stats['stderr'])}
$$

Приближённый 95\\% доверительный интервал:

$$
\\bar x \\pm 2m = {fmt(stats['mean'])} \\pm {fmt(ci_half_approx)}
$$
"""

    return f"""
# Лабораторная работа №4
## Расчёт числовых характеристик выборки

**Вариант:** `16 % 10 = 6`

### 1. Теоретическая часть

Для описания выборки используют:

$$
\\bar x = \\frac{{1}}{{n}} \\sum_{{i=1}}^n x_i,
\\qquad
S^2 = \\frac{{1}}{{n-1}} \\sum_{{i=1}}^n (x_i - \\bar x)^2,
\\qquad
S = \\sqrt{{S^2}}
$$

Также вычисляют медиану, моду, размах, асимметрию и эксцесс.

### 2. Алгоритм решения

1. Записать данные выборки.
2. Найти среднее, медиану, моду, дисперсию и стандартное отклонение.
3. Найти асимметрию, эксцесс, минимум, максимум, размах и сумму.
4. Проверить результаты вручную.

### 3. Реализация на Python

{block(code_text, "python")}

### 4. Разбор кода

* `mean` - среднее арифметическое.
* `median` - медиана.
* `modes` - список всех мод.
* `variance` - исправленная дисперсия с делением на $n-1$.
* `std_dev` - стандартное отклонение.
* `stderr` - стандартная ошибка среднего.
* `data_range` - размах $R = x_{{max}} - x_{{min}}$.

### 5. Пример выполнения

Вывод программы:

{table}

### 6. Пошаговый разбор вычислений

{manual}

### 7. Итог

Для варианта 6 получен полный набор числовых характеристик выборки. По значениям асимметрии и эксцесса можно сделать вывод о форме распределения.
"""


def lab5_section() -> str:
    sample = [18.7, 16.3, 18.4, 19.3, 18.8, 19.4, 18.7, 18.5, 20.6, 20.6, 19.4, 20.7, 16.3, 18.4, 19.3, 18.8, 18.4, 19.3, 19.3, 19.9, 23.1, 18.8, 17.4, 21.6, 19.1, 18.4, 19.3]
    stats = g.sample_stats(sample)
    ci_half_approx = 2 * stats["stderr"]
    ci_approx = (stats["mean"] - ci_half_approx, stats["mean"] + ci_half_approx)
    table = g.md_table(
        ["Показатель", "Значение"],
        [
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
            ["Счёт", str(stats["n"])],
            ["95% ДИ (прибл.)", f"[{fmt(ci_approx[0])}; {fmt(ci_approx[1])}]"],
        ],
    )

    code_text = f"""
import numpy as np
import statistics as st

sample = {sample}

data = np.asarray(sample, dtype=float)
n = data.size
mean = data.mean()
median = float(np.median(data))
modes = st.multimode(data.tolist())
variance = float(data.var(ddof=1))
std_dev = float(data.std(ddof=1))
stderr = std_dev / np.sqrt(n)
data_min = float(data.min())
data_max = float(data.max())
data_range = data_max - data_min
data_sum = float(data.sum())
print('Среднее:', mean)
print('Стандартная ошибка:', stderr)
print('Медиана:', median)
print('Мода:', modes)
print('Стандартное отклонение:', std_dev)
print('Дисперсия:', variance)
print('Эксцесс и асимметрия вычисляются по формулам')
print('Интервал:', data_range)
print('Минимум:', data_min)
print('Максимум:', data_max)
print('Сумма:', data_sum)
print('Счёт:', n)
"""

    manual = f"""
Дано:

$$
n = 27, \\qquad \\sum x_i = {fmt(stats['sum'])}
$$

Среднее:

$$
\\bar x = \\frac{{\\sum x_i}}{{n}} = \\frac{{{fmt(stats['sum'])}}}{{27}} = {fmt(stats['mean'])}
$$

Стандартная ошибка:

$$
m = \\frac{{S}}{{\\sqrt{{n}}}} = {fmt(stats['stderr'])}
$$

Приближённый 95\\% доверительный интервал:

$$
\\bar x \\pm 2m = {fmt(stats['mean'])} \\pm {fmt(ci_half_approx)}
$$
"""

    return f"""
# Лабораторная работа №5
## Описательная статистика выборки стандартными средствами ЭТ MS Excel

**Вариант:** `16 % 10 = 6`

### 1. Теоретическая часть

Пакет анализа Excel позволяет быстро получить основные числовые характеристики выборки:

$$
\\bar x,\\ Me,\\ Mo,\\ S,\\ S^2,\\ As,\\ Ek,\\ x_{{min}},\\ x_{{max}},\\ R,\\ n
$$

Эти величины помогают оценить центр распределения, разброс и форму выборки.

### 2. Алгоритм решения

1. Ввести исходные данные.
2. Найти основные статистики.
3. Сравнить результаты с формулами.
4. Сделать вывод о форме распределения.

### 3. Реализация на Python

{block(code_text, "python")}

### 4. Разбор кода

* `mean` - выборочное среднее.
* `stderr` - стандартная ошибка среднего.
* `median` - медиана.
* `modes` - мода.
* `variance` - исправленная дисперсия.
* `std_dev` - стандартное отклонение.
* `data_range` - размах выборки.

### 5. Пример выполнения

Вывод программы:

{table}

### 6. Пошаговый разбор вычислений

{manual}

### 7. Итог

Для варианта 6 выполнена описательная статистика выборки и получен полный набор числовых характеристик, достаточный для анализа центра, разброса и формы распределения.
"""


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    report = f"""
{lab1_section()}

---

{lab2_section()}

---

{lab4_section()}

---

{lab5_section()}
"""
    REPORT_PATH.write_text(report.strip() + "\n", encoding="utf-8")
    print(f"written: {REPORT_PATH}")


if __name__ == "__main__":
    main()
