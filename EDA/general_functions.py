

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import seaborn as sns
import numpy as np
import statsmodels.api as sm

import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm


def two_sample_ttest_ind_df(n1, n2):
    """
    Calculate the degrees of freedom for a two-sample t-test with equal variances.
    
    Parameters:
    n1 (int): Size of sample 1
    n2 (int): Size of sample 2
    
    Returns:
    int: Degrees of freedom
    """
    return n1 + n2 - 2
    

def welch_df(s1, s2, n1, n2):
    """
    Calculate the degrees of freedom for Welch's t-test.
    
    Parameters:
    s1 (float): Variance of sample 1
    s2 (float): Variance of sample 2
    n1 (int): Size of sample 1
    n2 (int): Size of sample 2
    
    Returns:
    float: Degrees of freedom
    """
    numerator = (s1/n1 + s2/n2) ** 2
    denominator = ((s1/n1) ** 2) / (n1 - 1) + ((s2/n2) ** 2) / (n2 - 1)
    return numerator / denominator


def is_p_drop(statistic_name, stat, p, stat_testing="mean", alpha=0.005):
    print(f"{statistic_name}: {stat:.3f}, p-value: {p:.4f}")
    if p < alpha:
        print(f"Drop H0 --> accept H1")
    else:
        print(f"Fail to drop H0 --> No significant difference among {stat_testing}s.")


def plot_distrubtion(movie_df, name, y, show=True, bins=30):
    plt.figure(figsize=(7, 5))
    sns.histplot(movie_df[y], bins=bins, kde=True, color='grey', edgecolor='black')

    # horizontal lines for mean and median
    # mean_rating = movie_df[y].mean()
    # median_rating = movie_df[y].median()

    # plt.axvline(mean_rating, color='red', linestyle='--', label=f'Mean: {mean_rating:.2f}')
    # plt.axvline(median_rating, color='green', linestyle='--', label=f'Median: {median_rating:.2f}')
    # plt.legend()

    plt.title(name)
    plt.xlabel(y.replace('_', ' ').title())
    plt.ylabel('Frequency')
    plt.grid(True)
    if show:
        plt.show()
        
    return plt

def plot_boxplot(movie_df, x, y, title, xlabel, ylabel, show=True):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x=x, y=y, data=movie_df)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
        
    # color the median lines
    for i, line in enumerate(plt.gca().lines[4::6]):
        line.set_color('red')
        line.set_linewidth(2)
    
    if show:
        plt.show()
    return plt


def graph_cdf(
    data,
    ax=None,
    label=None,
    color=None,
    show=True,
    add_title=True,
    jump_linestyle="--",
    jump_alpha=0.6,
    jump_linewidth=1.0,
    jump_color=None,          # default: match the CDF color
):
    value_counts = data.value_counts().sort_index()
    x = value_counts.index.values
    p_values = value_counts.values / value_counts.values.sum()

    if not np.isclose(p_values.sum(), 1.0):
        raise ValueError(f"The sum of p_values must be 1. Currently, it is {p_values.sum()}")

    cdf = np.cumsum(p_values)

    # step-friendly padding (so the first step starts at 0)
    x_left = x[0]
    x_right = x[-1]
    x_plot = np.concatenate(([x_left], x))
    cdf_plot = np.concatenate(([0.0], cdf))

    if ax is None:
        _, ax = plt.subplots()

    # draw CDF step
    (step_line,) = ax.step(x_plot, cdf_plot, where='post', label=label or 'CDF', color=color)

    # draw jump markers (optional dots)
    ax.plot(x, cdf, 'o', label=None if label else 'Jump points', color=step_line.get_color())

    # dashed vertical jump lines (from previous CDF level up to the jump)
    cdf_prev = np.concatenate(([0.0], cdf[:-1]))
    vcolor = jump_color or step_line.get_color()
    for xi, y0, y1 in zip(x, cdf_prev, cdf):
        ax.vlines(
            xi, y0, y1,
            linestyle=jump_linestyle,
            linewidth=jump_linewidth,
            alpha=jump_alpha,
            color=vcolor,
            label="_nolegend_",     # keep legend clean
        )

    if add_title:
        ax.set_title("CDF for a Discrete Random Variable")
    ax.set_xlabel("Random Variable ($x$)")
    ax.set_ylabel("Cumulative Probability")

    # guide lines at 0 and 1
    ax.hlines(0, x_left, x[0], linestyle='--', color=vcolor, alpha=0.5)
    ax.hlines(1, x[-1], x_right, linestyle='--', color=vcolor, alpha=0.5)

    ax.grid(True, linestyle='--', alpha=0.6)

    if show:
        plt.show()

    return ax, cdf

def plot_absolute_differences(x, cdf1, cdf2, ax=None, show=True):
    diff_cdf = np.abs(np.array(cdf1) - np.array(cdf2))

    if ax is None:
        _, ax = plt.subplots()

    ax.plot(x, diff_cdf, marker='o')
    ax.set_title('Absolute Differences Between Two CDFs')
    ax.set_xlabel('Value')
    ax.set_ylabel('Absolute Difference in CDF')
    ax.grid(True)

    if show:
        plt.show()

    return ax

def map_color_by_group(df, group_col, color_map):
    return df[group_col].map(color_map).values
