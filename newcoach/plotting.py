from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm


class Plotter:
    @staticmethod
    def plot_pre_post_mean(df: pd.DataFrame,
                           value_col: str,
                           rel_col: str = "relative_time",
                           post_col: str = "post",
                           title: Optional[str] = None,
                           ax: Optional[plt.Axes] = None,
                           xlabel: str = "Matches before (-) and after (+) coach change",
                           ylabel: Optional[str] = None):
        """
        draw relative mean trajectories on the relative time axis, and separately draw the average horizontal before and after the coach change.
        """
        d = df.dropna(subset=[rel_col, value_col]).copy()
        d = d.sort_values(rel_col)

        # relative time mean trajectory
        mean_by_rel = d.groupby(rel_col, dropna=False)[value_col].mean().sort_index()

        if ax is None:
            fig, ax = plt.subplots(figsize=(9, 5))

        ax.plot(mean_by_rel.index, mean_by_rel.values, marker="o", label=f"Actual avg {value_col}")

        # vertical line: t=0
        ax.axvline(0, color="red", linestyle="--", linewidth=1.5, label="Coach change")

        # calculate pre/post means + corresponding x ranges (data coordinates)
        pre_mask  = d[post_col] == 0
        post_mask = d[post_col] == 1

        if pre_mask.any():
            pre_mean = d.loc[pre_mask, value_col].mean()
            x_pre_min = d.loc[pre_mask, rel_col].min()
            x_pre_max = 0
            ax.hlines(pre_mean, xmin=x_pre_min, xmax=x_pre_max,
                      linestyles=":", linewidth=2, color="green",
                      label=f"Pre-change avg ({int(x_pre_min)}–0) = {pre_mean:.2f}")

        if post_mask.any():
            post_mean = d.loc[post_mask, value_col].mean()
            x_post_min = 0
            x_post_max = d.loc[post_mask, rel_col].max()
            ax.hlines(post_mean, xmin=x_post_min, xmax=x_post_max,
                      linestyles=":", linewidth=2, color="orange",
                      label=f"Post-change avg (0–{int(x_post_max)}) = {post_mean:.2f}")

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel or value_col.replace("_", " ").title())
        if title:
            ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend()
        return ax


    @staticmethod
    def plot_residual_hist_and_qq(residuals: np.ndarray,
                                  bins: int = 20,
                                  title_prefix: str = "OLS diagnostics"):
        """
        based on the residuals, draw histograms and QQ plots. Follow your process: first OLS diagnostics, then decide whether to use GLM.
        """
        fig1, ax1 = plt.subplots(figsize=(6, 4))
        ax1.hist(residuals, bins=bins, edgecolor="white")
        ax1.set_title(f"{title_prefix}: Residual histogram")
        ax1.set_xlabel("Residual")
        ax1.set_ylabel("Frequency")
        ax1.grid(True)

        fig2, ax2 = plt.subplots(figsize=(6, 4))
        sm.ProbPlot(residuals, fit=True).qqplot(ax=ax2, line="45")
        ax2.set_title(f"{title_prefix}: QQ plot")
        ax2.grid(True)
        return (ax1, ax2)
