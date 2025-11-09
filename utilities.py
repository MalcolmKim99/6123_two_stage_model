import yfinance as yf
from fredapi import Fred

# Series
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import PercentFormatter, MultipleLocator

from pathlib import Path
from dotenv import load_dotenv
import os



# Exploratory Data Analysis of Input Variables for the Financial Turbulence Model 
# Other pd.DatarFrame data is suitable too
def exploratory_data_analysis_pd_dataframe(data: pd.DataFrame):
    print("The shape:")
    print(data.shape)
    print("\nThe number of na data:")
    print(data.isna().sum())
    print("\nThe Info of the input dataframe:")
    print(data.info())
    print("\nThe numberic profile of the input data:")
    print(data.describe())
    print("\nThe skewness of the input dataframe:")
    print(data.skew())
    print("\nThe kurtosis of the input dataframe:")
    print(data.kurtosis())

# Plot the max drawdown for sp500 in the stage 1 before the k means clustering 
def plot_maxdrawdown_series(
    s,
    title: str = 'US Equity Drawdown',
    ymin: float = -0.45,
    ymax: float = 0.0,
    ystep: float = 0.05,
    year_freq: int = 1,                 # 1 = every year, 2 = every 2 years, etc.
    # date_fmt: str = '1/1/%Y',           # fallback to '01/01/%Y' on Windows if needed
    figsize=(14, 5),
    rotate_labels: int = 90
):
    """
    Plot a drawdown series with yearly x-axis ticks and percent y-axis.

    Parameters
    ----------
    s : pd.Series
        Time series (index must be datetime-like; values in decimal, e.g., -0.25 for -25%).
    title : str
        Plot title.
    ymin, ymax : float
        Y-axis limits.
    ystep : float
        Major tick step on Y axis (in decimal, e.g., 0.05 = 5%).
    year_freq : int
        Year tick spacing (1 = yearly, 2 = every two years, etc.).
    date_fmt : str
        Date label format for major ticks (e.g., '1/1/%Y' or '01/01/%Y').
    figsize : tuple
        Figure size.
    rotate_labels : int
        X tick label rotation in degrees.

    Returns
    -------
    (fig, ax) : matplotlib Figure and Axes
    """
    # --- ensure datetime index, sorted, and tz-naive ---
    s.index = pd.to_datetime(s.index)
    try:
        s.index = s.index.tz_convert(None)
    except Exception:
        pass
    s = s.sort_index()

    # --- plot ---
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(s.index, s.values, lw=2)

    ax.set_title(title)
    ax.set_xlabel('')
    ax.set_ylabel('')

    # Y-axis as percent
    ax.set_ylim(ymin, ymax)
    ax.yaxis.set_major_locator(MultipleLocator(ystep))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.grid(True, axis='y', alpha=0.35)

    # X-axis: yearly ticks on Jan 1
    ax.set_xlim(s.index.min(), s.index.max())
    ax.xaxis.set_major_locator(mdates.YearLocator(base=year_freq))
    # try:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter(date_fmt))
    # except ValueError:
    #     ax.xaxis.set_major_formatter(mdates.DateFormatter('01/01/%Y'))
    ax.tick_params(axis='x', labelrotation=rotate_labels)

    # Clean frame
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    return fig, ax
