# NewCoach  

## Table of Contents

- [Introduction](#introduction)
- [Dataset Requirements](#dataset_requirement)
- [Installation](#installation)
- [Usage](#usage)
- [Tutorial](#tutorial)
- [License](#license)

## Introduction

newcoach is a Python package for analyzing the effect of mid-season coach changes in football (soccer) teams.
It provides a full pipeline:

Data preprocessing: cleaning match data, computing metrics such as points, expected goal difference (xGD), and Elo difference.

Outlier detection: automatic 1.5×IQR filtering.

Event study variables: constructing pre-/post-event windows around coach changes.

Modeling: unified interface for OLS and GLM regression.

Visualization: mean trajectories before/after coach changes, residual diagnostics.

This package is designed for our Programming Final Project (TU Dortmund), but can be reused on any football dataset with basic match-level info.

## Dataset_requirement

Your dataset must be a CSV (or DataFrame) with the following columns:
| Column                 | Type         | Description                                            | Example                           |
| ---------------------- | ------------ | ------------------------------------------------------ | --------------------------------- |
| `Team`                 | str          | Team name                                              | `"Bayern Munich"`                 |
| `Date`                 | str/datetime | Match date                                             | `"2021-11-06"`                    |
| `Venue`                | str          | `"Home"` or `"Away"`                                   | `"Home"`                          |
| `Result`               | str          | `"W"`, `"D"`, `"L"`                                    | `"W"`                             |
| `xGF`                  | float        | Expected goals for                                     | `1.8`                             |
| `xGA`                  | float        | Expected goals against                                 | `1.2`                             |
| `elo_pre`              | float        | Team Elo rating before match                           | `1850.2`                          |
| `opp_elo_pre`          | float        | Opponent Elo rating before match                       | `1790.4`                          |
| `within_season_change` | int (0/1)    | Marks if this match is the **first under a new coach** | `1` for new coach start, else `0` |
| `SoT_percent_Standard` | float        | Share of shots on target                         | `45` (i.e. 45%)                 |

Note: Without these columns, some preprocessing or modeling functions will not run.

## Installation
In terminal, type:
```console
git clone https://github.com/0xtootoo/Intro_Python_HW.git
cd Intro_Python_HW
pip install -e . (or python -m pip install -e .)
```
In .py/notebook import:
```console
from newcoach import DataPreprocessor, Modeler, Plotter
```
## Usage
Here is a minimal workflow:
```console
import pandas as pd
from newcoach import DataPreprocessor, Modeler, Plotter
from newcoach import load_example
import matplotlib.pyplot as plt

# Example dataset (dataset with proper structure can also be used here)
df = load_example("New_Coach_Effect.csv")

# Preprocessing pipeline
dp = (
    DataPreprocessor(df)
    .add_basic_metrics()
    .add_event_time_variables(window=10)
)
out = dp.flag_outliers_iqr(cols=["xGD", "Diff_Elo"])

df_clean = dp.data.loc[~out.flags.any(axis=1)].copy()

# Modeling and check residual distribution and QQ plot
mod = Modeler(df_clean)
ols_res = mod.fit_ols("xGD ~ relative_time + post + time_post + Venue_home + Diff_Elo + C(Team)-1")

resid_res = ols_res.resid.reset_index(drop=True)
Plotter.plot_residual_hist_and_qq(resid_res, title_prefix="xGD OLS residuals");
plt.tight_layout()
plt.show()
print(ols_res.summary())

# Visualization of average values
Plotter.plot_pre_post_mean(df_clean, value_col="xGD",
                           title="Average xGD Around Coach Change")
plt.tight_layout()
plt.show()
```
## Tutorial
Here provide a Jupyter Notebook tutorial that explains each function step-by-step:

[Tutorial Notebook](tutorial.ipynb)

## License

`newcoach` is distributed under the terms of the [MIT](https://spdx.org/licenses/MIT.html) license.
