from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, Dict, List
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


def _to_datetime(df: pd.DataFrame, date_col: str = "Date") -> pd.DataFrame:
    """make sure the data has the form as datetime64[ns]."""
    if not np.issubdtype(df[date_col].dtype, np.datetime64):
        df = df.copy()
        df[date_col] = pd.to_datetime(df[date_col])
    return df


def _numeric_cols(df: pd.DataFrame, include: Optional[Iterable[str]] = None) -> List[str]:
    """Returns a numeric column (or a numeric column in a user-specified subset).w"""
    if include is None:
        mask = df.select_dtypes(include=[np.number]).columns.tolist()
        return mask
    include = list(include)
    return [c for c in include if pd.api.types.is_numeric_dtype(df[c])]


@dataclass
class OutlierResult:
    """save 1.5*IQR outlier detection results."""
    bounds: Dict[str, Tuple[float, float]]
    flags: pd.DataFrame 


class DataPreprocessor:
    """
    data preprocessing:
    1) points (W/D/L→3/1/0)
    2) xGD = xGF - xGA
    3) Diff_Elo = elo_pre - opp_elo_pre
    4) 1.5*IQR outlier detection
    5) based on the value of windows, set relative_time, post, time_post
    """

    def __init__(self, df: pd.DataFrame,
                 team_col: str = "Team",
                 date_col: str = "Date",
                 venue_col: str = "Venue",
                 change_flag_col: str = "within_season_change",
                 result_col: str = "Result",
                 SoT_col: str = "SoT_percent_Standard",
                 xgf_col: str = "xGF",
                 xga_col: str = "xGA",
                 elo_col: str = "elo_pre",
                 opp_elo_col: str = "opp_elo_pre"):
        self.df = df.copy()
        self.team_col = team_col
        self.date_col = date_col
        self.venue_col = venue_col
        self.change_flag_col = change_flag_col
        self.result_col = result_col
        self.SoT_col = SoT_col
        self.xgf_col = xgf_col
        self.xga_col = xga_col
        self.elo_col = elo_col
        self.opp_elo_col = opp_elo_col

    def add_basic_metrics(self) -> "DataPreprocessor":
        """add points, xGD, Diff_Elo."""
        self.df[self.result_col] = self.df[self.result_col].str.upper().str[0]
        map_points = {"W": 3, "D": 1, "L": 0}
        self.df["points"] = self.df[self.result_col].map(map_points).astype("Int64")

        self.df["xGD"] = self.df[self.xgf_col] - self.df[self.xga_col]
        self.df["Diff_Elo"] = self.df[self.elo_col] - self.df[self.opp_elo_col]
        self.df["Venue_home"] = (self.df[self.venue_col] == "Home").astype(int)
        return self

    def flag_outliers_iqr(self,
                          cols: Optional[Iterable[str]] = None,
                          k: float = 1.5) -> OutlierResult:
        """save 1.5*IQR outlier detection results.
        Returns bounds and boolean flag columns.
        By default, it is applied to all numeric columns (including newly created ones).
        """
        df = self.df
        cols = _numeric_cols(df, include=cols)

        bounds: Dict[str, Tuple[float, float]] = {}
        flags = pd.DataFrame(index=df.index)

        for c in cols:
            q1 = df[c].quantile(0.25)
            q3 = df[c].quantile(0.75)
            iqr = q3 - q1
            low = q1 - k * iqr
            high = q3 + k * iqr
            bounds[c] = (low, high)
            flags[f"{c}__is_outlier"] = (df[c] < low) | (df[c] > high)

        self.outlier_result_ = OutlierResult(bounds=bounds, flags=flags)
        return self.outlier_result_

    def add_event_time_variables(self, window: int = 10) -> "DataPreprocessor":
        """
    Per-event model window construction:
      - For each coaching change, take matches ±window around it
      - relative_time: t=0 = first match after change; negative before, positive after
      - post: 1{relative_time >= 0}
      - time_post: relative_time * post
      - event_global: "Team_eventid" (unique event label)
      - venue_home: 1{Venue == home}
    Note:
      - Same match may appear in multiple windows if multiple changes are close
      - Rows outside any ±window are excluded
        """
        df = _to_datetime(self.df, self.date_col).sort_values([self.team_col, self.date_col]).copy()

        records = []
        # find all coaching change points (within_season_change == 1)
        for idx, row in df[df[self.change_flag_col] == 1].iterrows():
            team = row[self.team_col]
            evt_id = int(df.loc[:idx, self.change_flag_col].sum())  # cumulative coaching changes
            sub = df[df[self.team_col] == team]  # all matches for this team
            pos = sub.index.get_loc(idx)         # position of the change match within the team

            # truncate window:前 WINDOW + 后 WINDOW
            win = sub.iloc[max(0, pos - window):pos + window + 1].copy()

            # relative_time: change point match = 0
            win["relative_time"] = np.arange(-min(window, pos),
                                         min(window, len(sub) - pos - 1) + 1)

            # post / time_post
            win["post"] = (win["relative_time"] >= 0).astype(int)
            win["time_post"] = win["relative_time"] * win["post"]

            # event_global
            win["event_global"] = f"{team}_{evt_id}"

            records.append(win)

        df_out = pd.concat(records, ignore_index=True)

        self.df = df_out
        return self


    @property
    def data(self) -> pd.DataFrame:
        """get the processed DataFrame."""
        return self.df.copy()


class Modeler:
    """
    using unified OLS / GLM fitting (based on statsmodels).
    Example:
        mod = Modeler(df)
        ols_res = mod.fit_ols("points ~ relative_time + post + time_post + Venue_home + Diff_Elo + C(Team)-1")
        glm_res = mod.fit_glm("SoT_percent_Standard ~ relative_time + post + time_post + Venue_home + Diff_Elo + C(Team)-1",
                              family="gaussian")
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def fit_ols(self, formula: str,
                weights: Optional[pd.Series] = None,
                cov_type: str = "HC3"):
        """
        OLS regression, default robust standard error HC3.
        weights: optional weighted OLS (e.g. weighted by number of shots)
        """
        if weights is None:
            model = smf.ols(formula=formula, data=self.df).fit(cov_type=cov_type)
        else:
            model = smf.wls(formula=formula, data=self.df, weights=weights).fit(cov_type=cov_type)
        return model

    def fit_glm(self, formula: str,
                family: str = "gaussian",
                var_weights: Optional[pd.Series] = None,
                cov_type: Optional[str] = None):
        """
        GLM fitting. family options: 'gaussian', 'binomial', 'poisson', 'gamma', etc.
        var_weights: variance weights (e.g. attempts in a shooting model)
        cov_type: optional robust covariance type (e.g. 'HC3'). Defaults to None.
        """
        fam = family.lower()
        if fam == "gaussian":
            fam_obj = sm.families.Gaussian()
        elif fam == "binomial":
            fam_obj = sm.families.Binomial()
        elif fam == "poisson":
            fam_obj = sm.families.Poisson()
        elif fam == "gamma":
            fam_obj = sm.families.Gamma()
        else:
            raise ValueError(f"Unsupported family: {family}")

        model = smf.glm(formula=formula, data=self.df,
                family=fam_obj, var_weights=var_weights)
        if cov_type:
            res = model.fit(cov_type=cov_type)  
        else:
            res = model.fit()
        return res

