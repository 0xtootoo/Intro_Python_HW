# SPDX-FileCopyrightText: 2025-present Yiming Zang <yiming.zang@tu-dortmund.de>
#
# SPDX-License-Identifier: MIT
"""
Project package for analyzing new coach effects.
Provides preprocessing, modeling, and plotting utilities.
"""

from importlib.resources import files
import pandas as pd

from .analysis import DataPreprocessor, Modeler, OutlierResult
from .plotting import Plotter

__all__ = [
    "DataPreprocessor", "Modeler", "OutlierResult",
    "Plotter",
    "load_example",
]

def load_example(name: str = "New_Coach_Data.csv") -> pd.DataFrame:
    """Load packaged example dataset located in newcoach/Dataset/."""
    path = files("newcoach") / "Dataset" / name
    return pd.read_csv(path)
