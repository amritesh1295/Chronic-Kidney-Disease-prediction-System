"""Synthetic CKD-like data - FOR TESTS ONLY (never use it to report model performance).

Reproduces the quirks of the real UCI/Kaggle file: stray tabs/spaces, '?' placeholders,
NaNs scattered across columns, an `id` column and tab-padded class labels.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def make_ckd_like(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ckd = rng.random(n) < 0.625
    pick = lambda a, b: np.where(ckd, a, b)  # noqa: E731
    flag = lambda p_ckd, p_ok, yes, no: np.where(rng.random(n) < pick(p_ckd, p_ok), yes, no)  # noqa: E731

    hemo = np.clip(pick(rng.normal(10.5, 2.3, n), rng.normal(15, 1.2, n)), 3, 18)
    df = pd.DataFrame({
        "id": np.arange(n),
        "age": np.clip(pick(rng.normal(57, 16, n), rng.normal(46, 15, n)), 2, 90).round(),
        "bp": np.clip(pick(rng.normal(80, 14, n), rng.normal(72, 8, n)), 50, 180).round(),
        "sg": np.where(ckd, rng.choice([1.005, 1.010, 1.015, 1.020], n, p=[.2, .35, .3, .15]),
                       rng.choice([1.020, 1.025], n, p=[.4, .6])),
        "al": np.where(ckd, rng.choice([0, 1, 2, 3, 4], n, p=[.15, .25, .25, .25, .1]), rng.choice([0, 1], n, p=[.95, .05])),
        "su": np.where(ckd, rng.choice([0, 1, 2, 3], n, p=[.7, .15, .1, .05]), 0),
        "rbc": flag(.5, .02, "abnormal", "normal"),
        "pc": flag(.5, .03, "abnormal", "normal"),
        "pcc": flag(.3, .01, "present", "notpresent"),
        "ba": flag(.15, .01, "present", "notpresent"),
        "bgr": np.clip(pick(rng.normal(150, 65, n), rng.normal(105, 15, n)), 60, 490).round(),
        "bu": np.clip(pick(rng.normal(75, 45, n), rng.normal(30, 8, n)), 10, 390).round(1),
        "sc": np.clip(pick(rng.lognormal(1.0, 0.8, n), rng.normal(0.9, 0.2, n)), 0.4, 76).round(1),
        "sod": np.clip(pick(rng.normal(134, 8, n), rng.normal(141, 3, n)), 104, 163).round(),
        "pot": np.clip(pick(rng.normal(4.9, 1.2, n), rng.normal(4.2, 0.4, n)), 2.5, 47).round(1),
        "hemo": hemo.round(1),
        "pcv": np.clip(hemo * 3 + rng.normal(0, 2, n), 9, 54).round(),
        "wc": np.clip(pick(rng.normal(8800, 3000, n), rng.normal(7500, 1500, n)), 2200, 26400).round(-2),
        "rc": np.clip(hemo / 3 + rng.normal(0, .4, n), 2.1, 8).round(1),
        "htn": flag(.7, .02, "yes", "no"),
        "dm": flag(.55, .02, "yes", "no"),
        "cad": flag(.15, 0, "yes", "no"),
        "appet": flag(.35, .02, "poor", "good"),
        "pe": flag(.4, .01, "yes", "no"),
        "ane": flag(.3, .01, "yes", "no"),
        "classification": np.where(ckd, "ckd", "notckd"),
    }).astype(object)

    # ---- raw-file quirks ----------------------------------------------------
    for col in df.columns.drop(["id", "classification"]):                 # missing values
        df.loc[rng.random(n) < rng.uniform(0.0, 0.25), col] = np.nan
    for col in ["pcv", "wc", "rc"]:                                        # numeric columns stored as text with "?" / tabs
        as_text = df[col].map(lambda v: v if pd.isna(v) else f"\t{v:g}")
        as_text[rng.random(n) < 0.08] = "\t?"
        df[col] = as_text
    for col in ["dm", "cad"]:                                              # tab / space padded categories
        df[col] = df[col].map(lambda v: v if pd.isna(v) or rng.random() > 0.2 else rng.choice(["\t", " "]) + v)
    df["classification"] = df["classification"].map(lambda v: v + "\t" if rng.random() < 0.05 else v)
    return df
