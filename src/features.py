"""Schema, cleaning, feature engineering and preprocessing for CKD prediction.

Shared by the training notebook AND the Flask app so that training and serving
apply exactly the same transformations.

Dataset: UCI / Kaggle "Chronic Kidney Disease" (400 patients, 24 clinical fields + class).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "classification"   # "ckd" / "notckd"
ID_COL = "id"
YES_NO = ["yes", "no"]
NORMAL = ["normal", "abnormal"]
PRESENT = ["present", "notpresent"]


def _num(name, label, group, unit, lo, hi, options=None):
    return dict(name=name, label=label, group=group, unit=unit, type="number", min=lo, max=hi, options=options)


def _cat(name, label, group, choices):
    return dict(name=name, label=label, group=group, unit="", type="category", choices=choices)


G1, G2, G3, G4 = "Patient & vitals", "Urinalysis", "Blood tests", "History & symptoms"

# Single source of truth: drives cleaning, validation, the HTML form and /api/fields.
FIELDS: list[dict] = [
    _num("age", "Age", G1, "years", 1, 120),
    _num("bp", "Blood pressure (diastolic)", G1, "mmHg", 20, 250),
    _num("sg", "Urine specific gravity", G2, "", 1.0, 1.05, [1.005, 1.010, 1.015, 1.020, 1.025]),
    _num("al", "Albumin in urine (0-5)", G2, "", 0, 5, [0, 1, 2, 3, 4, 5]),
    _num("su", "Sugar in urine (0-5)", G2, "", 0, 5, [0, 1, 2, 3, 4, 5]),
    _cat("rbc", "Red blood cells (urine)", G2, NORMAL),
    _cat("pc", "Pus cells (urine)", G2, NORMAL),
    _cat("pcc", "Pus cell clumps", G2, PRESENT),
    _cat("ba", "Bacteria", G2, PRESENT),
    _num("bgr", "Blood glucose, random", G3, "mg/dL", 20, 800),
    _num("bu", "Blood urea", G3, "mg/dL", 1, 400),
    _num("sc", "Serum creatinine", G3, "mg/dL", 0.1, 80),
    _num("sod", "Sodium", G3, "mEq/L", 90, 180),
    _num("pot", "Potassium", G3, "mEq/L", 1, 15),
    _num("hemo", "Hemoglobin", G3, "g/dL", 2, 20),
    _num("pcv", "Packed cell volume", G3, "%", 5, 70),
    _num("wc", "White blood cell count", G3, "cells/cumm", 500, 50000),
    _num("rc", "Red blood cell count", G3, "millions/cmm", 1, 9),
    _cat("htn", "Hypertension", G4, YES_NO),
    _cat("dm", "Diabetes mellitus", G4, YES_NO),
    _cat("cad", "Coronary artery disease", G4, YES_NO),
    _cat("appet", "Appetite", G4, ["good", "poor"]),
    _cat("pe", "Pedal edema", G4, YES_NO),
    _cat("ane", "Anemia", G4, YES_NO),
]
FIELD_BY_NAME = {f["name"]: f for f in FIELDS}
NUMERIC_RAW = [f["name"] for f in FIELDS if f["type"] == "number"]
CATEGORICAL_RAW = [f["name"] for f in FIELDS if f["type"] == "category"]
RAW_FEATURES = [f["name"] for f in FIELDS]

ENGINEERED_NUMERIC = [
    "UreaCreatinineRatio", "LowHemoglobin", "HighCreatinine", "Proteinuria",
    "ComorbidityCount", "SymptomCount", "AbnormalUrinalysisCount",
]
NUMERIC_FEATURES = NUMERIC_RAW + ENGINEERED_NUMERIC
CATEGORICAL_FEATURES = CATEGORICAL_RAW
MODEL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

MISSING_TOKENS = {"?", "", "nan", "none", "null", "<na>"}


# ---- Cleaning ---------------------------------------------------------------
def _text(series: pd.Series) -> pd.Series:
    """Strip tabs/spaces, lower-case and turn '?'/blank into NaN (the raw file has all of these)."""
    s = series.astype("string").str.strip().str.lower()
    s = s.where(~s.isin(MISSING_TOKENS), pd.NA)
    return s.astype(object).where(s.notna(), np.nan)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Raw CSV rows -> typed frame with exactly the 24 clinical columns (NaN = not measured)."""
    df = df.copy()
    df.columns = df.columns.str.strip()
    out = pd.DataFrame(index=df.index)
    for f in FIELDS:
        text = _text(df[f["name"]]) if f["name"] in df else pd.Series(np.nan, index=df.index, dtype=object)
        out[f["name"]] = pd.to_numeric(text, errors="coerce") if f["type"] == "number" else text
    return out


def encode_target(y: pd.Series) -> pd.Series:
    """'ckd' -> 1 (positive class), 'notckd' -> 0. Raises on unexpected labels."""
    mapped = _text(y).map({"ckd": 1, "notckd": 0})
    if mapped.isna().any():
        raise ValueError(f"Unexpected labels in target: {sorted(set(y[mapped.isna()].astype(str)))}")
    return mapped.astype(int)


# ---- Feature engineering ----------------------------------------------------
def _flag(condition: pd.Series, source: pd.Series) -> pd.Series:
    """0/1 flag that stays NaN where the underlying measurement is missing."""
    return condition.astype(float).where(source.notna())


def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically motivated features (thresholds are illustrative reference values)."""
    df = df.copy()
    df["UreaCreatinineRatio"] = df["bu"] / df["sc"].where(df["sc"] > 0)
    df["LowHemoglobin"] = _flag(df["hemo"] < 12, df["hemo"])
    df["HighCreatinine"] = _flag(df["sc"] > 1.2, df["sc"])
    df["Proteinuria"] = _flag(df["al"] >= 1, df["al"])
    df["ComorbidityCount"] = sum((df[c] == "yes").astype(int) for c in ["htn", "dm", "cad"])
    df["SymptomCount"] = (df["pe"] == "yes").astype(int) + (df["ane"] == "yes").astype(int) + (df["appet"] == "poor").astype(int)
    df["AbnormalUrinalysisCount"] = (
        (df["rbc"] == "abnormal").astype(int) + (df["pc"] == "abnormal").astype(int)
        + (df["pcc"] == "present").astype(int) + (df["ba"] == "present").astype(int)
    )
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """clean -> engineer -> model columns."""
    return engineer(clean(df))[MODEL_FEATURES]


# ---- Preprocessing ----------------------------------------------------------
def build_preprocessor() -> ColumnTransformer:
    """Impute (median / most-frequent), scale numerics, one-hot categoricals -> dense array.

    Imputation lives INSIDE the pipeline so it is fitted on training data only
    and so the API can accept patients with missing lab values.
    """
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent", keep_empty_features=True)),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", numeric, NUMERIC_FEATURES), ("cat", categorical, CATEGORICAL_FEATURES)],
        sparse_threshold=0.0,
    )


def risk_level(probability: float) -> str:
    if probability >= 0.70:
        return "High"
    if probability >= 0.30:
        return "Moderate"
    return "Low"
