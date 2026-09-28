"""Dataset loaders.

Two clinical tables with very different missingness problems:

* Horse colic (UCI): honest missingness, cells marked with ``?``. Roughly a
  quarter of the informative cells are absent, concentrated in the invasive
  measurements (nasogastric reflux pH, abdominocentesis).
* Pima diabetes (UCI): disguised missingness. The file contains no NA marker
  at all, but zeros in glucose, blood pressure, skin fold, insulin and BMI
  are physiologically impossible and are really unrecorded values.

Loaders parse with polars and hand numpy/pandas objects to scikit-learn.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

COLIC_COLUMNS = [
    "surgery",
    "age",
    "hospital_number",
    "rectal_temp",
    "pulse",
    "respiratory_rate",
    "temp_extremities",
    "peripheral_pulse",
    "mucous_membranes",
    "capillary_refill",
    "pain",
    "peristalsis",
    "abdominal_distension",
    "nasogastric_tube",
    "nasogastric_reflux",
    "nasogastric_reflux_ph",
    "rectal_exam_feces",
    "abdomen",
    "packed_cell_volume",
    "total_protein",
    "abdomino_appearance",
    "abdomino_protein",
    "outcome",
    "surgical_lesion",
    "lesion_1",
    "lesion_2",
    "lesion_3",
    "cp_data",
]

COLIC_NUMERIC = [
    "rectal_temp",
    "pulse",
    "respiratory_rate",
    "nasogastric_reflux_ph",
    "packed_cell_volume",
    "total_protein",
    "abdomino_protein",
]

COLIC_CATEGORICAL = [
    "surgery",
    "age",
    "temp_extremities",
    "peripheral_pulse",
    "mucous_membranes",
    "capillary_refill",
    "pain",
    "peristalsis",
    "abdominal_distension",
    "nasogastric_tube",
    "nasogastric_reflux",
    "rectal_exam_feces",
    "abdomen",
    "abdomino_appearance",
]

# Columns that never carry information about the horse itself, or that are
# assigned after the outcome is known (lesion codes come from surgery or
# autopsy; surgical_lesion is a retrospective label). Keeping them would leak.
COLIC_DROP = ["hospital_number", "surgical_lesion", "lesion_1", "lesion_2", "lesion_3", "cp_data"]

PIMA_COLUMNS = [
    "pregnancies",
    "glucose",
    "blood_pressure",
    "skin_thickness",
    "insulin",
    "bmi",
    "pedigree",
    "age",
    "diabetes",
]

# Zero is a physiologically impossible reading for these five measurements.
PIMA_DISGUISED_ZERO = ["glucose", "blood_pressure", "skin_thickness", "insulin", "bmi"]


@dataclass
class Dataset:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    numeric: list[str]
    categorical: list[str] = field(default_factory=list)

    @property
    def n_rows(self) -> int:
        return len(self.X)

    @property
    def complete_rows(self) -> int:
        return int(self.X.notna().all(axis=1).sum())


def load_colic(path: Path | None = None) -> Dataset:
    """Horse colic with outcome collapsed to survived (1) vs died/euthanised (0)."""
    path = path or DATA_DIR / "horse-colic.csv"
    df = pl.read_csv(
        path,
        has_header=False,
        new_columns=COLIC_COLUMNS,
        null_values=["?"],
        infer_schema_length=0,
    )
    df = df.with_columns(
        [pl.col(c).cast(pl.Float64, strict=False) for c in COLIC_NUMERIC + ["outcome"]]
    )
    df = df.filter(pl.col("outcome").is_not_null())
    y = (df["outcome"] == 1).cast(pl.Int8).to_numpy()
    keep = [c for c in COLIC_COLUMNS if c not in COLIC_DROP and c != "outcome"]
    pdf = df.select(keep).to_pandas()
    for c in COLIC_NUMERIC:
        pdf[c] = pd.to_numeric(pdf[c], errors="coerce")
    for c in COLIC_CATEGORICAL:
        pdf[c] = pdf[c].astype("object")
    return Dataset("colic", pdf, y, numeric=COLIC_NUMERIC, categorical=COLIC_CATEGORICAL)


def load_pima(path: Path | None = None, repair_zeros: bool = True) -> Dataset:
    """Pima diabetes. With ``repair_zeros`` the impossible zeros become NaN."""
    path = path or DATA_DIR / "pima-indians-diabetes.csv"
    df = pl.read_csv(path, has_header=False, new_columns=PIMA_COLUMNS)
    if repair_zeros:
        df = df.with_columns(
            [
                pl.when(pl.col(c) == 0).then(None).otherwise(pl.col(c)).alias(c)
                for c in PIMA_DISGUISED_ZERO
            ]
        )
    y = df["diabetes"].to_numpy().astype(np.int8)
    pdf = df.drop("diabetes").to_pandas().astype(float)
    name = "pima" if repair_zeros else "pima_raw"
    return Dataset(name, pdf, y, numeric=[c for c in PIMA_COLUMNS if c != "diabetes"])


def load_breast_cancer_complete() -> Dataset:
    """A fully observed table, used as ground truth for the amputation study."""
    from sklearn.datasets import load_breast_cancer

    bunch = load_breast_cancer(as_frame=True)
    X = bunch.data.copy()
    X.columns = [c.replace(" ", "_") for c in X.columns]
    return Dataset("breast_cancer", X, bunch.target.to_numpy(), numeric=list(X.columns))
