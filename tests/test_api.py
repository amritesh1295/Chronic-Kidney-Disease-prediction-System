"""Offline tests: trains a tiny Random Forest on synthetic data; no MongoDB / TensorFlow needed.

Run:  python -m unittest discover -s tests -t . -v      (or: pytest)
"""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from app import create_app
from predictor import CKDPredictor, ValidationError, validate_patient
from src import features as ft
from tests.synthetic import make_ckd_like

CKD_PATIENT = {
    "age": 62, "bp": 90, "sg": 1.010, "al": 3, "su": 1, "rbc": "abnormal", "pc": "abnormal", "pcc": "present",
    "ba": "notpresent", "bgr": 180, "bu": 80, "sc": 4.5, "sod": 128, "pot": 5.6, "hemo": 8.5, "pcv": 26,
    "wc": 9800, "rc": 3.1, "htn": "yes", "dm": "yes", "cad": "no", "appet": "poor", "pe": "yes", "ane": "yes",
}


class FakeStore:
    """In-memory stand-in for MongoStore."""
    def __init__(self):
        self.saved = []

    def save_prediction(self, patient_id, profile, result, model_version=""):
        self.saved.append((patient_id, profile, result))
        return f"pred-{len(self.saved)}"

    def recent_predictions(self, limit=20, patient_id=None, risk_level=None):
        return [{"patient_id": p, **r, "created_at": "2026-01-01T00:00:00"} for p, _, r in self.saved][:limit]

    def get_patient(self, pid):
        return next(({"patient_id": p, "profile": pr} for p, pr, _ in self.saved if p == pid), None)

    def stats(self):
        return {"total_predictions": len(self.saved), "by_risk_level": {}}

    def ping(self):
        return True


class FeatureTests(unittest.TestCase):
    def test_clean_handles_messy_raw_file(self):
        raw = make_ckd_like(400, seed=1)
        cleaned = ft.clean(raw)
        self.assertEqual(list(cleaned.columns), ft.RAW_FEATURES)
        for col in ft.NUMERIC_RAW:
            self.assertTrue(np.issubdtype(cleaned[col].dtype, np.number), col)
        for col in ft.CATEGORICAL_RAW:
            self.assertTrue(set(cleaned[col].dropna()) <= set(ft.FIELD_BY_NAME[col]["choices"]), col)
        self.assertEqual(set(ft.encode_target(raw[ft.TARGET])), {0, 1})

    def test_encode_target_rejects_unknown_labels(self):
        import pandas as pd
        with self.assertRaises(ValueError):
            ft.encode_target(pd.Series(["ckd", "maybe"]))

    def test_engineered_flags_stay_nan_when_measurement_missing(self):
        import pandas as pd
        row = pd.DataFrame([{**CKD_PATIENT, "hemo": np.nan, "sc": np.nan}])
        eng = ft.engineer(ft.clean(row))
        self.assertTrue(np.isnan(eng.loc[0, "LowHemoglobin"]) and np.isnan(eng.loc[0, "HighCreatinine"]))
        self.assertTrue(np.isnan(eng.loc[0, "UreaCreatinineRatio"]))
        self.assertEqual(eng.loc[0, "ComorbidityCount"], 2)

    def test_pipeline_output_has_no_nans(self):
        raw = make_ckd_like(300, seed=2)
        X = ft.prepare_features(raw)
        out = ft.build_preprocessor().fit_transform(X)
        self.assertFalse(np.isnan(out).any())


class ValidationTests(unittest.TestCase):
    def test_normalises_case_aliases_and_counts_fields(self):
        pid, rec, n = validate_patient({**CKD_PATIENT, "htn": "YES", "dm": True, "pcc": "Not Present", "patientID": " P1 "})
        self.assertEqual((pid, n), ("P1", 24))
        self.assertEqual((rec["htn"], rec["dm"], rec["pcc"]), ("yes", "yes", "notpresent"))

    def test_blank_and_question_mark_count_as_missing(self):
        _, rec, n = validate_patient({**CKD_PATIENT, "sc": "", "hemo": "?", "pcv": None})
        self.assertEqual(n, 21)
        self.assertTrue(all(np.isnan(rec[k]) for k in ("sc", "hemo", "pcv")))

    def test_reports_all_bad_fields(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_patient({**CKD_PATIENT, "sc": -1, "htn": "maybe", "age": "abc"})
        self.assertEqual(set(ctx.exception.errors), {"sc", "htn", "age"})

    def test_too_few_fields_rejected(self):
        with self.assertRaises(ValidationError) as ctx:
            validate_patient({"age": 50, "bp": 80, "hemo": 12})
        self.assertIn("_", ctx.exception.errors)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        model_dir = Path(cls.tmp.name)
        raw = make_ckd_like(400, seed=3)
        y = ft.encode_target(raw[ft.TARGET])
        pre = ft.build_preprocessor()
        rf = RandomForestClassifier(n_estimators=60, random_state=0).fit(pre.fit_transform(ft.prepare_features(raw)), y)
        joblib.dump(pre, model_dir / "preprocessor.joblib")
        joblib.dump(rf, model_dir / "rf_model.joblib")
        cls.predictor = CKDPredictor(model_dir)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def setUp(self):
        self.store = FakeStore()
        self.client = create_app(predictor=self.predictor, store=self.store).test_client()

    def test_predict_full_record(self):
        resp = self.client.post("/api/predict", json={**CKD_PATIENT, "patientID": "P-1"})
        body = resp.get_json()
        self.assertEqual(resp.status_code, 200)
        self.assertTrue(0 <= body["ckd_probability"] <= 1)
        self.assertEqual((body["patient_id"], body["stored"], body["fields_provided"], body["warnings"]), ("P-1", True, 24, []))
        self.assertGreater(body["ckd_probability"], 0.5)  # obviously abnormal synthetic profile
        self.assertEqual(len(self.store.saved), 1)

    def test_partial_record_works_with_warning_and_stores_only_measured_fields(self):
        sparse = {k: CKD_PATIENT[k] for k in ("age", "bp", "sc", "hemo", "htn", "dm", "appet")}
        body = self.client.post("/api/predict", json=sparse).get_json()
        self.assertEqual(body["fields_provided"], 7)
        self.assertEqual(len(body["warnings"]), 1)
        self.assertEqual(set(self.store.saved[0][1]), set(sparse))

    def test_generates_patient_id(self):
        body = self.client.post("/api/predict", json=CKD_PATIENT).get_json()
        self.assertTrue(body["patient_id"].startswith("PT-"))

    def test_invalid_input_is_400_and_not_stored(self):
        resp = self.client.post("/api/predict", json={**CKD_PATIENT, "al": 9})
        self.assertEqual(resp.status_code, 400)
        self.assertIn("al", resp.get_json()["fields"])
        self.assertEqual(self.store.saved, [])

    def test_too_few_fields_is_400(self):
        self.assertEqual(self.client.post("/api/predict", json={"age": 40}).status_code, 400)

    def test_missing_json_body_is_400(self):
        self.assertEqual(self.client.post("/api/predict", data="nope").status_code, 400)

    def test_ann_requested_but_unavailable_is_503(self):
        self.assertEqual(self.client.post("/api/predict?model=ann", json=CKD_PATIENT).status_code, 503)

    def test_batch_mixes_valid_and_invalid(self):
        resp = self.client.post("/api/predict/batch", json={"patients": [CKD_PATIENT, {"age": 3}]})
        body = resp.get_json()
        self.assertEqual((resp.status_code, len(body["results"]), len(body["errors"])), (200, 1, 1))

    def test_fields_endpoint_lists_schema(self):
        body = self.client.get("/api/fields").get_json()
        self.assertEqual(len(body["fields"]), 24)

    def test_web_form(self):
        self.assertEqual(self.client.get("/").status_code, 200)
        resp = self.client.post("/predict", data={k: str(v) for k, v in CKD_PATIENT.items()})
        self.assertEqual(resp.status_code, 200)
        self.assertIn(b"estimated probability of CKD", resp.data)
        bad = self.client.post("/predict", data={"age": "40"})
        self.assertEqual(bad.status_code, 400)

    def test_health_and_http_errors_are_json(self):
        self.assertEqual(self.client.get("/health").get_json()["mongodb"], "up")
        self.assertEqual(self.client.get("/nope").status_code, 404)
        self.assertEqual(self.client.get("/api/predict").status_code, 405)

    def test_app_without_models_degrades_gracefully(self):
        os.environ["MODEL_DIR"] = tempfile.mkdtemp()
        try:
            client = create_app(store=FakeStore()).test_client()
        finally:
            del os.environ["MODEL_DIR"]
        self.assertEqual(client.post("/api/predict", json=CKD_PATIENT).status_code, 503)
        self.assertEqual(client.get("/health").get_json()["status"], "degraded")


if __name__ == "__main__":
    unittest.main()
