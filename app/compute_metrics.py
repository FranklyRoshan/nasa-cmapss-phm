"""Compute RMSE and NASA score for every preloaded model and save them to artifacts/metrics.json.

Run once from the app folder (the one holding app.py and artifacts/):

    python compute_metrics.py

Put the official NASA files in the cmapss_data folder next to this script:
    test_FD001.txt ... test_FD004.txt   (the same engine files you upload in the app)
    RUL_FD001.txt  ... RUL_FD004.txt    (true remaining life of each test engine, one number per line)

The app then reads metrics.json and fills the RMSE and NASA score boxes. It uses the app's own
loading and prediction code, so the numbers match what the app predicts.
"""
import io
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import app  # the same loading + prediction code the web app uses

HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "cmapss_data"   # the folder holding test_FD00X.txt and RUL_FD00X.txt
CLIP_TRUE_RUL_TO_CAP = True   # the models predict capped RUL (cap = 125), so score against capped truth


def find(name):
    for folder in (DATA_DIR, HERE, app.MODEL_DIR):
        if (folder / name).is_file():
            return folder / name
    return None


class _File:  # just enough of Streamlit's UploadedFile for app.read_engine_file
    def __init__(self, path):
        self._bytes = Path(path).read_bytes()

    def getvalue(self):
        return self._bytes


def rmse(y_true, y_pred):
    return float(np.sqrt(np.mean((y_pred - y_true) ** 2)))


def nasa_score(y_true, y_pred):
    d = y_pred - y_true                       # predicted minus actual
    return float(np.sum(np.where(d < 0, np.exp(-d / 13.0) - 1, np.exp(d / 10.0) - 1)))


def main():
    results = {}
    for fleet in app.EXPECTED_CONDITIONS:
        test_path, rul_path = find(f"test_{fleet}.txt"), find(f"RUL_{fleet}.txt")
        if not (test_path and rul_path):
            print(f"{fleet}: skipped (need test_{fleet}.txt and RUL_{fleet}.txt)")
            continue
        raw = app.read_engine_file(_File(test_path))
        truth = pd.read_csv(rul_path, header=None).iloc[:, 0].to_numpy(dtype=float)  # engine 1, 2, 3...
        results[fleet] = {}
        for model in app.MODEL_SPECS:
            try:
                _names, sources = app._bundled_sources(fleet, model)
                art = app.load_artifacts(sources, ("metrics", fleet, model))
                table, _short = app.predict_engines(art, raw)
                table = table.sort_values("unit")
                y_true = truth[table["unit"].astype(int).to_numpy() - 1]
                if CLIP_TRUE_RUL_TO_CAP:
                    y_true = np.minimum(y_true, float(art["pre"]["rul_cap"]))
                y_pred = table["predicted_RUL"].to_numpy(dtype=float)
                results[fleet][model] = {"rmse": round(rmse(y_true, y_pred), 2),
                                         "nasa": round(nasa_score(y_true, y_pred), 0)}
                print(f"{fleet} {model:<13} RMSE {results[fleet][model]['rmse']:>7}  "
                      f"NASA {results[fleet][model]['nasa']:>10,.0f}")
            except Exception as exc:  # keep going; one broken model should not stop the rest
                print(f"{fleet} {model}: failed - {exc}", file=sys.stderr)

    out = app.MODEL_DIR / "metrics.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved {out}")


if __name__ == "__main__":
    main()
