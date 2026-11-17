"""Predict Li-ion cell cycle life from the first 100 cycles.

Data: Severson et al. (2019), "Data-driven prediction of battery cycle life before
capacity degradation", Nature Energy. 124 commercial LFP/graphite cells (A123 APR18650M1A)
cycled under fast-charging protocols. Batch .mat files from https://data.matr.io/1/

Usage:
    python cycle_life.py data/2017-05-12_batchdata.mat data/2017-06-30_batchdata.mat ...
"""
import json
import sys

import h5py
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

EARLY, REF = 99, 9          # cycle indices used for delta-Q(V): cycle 100 minus cycle 10
SHORT_LIFE = 550            # cells below this are flagged as "fail" at qualification


def load_batch(path):
    """Read one MATR batch file (MATLAB v7.3 / HDF5) into a list of cell dicts."""
    f = h5py.File(path, "r")
    batch = f["batch"]
    cells = []
    for i in range(batch["summary"].shape[0]):
        life = f[batch["cycle_life"][i, 0]][()].item()
        if not np.isfinite(life):
            continue
        summ = f[batch["summary"][i, 0]]
        cyc = f[batch["cycles"][i, 0]]
        if cyc["Qdlin"].shape[0] <= EARLY:
            continue
        cells.append({
            "id": f"{path.split('/')[-1][:10]}_{i}",
            "cycle_life": life,
            "qd": np.hstack(summ["QDischarge"][0, :]),
            "ir": np.hstack(summ["IR"][0, :]),
            "chargetime": np.hstack(summ["chargetime"][0, :]),
            "qdlin_ref": np.hstack(f[cyc["Qdlin"][REF, 0]][()]),
            "qdlin_early": np.hstack(f[cyc["Qdlin"][EARLY, 0]][()]),
        })
    return cells


def features(c):
    """Early-life features: delta-Q(V) statistics, capacity fade, resistance, charge time."""
    dq = c["qdlin_early"] - c["qdlin_ref"]
    qd = c["qd"][1:100]
    slope, intercept = np.polyfit(np.arange(2, 2 + len(qd)), qd, 1)
    return {
        "log_var_dq": np.log10(abs(np.var(dq))),
        "log_min_dq": np.log10(abs(np.min(dq))),
        "fade_slope": slope,
        "fade_intercept": intercept,
        "qd_cycle2": qd[0],
        "qd_max_minus_2": qd.max() - qd[0],
        "ir_cycle2": c["ir"][1],
        "ir_delta": c["ir"][99] - c["ir"][1],
        "charge_time_5": np.mean(c["chargetime"][1:6]),
    }


def main(paths):
    cells = [c for p in paths for c in load_batch(p)]
    X = pd.DataFrame([features(c) for c in cells])
    y = np.log10([c["cycle_life"] for c in cells])
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

    models = {
        "variance_only": make_pipeline(StandardScaler(), ElasticNetCV(cv=5)),
        "elastic_net": make_pipeline(StandardScaler(), ElasticNetCV(cv=5, l1_ratio=[.1, .5, .9, 1])),
        "gbr": GradientBoostingRegressor(n_estimators=300, max_depth=2, learning_rate=0.05),
    }
    report, true_life = {}, 10 ** yte
    for name, m in models.items():
        cols = ["log_var_dq"] if name == "variance_only" else X.columns
        m.fit(Xtr[cols], ytr)
        pred = 10 ** m.predict(Xte[cols])
        flag_true, flag_pred = true_life < SHORT_LIFE, pred < SHORT_LIFE
        report[name] = {
            "mape_pct": float(np.mean(np.abs(pred - true_life) / true_life) * 100),
            "rmse_cycles": float(np.sqrt(np.mean((pred - true_life) ** 2))),
            "short_life_recall": float((flag_true & flag_pred).sum() / max(flag_true.sum(), 1)),
        }

    # Qualification view: stop testing at 100 cycles instead of cycling to 80% capacity.
    report["test_time_reduction_x"] = float(np.mean(10 ** y) / (EARLY + 1))
    report["n_cells"] = len(cells)
    print(json.dumps(report, indent=2))
    with open("metrics.json", "w") as fh:
        json.dump(report, fh, indent=2)


if __name__ == "__main__":
    main(sys.argv[1:])
