#!/usr/bin/env python3
"""
HW2 — Overlapping Sliding Windows Anomaly Detection
- Fixed-size overlapping window (step=1)
- Percentile threshold per window with numpy.percentile(..., method="linear")
- One-sided (upper-tail) rule
- Labels only the newly added point at each step
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

DATA_PATH = "AG_NO3_fill_cells_remove_NAN-2.csv"  # expects Date, NO3N, Student_Flag
W = 600
Q = 90.0  # percentile

def predict_overlapping_percentile(x: np.ndarray, W: int, q: float):
    n = len(x)
    preds = np.full(n, np.nan)
    thresholds = np.full(n, np.nan)
    for i in range(0, n - W + 1):
        win = x[i:i+W]
        T = np.percentile(win, q, method="linear")
        j = i + W - 1  # index of the newly added point
        thresholds[j] = T
        preds[j] = 1.0 if x[j] >= T else 0.0
    return preds, thresholds

def compute_metrics(preds: np.ndarray, y_true: np.ndarray):
    mask = ~np.isnan(preds)
    p = preds[mask].astype(int)
    t = y_true[mask].astype(int)
    TP = int(((p==1) & (t==1)).sum())
    FP = int(((p==1) & (t==0)).sum())
    FN = int(((p==0) & (t==1)).sum())
    TN = int(((p==0) & (t==0)).sum())
    P = TP + FN
    N = TN + FP
    normal_acc = TN / N if N>0 else float("nan")
    anomaly_acc = TP / P if P>0 else float("nan")
    return TP, FP, FN, TN, P, N, normal_acc, anomaly_acc, mask

def main():
    df = pd.read_csv(DATA_PATH)
    df['Date'] = pd.to_datetime(df['Date'])
    x = df['NO3N'].astype(float).to_numpy()
    y_true = df['Student_Flag'].astype(int).to_numpy()

    preds, thresholds = predict_overlapping_percentile(x, W, Q)
    TP, FP, FN, TN, P, N, normal_acc, anomaly_acc, mask = compute_metrics(preds, y_true)

    # Print metrics
    print(f"W={W}, q={Q}")
    print(f"TP={TP}, FP={FP}, FN={FN}, TN={TN}")
    print(f"P={P}, N={N}")
    print(f"Normal accuracy = {normal_acc:.4f}")
    print(f"Anomaly accuracy = {anomaly_acc:.4f}")

    # Plot
    dates_eval = df['Date'].to_numpy()[mask]
    x_eval = x[mask]
    p_eval = preds[mask].astype(int)
    t_eval = y_true[mask].astype(int)

    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(dates_eval, x_eval, linewidth=1)
    ax.scatter(dates_eval[p_eval==1], x_eval[p_eval==1], s=8, label="Predicted anomaly")
    ax.scatter(dates_eval[t_eval==1], x_eval[t_eval==1], s=8, marker="x", label="Ground truth anomaly")
    ax.set_title(f"Sliding-Window Anomaly Detection (W={W}, q={Q})")
    ax.set_xlabel("Time"); ax.set_ylabel("NO3N"); ax.legend()
    fig.autofmt_xdate()

    out_plot = Path("anomaly_plot_W{}_q{}.png".format(W, int(Q) if Q.is_integer() else Q))
    plt.savefig(out_plot, bbox_inches="tight")
    print(f"Saved plot to {out_plot}")

if __name__ == "__main__":
    main()
