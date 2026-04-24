"""
Generate report figures from FINAL/training_logs.json, FINAL/results.csv, FINAL/riddles.csv.
Saves PNGs in the repository root. Run: python3 make_graphs.py
"""
import json
import os
import re
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(REPO, "FINAL")
A_COL = "Answer"


def _save(fig, out_path, dpi=150):
    fig.tight_layout()
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    return out_path


def _norm(s):
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", "", s)
    return re.sub(r"\s+", " ", s).strip()


def _em(g, p):
    return float(_norm(g) == _norm(p))


def _f1(g, p):
    a, b = _norm(g).split(), _norm(p).split()
    if not a or not b:
        return float(a == b)
    common = sum((Counter(a) & Counter(b)).values())
    if common == 0:
        return 0.0
    return 2 * common / (len(a) + len(b))


def _row_metrics(df, pred_col, gold_col="gold"):
    g, p = df[gold_col].astype(str), df[pred_col].astype(str)
    ems = [_em(gg, pp) for gg, pp in zip(g, p)]
    f1s = [_f1(gg, pp) for gg, pp in zip(g, p)]
    return float(np.mean(ems)), float(np.mean(f1s))


def load_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def plot_loss_curve():
    path = os.path.join(DATA, "training_logs.json")
    data = load_json(path)
    seeds = data.get("seeds") or []
    if not seeds:
        raise SystemExit("No 'seeds' in training_logs.json; run riddle_pipeline.py first.")

    train_rows = [s["train_loss"] for s in seeds]
    val_rows = [s["val_loss"] for s in seeds]
    n_ep = min(len(t) for t in train_rows)
    if n_ep < 1:
        raise SystemExit("Empty loss history.")
    tr = np.mean([t[:n_ep] for t in train_rows], axis=0)
    va = np.mean([t[:n_ep] for t in val_rows], axis=0)
    epochs = np.arange(1, n_ep + 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, tr, label="Average training loss", marker="o", markersize=4)
    ax.plot(epochs, va, label="Average validation loss", marker="o", markersize=4)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training and Validation Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    out = os.path.join(REPO, "loss_curve.png")
    return _save(fig, out)


def plot_model_comparison():
    """EM/F1 from run_metrics.json or results.csv (should match paper table)."""
    mpath = os.path.join(DATA, "run_metrics.json")
    rpath = os.path.join(DATA, "results.csv")
    if os.path.isfile(mpath):
        m = load_json(mpath)
        maj, pre, ft = m["majority"], m["pretrained"], m["finetuned"]
    elif os.path.isfile(rpath):
        df = pd.read_csv(rpath)
        em, f1v = _row_metrics(df, "majority_pred")
        maj = {"EM": em, "F1": f1v}
        em, f1v = _row_metrics(df, "zeroshot_pred")
        pre = {"EM": em, "F1": f1v}
        ft = {"EM": float(df["avg_em"].mean()), "F1": float(df["avg_f1"].mean())}
    else:
        raise SystemExit("Need FINAL/run_metrics.json or FINAL/results.csv")

    labels = [
        "Majority\nbaseline",
        "Pretrained\nT5-small",
        "Fine-tuned\nT5-small",
    ]
    ems = [maj["EM"], pre["EM"], ft["EM"]]
    f1s = [maj["F1"], pre["F1"], ft["F1"]]

    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - w / 2, ems, w, label="EM")
    ax.bar(x + w / 2, f1s, w, label="F1")
    ax.set_ylabel("Score")
    ax.set_title("Model Performance Comparison")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.legend()
    ax.set_ylim(0, 1.05)
    ax.grid(True, axis="y", alpha=0.3)
    out = os.path.join(REPO, "model_comparison.png")
    return _save(fig, out)


def plot_answer_lengths():
    path = os.path.join(DATA, "riddles.csv")
    if not os.path.isfile(path):
        raise SystemExit(f"Missing {path}")
    df = pd.read_csv(path)[[A_COL]].dropna()
    lengths = df[A_COL].astype(str).str.split().str.len()
    nmax = int(lengths.max()) if len(lengths) else 1

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(
        lengths,
        bins=range(1, nmax + 2),
        align="left",
        rwidth=0.85,
        edgecolor="black",
    )
    ax.set_xlabel("Number of words in answer")
    ax.set_ylabel("Count")
    ax.set_title("Answer Length Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    out = os.path.join(REPO, "answer_length_distribution.png")
    return _save(fig, out)


def _load_results():
    rpath = os.path.join(DATA, "results.csv")
    if not os.path.isfile(rpath):
        raise SystemExit(f"Missing {rpath}; run riddle_pipeline.py first.")
    return pd.read_csv(rpath)


def plot_prediction_quality():
    """
    Correct: avg_em == 1
    Partial: avg_em == 0 and avg_f1 > 0 (plus any remaining rows with 0<avg_em<1 and
    some F1, e.g. mixed seed votes — counted with Partial)
    Wrong: avg_f1 == 0
    """
    df = _load_results()
    em = df["avg_em"].astype(float)
    f1 = df["avg_f1"].astype(float)
    is_correct = np.isclose(em, 1.0)
    is_wrong = np.isclose(f1, 0.0)
    is_partial_core = np.isclose(em, 0.0) & (f1 > 0.0)
    covered = is_correct | is_wrong | is_partial_core
    n_correct = int(np.sum(is_correct))
    n_wrong = int(np.sum(is_wrong))
    # Strict partial + rows that are neither full match nor total failure (e.g. avg_em=2/3)
    n_partial = int(np.sum(is_partial_core) + np.sum(~covered))
    labels = ["Correct", "Partial", "Wrong"]
    counts = [n_correct, n_partial, n_wrong]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(labels, counts, color=["#2ca02c", "#ff7f0e", "#d62728"], edgecolor="black")
    ax.set_ylabel("Number of test examples")
    ax.set_title("Prediction Quality Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    out = os.path.join(REPO, "prediction_quality.png")
    return _save(fig, out)


def plot_f1_distribution():
    df = _load_results()
    f1 = df["avg_f1"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(f1, bins=20, range=(0, 1), edgecolor="black", alpha=0.85)
    ax.set_xlabel("F1 Score")
    ax.set_ylabel("Number of Examples")
    ax.set_title("F1 Score Distribution")
    ax.grid(True, axis="y", alpha=0.3)
    out = os.path.join(REPO, "f1_distribution.png")
    return _save(fig, out)


def plot_answer_length_vs_f1():
    df = _load_results()
    gold = df["gold"].astype(str)
    word_len = gold.str.split().str.len()
    f1 = df["avg_f1"].astype(float)
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.scatter(word_len, f1, alpha=0.6, edgecolors="k", linewidths=0.3)
    ax.set_xlabel("Answer length (words, gold)")
    ax.set_ylabel("F1 score (avg_f1)")
    ax.set_title("Answer Length vs F1 Score")
    ax.grid(True, alpha=0.3)
    out = os.path.join(REPO, "answer_length_vs_f1.png")
    return _save(fig, out)


def main():
    created = [
        plot_loss_curve(),
        plot_model_comparison(),
        plot_answer_lengths(),
        plot_prediction_quality(),
        plot_f1_distribution(),
        plot_answer_length_vs_f1(),
    ]
    print("\nGraphs created:")
    for p in created:
        print(f"  - {p}")


if __name__ == "__main__":
    main()
