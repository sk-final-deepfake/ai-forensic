"""GMFlow ffpp_vox 단일 프로필 시각화 (CM + score 분포 + range 상관).

ffpp_vox + celebdf 200개 CM/ROC는 gmflow_confusion_matrix_200.py 를 사용하세요.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

S3_INFER_SUMMARY = (
    "s3://forenshield-evidence-877044078824/"
    "cases/test/video-benchmark-datasets/gmflow/ffpp_vox/"
    "gmflow-ffpp-vox-benchmark-20260622-0544/infer_summary.json"
)

DOWNLOADS = Path.home() / "Downloads"
LOCAL_CANDIDATES = [
    DOWNLOADS / "gmflow_ffpp_vox_infer_summary.json",
    DOWNLOADS / "infer_summary.json",
]

OUT_DIR = Path(__file__).resolve().parent / "gmflow-ffpp-vox"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def _try_download_from_s3(target: Path) -> bool:
    try:
        subprocess.run(
            ["aws", "s3", "cp", S3_INFER_SUMMARY, str(target)],
            check=True,
            capture_output=True,
            text=True,
        )
        print(f"downloaded from s3: {target}")
        return True
    except Exception:
        return False


def _load_infer_summary() -> dict:
    for p in LOCAL_CANDIDATES:
        if p.is_file():
            print(f"using local file: {p}")
            return json.loads(p.read_text(encoding="utf-8"))

    downloaded = DOWNLOADS / "gmflow_ffpp_vox_infer_summary.json"
    if _try_download_from_s3(downloaded):
        return json.loads(downloaded.read_text(encoding="utf-8"))

    print("infer_summary.json을 찾지 못했습니다.")
    print("다음 중 하나를 준비하세요:")
    for p in LOCAL_CANDIDATES:
        print(f"  - {p}")
    print(f"또는 AWS CLI 설정 후 자동 다운로드 경로: {S3_INFER_SUMMARY}")
    raise SystemExit(1)


def _is_valid_item(item: dict) -> bool:
    score = item.get("fake_score")
    if score is None:
        return False
    try:
        return not np.isnan(float(score))
    except (TypeError, ValueError):
        return False


def _plot_confusion_matrix(items_ok: list[dict], threshold: float, title_suffix: str) -> None:
    labels = ["fake", "real"]
    y_true = [x["ground_truth_label"] for x in items_ok]
    y_pred = [x["pred_label"] for x in items_ok]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / max(1, tp + fn + fp + tn)

    print(f"\nCM [[TP, FN], [FP, TN]]:\n{cm}")
    print(f"Accuracy: {acc:.2%} ({tp + tn}/{tp + fn + fp + tn})")
    print(classification_report(y_true, y_pred, labels=labels, target_names=labels))

    annot = np.array([[f"{tp}\nTP", f"{fn}\nFN"], [f"{fp}\nFP", f"{tn}\nTN"]])
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=["Pred fake", "Pred real"],
        yticklabels=["Actual fake", "Actual real"],
        ax=ax,
    )
    ax.set_title(f"GMFlow ffpp_vox (n={len(items_ok)}, thr={threshold}) {title_suffix}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    out_png = OUT_DIR / "cm_gmflow_ffpp_vox.png"
    fig.savefig(out_png, dpi=150)
    print(f"saved: {out_png}")


def _plot_score_distribution(items_ok: list[dict], threshold: float, title_suffix: str) -> None:
    fake_scores = [float(x["fake_score"]) for x in items_ok if x["ground_truth_label"] == "fake"]
    real_scores = [float(x["fake_score"]) for x in items_ok if x["ground_truth_label"] == "real"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.linspace(0.0, 1.0, 26)
    ax.hist(fake_scores, bins=bins, alpha=0.6, label=f"fake (n={len(fake_scores)})")
    ax.hist(real_scores, bins=bins, alpha=0.6, label=f"real (n={len(real_scores)})")
    ax.axvline(threshold, color="red", linestyle="--", linewidth=1.2, label=f"threshold={threshold}")
    ax.set_title(f"GMFlow fake_score 분포 {title_suffix}")
    ax.set_xlabel("fake_score (= motion_anomaly_score)")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    out_png = OUT_DIR / "hist_gmflow_ffpp_vox_fake_score.png"
    fig.savefig(out_png, dpi=150)
    print(f"saved: {out_png}")


def _plot_range_vs_score(items_ok: list[dict], title_suffix: str) -> None:
    xs = []
    ys = []
    cs = []
    for x in items_ok:
        rng = x.get("flow_mag_pair_range")
        if rng is None:
            continue
        xs.append(float(rng))
        ys.append(float(x["fake_score"]))
        cs.append("tab:red" if x.get("ground_truth_label") == "fake" else "tab:blue")

    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.scatter(xs, ys, c=cs, alpha=0.7, s=18)
    ax.set_title(f"flow range vs fake_score {title_suffix}")
    ax.set_xlabel("flow_mag_pair_range (max - min)")
    ax.set_ylabel("fake_score")
    fig.tight_layout()
    out_png = OUT_DIR / "scatter_gmflow_ffpp_vox_range_vs_score.png"
    fig.savefig(out_png, dpi=150)
    print(f"saved: {out_png}")


def main() -> None:
    data = _load_infer_summary()
    items = data.get("items", [])
    run_id = data.get("run_id", "unknown")
    threshold = float(data.get("threshold", 0.5))

    items_ok = [it for it in items if _is_valid_item(it)]
    skipped = [it.get("file") for it in items if not _is_valid_item(it)]
    title_suffix = f"(run={run_id})"

    print(f"run_id: {run_id}")
    print(f"count: {len(items)}, used: {len(items_ok)}, skipped: {len(skipped)}")
    if skipped:
        print(f"skipped files: {skipped}")

    _plot_confusion_matrix(items_ok, threshold, title_suffix)
    _plot_score_distribution(items_ok, threshold, title_suffix)
    _plot_range_vs_score(items_ok, title_suffix)

    plt.show()


if __name__ == "__main__":
    main()

