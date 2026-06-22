"""GMFlow ffpp_vox + celebdf (200) — CM·ROC 시각화 (EfficientNet/Xception 노트북과 동일 레이아웃)."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

NOTEBOOK_DIR = Path(__file__).resolve().parent
OUT_DIR = NOTEBOOK_DIR / "output" / "gmflow-cm"
CACHE_DIR = NOTEBOOK_DIR / ".gmflow_cache"
OUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

S3_BASE = (
    "s3://forenshield-evidence-877044078824/"
    "cases/test/video-benchmark-datasets/gmflow"
)

PROFILES = {
    "ffpp_vox": {
        "run_id": "gmflow-ffpp-vox-benchmark-20260622-0544",
        "infer_summary": (
            f"{S3_BASE}/ffpp_vox/gmflow-ffpp-vox-benchmark-20260622-0544/infer_summary.json"
        ),
        "json_prefix": (
            f"{S3_BASE}/ffpp_vox/gmflow-ffpp-vox-benchmark-20260622-0544/json/"
        ),
        "needs_json_merge": False,
    },
    "celebdf": {
        "run_id": "gmflow-celebdf-benchmark-20260622-0142",
        "infer_summary": (
            f"{S3_BASE}/celebdf/gmflow-celebdf-benchmark-20260622-0142/infer_summary.json"
        ),
        "json_prefix": (
            f"{S3_BASE}/celebdf/gmflow-celebdf-benchmark-20260622-0142/json/"
        ),
        "needs_json_merge": True,
    },
}

LOCAL_FILES: dict[str, Path | None] = {
    "ffpp_vox": None,
    "celebdf": None,
}

LABELS_TEAM = ["fake", "real"]
THRESHOLD = 0.5
MODEL_LABEL = "GMFlow (motion heuristic)"


def _aws_cp(s3_uri: str, local_path: Path) -> None:
    local_path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "cp", s3_uri, str(local_path)],
        check=True,
        capture_output=True,
        text=True,
    )


def _aws_sync(s3_prefix: str, local_dir: Path) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["aws", "s3", "sync", s3_prefix, str(local_dir)],
        check=True,
        capture_output=True,
        text=True,
    )


def _resolve_infer_summary(profile: str) -> Path:
    local_override = LOCAL_FILES.get(profile)
    if local_override and local_override.is_file():
        return local_override

    cached = CACHE_DIR / profile / "infer_summary.json"
    if cached.is_file():
        return cached

    s3_uri = PROFILES[profile]["infer_summary"]
    print(f"downloading {profile} infer_summary from s3...")
    _aws_cp(s3_uri, cached)
    return cached


def _load_json_scores(profile: str) -> dict[str, dict]:
    cfg = PROFILES[profile]
    if not cfg["needs_json_merge"]:
        return {}

    json_dir = CACHE_DIR / profile / "json"
    if not any(json_dir.glob("*.json")):
        print(f"syncing {profile} json/ from s3...")
        _aws_sync(cfg["json_prefix"], json_dir)

    by_stem: dict[str, dict] = {}
    for path in json_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        by_stem[path.stem] = data
    return by_stem


def _merge_score_fields(item: dict, json_scores: dict[str, dict], threshold: float) -> dict:
    out = dict(item)
    if out.get("fake_score") is not None:
        return out

    stem = Path(out.get("file", "")).stem
    detail = json_scores.get(stem)
    if not detail:
        return out

    mas = detail.get("motion_anomaly_score")
    if mas is None:
        mas = detail.get("fake_score")
    if mas is not None:
        out["motion_anomaly_score"] = mas
        out["fake_score"] = mas
        out["pred_label"] = detail.get("pred_label") or ("fake" if float(mas) >= threshold else "real")

    if detail.get("flow_mag_pair_range") is not None:
        out["flow_mag_pair_range"] = detail["flow_mag_pair_range"]
    return out


def load_profile_items(profile: str, threshold: float = THRESHOLD) -> list[dict]:
    summary_path = _resolve_infer_summary(profile)
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    json_scores = _load_json_scores(profile)

    items: list[dict] = []
    for raw in data.get("items", []):
        item = _merge_score_fields(raw, json_scores, threshold)
        items.append(item)

    scored = [it for it in items if it.get("fake_score") is not None]
    print(
        f"gmflow / {profile} / n={len(items)} scored={len(scored)}  ({summary_path.name})"
    )
    if len(scored) < len(items):
        missing = [it.get("file") for it in items if it.get("fake_score") is None]
        print(f"  warning: missing fake_score for {len(missing)} items (showing up to 5): {missing[:5]}")
    return scored


def plot_team_cm(y_true, y_pred, title: str, out_path: Path | None = None):
    cm = confusion_matrix(y_true, y_pred, labels=LABELS_TEAM)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    annot = np.array([[f"{tp}\nTP", f"{fn}\nFN"], [f"{fp}\nFP", f"{tn}\nTN"]])

    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=True,
        xticklabels=["Pred 가짜 (Pos)", "Pred 진짜 (Neg)"],
        yticklabels=["Actual 가짜 (Pos)", "Actual 진짜 (Neg)"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Values")
    ax.set_ylabel("Actual Values")
    ax.set_title(title)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved {out_path}")
    plt.close(fig)
    return cm


def plot_cm_by_profile(items_ffpp: list[dict], items_celeb: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, name, items in [
        (axes[0], "ffpp_vox (100)", items_ffpp),
        (axes[1], "celebdf (100)", items_celeb),
    ]:
        yt = [it["ground_truth_label"] for it in items]
        yp = [it["pred_label"] for it in items]
        cm = confusion_matrix(yt, yp, labels=LABELS_TEAM)
        annot = np.array(
            [[f"{cm[0, 0]}\nTP", f"{cm[0, 1]}\nFN"], [f"{cm[1, 0]}\nFP", f"{cm[1, 1]}\nTN"]]
        )
        sns.heatmap(
            cm,
            annot=annot,
            fmt="",
            cmap="Blues",
            xticklabels=["Pred 가짜", "Pred 진짜"],
            yticklabels=["Actual 가짜", "Actual 진짜"],
            ax=ax,
            cbar=False,
        )
        ax.set_title(f"{MODEL_LABEL} / {name}")
    fig.suptitle(f"{MODEL_LABEL} — profile별 Confusion Matrix (thr={THRESHOLD})")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def items_to_roc_arrays(items) -> tuple[np.ndarray, np.ndarray]:
    y = np.array([1 if it["ground_truth_label"] == "fake" else 0 for it in items])
    s = np.array([float(it["fake_score"]) for it in items])
    return y, s


def plot_roc_single(items, title: str, out_path: Path | None = None, thr: float = THRESHOLD) -> float:
    y, s = items_to_roc_arrays(items)
    fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, lw=2, label=f"ROC (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.5)")
    if len(thresholds) > 0:
        idx = int(np.argmin(np.abs(thresholds - thr)))
        ax.scatter(fpr[idx], tpr[idx], s=80, zorder=5, label=f"thr={thr}")
    ax.set_xlabel("FPR (False Positive Rate, 오탐률)")
    ax.set_ylabel("TPR (True Positive Rate, 재현도)")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"saved {out_path}")
    plt.close(fig)
    return roc_auc


def plot_roc_by_profile(items_ffpp: list[dict], items_celeb: list[dict], out_path: Path) -> list[dict]:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    auc_rows: list[dict] = []
    for ax, name, items in [
        (axes[0], "ffpp_vox (100)", items_ffpp),
        (axes[1], "celebdf (100)", items_celeb),
    ]:
        y, s = items_to_roc_arrays(items)
        fpr, tpr, thresholds = roc_curve(y, s, pos_label=1)
        a = auc(fpr, tpr)
        ax.plot(fpr, tpr, lw=2, label=f"AUC = {a:.3f}")
        ax.plot([0, 1], [0, 1], "k--", lw=1)
        if len(thresholds) > 0:
            idx = int(np.argmin(np.abs(thresholds - THRESHOLD)))
            ax.scatter(fpr[idx], tpr[idx], s=60, zorder=5, label=f"thr={THRESHOLD}")
        ax.set_xlabel("FPR (오탐률)")
        ax.set_ylabel("TPR (재현도)")
        ax.set_title(f"{MODEL_LABEL} / {name}")
        ax.legend(loc="lower right")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
        profile = name.split()[0]
        auc_rows.append({"구분": profile, "AUC": f"{a:.4f}", "n": len(items)})

    fig.suptitle(f"{MODEL_LABEL} — profile별 ROC (score=fake_score)")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
    return auc_rows


def main() -> None:
    items_ffpp = load_profile_items("ffpp_vox")
    items_celeb = load_profile_items("celebdf")
    items_all = items_ffpp + items_celeb
    print(f"total scored: {len(items_all)}")

    y_true = [it["ground_truth_label"] for it in items_all]
    y_pred = [it["pred_label"] for it in items_all]

    cm200 = plot_team_cm(
        y_true,
        y_pred,
        title=f"{MODEL_LABEL} — Confusion Matrix (n={len(items_all)}, thr={THRESHOLD})",
        out_path=OUT_DIR / "cm_gmflow_200_combined.png",
    )
    print("CM (rows=fake,real):\n", cm200)
    print(classification_report(y_true, y_pred, labels=LABELS_TEAM, digits=3))

    plot_cm_by_profile(items_ffpp, items_celeb, OUT_DIR / "cm_gmflow_200_by_profile.png")

    auc200 = plot_roc_single(
        items_all,
        f"{MODEL_LABEL} — ROC Curve (n={len(items_all)}, score=fake_score)",
        OUT_DIR / "roc_gmflow_200_combined.png",
    )
    auc_rows = plot_roc_by_profile(items_ffpp, items_celeb, OUT_DIR / "roc_gmflow_200_by_profile.png")

    print("\nROC AUC summary:")
    print(f"  200 combined: {auc200:.4f} (n={len(items_all)})")
    for row in auc_rows:
        print(f"  {row['구분']}: {row['AUC']} (n={row['n']})")


if __name__ == "__main__":
    main()
