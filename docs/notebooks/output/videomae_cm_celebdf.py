"""VideoMAE celebdf Confusion Matrix from predictions.json (100 infer)."""

import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

mpl.rcParams["font.family"] = "Malgun Gothic"
mpl.rcParams["axes.unicode_minus"] = False

DOWNLOADS = Path(r"C:\Users\user\Downloads")
CANDIDATES = [
    DOWNLOADS / "videomae_celebdf_predictions.json",
    DOWNLOADS / "predictions.json",
]
PRED_PATH = next((p for p in CANDIDATES if p.is_file()), CANDIDATES[0])
OUT_DIR = Path(__file__).resolve().parent / "videomae-cm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def is_valid_score(score) -> bool:
    if score is None:
        return False
    try:
        return not np.isnan(float(score))
    except (TypeError, ValueError):
        return False


def main() -> None:
    if not PRED_PATH.is_file():
        print("predictions.json을 찾을 수 없습니다.")
        print("다음 중 하나에 파일을 두세요:")
        for path in CANDIDATES:
            print(f"  - {path}")
        raise SystemExit(1)

    print(f"using: {PRED_PATH}")
    data = json.loads(PRED_PATH.read_text(encoding="utf-8"))
    items = data["items"]
    threshold = data.get("threshold", "?")
    print(f"run_id: {data.get('run_id')}, count: {data.get('count')}, threshold: {threshold}")

    items_ok = [it for it in items if is_valid_score(it.get("fake_score"))]
    skipped = [it["file"] for it in items if not is_valid_score(it.get("fake_score"))]
    print(f"used: {len(items_ok)}, skipped: {len(skipped)} {skipped}")

    labels = ["fake", "real"]
    y_true = [it["ground_truth_label"] for it in items_ok]
    y_pred = [it["pred_label"] for it in items_ok]

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    tp, fn, fp, tn = cm[0, 0], cm[0, 1], cm[1, 0], cm[1, 1]
    acc = (tp + tn) / (tp + fn + fp + tn)

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
        xticklabels=["Pred 가짜", "Pred 진짜"],
        yticklabels=["Actual 가짜", "Actual 진짜"],
        ax=ax,
    )
    ax.set_title(f"VideoMAE celebdf (n={len(items_ok)}, thr={threshold})")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    out_png = OUT_DIR / "cm_videomae_celebdf.png"
    fig.savefig(out_png, dpi=150)
    print(f"\nsaved: {out_png}")
    plt.show()


if __name__ == "__main__":
    main()
