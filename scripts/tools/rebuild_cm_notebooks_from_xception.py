#!/usr/bin/env python3
"""Rebuild convnext/video_swin notebooks from xception with visible ROC section."""
from __future__ import annotations

import copy
import json
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "docs/notebooks/output"
XCEPTION = OUT / "xception_confusion_matrix_200.ipynb"

NOTEBOOK_DIR = r"C:\Users\user\Desktop\sk-final-forensic\ai-forensic\docs\notebooks"

TARGETS = [
    {
        "file": "convnext_confusion_matrix_200.ipynb",
        "title": "# ConvNeXt-S — Confusion Matrix · ROC (200개) · **로컬 노트북**",
        "data_subdir": "convnext",
        "out_subdir": "convnext-cm",
        "model_id": "convnext/v1.0.0",
        "model_label": "ConvNeXt-S",
        "roc_slug": "convnext",
        "ffpp_json": "infer_summary_convnext_ffpp_vox.json",
        "celeb_json": "infer_summary_convnext_celebdf.json",
        "nan_filter": False,
    },
    {
        "file": "video_swin_confusion_matrix_200.ipynb",
        "title": "# Video Swin — Confusion Matrix · ROC (200개) · **로컬 노트북**",
        "data_subdir": "video-swin",
        "out_subdir": "video-swin-cm",
        "model_id": "video-swin/v1.0.0",
        "model_label": "Video Swin",
        "roc_slug": "video_swin",
        "ffpp_json": "infer_summary_video_swin_ffpp_vox.json",
        "celeb_json": "infer_summary_video_swin_celebdf.json",
        "nan_filter": True,
    },
]

SECTIONS = """
## 노트북 섹션 (Run All)

| 순서 | 내용 |
|------|------|
| 1 | 환경 설정 + JSON 경로 |
| 2 | 데이터 로드 (200개) |
| 3 | Confusion Matrix (200 combined) |
| 4 | Confusion Matrix (profile별) |
| **5** | **ROC Curve 설명** |
| **6** | **ROC Curve 그래프 + AUC 표** |
| 7–8 | Accuracy / Precision / Recall / F1 |
"""

DOWNLOAD_CONV = """
### JSON 받기 (GPU SSH에서 `unset AWS_PROFILE` 후)

**PC로 복사** (PowerShell):
```powershell
scp sk4team@58.127.241.84:~/notebook-data/convnext/*.json `
  C:\\Users\\user\\Desktop\\sk-final-forensic\\ai-forensic\\docs\\notebooks\\data\\convnext\\
```

경로가 다르면 **셀 2**의 `LOCAL_FILES`만 수정하세요.

팀 CM 표기: [09-CNN-벤치마크-Confusion-Matrix-ROC.md](../09-CNN-벤치마크-Confusion-Matrix-ROC.md) 부록
"""

DOWNLOAD_VS = """
### JSON 받기 (GPU SSH에서 `unset AWS_PROFILE` 후)

**PC로 복사** (PowerShell):
```powershell
scp sk4team@58.127.241.84:~/notebook-data/video-swin/*.json `
  C:\\Users\\user\\Desktop\\sk-final-forensic\\ai-forensic\\docs\\notebooks\\data\\video-swin\\
```

경로가 다르면 **셀 2**의 `LOCAL_FILES`만 수정하세요.

팀 CM 표기: [09-CNN-벤치마크-Confusion-Matrix-ROC.md](../09-CNN-벤치마크-Confusion-Matrix-ROC.md) 부록
"""


def _lines(text: str) -> list[str]:
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    return lines


def intro_md(cfg: dict) -> str:
    dl = DOWNLOAD_CONV if cfg["data_subdir"] == "convnext" else DOWNLOAD_VS
    return f"""{cfg["title"]}

GPU / S3 없이 **PC에서 이 노트북만** 실행합니다.
{SECTIONS}
## 준비
1. 아래 **JSON 2개**를 `docs/notebooks/data/{cfg["data_subdir"]}/`에 둡니다
2. 터미널: `pip install matplotlib seaborn scikit-learn numpy pandas`
3. Cursor에서 이 `.ipynb` 열기 → **Run All** (셀 6까지 ROC 포함)

| 파일 | model | profile |
|------|-------|---------|
| `{cfg["ffpp_json"]}` | {cfg["model_id"]} | ffpp_vox |
| `{cfg["celeb_json"]}` | {cfg["model_id"]} | celebdf |

{dl}
"""


def setup_cell(cfg: dict) -> str:
    return f'''# 첫 실행 시 주석 해제:
# !pip install matplotlib seaborn scikit-learn numpy pandas

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import auc, classification_report, confusion_matrix, roc_curve

import matplotlib as mpl
import matplotlib.font_manager as fm

# 노트북에서 그래프(ROC 포함)가 셀 아래에 표시되도록
%matplotlib inline


def _setup_korean_font() -> str:
    """CM 축 라벨(가짜/진짜) 한글이 □로 깨지지 않게 폰트 설정."""
    candidates = [
        "Malgun Gothic",
        "NanumGothic",
        "Nanum Gothic",
        "AppleGothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    available = {{f.name for f in fm.fontManager.ttflist}}
    for name in candidates:
        if name in available:
            mpl.rcParams["font.family"] = name
            mpl.rcParams["axes.unicode_minus"] = False
            return name
    mpl.rcParams["axes.unicode_minus"] = False
    return "default (Korean font not found — install Malgun Gothic or NanumGothic)"


_kr_font = _setup_korean_font()
print(f"matplotlib font: {{_kr_font}}")

NOTEBOOK_DIR = Path(r"{NOTEBOOK_DIR}")
DATA_DIR = NOTEBOOK_DIR / "data" / "{cfg["data_subdir"]}"
DATA_DIR.mkdir(parents=True, exist_ok=True)

LOCAL_FILES = {{
    "ffpp_vox": DATA_DIR / "{cfg["ffpp_json"]}",
    "celebdf": DATA_DIR / "{cfg["celeb_json"]}",
}}

OUT_DIR = NOTEBOOK_DIR / "output" / "{cfg["out_subdir"]}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LABELS_TEAM = ["fake", "real"]

missing = [f"{{k}}: {{p}}" for k, p in LOCAL_FILES.items() if not p.is_file()]
if missing:
    print("MISSING — markdown 'JSON 받기' 절차로 아래 파일을 준비하세요:")
    for line in missing:
        print(f"  {{line}}")
    raise FileNotFoundError(missing[0])
print("OK: both JSON files found")
'''


def load_cell(cfg: dict) -> str:
    return f'''def load_items(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if "model" not in data:
        data["model"] = "{cfg["model_id"]}"
    if "profile" not in data:
        name = path.name.lower()
        data["profile"] = "ffpp_vox" if "ffpp" in name else "celebdf"
    if "count" not in data:
        data["count"] = len(data["items"])
    assert data["model"] == "{cfg["model_id"]}", f"expected {cfg["model_id"]}, got {{data.get('model')}} in {{path.name}}"
    print(f"{{data['model']}} / {{data['profile']}} / n={{data['count']}}  ({{path.name}})")
    return data["items"]


items_ffpp = load_items(LOCAL_FILES["ffpp_vox"])
items_celeb = load_items(LOCAL_FILES["celebdf"])
items_all = items_ffpp + items_celeb

print(f"total: {{len(items_all)}}")

y_true = [it["ground_truth_label"] for it in items_all]
y_pred = [it["pred_label"] for it in items_all]
'''


def patch_roc_code(code: str, cfg: dict) -> str:
    code = code.replace('MODEL_LABEL = "Xception"', f'MODEL_LABEL = "{cfg["model_label"]}"')
    code = code.replace("roc_xception_", f"roc_{cfg['roc_slug']}_")
    banner = (
        'print("\\n" + "=" * 60)\n'
        'print("ROC Curve (200개) — prob_fake score")\n'
        'print("=" * 60)\n\n'
    )
    if banner.strip() not in code:
        code = code.replace(
            "from sklearn.metrics import roc_curve, auc\n",
            "from sklearn.metrics import roc_curve, auc\n" + banner,
        )
    if cfg["nan_filter"]:
        old_fn = '''def items_to_roc_arrays(items) -> tuple[np.ndarray, np.ndarray]:
    """Positive=fake(1), score=prob_fake."""
    y = np.array([1 if it["ground_truth_label"] == "fake" else 0 for it in items])
    s = np.array([float(it["prob_fake"]) for it in items])
    return y, s'''
        new_fn = '''def items_to_roc_arrays(items) -> tuple[np.ndarray, np.ndarray]:
    """Positive=fake(1), score=prob_fake. NaN/inf scores are excluded."""
    y_list, s_list = [], []
    for it in items:
        s = float(it["prob_fake"])
        if np.isfinite(s):
            y_list.append(1 if it["ground_truth_label"] == "fake" else 0)
            s_list.append(s)
    return np.array(y_list), np.array(s_list)'''
        code = code.replace(old_fn, new_fn)
        code = code.replace(
            "auc200 = plot_roc_single(\n    items_all,",
            "y200, _ = items_to_roc_arrays(items_all)\nauc200 = plot_roc_single(\n    items_all,",
        )
        code = code.replace(
            'f"{MODEL_LABEL} — ROC Curve (n={len(items_all)}, score=prob_fake)"',
            'f"{MODEL_LABEL} — ROC Curve (n={len(y200)}, score=prob_fake)"',
        )
        code = code.replace(
            'auc_rows.append({"구분": profile, "AUC": f"{a:.4f}", "n": len(items)})',
            'auc_rows.append({"구분": profile, "AUC": f"{a:.4f}", "n": len(y)})',
        )
        code = code.replace(
            '[{"구분": "200 combined", "AUC": f"{auc200:.4f}", "n": len(items_all)}] + auc_rows',
            '[{"구분": "200 combined", "AUC": f"{auc200:.4f}", "n": len(y200)}] + auc_rows',
        )
    return code


def patch_cm_titles(code: str, cfg: dict) -> str:
    return (
        code.replace("Xception-S", cfg["model_label"])
        .replace("Xception", cfg["model_label"])
        .replace("cm_xception_", f"cm_{cfg['roc_slug']}_")
        .replace("xception_200", f"{cfg['roc_slug']}_200")
    )


def rebuild(cfg: dict, xception: dict) -> None:
    cells = []
    # 0 intro
    cells.append({"cell_type": "markdown", "metadata": {}, "source": _lines(intro_md(cfg))})
    # 1 setup
    cells.append({"cell_type": "code", "metadata": {}, "source": _lines(setup_cell(cfg)), "outputs": [], "execution_count": None})
    # 2 load
    cells.append({"cell_type": "code", "metadata": {}, "source": _lines(load_cell(cfg)), "outputs": [], "execution_count": None})
    # 3-4 CM from xception
    for idx in [3, 4]:
        src = patch_cm_titles("".join(xception["cells"][idx]["source"]), cfg)
        cells.append({"cell_type": "code", "metadata": {}, "source": _lines(src), "outputs": [], "execution_count": None})
    # 5 ROC markdown (xception)
    cells.append(copy.deepcopy(xception["cells"][5]))
    cells[-1].pop("outputs", None)
    cells[-1].pop("execution_count", None)
    # 6 ROC code (xception)
    roc_src = patch_roc_code("".join(xception["cells"][6]["source"]), cfg)
    cells.append({"cell_type": "code", "metadata": {}, "source": _lines(roc_src), "outputs": [], "execution_count": None})
    # 7-8 metrics from xception
    for idx in [7, 8]:
        src = patch_cm_titles("".join(xception["cells"][idx]["source"]), cfg)
        if idx == 8:
            src = src.replace("Xception 200", f"{cfg['model_label']} 200")
        cells.append({"cell_type": "code" if xception["cells"][idx]["cell_type"] == "code" else "markdown", "metadata": {}, "source": _lines(src), "outputs": [], "execution_count": None})
        if cells[-1]["cell_type"] == "markdown":
            cells[-1].pop("outputs", None)
            cells[-1].pop("execution_count", None)

    for cell in cells:
        cell["id"] = uuid.uuid4().hex[:8]

    nb = {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": xception.get("metadata", {}).get("kernelspec", {"display_name": "Python 3", "language": "python", "name": "python3"}),
            "language_info": xception.get("metadata", {}).get("language_info", {"name": "python", "version": "3.11.0"}),
        },
        "cells": cells,
    }
    path = OUT / cfg["file"]
    path.write_text(json.dumps(nb, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"wrote {path.name} ({len(cells)} cells)")


def main() -> None:
    xception = json.loads(XCEPTION.read_text(encoding="utf-8"))
    for cfg in TARGETS:
        rebuild(cfg, xception)


if __name__ == "__main__":
    main()
