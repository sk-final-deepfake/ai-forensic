# Deepfake Model Tuning Execution Report

## Model Ranking

| rank | model | profiles | baseline accuracy | fake recall | real recall | balanced-threshold accuracy | worst FN |
|---:|---|---|---:|---:|---:|---:|---:|
| 1 | convnext | celebdf, ffpp_vox | 0.720 | 0.500 | 0.940 | 0.815 | 50 |
| 2 | video-swin/v1.0.0 | celebdf, ffpp_vox | 0.705 | 0.630 | 0.780 | 0.700 | 35 |
| 3 | videomae | celebdf, ffpp_vox | 0.505 | 0.890 | 0.120 | 0.610 | 11 |

## Per-Profile Threshold Sweep

### convnext / celebdf
- Baseline: thr=0.50, acc=0.460, fake_recall=0.000, real_recall=0.920, FN=50, FP=4
- Best accuracy: thr=0.30, acc=0.630, fake_recall=0.620, real_recall=0.640, FN=19, FP=18
- Best balanced recall: thr=0.30, acc=0.630, fake_recall=0.620, real_recall=0.640, FN=19, FP=18
- Source: `C:\FINAL\ai-forensic\docs\notebooks\data\convnext\infer_summary_convnext_celebdf.json`

### convnext / ffpp_vox
- Baseline: thr=0.50, acc=0.980, fake_recall=1.000, real_recall=0.960, FN=0, FP=2
- Best accuracy: thr=0.62, acc=1.000, fake_recall=1.000, real_recall=1.000, FN=0, FP=0
- Best balanced recall: thr=0.62, acc=1.000, fake_recall=1.000, real_recall=1.000, FN=0, FP=0
- Source: `C:\FINAL\ai-forensic\docs\notebooks\data\convnext\infer_summary_convnext_ffpp_vox.json`

### videomae / ffpp_vox
- Baseline: thr=0.50, acc=0.540, fake_recall=1.000, real_recall=0.080, FN=0, FP=46
- Best accuracy: thr=0.64, acc=0.730, fake_recall=0.760, real_recall=0.700, FN=12, FP=15
- Best balanced recall: thr=0.64, acc=0.730, fake_recall=0.760, real_recall=0.700, FN=12, FP=15
- Source: `C:\FINAL\ai-forensic\docs\notebooks\data\convnext\infer_summary_videomae_ffpp_vox.json.json`

### videomae / celebdf
- Baseline: thr=0.50, acc=0.470, fake_recall=0.780, real_recall=0.160, FN=11, FP=42
- Best accuracy: thr=0.06, acc=0.500, fake_recall=0.960, real_recall=0.040, FN=2, FP=48
- Best balanced recall: thr=0.95, acc=0.490, fake_recall=0.580, real_recall=0.400, FN=21, FP=30
- Source: `C:\FINAL\ai-forensic\docs\notebooks\data\convnext\infer_summary_vidiomae_celebdf.json.json`

### video-swin/v1.0.0 / celebdf
- Baseline: thr=0.50, acc=0.490, fake_recall=0.300, real_recall=0.680, FN=35, FP=16
- Best accuracy: thr=0.22, acc=0.560, fake_recall=0.920, real_recall=0.200, FN=4, FP=40
- Best balanced recall: thr=0.37, acc=0.480, fake_recall=0.480, real_recall=0.480, FN=26, FP=26
- Source: `C:\FINAL\ai-forensic\docs\notebooks\data\video-swin\infer_summary_video_swin_celebdf.json`

### video-swin/v1.0.0 / ffpp_vox
- Baseline: thr=0.50, acc=0.920, fake_recall=0.960, real_recall=0.880, FN=2, FP=6
- Best accuracy: thr=0.48, acc=0.920, fake_recall=1.000, real_recall=0.840, FN=0, FP=8
- Best balanced recall: thr=0.50, acc=0.920, fake_recall=0.960, real_recall=0.880, FN=2, FP=6
- Source: `C:\FINAL\ai-forensic\docs\notebooks\data\video-swin\infer_summary_video_swin_ffpp_vox.json`

## Recommendation

- Primary fine-tune target: `convnext`.
- Use threshold sweep as a gate before promotion; if a threshold-only change trades too much real recall for fake recall, prefer a short head-only fine-tune.
- Keep the tuned checkpoint in `models/dev` first, then promote only after both `celebdf` and `ffpp_vox` improve or remain within the accepted regression budget.
