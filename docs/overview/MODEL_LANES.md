# ForenShield AI — Model Lanes

정식 레인(배포/gpu_worker) · 벤치/학습 · 폐기(archive) 를 **파일로 구분**한 기준 문서입니다.
경로를 크게 바꾸면 GPU/`gpu_worker` import가 깨지므로, **prod는 기존 경로 유지**하고 폐기분만 `scripts/_archive/`, `docs/_archive/` 로 옮깁니다.

목차: [docs/README.md](../README.md)

---

## 1) Deepfake lane (1차)

### PROD (gpu_worker가 import)

| 역할 | 경로 |
|------|------|
| CNN | `scripts/infer/video_xception_infer.py`, `face_crop.py`, `xception_operating_defaults.py` |
| Temporal | `scripts/infer/video_timesformer_infer.py`, `video_clip_transformer_common.py` |
| Optical | `scripts/infer/optical_flow_*.py`, `gmflow_scoring.py`, `gmflow_feature_extract.py`, `gmflow_learned_head_infer.py`, `backends/{base,raft,gmflow}_*` |
| Fusion | `config/fusion_v4_ts_gated.json` (+ tests용 `config/test/fusion_v1_tuned.json`) |
| Cohort | `config/optical_flow_cohort_v0.json` |

**배포 가중치 (참고):** Xception `xception_finetuned_celeb1k.pth`, TimeSformer deepfake finetune, GMFlow + learned head.

### TRAIN / BENCH (재현용, prod 아님)

| 구분 | 경로 |
|------|------|
| Finetune | `scripts/infer/video_xception_finetune.py`, `xception_finetune_*.py`, `run_*_finetune*.sh`, `download/data/prepare_xception_*`, `deploy/install_xception_finetune_bundle.sh` |
| 벤치 | `*_benchmark_infer.py`, `run_*_celebdf*`, VideoMAE/ConvNeXt/Swin 등 |
| Eval | `scripts/eval/eval_field_late_fusion.py`, `*_fusion_v4c*`, `analyze_optical_flow_threshold.py`, `gmflow_motion_score.py`, `train_gmflow_learned_head.py` |
| Data | `scripts/download/**` |

### ARCHIVE (폐기)

| 내용 | 경로 |
|------|------|
| PWC-Net backends | `scripts/_archive/deepfake/backends/pwcnet_*` |
| 미사용 scoring config | `scripts/_archive/deepfake/config/gmflow_scoring_v1.json` |
| 미사용 util | `scripts/_archive/deepfake/app/hash_utils.py` |
| PWC CM 노트북 | `docs/_archive/pwcnet_confusion_matrix_200.ipynb` |

---

## 2) Forgery lane (2차)

### PROD (gpu_worker forgery_infer)

| 역할 | 경로 | 버전 |
|------|------|------|
| Spatial TruFor | `scripts/forgery/infer/spatial_mvtamperbench_benchmark.py` + vendor TruFor ckpt | **videocof-v2** (T≈0.515) |
| Temporal TS | `scripts/forgery/infer/timesformer_forgery_features.py` | **v1.9-hardneg** (T≈0.173) |
| 공용 | `tamper_segment_labels.py`, `video_decode_robust.py`, `timesformer_forgery_benchmark.py` | |

환경변수: `TRUFOR_CKPT`, `FORGERY_TS_CKPT` (기본 경로는 `forgery_infer.py` 참고).

### TRAIN / REPRO (재학습·사다리)

| 구분 | 경로 | 비고 |
|------|------|------|
| TruFor 최종 | `run_trufor_forgery_train_r5_calibrated.sh`, `s1/s2_calibrated`, `f16_calibrated`, `run_trufor_*_videocof_v2*`, yaml `r5`/`s1`/`s2` | **유지** |
| TS 사다리 | `train_timesformer_forgery_*.py`, `run_timesformer_forgery_v1.6~v1.9*.sh`, `v1.4_prepare_*` | 재현용 |
| Helpers | `sweep_*_threshold.py`, `mine_timesformer_*`, `prepare_gmflow_temporal_dataset.py` | |
| Calib JSON | `config/forgery/trufor_r5_*`, `trufor_videocof_v2_*` | |

### ARCHIVE (폐기)

| 내용 | 경로 |
|------|------|
| TruFor 구 recipe | `scripts/_archive/forgery/trufor_legacy/` |
| TS 초기·깨진 shell | `scripts/_archive/forgery/timesformer_early/` |
| r3 calib | `scripts/_archive/forgery/config/trufor_r3_*` |

---

## 3) 폴더 규칙

```text
scripts/
  infer/           # deepfake PROD + train/bench
  forgery/infer/   # forgery PROD
  forgery/train/   # forgery TRAIN
  forgery/data/
  _archive/
  eval/ download|upload|deploy|...
docs/
  overview/ MODEL_LANES.md
  ops/ deepfake/ forgery/ contracts/
  notebooks/ _archive/
config/
  fusion_v4_ts_gated.json
  test/fusion_v1_tuned.json
  forgery/
```

---

## 4) 버전 한눈에

| Lane | PROD 모델 | 폐기/대체 |
|------|-----------|-----------|
| Deepfake CNN | Xception celeb1k FT | EfficientNet/ConvNeXt 벤치만 |
| Deepfake Temporal | TimeSformer deepfake FT | — |
| Deepfake Optical | GMFlow + learned head | PWC-Net → archive |
| Forgery Spatial | TruFor videocof-v2 | r1–r4 / v2–v5 → archive |
| Forgery Temporal | TimeSformer-forgery v1.9-hardneg | GMFlow discontinuity → archive |
