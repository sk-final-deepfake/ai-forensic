# scripts/forgery/

Lane map: [`docs/overview/MODEL_LANES.md`](../../docs/overview/MODEL_LANES.md)

## Layout

| Path | Role |
|------|------|
| `infer/` | **PROD** — TruFor spatial + TimeSformer-forgery features (imported by `gpu_worker`) |
| `train/` | **TRAIN** — calibrated TruFor + TS v1.6–v1.9 recipes |
| `data/` | Dataset prepare helpers (`prepare_gmflow_temporal_dataset.py`) |
| `../_archive/forgery/` | **Deprecated** TruFor v2–v5 / early TS shells |

## PROD versions

- Spatial: TruFor **videocof-v2** (`TRUFOR_CKPT`)
- Temporal: TimeSformer-forgery **v1.9-hardneg** (`FORGERY_TS_CKPT`)

GPU mirror: `~/forenShield-ai/forgery/scripts/{infer,train,data}/`  
Deploy helper: `train/deploy_to_gpu.ps1`
