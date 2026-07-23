# scripts/

Lane map: [`docs/overview/MODEL_LANES.md`](../docs/overview/MODEL_LANES.md)

| Dir | Role |
|-----|------|
| `infer/` | **Deepfake PROD** (+ finetune/bench). Do not rename — `gpu_worker` imports here. |
| `forgery/` | **Forgery PROD/TRAIN** — see `forgery/README.md` |
| `eval/` | Offline fusion / optical eval |
| `download/` `upload/` `deploy/` `messaging/` | Data & infra |
| `_archive/` | **Deprecated** — not on prod import path |

Restore something from `_archive/` only for historical replay; prefer current PROD paths for new work.
