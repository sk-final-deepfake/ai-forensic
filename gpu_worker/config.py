"""Worker configuration from environment variables."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _load_dotenv() -> None:
    candidates: list[Path] = [
        Path(__file__).resolve().parent / ".env",
    ]
    forenshield_root = os.getenv("FORENSHIELD_AI_ROOT", "").strip()
    if forenshield_root:
        root = Path(forenshield_root)
        # Typical welabs layout: FORENSHIELD_AI_ROOT=.../forenShield-ai/deepfake
        candidates.append(root.parent / "config" / "env.local")
        candidates.append(root / "config" / "env.local")
    for env_path in candidates:
        if not env_path.is_file():
            continue
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
            return
        except ImportError:
            pass
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip("'\"")
            if key and key not in os.environ:
                os.environ[key] = value
        return


def _env(name: str, default: str = "") -> str:
    return os.getenv(name, default).strip()


_load_dotenv()


@dataclass(frozen=True)
class WorkerConfig:
    """RabbitMQ / S3 / GPU 경로 — backend-forensic RabbitMqConfig 와 동일한 큐 이름."""

    rabbit_host: str = _env("RABBITMQ_HOST", "localhost")
    rabbit_port: int = int(_env("RABBITMQ_PORT", "5672"))
    rabbit_user: str = _env("RABBITMQ_USER", "forenshield")
    rabbit_password: str = _env("RABBITMQ_PASSWORD", "")
    rabbit_vhost: str = _env("RABBITMQ_VHOST", "/")

    analysis_queue: str = "forenshield.analysis.queue"
    overlay_queue: str = _env("OVERLAY_QUEUE", "forenshield.overlay.queue")
    result_queue: str = "backend.ai.result.queue"
    result_exchange: str = _env("AI_RESULT_EXCHANGE", "ai.result.exchange")
    result_routing_key: str = _env("AI_RESULT_ROUTING_KEY", "result.video")
    overlay_result_routing_key: str = _env("AI_OVERLAY_RESULT_ROUTING_KEY", "result.overlay")

    aws_region: str = _env("AWS_REGION", "ap-northeast-2")
    evidence_bucket: str = _env("S3_EVIDENCE_BUCKET", "")

    project_root: Path = Path(_env("FORENSHIELD_AI_ROOT", str(Path.home() / "forenShield-ai")))
    deepfake_root: Path = Path(_env("DEEPFAKE_ROOT", "")) if _env("DEEPFAKE_ROOT", "") else project_root / "deepfake"
    work_dir: Path = project_root / "work"
    samples_dir: Path = project_root / "samples"
    models_test_dir: Path = deepfake_root / "models" / "test"
    results_dir: Path = project_root / "results"

    inference_mode: str = _env("INFERENCE_MODE", "test")  # test | gateway | local_model
    use_mock_infer: bool = _env("USE_MOCK_INFER", "0") in ("1", "true", "TRUE", "yes")
    gpu_gateway_url: str = _env("AI_GATEWAY_URL", "http://127.0.0.1:8000")
    device: str = _env("INFER_DEVICE", "") or _env("INFERENCE_DEVICE", "cpu")

    model_id: str = _env("INFERENCE_MODEL_ID", "xception")
    model_version: str = _env("INFERENCE_MODEL_VERSION", "test")
    model_checkpoint: str = _env("MODEL_CHECKPOINT_PATH", "") or _env("XCEPTION_WEIGHTS", "")
    timesformer_weights: str = _env("TIMESFORMER_WEIGHTS", "")
    gmflow_pretrained: str = _env(
        "GMFLOW_PRETRAINED",
        "models/test/video/optical-flow/gmflow/pretrained/gmflow_things-e9887eda.pth",
    )
    gmflow_learned_head: str = _env(
        "GMFLOW_LEARNED_HEAD",
        "models/test/video/optical-flow/gmflow/v1.0.0/gmflow_learned_head.joblib",
    )
    gmflow_meta: str = _env(
        "GMFLOW_META",
        "models/test/video/optical-flow/gmflow/v1.0.0/gmflow_best.meta.json",
    )
    fusion_config_path: str = _env("FUSION_CONFIG_PATH", "config/fusion_v4_ts_gated.json")
    sample_fps: float = float(_env("INFERENCE_SAMPLE_FPS", "1"))
    max_frames: int = int(_env("INFERENCE_MAX_FRAMES", "32"))
    deepfake_threshold: float = float(_env("DEEPFAKE_THRESHOLD", "0.5"))

    # Soft-gate forgery continuation (TruFor spatial) — best-effort, never hard-fails the job.
    trufor_weights: str = _env("TRUFOR_WEIGHTS", "models/test/spatial/trufor/v1.0.0/trufor.pth.tar")
    trufor_experiment: str = _env("TRUFOR_EXPERIMENT", "trufor_ph3")
    trufor_frames_per_video: int = int(_env("TRUFOR_FRAMES_PER_VIDEO", "8"))
    trufor_threshold: float = float(_env("TRUFOR_THRESHOLD", "0.515"))

    prefetch_count: int = 1


def load_config() -> WorkerConfig:
    cfg = WorkerConfig()
    cfg.work_dir.mkdir(parents=True, exist_ok=True)
    cfg.results_dir.mkdir(parents=True, exist_ok=True)
    (cfg.results_dir / "infer").mkdir(parents=True, exist_ok=True)
    return cfg
