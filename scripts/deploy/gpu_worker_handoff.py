#!/usr/bin/env python3
"""GPU worker 작업자에게 보낼 배포 안내(경로·환경변수)를 카톡 복붙용으로 출력합니다.

Usage:
  python scripts/deploy/gpu_worker_handoff.py
  python scripts/deploy/gpu_worker_handoff.py --copy
  python scripts/deploy/gpu_worker_handoff.py --branch fix/dynamic-weighted-risk-lane
  python scripts/deploy/gpu_worker_handoff.py --topic dynamic-risk
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
GITHUB_REPO = "https://github.com/sk-final-deepfake/ai-forensic"

WELABS = {
    "ssh": "sk4team@58.151.205.220",
    "ai_repo": "/home/sk4team/ai-forensic",
    "forenshield_root": "/home/sk4team/forenShield-ai/deepfake",
    "env_file": "/home/sk4team/forenShield-ai/config/env.local",
    "worker_log": "/home/sk4team/forenShield-ai/logs/gpu_worker.log",
}

TOPICS: dict[str, dict[str, str]] = {
    "dynamic-risk": {
        "title": "riskScore 동적 가중 + forgery lane_ran 보정",
        "purpose": (
            "최종 riskScore = Late Fusion(F) + 위변조 max(G) 동적 가중 (F²+G²)/(F+G)×100.\n"
            "forgery merge 시 lane_ran=False(비활성)이면 forgery 0점을 넣지 않음.\n"
            "spatial+temporal merge 후 riskScore/riskLevel 재계산."
        ),
        "changed_files": "\n".join(
            [
                "- app/services/integrated_risk.py",
                "- gpu_worker/pipeline/forgery_infer.py (lane_ran)",
                "- gpu_worker/pipeline/forgery_merge.py",
                "- tests/test_integrated_risk*.py",
            ]
        ),
        "default_branch": "fix/dynamic-weighted-risk-lane",
        "smoke": "\n".join(
            [
                "1) FORGERY_ENABLED=1 상태에서 영상 1건 재분석",
                "2) worker 로그에 Recomputed riskScore=... method=dynamic_weighted_deepfake_forgery",
                "3) FORGERY_ENABLED=0 이면 deepfake(fusion)만 최종 점수",
            ]
        ),
    },
    "fusion-v4c": {
        "title": "Late Fusion v4c-field-tuned",
        "purpose": "fusion-v4c 게이트(T=0.578) + 멀티페이스 CNN/오버레이.",
        "changed_files": "- config/fusion_v4_ts_gated.json\n- gpu_worker/pipeline/fusion.py",
        "default_branch": "develop",
        "smoke": "1) fusion T=0.578 로드 확인\n2) CNN/TS/GMF 모듈 점수 + fusion 점수 확인",
    },
}


def _git_branch() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "develop"


def _git_commit_short() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=REPO_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        )
        return out.strip()
    except Exception:
        return "unknown"


def _load_fusion_meta() -> dict[str, str]:
    cfg_path = REPO_ROOT / "config" / "fusion_v4_ts_gated.json"
    if not cfg_path.is_file():
        return {"version": "fusion-v4c-field-tuned", "threshold": "0.578"}
    data = json.loads(cfg_path.read_text(encoding="utf-8"))
    return {
        "version": str(data.get("fusion_version", "fusion-v4c-field-tuned")),
        "threshold": str(data.get("threshold", "0.578")),
    }


def build_handoff(*, branch: str, topic: str) -> str:
    meta = TOPICS.get(topic, TOPICS["dynamic-risk"])
    fusion = _load_fusion_meta()
    commit = _git_commit_short()
    root = WELABS["forenshield_root"]
    ai_repo = WELABS["ai_repo"]

    env_block = f"""# --- 운영 필수 (env.local) ---
INFERENCE_MODE=local_model
USE_MOCK_INFER=0
INFER_DEVICE=cuda

FORENSHIELD_AI_ROOT={root}
DEEPFAKE_ROOT={ai_repo}

FUSION_CONFIG_PATH=config/fusion_v4_ts_gated.json
# 또는 절대경로:
# FUSION_CONFIG_PATH={root}/config/fusion_v4_ts_gated.json

# 딥페이크 weights (서버 실경로로 교체)
XCEPTION_WEIGHTS={root}/models/test/video/xception/v1.0.0/xception_finetuned_celeb1k.pth
TIMESFORMER_WEIGHTS={root}/models/test/video/timesformer/v1.0.0/timesformer_finetuned_celeb1k.pth
GMFLOW_PRETRAINED={root}/models/test/video/optical-flow/gmflow/pretrained/gmflow_things-e9887eda.pth
GMFLOW_LEARNED_HEAD={root}/models/test/video/optical-flow/gmflow/v1.0.0/gmflow_learned_head.joblib

# 위변조 lane (동적 가중 riskScore에 temporal 반영하려면 1 필수)
FORGERY_ENABLED=1
TRUFOR_CKPT=<TruFor ckpt 절대경로>
FORGERY_TS_CKPT=<forgery TimeSformer ckpt 절대경로>
FORGERY_ROOT=<forgery 코드 루트, 필요 시>

TRUFOR_THRESHOLD=0.515
FORGERY_TS_THRESHOLD=0.173386

# 시각화/오버레이 (기존 값 유지)
AI_VISUALIZATION_ENABLED=1
AI_VISUALIZATION_UPLOAD=1
AI_VISUALIZATION_OVERLAY=1

# RabbitMQ / AWS - 기존 env.local 값 그대로 유지 (여기 적지 말 것)"""

    return f"""[GPU Worker 배포 요청] {meta['title']}

■ 목적
{meta['purpose']}

■ Git
repo: {GITHUB_REPO}
branch: {branch}
local commit: {commit}

ssh {WELABS['ssh']}
cd {ai_repo}
git fetch origin
git checkout {branch}
git pull origin {branch}

■ 서버 경로
AI repo        : {ai_repo}
FORENSHIELD    : {root}
env.local      : {WELABS['env_file']}
worker log     : {WELABS['worker_log']}

■ Fusion (파일명 그대로)
config/fusion_v4_ts_gated.json
version   : {fusion['version']}
threshold : {fusion['threshold']}

■ 변경 파일 (핵심)
{meta['changed_files']}

■ 환경변수 템플릿 (env.local에 반영·확인)
{env_block}

■ worker 재시작
cd {ai_repo}
bash scripts/deploy/welabs-gpu-worker.sh
# 또는
pkill -f 'gpu_worker.rabbitmq_worker' || true
source {WELABS['env_file']}
nohup python -m gpu_worker.rabbitmq_worker >>{WELABS['worker_log']} 2>&1 &

■ 배포 후 스모크
{meta['smoke']}

■ 참고
- AI GitHub Actions는 FastAPI만 배포. GPU worker는 수동 pull+restart 필수.
- 이미 COMPLETED된 분석은 DB 점수 유지 → 확인은 재분석 필요.
"""


def copy_to_clipboard(text: str) -> bool:
    try:
        if sys.platform == "win32":
            subprocess.run(["clip"], input=text, text=True, check=True)
            return True
        if sys.platform == "darwin":
            subprocess.run(["pbcopy"], input=text, text=True, check=True)
            return True
        subprocess.run(["xclip", "-selection", "clipboard"], input=text, text=True, check=True)
        return True
    except Exception:
        return False


def main() -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    parser = argparse.ArgumentParser(description="GPU worker 배포 안내 복붙 텍스트 생성")
    parser.add_argument(
        "--branch",
        default="",
        help="배포 브랜치 (미지정 시 현재 git branch)",
    )
    parser.add_argument(
        "--topic",
        choices=sorted(TOPICS),
        default="dynamic-risk",
        help="배포 주제 템플릿",
    )
    parser.add_argument(
        "--copy",
        action="store_true",
        help="출력 후 클립보드에 복사 (Windows: clip)",
    )
    args = parser.parse_args()

    branch = args.branch.strip() or _git_branch()
    if branch == "HEAD":
        branch = TOPICS[args.topic]["default_branch"]

    text = build_handoff(branch=branch, topic=args.topic)
    print(text)

    if args.copy:
        if copy_to_clipboard(text):
            print("\n[clipboard] 복사 완료 - 카톡에 붙여넣기 하세요.", file=sys.stderr)
        else:
            print("\n[clipboard] 자동 복사 실패 - 위 텍스트를 직접 복사하세요.", file=sys.stderr)
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
