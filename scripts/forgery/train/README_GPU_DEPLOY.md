# TruFor 학습 스크립트 → GPU 서버 배포

로컬(`ai-forensic`)에서 작성한 파일을 GPU 서버 `~/forenShield-ai/forgery/scripts/train/` 로 옮긴 뒤 학습합니다.

## 복사할 파일 목록

```text
ai-forensic/scripts/forgery/train/
├── prepare_trufor_video_frames.py
├── train_trufor_video_forgery.py
├── trufor_video_common.py
├── run_trufor_forgery_train.sh
└── vendor_patches/
    ├── dataset_ForenShieldVideo.py
    ├── trufor_forgery_video.yaml
    └── data_core_FSVIDEO.patch.md   # 참고용, 패치 필수 아님
```

서버에 이미 있는 GMFlow 스크립트(`train_gmflow_*.py`)는 **덮어쓰지 않음**.

---

## 방법 A — Windows PowerShell에서 scp (권장)

로컬 PC에서 실행 (`c:\FINAL` 기준):

```powershell
$REMOTE = "sk4team@58.127.241.84"
$LOCAL  = "c:\FINAL\ai-forensic\scripts\forgery\train"
$REMOTE_DIR = "~/forenShield-ai/forgery/scripts/train"

scp "$LOCAL\prepare_trufor_video_frames.py" "${REMOTE}:${REMOTE_DIR}/"
scp "$LOCAL\train_trufor_video_forgery.py"  "${REMOTE}:${REMOTE_DIR}/"
scp "$LOCAL\trufor_video_common.py"         "${REMOTE}:${REMOTE_DIR}/"
scp "$LOCAL\run_trufor_forgery_train.sh"     "${REMOTE}:${REMOTE_DIR}/"

ssh $REMOTE "mkdir -p ~/forenShield-ai/forgery/scripts/train/vendor_patches"
scp "$LOCAL\vendor_patches\dataset_ForenShieldVideo.py" "${REMOTE}:${REMOTE_DIR}/vendor_patches/"
scp "$LOCAL\vendor_patches\trufor_forgery_video.yaml"   "${REMOTE}:${REMOTE_DIR}/vendor_patches/"
```

Windows에서 복사한 `.sh`는 CRLF 줄바꿈 때문에 Linux에서 깨질 수 있습니다. 서버에서 한 번 실행:

```bash
sed -i 's/\r$//' ~/forenShield-ai/forgery/scripts/train/run_trufor_forgery_train.sh
```

또는 셸 스크립트 없이 아래 **python 명령만** 직접 실행해도 됩니다.

---



## 방법 B — GPU 서버에서 git pull

레포에 푸시되어 있다면:

```bash
cd ~/forenShield-ai
git pull
# 레포 구조에 따라 forgery/scripts/train/ 로 symlink 또는 cp
```

---



## 서버에서 학습 (전체 한 번에)

SSH 접속 후:

```bash
cd ~/forenShield-ai/forgery
source ../.venv/bin/activate
chmod +x scripts/train/run_trufor_forgery_train.sh

# vendor 패치 (1회)
cp scripts/train/vendor_patches/dataset_ForenShieldVideo.py \
   vendor/TruFor/TruFor_train_test/dataset/
cp scripts/train/vendor_patches/trufor_forgery_video.yaml \
   vendor/TruFor/TruFor_train_test/lib/config/

# prepare + smoke train
bash scripts/train/run_trufor_forgery_train.sh
```

또는 단계별:

```bash
# 1) 프레임 캐시
python3 scripts/train/prepare_trufor_video_frames.py \
  --data-root data/train/video/forgery-gmflow-train-400 \
  --out-dir data/processed/trufor-gmflow-train-400 \
  --frames-per-video 8

# 2) 학습
python3 scripts/train/train_trufor_video_forgery.py \
  -exp trufor_forgery_video \
  --run-name forgery-$(date +%Y%m%d-%H%M) \
  -g 0 \
  --cache-root data/processed/trufor-gmflow-train-400 \
  --pretrained-checkpoint /home/sk4team/forenShield-ai/forgery/models/test/spatial/trufor/v1.0.0/trufor.pth.tar \
  TRAIN.END_EPOCH 2 TRAIN.BATCH_SIZE_PER_GPU 4
```

`-exp`는 **config yaml 파일 이름**(stem)입니다. `trufor-forgery-smoke-20260701`처럼 날짜를 넣으면 안 됩니다.  
체크포인트 폴더 이름은 `--run-name`으로 지정합니다.



## 결과 위치

```text
data/processed/trufor-gmflow-train-400/meta.json     # prepare 요약
vendor/TruFor/TruFor_train_test/log/train/<실험명>/best.pth.tar
```



## 트러블슈팅

### `np.sctypes was removed in NumPy 2.0` (albumentations/imgaug)

smoke 학습은 augmentation 없이 돌립니다 (`trufor_forgery_video.yaml`의 `AUG: null`).

서버에서 yaml 갱신:

```bash
cp scripts/train/vendor_patches/trufor_forgery_video.yaml \
   vendor/TruFor/TruFor_train_test/lib/config/
```

`train_trufor_video_forgery.py`도 최신본으로 scp 후 재실행.

또는 venv에서 (다른 패키지에 영향 가능):

```bash
pip install 'numpy<2.0'
```

### `No module named 'tensorboardX'`

TruFor vendor 학습이 TensorBoard 로깅에 씁니다. 설치:

```bash
source ~/forenShield-ai/.venv/bin/activate
pip install tensorboardX
```

최신 `train_trufor_video_forgery.py`는 없어도 동작하지만(로그만 생략), 설치를 권장합니다.

### `np.int` / validation crash (NumPy 2.x)

TruFor vendor `lib/utils.py`가 `np.int`를 씁니다. 최신 `train_trufor_video_forgery.py`는 import 시 alias를 복구합니다. scp 후 **처음부터** 재실행.

GPU에서 vendor 직접 패치 (scp 없을 때):

```bash
sed -i 's/dtype=np\.int)/dtype=int)/g' vendor/TruFor/TruFor_train_test/lib/utils.py
grep -n "np\.int" vendor/TruFor/TruFor_train_test/lib/utils.py
```

---

`trufor.pth.tar` 로드 시 `weights_only=True` 기본값 때문에 실패할 수 있습니다.  
최신 `train_trufor_video_forgery.py`는 `weights_only=False`로 로드합니다. scp 후 재실행.

---

- [ ] 스크립트 4개 + vendor_patches 2개 서버에 있음
- [ ] `data/train/video/forgery-gmflow-train-400/` 존재
- [ ] `models/test/spatial/trufor/v1.0.0/trufor.pth.tar` 존재 (없으면 `--pretrained-checkpoint` 생략)
- [ ] 테스트 400개는 학습에 안 넣음 (prepare는 train-400만 사용)

---

## 중단 후 이어하기 (resume)

### 1) prepare (프레임 추출) — 자동 스킵

`prepare_trufor_video_frames.py`는 이미 저장된 `frames/*.jpg`가 있으면 **건너뜁니다**.
끊겼으면 **같은 명령을 다시 실행**하면 됩니다.

```bash
python3 scripts/train/prepare_trufor_video_frames.py \
  --data-root data/train/video/forgery-gmflow-train-400 \
  --out-dir data/processed/trufor-gmflow-train-400 \
  --frames-per-video 8
```

### 2) train — epoch 단위 checkpoint

매 epoch 끝에 저장됩니다:

```text
vendor/TruFor/TruFor_train_test/log/train/<실험명>/
├── checkpoint.pth.tar   # 마지막 epoch (resume용)
└── best.pth.tar         # valid 지표 최고 (배포/infer용)
```

**이어서 학습** — 반드시 **처음과 같은 `-exp` 이름** + `--resume`:

```bash
python3 scripts/train/train_trufor_video_forgery.py \
  -exp trufor-forgery-smoke-20260701 \
  -g 0 \
  --cache-root data/processed/trufor-gmflow-train-400 \
  --resume \
  TRAIN.END_EPOCH 10
```

- `--resume`: `checkpoint.pth.tar`에서 **모델+optimizer+epoch** 복구
- `TRAIN.END_EPOCH`를 늘리면 그 epoch까지 추가 학습 (예: 2에서 끊겼으면 `END_EPOCH 10`으로 8 epoch 더)

주의:

- **epoch 중간**(한 epoch 도는 도중)에 끊기면 그 epoch 진행분은 **저장 안 됨** → 마지막 완료 epoch부터 재개
- `-exp` 이름이 다르면 **새 폴더**라 resume 안 됨
- 장시간 학습은 `tmux` / `screen` 권장: `tmux new -s trufor`

```bash
tmux new -s trufor
# 위 train 명령 실행 후 Ctrl+B, D 로 detach
# 재접속: tmux attach -t trufor
```

## infer용 checkpoint merge (필수)

학습 yaml의 `MODULES: ['NP++','backbone','loc_head']` 때문에 `best.pth.tar`에는
`decode_head_conf`, `detection` 가중치가 **없습니다**.  
infer(`test.py -exp trufor_ph3`)는 **전체 모델**을 strict 로드하므로, baseline 위에 튜닝 키만 덮어써야 합니다.

```bash
cd ~/forenShield-ai/forgery

python3 scripts/train/merge_trufor_infer_checkpoint.py \
  --base models/test/spatial/trufor/v1.0.0/trufor.pth.tar \
  --tuned vendor/TruFor/TruFor_train_test/weights/forgery-20260701-0430/best.pth.tar \
  --out models/dev/spatial/trufor/v1.0.0/smoke-forgery-20260701/trufor.pth.tar
```

merge 후 `--trufor-weights`에 위 `trufor.pth.tar`를 넘겨 infer 재실행:

```bash
RUN_DATE=$(date +%Y%m%d-%H%M)
CKPT="models/dev/spatial/trufor/v1.0.0/smoke-forgery-20260701/trufor.pth.tar"

python3 scripts/infer/spatial_mvtamperbench_benchmark.py \
  --root ~/forenShield-ai/forgery \
  --data-root data/pull/evidence/csvted-200-balanced \
  --model trufor --num-frames 8 --threshold 0.5 \
  --trufor-weights "$CKPT" \
  --run-id "trufor-csvted200-tuned-${RUN_DATE}"
```