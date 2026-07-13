# Step A — GPU SSH (이미 열린 세션)에서 실행:
cd ~/forenShield-ai/forgery/data/pull/evidence
tar czf /tmp/mvtamperbench-200-s3.tar.gz mvtamperbench-200-s3
tar czf /tmp/csvted-200-balanced.tar.gz csvted-200-balanced
du -sh /tmp/mvtamperbench-200-s3.tar.gz /tmp/csvted-200-balanced.tar.gz

# Step B — 로컬 PowerShell (SSH 안 들어간 창)에서 실행:
# 아래 pull_testsets.ps1 참고
