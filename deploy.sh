#!/usr/bin/env bash
set -euo pipefail

SSH_KEY="/Users/markpinnuck/OCI/Key/mpoci.key"
REMOTE="ubuntu@168.138.106.96"
DEST="/var/www/tonightsky/"

DRY_RUN=""
if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN="--dry-run"
  echo "DRY RUN - no files will be transferred."
fi

if ! command -v rsync >/dev/null 2>&1; then
  echo "Local rsync is missing. Install it first (macOS: brew install rsync)."
  exit 127
fi

if [[ ! -f "$SSH_KEY" ]]; then
  echo "SSH key not found: $SSH_KEY"
  exit 1
fi

if ! ssh -i "$SSH_KEY" -o BatchMode=yes "$REMOTE" "command -v rsync >/dev/null 2>&1"; then
  echo "Remote rsync is missing on $REMOTE."
  echo "Install it with: sudo apt update && sudo apt install -y rsync"
  exit 127
fi

rsync -avz --delete $DRY_RUN \
  --chmod=D755,F644 \
  --exclude='.git/' \
  --exclude='.vscode/' \
  --exclude='venv/' \
  --exclude='__pycache__/' \
  --exclude='*.pyc' \
  --exclude='*.pyo' \
  --exclude='*.zip' \
  --exclude='.DS_Store' \
  --exclude='logs/' \
  --exclude='test/' \
  --exclude='.gitignore' \
  --exclude='Makefile' \
  --exclude='README.md' \
  --exclude='TonightSky.code-workspace' \
  --exclude='zip_source.sh' \
  --exclude='deploy.sh' \
  -e "ssh -i $SSH_KEY" \
  ./ "$REMOTE:$DEST"

echo "Deploy complete to $REMOTE:$DEST"