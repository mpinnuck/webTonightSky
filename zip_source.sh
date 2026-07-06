#!/usr/bin/env bash
set -euo pipefail

project_name="TonightSky"
output_zip="${project_name}_source.zip"

rm -f "$output_zip"

zip -r "$output_zip" . \
  -x ".git/*" \
  -x "venv/*" \
  -x "__pycache__/*" \
  -x "*/__pycache__/*" \
  -x "*.pyc" \
  -x "*.pyo" \
  -x ".DS_Store" \
  -x "*/.DS_Store" \
  -x "logs/*" \
  -x "*.log" \
  -x "*/logs/*" \
  -x "*.zip"

echo "Created: $output_zip"