#!/usr/bin/env bash
set -euo pipefail
python -m pip install -r requirements.txt
python -m src.run_pipeline
python reproducibility/capture_provenance.py
