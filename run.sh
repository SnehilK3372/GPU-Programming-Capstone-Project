#!/usr/bin/env bash
# run.sh - full demo launcher for Edge-ready Real-time Perception project
# Creates venv, installs deps, (optionally) builds CUDA extension, downloads model,
# creates a small sample video if none exists, then runs the demo.
#
# Usage:
#   ./run.sh                 # uses defaults
#   ./run.sh -i input.mp4 -o outputs/demo.mp4 -m models/fasterrcnn_resnet50_fpn_ts.pt
#
# Flags:
#   -i INPUT      input video path or webcam index (default: data/samples/sample_video.mp4)
#   -o OUTPUT     output annotated video path (default: outputs/annotated_demo.mp4)
#   -m MODEL      path to TorchScript model (default: models/fasterrcnn_resnet50_fpn_ts.pt)
#   -s SCORE      detection score threshold (default: 0.5)
#   -f MAX_FRAMES max frames to process (0 = all) (default: 0)
#   -h            print this help

set -euo pipefail
IFS=$'\n\t'

# --- defaults ---
INPUT="data/samples/sample_video.mp4"
OUTPUT="outputs/annotated_demo.mp4"
MODEL="models/fasterrcnn_resnet50_fpn_ts.pt"
SCORE=0.5
MAX_FRAMES=0
VENV_DIR="venv"
REQS="requirements.txt"

# --- helpers ---
info(){ echo -e "\033[1;34m[INFO]\033[0m $*"; }
warn(){ echo -e "\033[1;33m[WARN]\033[0m $*"; }
err(){ echo -e "\033[1;31m[ERROR]\033[0m $*" >&2; }

usage(){
  sed -n '1,120p' "$0" | sed -n '1,38p'
}

# --- parse args ---
while getopts ":i:o:m:s:f:h" opt; do
  case ${opt} in
    i ) INPUT="$OPTARG" ;;
    o ) OUTPUT="$OPTARG" ;;
    m ) MODEL="$OPTARG" ;;
    s ) SCORE="$OPTARG" ;;
    f ) MAX_FRAMES="$OPTARG" ;;
    h ) usage; exit 0 ;;
    \? ) err "Invalid option: -$OPTARG"; usage; exit 1 ;;
    : ) err "Option -$OPTARG requires an argument."; usage; exit 1 ;;
  esac
done
shift $((OPTIND -1))

# ensure project-root relative execution works even when called from elsewhere
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

info "Using input: $INPUT"
info "Using output: $OUTPUT"
info "Using model: $MODEL"
info "Score threshold: $SCORE"
info "Max frames: $MAX_FRAMES"

# --- setup venv & install deps ---
if [ ! -d "$VENV_DIR" ]; then
  info "Creating Python venv in $VENV_DIR ..."
  python3 -m venv "$VENV_DIR"
fi

# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"

info "Upgrading pip and installing requirements from $REQS ..."
python -m pip install --upgrade pip >/dev/null
if [ -f "$REQS" ]; then
  pip install -r "$REQS"
else
  warn "requirements.txt not found in repo root. Installing minimal deps."
  pip install torch torchvision numpy opencv-python tqdm
fi

# --- ensure directories ---
mkdir -p "$(dirname "$INPUT")"
mkdir -p "$(dirname "$OUTPUT")"
mkdir -p models
mkdir -p data/samples
mkdir -p outputs

# --- optional: download or create sample video if input missing ---
if [ "$INPUT" = "0" ] || [ "$INPUT" = "1" ]; then
  info "Using webcam index $INPUT"
else
  if [ ! -f "$INPUT" ]; then
    info "Input file '$INPUT' not found. Creating a short synthetic sample video at $INPUT ..."
    python - <<PY
import cv2, numpy as np, os
path = "${INPUT}"
os.makedirs(os.path.dirname(path), exist_ok=True)
w,h = 640,360
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(path, fourcc, 25.0, (w,h))
for i in range(200):
    frame = np.full((h,w,3), 30, dtype=np.uint8)
    # moving rectangles
    x = int((i*4) % (w-120))
    cv2.rectangle(frame, (x,50), (x+100,180), (0,200,0), -1)
    x2 = int((i*2 + 200) % (w-160))
    cv2.rectangle(frame, (x2,220), (x2+140,320), (200,0,0), -1)
    cv2.putText(frame, f"frame {i+1}", (10,20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 1)
    out.write(frame)
out.release()
print("Wrote synthetic video:", path)
PY
  else
    info "Input file found."
  fi
fi

# --- download model if missing ---
if [ ! -f "$MODEL" ]; then
  if [ -x "models/download_models.sh" ]; then
    info "Model not found. Attempting to download/export model using models/download_models.sh ..."
    bash models/download_models.sh || warn "models/download_models.sh failed; continue with DummyDetector or torchvision fallback"
  else
    warn "Model $MODEL not found and models/download_models.sh is missing/executable. Demo will try torchvision fallback or DummyDetector."
  fi
else
  info "Model file exists: $MODEL"
fi

# --- try to build CUDA extension (non-fatal) ---
info "Attempting to build CUDA extension (if no CUDA or build tools present this may fail but script will continue)..."
python - <<PY || warn "Build step failed; continuing without CUDA extension"
import runpy, sys
runpy.run_path('src/build_ext.py', run_name="__main__")
PY

# --- run demo ---
CMD=(python -m src.demo --input "$INPUT" --output "$OUTPUT" --model "$MODEL" --score-thresh "$SCORE")
if [ "$MAX_FRAMES" != "0" ]; then
  CMD+=(--max-frames "$MAX_FRAMES")
fi

info "Running demo: ${CMD[*]}"
"${CMD[@]}"

info "Demo finished. Outputs (if created):"
ls -l "$(dirname "$OUTPUT")" || true

info "Done."
