# src/demo.py
import argparse
import time
import os
import sys
import cv2
import numpy as np

from pathlib import Path

# Build/import CUDA extension if available. We'll try to import; if fails, attempt to build.
def ensure_extension():
    try:
        import postproc_cuda  # noqa
        return True
    except Exception:
        pass
    # Try to build
    try:
        this_dir = Path(__file__).parent
        build = (this_dir / 'build_ext.py').resolve()
        print("Attempting to build CUDA extension...")
        # run as module
        import runpy
        runpy.run_path(str(build), run_name="__main__")
        import postproc_cuda  # noqa
        return True
    except Exception as e:
        print("Could not build CUDA extension (falling back to CPU). Error:", e)
        return False

# Import project modules
sys.path.append(str(Path(__file__).parent))
from detector import make_detector
from tracker import GreedyTracker
from utils import draw_boxes, save_detections_csv, ensure_dir

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--input', type=str, default='0', help='Input video path or webcam index (0) or sample file')
    p.add_argument('--output', type=str, default='outputs/annotated_demo.mp4', help='Output annotated video file')
    p.add_argument('--model', type=str, default='models/fasterrcnn_resnet50_fpn_ts.pt', help='TorchScript model path')
    p.add_argument('--score-thresh', type=float, default=0.5, help='Detection score threshold')
    p.add_argument('--max-frames', type=int, default=0, help='Max frames to process (0 = all)')
    return p.parse_args()

def open_input(path):
    if path.isdigit():
        return cv2.VideoCapture(int(path))
    else:
        return cv2.VideoCapture(path)

def main():
    args = parse_args()
    # try to build ext (non-fatal)
    has_ext = ensure_extension()
    if has_ext:
        print("CUDA extension available.")
    else:
        print("CUDA extension not available; GPU IoU will not be used.")

    model_path = args.model if os.path.exists(args.model) else None
    detector = make_detector(model_path=model_path)
    tracker = GreedyTracker(iou_thresh=0.3, max_age=30)

    input_src = args.input
    cap = None
    if input_src.isdigit():
        cap = cv2.VideoCapture(int(input_src))
    else:
        cap = cv2.VideoCapture(input_src)

    if not cap.isOpened():
        print("Failed to open input:", input_src)
        return

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    print(f"Input opened: {input_src} ({w}x{h}, fps={fps}, frames={total_frames})")

    ensure_dir(os.path.dirname(args.output) or '.')
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out_vid = cv2.VideoWriter(args.output, fourcc, fps, (w,h))

    detections_log = []
    frame_idx = 0
    start_time = time.time()
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            if args.max_frames and frame_idx > args.max_frames:
                break

            # detect
            try:
                boxes, scores, labels = detector.predict(frame)
            except Exception as e:
                print("Detector failed on frame", frame_idx, ":", e)
                boxes, scores, labels = np.empty((0,4)), np.empty((0,)), np.empty((0,))

            # filter by score
            if boxes.shape[0] > 0:
                keep = scores >= args.score_thresh
                boxes = boxes[keep]
                scores = scores[keep]
                labels = labels[keep]
            else:
                boxes = np.empty((0,4))
                scores = np.empty((0,))
                labels = np.empty((0,))

            # update tracker
            tracks = tracker.update(boxes, scores, labels)

            # collect logs
            for t in tracks:
                tid, box, label, score = t
                x1,y1,x2,y2 = [int(round(v)) for v in box]
                detections_log.append([frame_idx, tid, label, x1, y1, x2, y2, score])

            # draw and write frame
            vis = draw_boxes(frame, tracks)
            out_vid.write(vis)

            if frame_idx % 20 == 0:
                print(f"Frame {frame_idx}: detections {len(boxes)}, tracks {len(tracks)}")

    finally:
        cap.release()
        out_vid.release()
        elapsed = time.time() - start_time
        fps_proc = frame_idx / elapsed if elapsed > 0 else 0.0
        print(f"Processed {frame_idx} frames in {elapsed:.2f}s, ~{fps_proc:.2f} FPS")

        out_csv = os.path.join(os.path.dirname(args.output) or '.', 'outputs', 'detections.csv')
        ensure_dir(os.path.dirname(out_csv))
        save_detections_csv(out_csv, detections_log)
        print("Wrote detections CSV to", out_csv)
        # also write a small benchmark
        with open(os.path.join(os.path.dirname(args.output) or '.', 'outputs', 'benchmark.txt'), 'w') as f:
            f.write(f"frames={frame_idx}\nseconds={elapsed:.4f}\nfps={fps_proc:.4f}\n")
        print("Wrote benchmark to outputs/benchmark.txt")

if __name__ == '__main__':
    main()
