# src/utils.py
import cv2
import numpy as np
import csv
import os

def draw_boxes(img, detections, color=(0,255,0)):
    out = img.copy()
    for det in detections:
        tid, box, label, score = det
        x1,y1,x2,y2 = [int(round(v)) for v in box]
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        cv2.putText(out, f"ID:{tid} L:{label} S:{score:.2f}", (x1, max(10,y1-6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
    return out

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_detections_csv(path, rows):
    # rows: list of dicts or tuples: frame,track_id,class,x1,y1,x2,y2,score
    ensure_dir(os.path.dirname(path) or '.')
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame','track_id','class','x1','y1','x2','y2','score'])
        for r in rows:
            writer.writerow(r)
