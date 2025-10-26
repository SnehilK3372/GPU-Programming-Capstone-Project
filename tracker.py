# src/tracker.py
import numpy as np
import time

# try to import CUDA extension; builder will compile at runtime if missing
try:
    from postproc_cuda import compute_iou_matrix
    has_cuda_ext = True
except Exception:
    has_cuda_ext = False

class Track:
    def __init__(self, tid, box, label, score):
        self.tid = tid
        self.box = box.copy()
        self.label = label
        self.score = score
        self.time_since_update = 0

class GreedyTracker:
    def __init__(self, iou_thresh=0.3, max_age=30):
        self.next_id = 1
        self.tracks = []
        self.iou_thresh = iou_thresh
        self.max_age = max_age

    def _iou_cpu(self, boxes_a, boxes_b):
        N = boxes_a.shape[0]
        M = boxes_b.shape[0]
        iou = np.zeros((N, M), dtype=np.float32)
        for i in range(N):
            ax1, ay1, ax2, ay2 = boxes_a[i]
            area_a = max(0.0, ax2-ax1) * max(0.0, ay2-ay1)
            for j in range(M):
                bx1, by1, bx2, by2 = boxes_b[j]
                x1 = max(ax1, bx1)
                y1 = max(ay1, by1)
                x2 = min(ax2, bx2)
                y2 = min(ay2, by2)
                w = max(0.0, x2 - x1)
                h = max(0.0, y2 - y1)
                inter = w*h
                area_b = max(0.0, bx2-bx1) * max(0.0, by2-by1)
                uni = area_a + area_b - inter + 1e-6
                iou[i,j] = inter / uni
        return iou

    def _iou_gpu(self, boxes_a, boxes_b):
        # stack and send to CUDA ext; compute_iou_matrix returns [iou_mat]
        import torch
        boxes = np.vstack([boxes_a, boxes_b]).astype(np.float32)
        t = torch.from_numpy(boxes).cuda()
        iou_mat = compute_iou_matrix(t)[0]
        N = boxes_a.shape[0]
        M = boxes_b.shape[0]
        iou = iou_mat[:N, N:N+M].cpu().numpy()
        return iou

    def update(self, boxes, scores, labels):
        # boxes: (M,4)
        M = boxes.shape[0]
        if len(self.tracks) == 0:
            for i in range(M):
                tr = Track(self.next_id, boxes[i], int(labels[i]), float(scores[i]))
                self.tracks.append(tr)
                self.next_id += 1
            return [(t.tid, t.box, t.label, t.score) for t in self.tracks]

        tr_boxes = np.array([t.box for t in self.tracks], dtype=np.float32)
        if M == 0:
            # increment age
            for t in self.tracks:
                t.time_since_update += 1
            # remove old
            self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
            return []

        # compute IoU
        try:
            if has_cuda_ext:
                ious = self._iou_gpu(tr_boxes, boxes)
            else:
                ious = self._iou_cpu(tr_boxes, boxes)
        except Exception:
            ious = self._iou_cpu(tr_boxes, boxes)

        matched_tracks = set()
        matched_dets = set()
        matches = []
        for ti in range(ious.shape[0]):
            best_j = -1
            best_iou = 0.0
            for j in range(ious.shape[1]):
                if j in matched_dets: continue
                if ious[ti,j] > best_iou:
                    best_iou = ious[ti,j]
                    best_j = j
            if best_j != -1 and best_iou >= self.iou_thresh:
                matches.append((ti, best_j))
                matched_tracks.add(ti)
                matched_dets.add(best_j)

        # update matched
        for ti, j in matches:
            tr = self.tracks[ti]
            tr.box = boxes[j].copy()
            tr.score = float(scores[j])
            tr.label = int(labels[j])
            tr.time_since_update = 0

        # spawn unmatched detections
        for j in range(M):
            if j not in matched_dets:
                tr = Track(self.next_id, boxes[j], int(labels[j]), float(scores[j]))
                self.tracks.append(tr)
                self.next_id += 1

        # age unmatched tracks
        for idx, tr in enumerate(self.tracks):
            if idx not in matched_tracks:
                tr.time_since_update += 1
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        return [(t.tid, t.box, t.label, t.score) for t in self.tracks]
