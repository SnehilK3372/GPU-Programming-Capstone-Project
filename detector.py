"""
Detector abstraction. Tries to load a TorchScript model from models/,
otherwise uses torchvision Faster R-CNN or a DummyDetector fallback.
"""
import os
import torch
import numpy as np
import cv2

class DummyDetector:
    def __init__(self, device='cpu'):
        self.device = device

    def predict(self, image):
        # synthetic boxes for demo (x1,y1,x2,y2)
        H, W = image.shape[:2]
        boxes = np.array([
            [W*0.1, H*0.2, W*0.4, H*0.6],
            [W*0.5, H*0.2, W*0.9, H*0.8]
        ], dtype=np.float32)
        scores = np.array([0.9, 0.85], dtype=np.float32)
        labels = np.array([1, 2], dtype=np.int64)
        return boxes, scores, labels

class TorchDetector:
    def __init__(self, model_path=None, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.model = None
        if model_path and os.path.exists(model_path):
            try:
                self.model = torch.jit.load(model_path, map_location=self.device)
                self.model.eval()
                print(f"Loaded TorchScript model from {model_path} on {self.device}")
            except Exception as e:
                print("Failed to load TorchScript model:", e)
                self.model = None
        if self.model is None:
            # try torchvision model
            try:
                from torchvision.models.detection import fasterrcnn_resnet50_fpn
                self.model = fasterrcnn_resnet50_fpn(pretrained=True).to(self.device).eval()
                print("Loaded torchvision Faster R-CNN on device:", self.device)
            except Exception as e:
                print("Failed to load torchvision model:", e)
                self.model = None

    def preprocess(self, image):
        img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2,0,1).unsqueeze(0).float().to(self.device)
        return [img[0]]  # torchvision expects list[Tensor]

    def postprocess(self, outputs, orig_shape):
        if isinstance(outputs, list):
            out = outputs[0]
            boxes = out['boxes'].detach().cpu().numpy()
            scores = out['scores'].detach().cpu().numpy()
            labels = out['labels'].detach().cpu().numpy()
            return boxes, scores, labels
        else:
            # unknown format
            try:
                boxes = outputs[0][0].detach().cpu().numpy()
                scores = outputs[1][0].detach().cpu().numpy()
                labels = outputs[2][0].detach().cpu().numpy()
                return boxes, scores, labels
            except Exception:
                return np.empty((0,4)), np.empty((0,)), np.empty((0,))

    def predict(self, image):
        if self.model is None:
            raise RuntimeError("No model loaded")
        x = self.preprocess(image)
        with torch.no_grad():
            outputs = self.model(x)
        return self.postprocess(outputs, image.shape)

def make_detector(model_path=None, device=None):
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    if model_path and os.path.exists(model_path):
        try:
            return TorchDetector(model_path=model_path, device=device)
        except Exception as e:
            print("Failed to create TorchDetector:", e)
    try:
        det = TorchDetector(model_path=None, device=device)
        if det.model is not None:
            return det
    except Exception:
        pass
    print("Using DummyDetector fallback")
    return DummyDetector(device=device)
