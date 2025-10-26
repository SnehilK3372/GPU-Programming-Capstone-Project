#!/usr/bin/env bash
set -e
mkdir -p models
python3 - <<'PY'
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
print('Downloading torchvision Faster R-CNN pretrained...')
model = fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()
# create dummy input for tracing (smaller to speed up)
example = [torch.randn(3, 400, 400)]
try:
    traced = torch.jit.trace(model, example, strict=False)
    traced.save('models/fasterrcnn_resnet50_fpn_ts.pt')
    print('Saved TorchScript model to models/fasterrcnn_resnet50_fpn_ts.pt')
except Exception as e:
    print('Tracing failed:', e)
PY
