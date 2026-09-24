import cv2
import numpy as np

for name in ['spot/sample_002', 'spot/sample_000', 'venus/sample_000', 'spain_crops/sample_001']:
    lr = cv2.imread(f'data/benchmark/{name}/output/lr_preview.png')
    sr = cv2.imread(f'data/benchmark/{name}/output/sr_preview.png')
    hr = cv2.imread(f'data/benchmark/{name}/output/hr_preview.png')
    
    print(f"\n=== {name} ===")
    print("LR shape:", lr.shape, "mean:", lr.mean(axis=(0,1)).astype(int))
    print("SR shape:", sr.shape, "mean:", sr.mean(axis=(0,1)).astype(int), "sharpness:", cv2.Laplacian(sr, cv2.CV_64F).var())
    print("HR shape:", hr.shape, "mean:", hr.mean(axis=(0,1)).astype(int), "sharpness:", cv2.Laplacian(hr, cv2.CV_64F).var())
