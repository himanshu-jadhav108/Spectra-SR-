from pathlib import Path
import json
import cv2
import numpy as np

base = Path('data/benchmark')
for ds in sorted(base.iterdir()):
    if ds.is_dir():
        samples = sorted(list(ds.iterdir()))
        print(f"\n=== Dataset: {ds.name} ({len(samples)} samples) ===")
        for s in samples:
            out = s / "output"
            lr = out / "lr_preview.png"
            sr = out / "sr_preview.png"
            hr = out / "hr_preview.png"
            if lr.exists() and sr.exists() and hr.exists():
                hr_img = cv2.imread(str(hr))
                lr_img = cv2.imread(str(lr))
                sr_img = cv2.imread(str(sr))
                
                # Check mean color and contrast
                hr_var = cv2.Laplacian(hr_img, cv2.CV_64F).var()
                mean_bgr = hr_img.mean(axis=(0, 1))
                # Check greenness
                green_ratio = mean_bgr[1] / (mean_bgr[0] + mean_bgr[2] + 1e-5)
                print(f"  {s.name}: HR shape={hr_img.shape}, HR var={hr_var:.1f}, mean BGR={mean_bgr.astype(int)}, green ratio={green_ratio:.2f}")
