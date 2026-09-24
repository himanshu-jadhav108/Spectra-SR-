from pathlib import Path
import json
import cv2

candidates = [
    'spot/sample_000',
    'spot/sample_001',
    'spot/sample_002',
    'venus/sample_000',
    'venus/sample_001',
    'spain_crops/sample_001',
    'spain_crops/sample_002',
    'spain_crops/sample_003',
]

for p in candidates:
    base = Path('data/benchmark') / p
    if not base.exists():
        continue
    with open(base / 'metadata.json', 'r', encoding='utf-8') as f:
        meta = json.load(f)
    hr = cv2.imread(str(base / 'output' / 'hr_preview.png'))
    sr = cv2.imread(str(base / 'output' / 'sr_preview.png'))
    lr = cv2.imread(str(base / 'output' / 'lr_preview.png'))
    if hr is None or sr is None or lr is None:
        continue
    b, g, r = hr.mean(axis=(0, 1))
    std = hr.std()
    lap = cv2.Laplacian(hr, cv2.CV_64F).var()
    print(f"[{p}]")
    print(f"  Desc: {meta.get('description')}")
    print(f"  Mean RGB: ({int(r)}, {int(g)}, {int(b)}), Contrast std: {std:.1f}, Laplacian sharpness: {lap:.1f}")
