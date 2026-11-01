"""Post hoc shortcut-mitigation experiment (post hoc).
Same leave-one-site-out folds as A1, but training images are resampled so that every institution has the pooled
label distribution (weight = pooled p(y) / p(y | institution), clipped at 10), removing the institution-label association.
Everything else is identical to A1 (DINOv3 ViT-S/16 last 2 blocks, 8 epochs, patience 3, seed 2026)."""
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import run_site_split as R
import run_ablations as A

OUT = R.WORK / 'site_balanced'; OUT.mkdir(exist_ok=True)
A.WORK = OUT

def balanced_loader(rows, training=False, batch=32):
    if not training: return DataLoader(R.Images(rows, False), batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)
    y = [int(r['folder_label'][1:]) for r in rows]; s = [r['site'] for r in rows]
    pooled = Counter(y); joint = Counter(zip(s, y)); site_n = Counter(s); n = len(rows)
    w = np.array([(pooled[yi] / n) / (joint[(si, yi)] / site_n[si]) for si, yi in zip(s, y)])
    w = np.minimum(w, 10.0)
    sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=n, replacement=True, generator=torch.Generator().manual_seed(2026))
    return DataLoader(R.Images(rows, True), batch_size=batch, sampler=sampler, num_workers=0, pin_memory=True)

def main():
    import csv, ctypes, time
    with (R.WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    R.loader = balanced_loader
    for s in A.EXTERNAL:
        A.train_dinov3(f'SB_loso_sitebalanced_heldout{s}', A.loso_parts(rows, s))
    R.save_json(OUT / 'site_balanced_complete.json', {'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')})

if __name__ == '__main__':
    main()
