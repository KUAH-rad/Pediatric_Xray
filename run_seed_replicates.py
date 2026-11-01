"""Seed replicates of the leave-one-site-out multi-institution models (A1) and their institution-balanced versions (SB)
(post hoc: show whether the reduction in the normal-label association repeats across training runs).
Identical to run_ablations.A1 / run_site_balanced except the seed (2027, 2028; the sampler generator uses the same seed).
Outputs go to experiments/seed_replicates; seed-2026 runs are not touched."""
import csv, ctypes, time
from collections import Counter
import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import run_site_split as R
import run_ablations as A

OUT = R.WORK / 'seed_replicates'; OUT.mkdir(exist_ok=True)
A.WORK = OUT
PLAIN = R.loader

def balanced_loader_for(seed):
    def loader(rows, training=False, batch=32):
        if not training: return PLAIN(rows, False, batch)
        y = [int(r['folder_label'][1:]) for r in rows]; s = [r['site'] for r in rows]
        pooled = Counter(y); joint = Counter(zip(s, y)); site_n = Counter(s); n = len(rows)
        w = np.minimum([(pooled[yi] / n) / (joint[(si, yi)] / site_n[si]) for si, yi in zip(s, y)], 10.0)
        sampler = WeightedRandomSampler(torch.tensor(w, dtype=torch.double), num_samples=n, replacement=True, generator=torch.Generator().manual_seed(seed))
        return DataLoader(R.Images(rows, True), batch_size=batch, sampler=sampler, num_workers=0, pin_memory=True)
    return loader

def main(seed, condition):
    """One (seed, condition) stream per process, so several streams can share the GPU without touching the same folders."""
    with (R.WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    for s in A.EXTERNAL:
        parts = A.loso_parts(rows, s)
        if condition == 'A1': R.loader = PLAIN; A.train_dinov3(f'A1_loso_heldout{s}_seed{seed}', parts, seed=seed)
        else: R.loader = balanced_loader_for(seed); A.train_dinov3(f'SB_loso_sitebalanced_heldout{s}_seed{seed}', parts, seed=seed)
    R.save_json(OUT / f'complete_{condition}_seed{seed}.json', {'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')})

if __name__ == '__main__':
    import sys
    main(int(sys.argv[1]), sys.argv[2])   # e.g. 2027 A1 | 2027 SB | 2028 A1 | 2028 SB
