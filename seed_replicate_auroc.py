"""Held-out-institution macro AUROC of the multi-institution models (A1) and their institution-balanced versions (SB) for
seeds 2026-2028, with paired patient-image group bootstrap CIs for the difference (post hoc).
Reads predictions only. Weighted-AUROC bootstrap as in analyze_all.py."""
import csv, json
from pathlib import Path
import numpy as np

import config; W = config.WORK_DIR
EXT = ['02', '03', '04', '06', '08', '09']; B = 1000
rng = np.random.default_rng(2026)
RUNS = {2026: ('ablations/A1_loso_heldout{s}', 'site_balanced/SB_loso_sitebalanced_heldout{s}'),
        2027: ('seed_replicates/A1_loso_heldout{s}_seed2027', 'seed_replicates/SB_loso_sitebalanced_heldout{s}_seed2027'),
        2028: ('seed_replicates/A1_loso_heldout{s}_seed2028', 'seed_replicates/SB_loso_sitebalanced_heldout{s}_seed2028')}

def read(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
def load(d):
    r = read(d / 'test_predictions.csv')
    return np.array([a['image_id'] for a in r]), np.array([a['split_group'] for a in r]), np.array([int(a['folder_label'][1:]) for a in r]), np.array([[float(a[f'prob_D{c:02d}']) for c in range(11)] for a in r])
def auc_c(y, s, w):
    o = np.argsort(s, kind='mergesort'); _, inv = np.unique(s[o], return_inverse=True); pos = y[o]; ww = w[o]
    negw = ww * ~pos; posw = ww * pos; Wp, Wn = posw.sum(), negw.sum()
    if Wp == 0 or Wn == 0: return np.nan
    g = np.bincount(inv, negw); below = np.cumsum(g) - g
    return (posw * (below[inv] + .5 * g[inv])).sum() / (Wp * Wn)
def macro(y, p, w): return np.nanmean([auc_c(y == c, p[:, c], w) for c in range(11) if (y == c).any() and not (y == c).all()])

out = {}
for seed, (a, b) in RUNS.items():
    for s in EXT:
        da, db = W / a.format(s=s), W / b.format(s=s)
        if not ((da / 'test_predictions.csv').exists() and (db / 'test_predictions.csv').exists()): continue
        ia, g, y, pa = load(da); ib, _, _, pb = load(db); assert (ia == ib).all()
        gid = np.unique(g, return_inverse=True)[1]; K = gid.max() + 1; one = np.ones(len(y))
        ea, eb = macro(y, pa, one), macro(y, pb, one); d = []
        for _ in range(B):
            w = np.bincount(rng.integers(0, K, K), minlength=K)[gid].astype(float); d.append(macro(y, pb, w) - macro(y, pa, w))
        out[f'heldout{s}_seed{seed}'] = {'heldout': s, 'seed': seed, 'auc_unbalanced': ea, 'auc_balanced': eb, 'diff': eb - ea,
                                         'diff_ci': [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]}
        print(seed, s, f'{ea:.3f} -> {eb:.3f}  diff {eb - ea:+.3f} [{np.percentile(d, 2.5):+.3f}, {np.percentile(d, 97.5):+.3f}]', flush=True)
summary = {}
for s in EXT:
    v = [x for x in out.values() if x['heldout'] == s]
    if v: summary[s] = {'n_seeds': len(v), 'mean_diff': float(np.mean([x['diff'] for x in v])), 'min_diff': min(x['diff'] for x in v), 'max_diff': max(x['diff'] for x in v),
                        'mean_auc_unbalanced': float(np.mean([x['auc_unbalanced'] for x in v])), 'mean_auc_balanced': float(np.mean([x['auc_balanced'] for x in v]))}
json.dump({'runs': out, 'by_heldout': summary}, open(W / 'seed_replicates' / 'seed_replicate_auroc.json', 'w', encoding='utf-8'), indent=1, default=float)
print(json.dumps(summary, indent=1))
