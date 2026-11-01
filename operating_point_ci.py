"""95% CIs for external sensitivity, specificity and PPV at the validation-fixed Youden thresholds (DINOv3 ensemble), and
external counts by institution (images, patient-image groups, positives per label). Post hoc.
Patient-image group bootstrap (1,000 resamples) as in analyze_all.py. Reads predictions and results/all_results.json only."""
import csv, json
from pathlib import Path
import numpy as np

import config; W = config.WORK_DIR; RES = config.RESULTS_DIR
EN = ['Normal', 'Pneumoperitoneum', 'Abdominal mass', 'Intussusception', 'Air-fluid level', 'Foreign body', 'Constipation',
      'Calcified lesion', 'Splenomegaly', 'Hepatomegaly', 'Situs inversus totalis']
EXT = ['02', '03', '04', '06', '08', '09']
rng = np.random.default_rng(2026)
A = json.load(open(RES / 'all_results.json', encoding='utf-8'))

def read(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))
rs = [read(W / f'dinov3_last2blocks_finetune_seed{s}' / 'test_predictions.csv') for s in (2026, 2027, 2028)]
P = np.mean([[[float(a[f'prob_D{c:02d}']) for c in range(11)] for a in r] for r in rs], 0)
y = np.array([int(a['folder_label'][1:]) for a in rs[0]]); site = np.array([a['site'] for a in rs[0]]); group = np.array([a['split_group'] for a in rs[0]])
gid = np.unique(group, return_inverse=True)[1]; K = gid.max() + 1

out = {'operating_points': {}, 'counts_by_institution': {}}
W_ = [np.bincount(rng.integers(0, K, K), minlength=K)[gid].astype(float) for _ in range(1000)]
for c in range(11):
    t = A['per_finding'][EN[c]]['thresholds']['youden']; pos = y == c; pr = P[:, c] >= t
    def m(w):
        tp, fn, fp, tn = (w * (pos & pr)).sum(), (w * (pos & ~pr)).sum(), (w * (~pos & pr)).sum(), (w * (~pos & ~pr)).sum()
        return tp / (tp + fn) if tp + fn else np.nan, tn / (tn + fp), tp / (tp + fp) if tp + fp else np.nan
    est = m(np.ones(len(y))); b = np.array([m(w) for w in W_])
    out['operating_points'][EN[c]] = {'n_pos': int(pos.sum()), 'threshold': t,
        **{k: {'est': float(est[i]), 'lo': float(np.nanpercentile(b[:, i], 2.5)), 'hi': float(np.nanpercentile(b[:, i], 97.5))} for i, k in enumerate(('sens', 'spec', 'ppv'))}}
for s in EXT:
    msk = site == s
    out['counts_by_institution'][s] = {'images': int(msk.sum()), 'groups': int(len(np.unique(group[msk]))), 'positives': {EN[c]: int((y[msk] == c).sum()) for c in range(11)}}
json.dump(out, open(RES / 'operating_point_ci.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False)
for k, v in out['operating_points'].items():
    print(f"{k:24s} n={v['n_pos']:5d} " + '  '.join(f"{m} {100*v[m]['est']:.0f} ({100*v[m]['lo']:.0f}-{100*v[m]['hi']:.0f})" for m in ('sens', 'spec', 'ppv')))
