"""Robustness of the normal-label institutional association (post hoc). No GPU.
1. Institutional label proportions from the actual training partition of each fold (not training + validation pool).
2. Permutation of institution at the patient-image group level (groups keep their images together).
3. Patient-image group bootstrap (stratified by institution) for rho, and paired bootstrap for delta-rho (balanced - unbalanced, same images).
4. Leave-one-institution-out rho (direction check) and number of labeled-normal images per institution.
5. Seed replicates (experiments/seed_replicates) when available.
Reads predictions only."""
import csv, json, sys
from pathlib import Path
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import config; WORK = config.WORK_DIR; OUT = WORK / 'shortcut_mechanism'
OLD = config.COHORT_DIR
SITES = ['01', '02', '03', '04', '06', '08', '09']; EXT = SITES[1:]
NB, NP = 1000, 2000
rng = np.random.default_rng(20261027)

def read(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def prev(rows):
    out = {}
    for s in SITES:
        lab = [int(r['folder_label'][1:]) for r in rows if r['patient_id'][:2] == s]
        if lab: out[s] = np.bincount(lab, minlength=11) / len(lab)
    return out

class Normals:
    """Labeled-normal images of one experiment: probabilities, institution and patient-image group."""
    def __init__(self, pred_rows, train_prev, min_n=30):
        rows = [r for r in pred_rows if r['folder_label'] == 'D00' and r['patient_id'][:2] in train_prev]
        site = np.array([r['patient_id'][:2] for r in rows])
        self.sites = [s for s in SITES if (site == s).sum() >= min_n]
        keep = np.isin(site, self.sites); rows = [r for r, k in zip(rows, keep) if k]
        self.site = site[keep]; self.P = np.array([[float(r[f'prob_D{c:02d}']) for c in range(1, 11)] for r in rows])
        self.group = np.array([r.get('split_group') or r['patient_id'] for r in rows])
        self.X = np.array([train_prev[s][1:] for s in self.sites]); self.n = {s: int((self.site == s).sum()) for s in self.sites}
        # group index per institution for cluster resampling
        self.gidx = {s: [np.where((self.group == g) & (self.site == s))[0] for g in np.unique(self.group[self.site == s])] for s in self.sites}
    def rho(self, idx=None, site=None, sites=None):
        site = self.site if site is None else site; sites = sites or self.sites
        idx = np.arange(len(site)) if idx is None else idx
        Y = np.array([self.P[idx][site[idx] == s].mean(0) for s in sites]); X = np.array([self.X[self.sites.index(s)] for s in sites])
        return float(spearmanr((X - X.mean(0)).ravel(), (Y - Y.mean(0)).ravel()).statistic)
    def boot_idx(self, r):
        return np.concatenate([np.concatenate([g[i] for i in r.integers(0, len(g), len(g))]) for g in self.gidx.values()])
    def group_perm_p(self, obs):
        groups = np.unique(self.group); gsite = {g: self.site[self.group == g][0] for g in groups}
        labels = np.array([gsite[g] for g in groups]); pos = {g: i for i, g in enumerate(groups)}; gi = np.array([pos[g] for g in self.group])
        perm = [self.rho(site=rng.permutation(labels)[gi]) for _ in range(NP)]
        return float((np.sum(np.array(perm) >= obs) + 1) / (NP + 1))
    def loo(self):
        return {s: self.rho(sites=[t for t in self.sites if t != s]) for s in self.sites}

def ci(v): return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]

def summarize(nm, train_prev):
    out = {'n_normal_by_institution': nm.n, 'rho': nm.rho()}
    out['group_permutation_p'] = nm.group_perm_p(out['rho'])
    b = [nm.rho(nm.boot_idx(rng)) for _ in range(NB)]; out['rho_ci'] = ci(b)
    lo = nm.loo(); out['leave_one_institution_out'] = lo; out['loo_range'] = [min(lo.values()), max(lo.values())]
    return out

man = read(WORK / 'manifest.csv'); man_by_id = {r['image_id']: r for r in man}
import run_ablations as A   # loso_parts only (no training)
R = {}
# Mixed random-split model and single-institution comparator (unchanged definitions)
old = read(OLD / 'manifest.csv')
nm = Normals(read(OLD / 'dinov3_last2blocks_finetune_seed2026' / 'test_predictions.csv'), prev([r for r in old if r['split'] == 'train']))
R['mixed_random_split'] = summarize(nm, None)
def ens(dirs, split):
    rows = [read(d / f'{split}_predictions.csv') for d in dirs]; out = []
    for i, base in enumerate(rows[0]):
        rec = dict(base); rec.update({f'prob_D{c:02d}': str(np.mean([float(r[i][f'prob_D{c:02d}']) for r in rows])) for c in range(11)}); out.append(rec)
    return out
ku = ens([WORK / f'dinov3_last2blocks_finetune_seed{s}' for s in (2026, 2027, 2028)], 'test')
R['single_institution'] = summarize(Normals(ku, prev([r for r in man if r['split'] in ('train', 'validation', 'internal_test', 'test')])), None)

# Multi-institution folds: unbalanced (A1) vs institution-balanced (SB), per seed, with training-partition proportions
runs = {2026: (WORK / 'ablations' / 'A1_loso_heldout{s}', WORK / 'site_balanced' / 'SB_loso_sitebalanced_heldout{s}'),
        2027: (WORK / 'seed_replicates' / 'A1_loso_heldout{s}_seed2027', WORK / 'seed_replicates' / 'SB_loso_sitebalanced_heldout{s}_seed2027'),
        2028: (WORK / 'seed_replicates' / 'A1_loso_heldout{s}_seed2028', WORK / 'seed_replicates' / 'SB_loso_sitebalanced_heldout{s}_seed2028')}
R['folds'] = {}
for seed, (a1, sb) in runs.items():
    for s in EXT:
        da, db = Path(str(a1).format(s=s)), Path(str(sb).format(s=s))
        if not ((da / 'validation_predictions.csv').exists() and (db / 'validation_predictions.csv').exists()): continue
        parts = A.loso_parts(man, s); tp = prev(parts['train']); pp = prev(parts['train'] + parts['validation'])
        na, nb = Normals(read(da / 'validation_predictions.csv'), tp), Normals(read(db / 'validation_predictions.csv'), tp)
        assert list(na.group) == list(nb.group)   # same validation images -> paired
        ra, rb = summarize(na, tp), summarize(nb, tp)
        ra['rho_train_plus_validation_proportions'] = Normals(read(da / 'validation_predictions.csv'), pp).rho()
        d = []
        for _ in range(NB):
            idx = na.boot_idx(rng); d.append(nb.rho(idx) - na.rho(idx))
        R['folds'][f'heldout{s}_seed{seed}'] = {'heldout': s, 'seed': seed, 'unbalanced': ra, 'balanced': rb,
                                               'delta_rho': rb['rho'] - ra['rho'], 'delta_rho_ci': ci(d)}
        print(seed, s, 'rho %.2f -> %.2f  delta %.2f %s  perm p %.4f / %.4f' % (ra['rho'], rb['rho'], rb['rho'] - ra['rho'], np.round(ci(d), 2), ra['group_permutation_p'], rb['group_permutation_p']), flush=True)
# Pooled across folds per seed: mean delta-rho
for seed in runs:
    f = [v for v in R['folds'].values() if v['seed'] == seed]
    if len(f) == 6:
        R.setdefault('mean_delta_rho_by_seed', {})[seed] = float(np.mean([v['delta_rho'] for v in f]))
R['mean_delta_rho_all'] = float(np.mean([v['delta_rho'] for v in R['folds'].values()]))
R['n_fold_runs'] = len(R['folds']); R['n_negative_delta'] = int(sum(v['delta_rho'] < 0 for v in R['folds'].values()))
R['n_delta_ci_below_zero'] = int(sum(v['delta_rho_ci'][1] < 0 for v in R['folds'].values()))
(OUT / 'shortcut_robustness.json').write_text(json.dumps(R, indent=1, ensure_ascii=False, default=float), encoding='utf-8')
print(json.dumps({k: R[k] for k in R if k != 'folds'}, indent=1, default=float))
