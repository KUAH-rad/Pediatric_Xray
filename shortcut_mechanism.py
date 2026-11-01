"""Post hoc site-shortcut mechanism analysis (post hoc). No GPU.
Among truly normal images, does the predicted probability of each finding follow that hospital's training prevalence?
Reads predictions only. The mixed-site comparator predictions are read from COHORT_DIR (earlier random-split run)."""
import csv, json, sys
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import config; WORK = config.WORK_DIR; OUT = WORK / 'shortcut_mechanism'; OUT.mkdir(exist_ok=True)
OLD = config.COHORT_DIR
EN = ['Normal', 'Pneumoperitoneum', 'Abdominal mass', 'Intussusception', 'Air-fluid level', 'Foreign body', 'Constipation',
      'Calcified lesion', 'Splenomegaly', 'Hepatomegaly', 'Situs inversus totalis']
SITES = ['01', '02', '03', '04', '06', '08', '09']
rng = np.random.default_rng(2026)

def read(path):
    with open(path, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def prevalence(rows):
    out = {}
    for s in SITES:
        lab = [int(r['folder_label'][1:]) for r in rows if r['patient_id'][:2] == s]
        if lab: out[s] = np.bincount(lab, minlength=11) / len(lab)
    return out

def analyse(name, pred_rows, train_prev, note):
    normals = [r for r in pred_rows if r['folder_label'] == 'D00' and r['patient_id'][:2] in train_prev]
    site = np.array([r['patient_id'][:2] for r in normals]); P = np.array([[float(r[f'prob_D{c:02d}']) for c in range(11)] for r in normals])
    sites = [s for s in SITES if (site == s).sum() >= 30]
    def stat(site_labels):
        Y = np.array([P[site_labels == s, 1:].mean(0) for s in sites]); X = np.array([train_prev[s][1:] for s in sites])
        Y = Y - Y.mean(0); X = X - X.mean(0)
        return spearmanr(X.ravel(), Y.ravel()).statistic, Y, X
    rho, Y, X = stat(site)
    rho_no_situs = float(spearmanr(X[:, :9].ravel(), Y[:, :9].ravel()).statistic)  # sensitivity: situs inversus excluded
    perm = np.array([stat(rng.permutation(site))[0] for _ in range(10000)])
    points = [[float(X[i, c]), float(Y[i, c]), sites[i], EN[c + 1]] for i in range(len(sites)) for c in range(10)]
    p = float((np.sum(perm >= rho) + 1) / (len(perm) + 1))
    # Normal images predicted abnormal: does the predicted label match the hospital's most common abnormal training label?
    pred = P.argmax(1); fp = pred != 0
    top = {s: int(np.argmax(train_prev[s][1:]) + 1) for s in sites}
    in_sites = np.isin(site, sites); match = np.array([site[i] in top and pred[i] == top[site[i]] for i in range(len(pred))])
    pooled = Counter(pred[fp & in_sites]); n_fp = int((fp & in_sites).sum())
    expected = float(np.mean([pooled[top[site[i]]] / n_fp for i in np.where(fp & in_sites)[0]])) if n_fp else None
    res = {'model': name, 'note': note, 'n_normal_images': int(in_sites.sum()), 'sites': sites, 'points': points,
           'spearman_rho_centered': float(rho), 'spearman_rho_no_situs': rho_no_situs,'permutation_p_one_sided': p,
           'false_positive_n': n_fp, 'fp_matching_site_top_label': float(match[fp & in_sites].mean()) if n_fp else None,
           'fp_matching_expected_if_site_independent': expected,
           'site_top_abnormal_label': {s: EN[top[s]] for s in sites},
           'normal_mean_prob_centered': {s: {EN[c + 1]: round(float(Y[i, c]), 4) for c in range(10)} for i, s in enumerate(sites)}}
    print(name, 'rho=%.3f p=%.4f fp_match=%s expected=%s' % (rho, p, res['fp_matching_site_top_label'], expected))
    return res

def ensemble(dirs, split):
    rows = [read(d / f'{split}_predictions.csv') for d in dirs]
    for r in rows[1:]: assert [a['image_id'] for a in r] == [a['image_id'] for a in rows[0]]
    out = []
    for i, base in enumerate(rows[0]):
        rec = dict(base)
        for c in range(11): rec[f'prob_D{c:02d}'] = str(np.mean([float(r[i][f'prob_D{c:02d}']) for r in rows]))
        out.append(rec)
    return out

results = []
# 1. Mixed random split (all hospitals in training), 2026-10-22 DINOv3; prevalence = each hospital's training rows.
old = read(OLD / 'manifest.csv')
results.append(analyse('mixed_random_split_dinov3', read(OLD / 'dinov3_last2blocks_finetune_seed2026' / 'test_predictions.csv'),
                       prevalence([r for r in old if r['split'] == 'train']), 'All hospitals in training; site cues learnable.'))
# 2. Development institution only (main run, 3-seed ensemble); external normals vs each hospital's own label mix (never seen).
new = read(WORK / 'manifest.csv')
ku = ensemble([WORK / f'dinov3_last2blocks_finetune_seed{s}' for s in (2026, 2027, 2028)], 'test')
results.append(analyse('development_institution_only_dinov3', ku, prevalence([r for r in new if r['split'] in ('train', 'validation', 'internal_test', 'test')]),
                       'Negative control: external hospitals never seen in training.'))
# 3. Leave-one-site-out multi-institution models (A1), pooled over held-out folds when available.
ab = WORK / 'ablations'
import run_ablations as A   # loso_parts only
# Within each fold, validation images come from the institutions seen in training: site cues are learnable there.
for d in sorted(ab.glob('A1_loso_heldout*')):
    held = d.name[-2:]
    val = [r for r in read(d / 'validation_predictions.csv')]
    train_rows = A.loso_parts(new, held)['train']   # v2: proportions from the fold's actual training partition
    results.append(analyse(f'multi_institution_heldout{held}_validation', val, prevalence(train_rows), f'Multi-institution model (institution {held} held out); validation images from training institutions.'))
# 4. Same folds retrained with institution-balanced label sampling (run_site_balanced.py, post hoc mitigation, 2026-10-27).
for d in sorted((WORK / 'site_balanced').glob('SB_loso_sitebalanced_heldout*')):
    if not (d / 'validation_predictions.csv').exists(): continue
    held = d.name[-2:]
    train_rows = A.loso_parts(new, held)['train']
    results.append(analyse(f'site_balanced_heldout{held}_validation', read(d / 'validation_predictions.csv'), prevalence(train_rows), f'Site-balanced sampling (institution {held} held out); validation images from training institutions.'))
(OUT / 'shortcut_mechanism_v2.json').write_text(json.dumps(results, indent=1, ensure_ascii=False), encoding='utf-8')
