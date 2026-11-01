"""Final statistics for the manuscript. No GPU. Reads saved predictions/metrics only; writes ../results.
Primary model: DINOv3 ViT-S/16 partial fine-tuning, 3-seed ensemble (mean of temperature-scaled probabilities).
Comparator: ConvNeXt V2 Atto, 3-seed ensemble. CIs: patient-image-group cluster bootstrap (1,000 resamples, seed 2026)."""
import csv, json
from pathlib import Path
from collections import Counter, defaultdict
import numpy as np
from sklearn.metrics import roc_auc_score
from scipy.stats import binomtest

import config
W = config.WORK_DIR; OUT = config.RESULTS_DIR; OUT.mkdir(parents=True, exist_ok=True)
EN = ['Normal', 'Pneumoperitoneum', 'Abdominal mass', 'Intussusception', 'Air-fluid level', 'Foreign body', 'Constipation',
      'Calcified lesion', 'Splenomegaly', 'Hepatomegaly', 'Situs inversus totalis']
SITE = {'01': 'A', '02': 'B', '03': 'C', '04': 'D', '06': 'E', '08': 'F', '09': 'G'}  # anonymized institution letters
EXT = ['02', '03', '04', '06', '08', '09']
B = 1000; rng = np.random.default_rng(2026)
TURING = config.READER_STUDY_DIR

def read(p):
    with open(p, encoding='utf-8-sig', newline='') as f: return list(csv.DictReader(f))

def load(dirs, split):
    rs = [read(d / f'{split}_predictions.csv') for d in dirs]
    for r in rs[1:]: assert [a['image_id'] for a in r] == [a['image_id'] for a in rs[0]]
    P = np.mean([[[float(a[f'prob_D{c:02d}']) for c in range(11)] for a in r] for r in rs], 0)
    base = rs[0]
    return {'id': np.array([a['image_id'] for a in base]), 'site': np.array([a['site'] for a in base]), 'group': np.array([a['split_group'] for a in base]),
            'y': np.array([int(a['folder_label'][1:]) for a in base]), 'p': P}

# Bootstrap by integer case weights (multiplicity of each image in a cluster resample); equivalent to row resampling
# and much faster because each score column is sorted once. Tie-aware Mann-Whitney AUROC.
_CACHE = {}
def _prep(p, c):
    k = (id(p), c)
    if k not in _CACHE:
        o = np.argsort(p[:, c], kind='mergesort'); _, inv = np.unique(p[o, c], return_inverse=True); _CACHE[k] = (o, inv, p)
    return _CACHE[k][:2]
def auc_c(y, p, c, w=None):
    o, inv = _prep(p, c); pos = y[o] == c; ww = np.ones(len(o)) if w is None else w[o]
    negw = ww * ~pos; posw = ww * pos; Wp, Wn = posw.sum(), negw.sum()
    if Wp == 0 or Wn == 0: raise ValueError('one class')
    g = np.bincount(inv, negw); below = np.cumsum(g) - g
    return float((posw * (below[inv] + .5 * g[inv])).sum() / (Wp * Wn))
def macro_auc(y, p, w=None):
    ww = np.ones(len(y)) if w is None else w
    return float(np.mean([auc_c(y, p, c, w) for c in range(11) if 0 < ww[y == c].sum() < ww.sum()]))
def macro_sens(y, p, w=None):
    pr = p.argmax(1); ww = np.ones(len(y)) if w is None else w
    return float(np.mean([ww[(y == c) & (pr == c)].sum() / ww[y == c].sum() for c in range(11) if ww[y == c].sum() > 0]))
def acc(y, p, w=None):
    ww = np.ones(len(y)) if w is None else w; return float((ww * (p.argmax(1) == y)).sum() / ww.sum())
def ece(y, p, bins=15):
    conf = p.max(1); ok = p.argmax(1) == y; e = 0
    for lo, hi in zip(np.linspace(0, 1, bins + 1)[:-1], np.linspace(0, 1, bins + 1)[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any(): e += m.mean() * abs(ok[m].mean() - conf[m].mean())
    return float(e)

def clusters(groups):
    idx = defaultdict(list)
    for i, g in enumerate(groups): idx[g].append(i)
    return [np.array(v) for v in idx.values()]

def boot(d, fns, others=None):
    """fns: name -> f(y, p). others: list of extra prediction arrays aligned with d (paired). Returns estimate + CI per name."""
    gid = np.unique(d['group'], return_inverse=True)[1]; K = int(gid.max()) + 1; res = {k: [] for k in fns}
    for _ in range(B):
        w = np.bincount(rng.integers(0, K, K), minlength=K)[gid].astype(float)
        for k, f in fns.items():
            try: res[k].append(f(d['y'], d['p'], *(others or []), w=w))
            except (ValueError, ZeroDivisionError): pass
    out = {}
    for k, f in fns.items():
        est = f(d['y'], d['p'], *(others or [])); lo, hi = np.percentile(res[k], [2.5, 97.5]); v = np.array(res[k])
        out[k] = {'est': est, 'lo': float(lo), 'hi': float(hi), 'p_two_sided_vs_0': float(min(1, 2 * min(((v <= 0).sum() + 1) / (len(v) + 1), ((v >= 0).sum() + 1) / (len(v) + 1))))}
    return out

def sub(d, m): return {k: (v[m] if isinstance(v, np.ndarray) else v) for k, v in d.items()}

R = {}
DIN = [W / f'dinov3_last2blocks_finetune_seed{s}' for s in (2026, 2027, 2028)]
CNX = [W / f'convnextv2_atto_finetune_seed{s}' for s in (2026, 2027, 2028)]
din = {s: load(DIN, s) for s in ('validation', 'internal_test', 'test', 'turing363')}
cnx = {s: load(CNX, s) for s in ('validation', 'internal_test', 'test', 'turing363')}

# ---------- Table 2: overall performance
def overall(d):
    b = boot(d, {'macro_auc': macro_auc, 'macro_sens': macro_sens, 'acc': acc}); b['ece'] = ece(d['y'], d['p']); return b
R['overall'] = {}
for name, dirs in [('DINOv3 ViT-S/16 (partial fine-tuning)', DIN), ('ConvNeXt V2 Atto (full fine-tuning)', CNX),
                   ('DINOv3 frozen linear probe, weighted', [W / 'transformer_linear_global_balanced_seed2026']),
                   ('DINOv3 frozen linear probe, unweighted', [W / 'transformer_linear_global_unweighted_seed2026']),
                   ('Metadata only (weight group, weight, sex)', [W / 'metadata_only_negative_control'])]:
    R['overall'][name] = {s: overall(load(dirs, s)) for s in ('internal_test', 'test')}
    print(name, {s: round(v['macro_auc']['est'], 3) for s, v in R['overall'][name].items()})
# seed-level spread
R['seed_spread'] = {nm: {s: [json.load(open(d / 'metrics.json', encoding='utf-8'))[s]['macro_auc'] for d in dirs] for s in ('internal_test', 'test')} for nm, dirs in [('DINOv3', DIN), ('ConvNeXt', CNX)]}
# primary comparison: external macro AUROC, DINOv3 minus ConvNeXt (paired)
assert (din['test']['id'] == cnx['test']['id']).all()
R['primary_comparison'] = boot(din['test'], {'diff_macro_auc': lambda y, p, q, w=None: macro_auc(y, p, w) - macro_auc(y, q, w),
                                             'diff_macro_sens': lambda y, p, q, w=None: macro_sens(y, p, w) - macro_sens(y, q, w)}, [cnx['test']['p']])
# internal to external drop (unpaired, separate bootstraps)
R['internal_external_drop'] = {'dinov3_internal': R['overall']['DINOv3 ViT-S/16 (partial fine-tuning)']['internal_test']['macro_auc'],
                               'dinov3_external': R['overall']['DINOv3 ViT-S/16 (partial fine-tuning)']['test']['macro_auc']}
print('primary', R['primary_comparison'])
# Sensitivity analysis (added 2026-10-27): situs inversus excluded (orientation-dependent label); 10-label macro AUROC on non-situs images.
def macro_auc10(y, p, w=None):
    ww = np.ones(len(y)) if w is None else w
    return float(np.mean([auc_c(y, p, c, w) for c in range(10) if 0 < ww[y == c].sum() < ww.sum()]))
ns = {k: {s: sub(d[s], d[s]['y'] != 10) for s in ('internal_test', 'test')} for k, d in (('din', din), ('cnx', cnx))}
R['sensitivity_no_situs'] = {'dinov3': {s: boot(ns['din'][s], {'auc': macro_auc10})['auc'] for s in ('internal_test', 'test')},
                             'convnext': {s: boot(ns['cnx'][s], {'auc': macro_auc10})['auc'] for s in ('internal_test', 'test')},
                             'diff_dinov3_minus_convnext_external': boot(ns['din']['test'], {'d': lambda y, p, q, w=None: macro_auc10(y, p, w) - macro_auc10(y, q, w)}, [ns['cnx']['test']['p']])['d']}
print('no situs', {k: (v if k.startswith('diff') else {s: round(x['est'], 3) for s, x in v.items()}) for k, v in R['sensitivity_no_situs'].items()})

# ---------- per-site (Figure 2)
R['per_site'] = {}
for s in EXT:
    m = din['test']['site'] == s
    R['per_site'][s] = {'n': int(m.sum()), 'dinov3': boot(sub(din['test'], m), {'macro_auc': macro_auc, 'macro_sens': macro_sens}),
                        'convnext': boot(sub(cnx['test'], m), {'macro_auc': macro_auc, 'macro_sens': macro_sens}),
                        'labels_present': [EN[c] for c in range(11) if (din['test']['y'][m] == c).any()]}

# ---------- per-finding (Table 3, Figure 3) with validation-fixed operating points
def op_thresholds(c):
    yv = din['validation']['y'] == c; pv = din['validation']['p'][:, c]
    ts = np.unique(pv); sens = np.array([(pv[yv] >= t).mean() for t in ts]); spec = np.array([(pv[~yv] < t).mean() for t in ts])
    youden = float(ts[np.argmax(sens + spec - 1)]); s90 = float(ts[np.where(sens >= .9)[0].max()]) if (sens >= .9).any() else float(ts.min())
    return {'youden': youden, 'sens90': s90}
def opm(y, p, t):
    pos = y; pr = p >= t; tp = (pos & pr).sum(); fn = (pos & ~pr).sum(); fp = (~pos & pr).sum(); tn = (~pos & ~pr).sum()
    f = lambda a, b: float(a / (a + b)) if a + b else None
    return {'sens': f(tp, fn), 'spec': f(tn, fp), 'ppv': f(tp, fp), 'npv': f(tn, fn), 'tp': int(tp), 'fn': int(fn), 'fp': int(fp), 'tn': int(tn)}
occ = json.load(open(W / 'cam_flip' / 'occlusion_localization_summary.json', encoding='utf-8'))
KO = ['정상', '기복증', '종괴', '장중첩증', '공기-액체음영', '이물', '변비', '석회화병변', '비장비대', '간비대', '전내장역위증']
R['per_finding'] = {}
for c in range(11):
    th = op_thresholds(c); row = {'thresholds': th}
    for s in ('internal_test', 'test'):
        d = din[s]
        b = boot(d, {'auc': lambda y, p, w=None, c=c: auc_c(y, p, c, w)})
        row[s] = {'n_pos': int((d['y'] == c).sum()), 'n': int(len(d['y'])), 'auc': b['auc'],
                  **{k: opm(d['y'] == c, d['p'][:, c], t) for k, t in th.items()}}
    row['ext_auc_by_site'] = {s: (float(roc_auc_score(din['test']['y'][din['test']['site'] == s] == c, din['test']['p'][din['test']['site'] == s, c]))
                                  if 0 < (din['test']['y'][din['test']['site'] == s] == c).sum() < (din['test']['site'] == s).sum() else None) for s in EXT}
    if KO[c] in occ: row['occlusion'] = occ[KO[c]]
    R['per_finding'][EN[c]] = row
# normal triage: abnormal-vs-normal at 95% abnormal sensitivity (validation-fixed)
pa_v = 1 - din['validation']['p'][:, 0]; ab_v = din['validation']['y'] != 0
t95 = float(np.sort(pa_v[ab_v])[int(np.floor(.05 * ab_v.sum()))])
R['triage'] = {}
for s in ('internal_test', 'test'):
    d = din[s]; pa = 1 - d['p'][:, 0]; ab = d['y'] != 0
    R['triage'][s] = {'threshold': t95, 'abnormal_sensitivity': float((pa[ab] >= t95).mean()), 'normal_specificity': float((pa[~ab] < t95).mean()),
                      'fraction_flagged_normal': float((pa < t95).mean()), 'npv': float((~ab & (pa < t95)).sum() / max((pa < t95).sum(), 1)),
                      'auc_abnormal_vs_normal': float(roc_auc_score(ab, pa))}

# ---------- readers (Table 4, Figure 4)
key = {}
import pandas as pd
k = pd.read_excel(TURING / config.READER_KEY_FILE)
num = dict(zip(k['Original Filename'].str.replace('.png', '', regex=False), k['Renamed Filename'].str[:3].astype(int)))
t = din['turing363']; y = t['y']
readers = {}
for f, who in config.READER_FILES:
    x = pd.read_excel(TURING / f, header=None).iloc[2:]; x.index = x[0].astype(str).str[:3].astype(int)
    a = np.array([pd.to_numeric(x.loc[num[i], [1, 2, 3]], errors='coerce').values for i in t['id']])
    readers[who] = a
order = np.argsort(-t['p'], 1)
def kappa(a, b):
    po = (a == b).mean(); pe = sum((a == i).mean() * (b == i).mean() for i in range(11)); return float((po - pe) / (1 - pe))
ai_ok = order[:, 0] == y
R['readers'] = {'n': int(len(y)), 'AI (DINOv3 ensemble)': {'top1': float(ai_ok.mean()), 'top2': float((order[:, :2] == y[:, None]).any(1).mean()),
                                                           'macro_sens': macro_sens(y, t['p']), 'per_label_sens': [float((order[y == c, 0] == c).mean()) for c in range(11)],
                                                           'macro_auc': macro_auc(y, t['p'])}}
for who, a in readers.items():
    ok = a[:, 0] == y; ok2 = np.array([yy in aa[:2] for yy, aa in zip(y, a)])
    b01 = int((ai_ok & ~ok).sum()); b10 = int((~ai_ok & ok).sum())
    R['readers'][who] = {'top1': float(ok.mean()), 'top2': float(ok2.mean()), 'n_second_choice': int(np.isfinite(a[:, 1]).sum()),
                         'per_label_sens': [float((a[y == c, 0] == c).mean()) for c in range(11)],
                         'mcnemar_vs_ai': {'ai_only_correct': b01, 'reader_only_correct': b10, 'p': float(binomtest(b01, b01 + b10, .5).pvalue) if b01 + b10 else 1.0}}
names = list(readers)
R['readers']['kappa'] = {f'{a} vs {b}': kappa(readers[a][:, 0], readers[b][:, 0]) for i, a in enumerate(names) for b in names[i + 1:]}
R['readers']['kappa'].update({f'AI vs {a}': kappa(order[:, 0], readers[a][:, 0]) for a in names})
spec = readers['Pediatric radiologist']; flipped_note = 'Radiologist commented "image flipped" on 24 situs inversus cases (scored by recorded first answer).'
both = (order[:, 0] != y) & (order[:, 1] == y) & np.isfinite(spec[:, 1])
R['readers']['ai_error_truth2nd_with_radiologist_two_answers'] = {'n': int(both.sum()), 'radiologist_listed_ai_top1': float(np.mean([order[i, 0] in spec[i, :2] for i in np.where(both)[0]])) if both.any() else None, 'note': flipped_note}

# ---------- ablations
def ext_auc(dirpath, split='test', mask_site=None):
    d = load([dirpath], split)
    if mask_site: d = sub(d, d['site'] == mask_site)
    return d
ref = load([DIN[0]], 'test')
# Ablations are single-seed (2026): compare with the single-institution seed-2026 model, not the 3-seed ensemble.
R['ablation_site'] = {}
for s in EXT:
    ku = sub(ref, ref['site'] == s)
    for nm, folder in [('multi_institution', 'ablations/A1_loso_heldout' + s), ('multi_institution_size_matched', 'ablations/A1m_loso_sizematched_heldout' + s),
                       ('site_balanced', 'site_balanced/SB_loso_sitebalanced_heldout' + s)]:
        if not (W / folder / 'test_predictions.csv').exists(): continue
        a = load([W / folder], 'test'); order_ix = {v: i for i, v in enumerate(a['id'])}; ix = np.array([order_ix[v] for v in ku['id']])
        R['ablation_site'].setdefault(s, {'single_institution_seed2026': boot(ku, {'auc': macro_auc})['auc']})[nm] = boot(ku, {'auc': lambda y, p, q, w=None: macro_auc(y, q, w), 'diff': lambda y, p, q, w=None: macro_auc(y, q, w) - macro_auc(y, p, w)}, [a['p'][ix]])
    if (W / 'site_balanced' / ('SB_loso_sitebalanced_heldout' + s) / 'test_predictions.csv').exists():
        a1 = load([W / 'ablations' / ('A1_loso_heldout' + s)], 'test'); sb = load([W / 'site_balanced' / ('SB_loso_sitebalanced_heldout' + s)], 'test')
        assert (a1['id'] == sb['id']).all()
        R['ablation_site'][s]['site_balanced_minus_multi'] = boot(a1, {'diff': lambda y, p, q, w=None: macro_auc(y, q, w) - macro_auc(y, p, w)}, [sb['p']])['diff']
        R['ablation_site'][s]['ece'] = {'single_institution_seed2026': ece(ku['y'], ku['p']), 'multi_institution': ece(a1['y'], a1['p']), 'site_balanced': ece(sb['y'], sb['p'])}
R['ablation_other'] = {}
for folder, label in [('ablations/A2_no_label_cleansing', 'A2 No label curation'), ('ablations/A3_last4blocks', 'A3 Last 4 blocks fine-tuned'),
                      ('ablations/A3_full_finetune', 'A3 Full fine-tuning'), ('ablations/A4_unweighted_ce', 'A4 Unweighted loss'),
                      ('ablations/A5_longer_training_30ep', 'A5 Up to 30 epochs'), ('ablations_bc/B1_dinov3_448', 'B1 448 x 448 input'),
                      ('ablations_bc/B2_dinov3_clahe', 'B2 CLAHE preprocessing'), ('ablations_bc/C1_resnet50', 'C1 ResNet-50'),
                      ('ablations_bc/C2_efficientnet_b0', 'C2 EfficientNet-B0'), ('ablations_bc/C3_dinov2_vits14_reg', 'C3 DINOv2 ViT-S/14'),
                      ('ablations_bc/C4_dinov3_vitb16', 'C4 DINOv3 ViT-B/16')]:
    a = load([W / folder], 'test'); assert (a['id'] == ref['id']).all()
    ia = load([W / folder], 'internal_test')
    R['ablation_other'][label] = {'internal_auc': macro_auc(ia['y'], ia['p']), **boot(ref, {'external_auc': lambda y, p, q, w=None: macro_auc(y, q, w), 'diff_vs_main_seed2026': lambda y, p, q, w=None: macro_auc(y, q, w) - macro_auc(y, p, w)}, [a['p']])}
    print(label, round(R['ablation_other'][label]['external_auc']['est'], 3))
R['ablation_reference_main_seed2026'] = {'external_auc': macro_auc(ref['y'], ref['p'])}

# ---------- mechanism, flip, site probe, top-k, Table 1
R['shortcut_mechanism'] = json.load(open(W / 'shortcut_mechanism' / 'shortcut_mechanism.json', encoding='utf-8'))
R['site_probe'] = json.load(open(W / 'site_shortcut_probe' / 'metrics.json', encoding='utf-8'))
R['flip_test'] = json.load(open(W / 'cam_flip' / 'flip_test.json', encoding='utf-8'))
R['saliency_methods'] = json.load(open(W / 'cam_flip' / 'saliency_method_comparison.json', encoding='utf-8'))
man = read(W / 'manifest.csv'); audit = json.load(open(W / 'site_split_audit.json', encoding='utf-8'))
t1 = {}
for s in ['01'] + EXT:
    rr = [r for r in man if r['site'] == s and r['split'] in ('train', 'validation', 'internal_test', 'test')]
    w = [float(r['body_weight']) for r in rr if r['body_weight'] not in ('', 'None') and 0.3 < float(r['body_weight']) <= 150]
    t1[s] = {'images': len(rr), 'groups': len({r['split_group'] for r in rr}), 'female': sum(r['gender'] == 'F' for r in rr), 'male': sum(r['gender'] == 'M' for r in rr),
             'weight_median_iqr': [float(np.median(w)), float(np.percentile(w, 25)), float(np.percentile(w, 75))] if w else None,
             'weight_group': dict(Counter(r['batch'] for r in rr)), 'labels': [sum(r['folder_label'] == f'D{c:02d}' for r in rr) for c in range(11)]}
R['table1'] = t1; R['split_audit'] = audit['partitions']
json.dump(R, open(OUT / 'all_results.json', 'w', encoding='utf-8'), indent=1, ensure_ascii=False, default=float)
print('saved', OUT / 'all_results.json')
