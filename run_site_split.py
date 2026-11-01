"""Institution-split training (main models, metadata-only control, frozen-feature linear probes, institution probe).

DINOv3 ViT-S/16 (last two blocks fine-tuned) and ConvNeXt V2 Atto, three training runs (seeds) each, class-weighted
cross-entropy, temperature scaling on validation; predictions are exported for every partition and the reader set.
Validation alone selects checkpoints, temperature and thresholds. Test partitions are not used for any choice.
"""
import os
os.environ.setdefault('HF_HUB_DISABLE_SYMLINKS_WARNING', '1')
os.environ.setdefault('MPLBACKEND', 'Agg')
from pathlib import Path
import csv, json, time, random, hashlib, traceback, platform, sys
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from torchvision import transforms as T
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, f1_score

import config
CODE = Path(__file__).resolve().parent; WORK = config.WORK_DIR; DEVICE = 'cuda'
CACHE = config.CACHE224
SEEDS = [2026, 2027, 2028]
EVAL_SPLITS = ['validation', 'internal_test', 'test', 'reader_reserved']
DRY_RUN = '--dry-run' in sys.argv
torch.set_num_threads(4)

def log(x): print(time.strftime('%Y-%m-%d %H:%M:%S'), x, flush=True)
def save_json(path, obj):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=2, default=str), encoding='utf-8'); tmp.replace(path)
def save_checkpoint(obj, path):
    tmp = path.with_suffix(path.suffix + '.tmp'); torch.save(obj, tmp); tmp.replace(path)
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

class Images(Dataset):
    def __init__(self, rows, training=False):
        self.rows = rows
        aug = [T.RandomRotation(5), T.ColorJitter(brightness=.1, contrast=.1)] if training else []
        self.transform = T.Compose(aug + [T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        with Image.open(r['cache_path']) as im: x = self.transform(im.convert('RGB'))
        return x, int(r['folder_label'][1:]), i

def loader(rows, training=False, batch=32):
    return DataLoader(Images(rows, training), batch_size=batch, shuffle=training, num_workers=0, pin_memory=True)

def metrics(y, p):
    pred = p.argmax(1); cm = confusion_matrix(y, pred, labels=np.arange(11)); sup = cm.sum(1)
    recall = np.divide(cm.diagonal(), sup, out=np.full(11, np.nan), where=sup > 0)
    auc = []; ap = []
    for c in range(11):
        a = (y == c).astype(int)
        auc.append(float(roc_auc_score(a, p[:, c])) if len(np.unique(a)) == 2 else None)
        ap.append(float(average_precision_score(a, p[:, c])) if a.sum() else None)
    return {'n_images': len(y), 'top1_accuracy': float((pred == y).mean()), 'macro_recall': float(np.nanmean(recall)),
            'macro_f1': float(f1_score(y, pred, labels=np.unique(y), average='macro', zero_division=0)),
            'macro_auc': float(np.mean([v for v in auc if v is not None])),
            'class_auc': auc, 'class_auprc': ap, 'class_recall': recall.tolist(), 'class_support': sup.tolist(),
            'confusion_matrix': cm.tolist(), 'target_note': 'Recorded representative folder label; classes absent from a partition are excluded from macro averages.'}

@torch.inference_mode()
def predict(model, rows, batch=32):
    model.eval(); ys = []; ps = []
    for x, y, _ in loader(rows, batch=batch):
        with torch.autocast('cuda', dtype=torch.float16): z = model(x.to(DEVICE, non_blocking=True))
        ps.append(z.float().cpu()); ys.append(y)
    return torch.cat(ys).numpy(), torch.cat(ps).numpy()

def calibrate_temperature(logits, y):
    ts = np.geomspace(.5, 5, 81); z = torch.tensor(logits); g = torch.tensor(y, dtype=torch.long)
    losses = [float(nn.functional.cross_entropy(z / float(t), g)) for t in ts]
    return float(ts[int(np.argmin(losses))])

FIELDS = ['image_id', 'patient_id', 'split_group', 'site', 'batch', 'folder_label']
def export_predictions(path, rows, logits, temperature):
    probs = torch.tensor(logits / temperature).softmax(1).numpy()
    fields = FIELDS + ['pred_label', 'confidence', 'entropy'] + [f'prob_D{c:02d}' for c in range(11)]
    with path.open('w', encoding='utf-8-sig', newline='') as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r, p in zip(rows, probs):
            rec = {k: r.get(k, '') for k in FIELDS}; rec.update(pred_label=f'D{p.argmax():02d}', confidence=float(p.max()), entropy=float(-(p * np.log(p + 1e-12)).sum()))
            rec.update({f'prob_D{c:02d}': float(p[c]) for c in range(11)}); w.writerow(rec)
    return probs

def finish_model(name, parts, logits, extra):
    """parts/logits: dicts keyed by split name (validation, internal_test, test, reader_reserved, turing363)."""
    target = WORK / name; target.mkdir(exist_ok=True)
    yv = np.array([int(r['folder_label'][1:]) for r in parts['validation']])
    temp = calibrate_temperature(logits['validation'], yv)
    summary = {'temperature': temp, **extra}
    for s, z in logits.items():
        if z is None or not len(parts[s]): continue
        p = export_predictions(target / f'{s}_predictions.csv', parts[s], z, temp)
        if s in ('validation', 'internal_test', 'test', 'turing363'):
            summary[s] = metrics(np.array([int(r['folder_label'][1:]) for r in parts[s]]), p)
    ext = parts['test']; site = np.array([r['site'] for r in ext]); yt = np.array([int(r['folder_label'][1:]) for r in ext])
    pt = torch.tensor(logits['test'] / temp).softmax(1).numpy()
    summary['test_by_site'] = {s: metrics(yt[site == s], pt[site == s]) for s in sorted(set(site))}
    save_json(target / 'metrics.json', summary)
    log(name + ' RESULT ' + json.dumps({s: round(summary[s]['macro_auc'], 4) for s in ('internal_test', 'test', 'turing363') if s in summary}))

def train_classifier(parts, transformer, seed):
    model_id = 'vit_small_patch16_dinov3.lvd1689m' if transformer else 'convnextv2_atto.fcmae_ft_in1k'
    name = ('dinov3_last2blocks_finetune_seed' if transformer else 'convnextv2_atto_finetune_seed') + str(seed); target = WORK / name; target.mkdir(exist_ok=True)
    if (target / 'metrics.json').exists(): log('skip completed ' + name); return
    seed_all(seed); kwargs = {'img_size': 224} if transformer else {}
    model = timm.create_model(model_id, pretrained=True, num_classes=11, **kwargs).to(DEVICE)
    if transformer:
        for p in model.parameters(): p.requires_grad = False
        for block in model.blocks[-2:]:
            for p in block.parameters(): p.requires_grad = True
        for module in [model.norm, model.get_classifier()]:
            for p in module.parameters(): p.requires_grad = True
    tr, va = parts['train'], parts['validation']
    counts = np.bincount([int(r['folder_label'][1:]) for r in tr], minlength=11)
    weights = 1 / np.sqrt(counts); weights = weights / weights.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=DEVICE))
    if transformer:
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': 1e-5}, {'params': model.get_classifier().parameters(), 'lr': 1e-4}]
    else: groups = [{'params': model.parameters(), 'lr': 1e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=.05); scaler = torch.amp.GradScaler('cuda')
    best = -1; start = 0; patience = 0; history = []
    if (target / 'last.pt').exists():
        ck = torch.load(target / 'last.pt', map_location=DEVICE, weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start = ck['epoch'] + 1; best = ck['best']; patience = ck['patience']; history = ck['history']
    for epoch in range(start, 8 if transformer else 12):
        model.train(); loss_sum = 0; n = 0; t0 = time.time()
        for step, (x, y, _) in enumerate(loader(tr, True, batch=16)):
            x = x.to(DEVICE, non_blocking=True); y = y.to(DEVICE, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16): z = model(x); loss = criterion(z, y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1); scaler.step(opt); scaler.update()
            loss_sum += float(loss.detach()) * len(y); n += len(y)
            if step % 100 == 0: log(f'{name} epoch={epoch+1} step={step} loss={loss_sum/n:.4f}')
        yv, zv = predict(model, va); m = metrics(yv, torch.tensor(zv).softmax(1).numpy())
        history.append({'epoch': epoch + 1, 'loss': loss_sum / n, 'validation_macro_auc': m['macro_auc'], 'validation_macro_recall': m['macro_recall'], 'seconds': time.time() - t0})
        if m['macro_auc'] > best: best = m['macro_auc']; patience = 0; save_checkpoint(model.state_dict(), target / 'best.pt')
        else: patience += 1
        save_checkpoint({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'patience': patience, 'history': history}, target / 'last.pt')
        save_json(target / 'training_history.json', history); log(history[-1])
        if patience >= (3 if transformer else 4): break
    model.load_state_dict(torch.load(target / 'best.pt', map_location=DEVICE, weights_only=True))
    logits = {s: predict(model, parts[s])[1] for s in EVAL_SPLITS + ['turing363'] if len(parts[s])}
    finish_model(name, parts, logits, {'encoder': model_id, 'seed': seed, 'validation_selection': 'macro AUROC', 'epochs_completed': len(history),
                                       'training': 'last two transformer blocks, final norm and head' if transformer else 'supervised full fine-tuning'})
    del model; torch.cuda.empty_cache()

def linear_probes(parts, rows):
    """Frozen DINOv3 CLS linear probes on fixed, label-independent features (secondary)."""
    info = json.loads((WORK / 'stream_features_complete.json').read_text()); allf = np.load(info['path'], mmap_mode='r')
    with (WORK / 'metadata_manifest.csv').open(encoding='utf-8-sig', newline='') as f: index = {r['image_id']: i for i, r in enumerate(csv.DictReader(f))}
    gd = info['global_dim']
    X = {s: torch.tensor(np.asarray(allf[[index[r['image_id']] for r in parts[s]]])[:, :gd], device=DEVICE) for s in parts if len(parts[s])}
    mu = X['train'].mean(0); sd = X['train'].std(0).clamp_min(.01); X = {s: (v - mu) / sd for s, v in X.items()}
    y = {s: torch.tensor([int(r['folder_label'][1:]) for r in parts[s]], device=DEVICE) for s in X}
    for balanced in (False, True):
        name = 'transformer_linear_global_' + ('balanced' if balanced else 'unweighted') + '_seed2026'; target = WORK / name; target.mkdir(exist_ok=True)
        if (target / 'metrics.json').exists(): continue
        seed_all(2026); head = nn.Linear(gd, 11).to(DEVICE); opt = torch.optim.AdamW(head.parameters(), lr=.005, weight_decay=.01)
        counts = np.bincount(y['train'].cpu().numpy(), minlength=11); w = 1 / np.sqrt(counts); w /= w.mean()
        crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=DEVICE) if balanced else None)
        best = -1; state = None
        for epoch in range(100):
            for ids in torch.randperm(len(y['train']), device=DEVICE).split(512):
                opt.zero_grad(set_to_none=True); crit(head(X['train'][ids]), y['train'][ids]).backward(); opt.step()
            with torch.no_grad(): m = metrics(y['validation'].cpu().numpy(), head(X['validation']).softmax(1).cpu().numpy())
            if m['macro_auc'] > best: best = m['macro_auc']; state = {k: v.detach().clone() for k, v in head.state_dict().items()}
        head.load_state_dict(state); torch.save({'head': state, 'feature_mean': mu.cpu(), 'feature_std': sd.cpu()}, target / 'best.pt')
        with torch.no_grad(): logits = {s: head(X[s]).cpu().numpy() for s in X if s != 'train'}
        finish_model(name, parts, logits, {'encoder': info['model'], 'training': 'frozen encoder linear probe (CLS)', 'balanced_loss': balanced, 'seed': 2026})

def site_shortcut_probe(rows):
    """Can the institution be predicted from frozen image features alone? Fitted on development+external training-style split
    of all included images by group hash; reported as balanced accuracy. No disease label is used."""
    target = WORK / 'site_shortcut_probe'; target.mkdir(exist_ok=True)
    if (target / 'metrics.json').exists(): return
    info = json.loads((WORK / 'stream_features_complete.json').read_text()); allf = np.load(info['path'], mmap_mode='r')
    with (WORK / 'metadata_manifest.csv').open(encoding='utf-8-sig', newline='') as f: index = {r['image_id']: i for i, r in enumerate(csv.DictReader(f))}
    use = [r for r in rows if r['split'] in ('train', 'validation', 'internal_test', 'test')]
    sites = sorted({r['site'] for r in use}); s2i = {s: i for i, s in enumerate(sites)}
    u = np.array([int(hashlib.sha256(('site-probe|' + r['split_group']).encode()).hexdigest()[:8], 16) / 2**32 for r in use])
    x = torch.tensor(np.asarray(allf[[index[r['image_id']] for r in use]])[:, :info['global_dim']], device=DEVICE)
    ys = torch.tensor([s2i[r['site']] for r in use], device=DEVICE); yd = np.array([int(r['folder_label'][1:]) for r in use])
    tr = torch.tensor(u < .7, device=DEVICE); te = ~tr
    mu = x[tr].mean(0); sd = x[tr].std(0).clamp_min(.01); x = (x - mu) / sd
    seed_all(2026); head = nn.Linear(x.shape[1], len(sites)).to(DEVICE); opt = torch.optim.AdamW(head.parameters(), lr=.005, weight_decay=.01)
    cnt = torch.bincount(ys[tr], minlength=len(sites)).float(); crit = nn.CrossEntropyLoss(weight=(1 / cnt) / (1 / cnt).mean())
    idx = torch.where(tr)[0]
    for epoch in range(60):
        for ids in idx[torch.randperm(len(idx), device=DEVICE)].split(512):
            opt.zero_grad(set_to_none=True); crit(head(x[ids]), ys[ids]).backward(); opt.step()
    with torch.no_grad(): pred = head(x[te]).argmax(1).cpu().numpy()
    yt = ys[te].cpu().numpy(); ydt = yd[te.cpu().numpy()]
    bal = float(np.mean([(pred[yt == k] == k).mean() for k in range(len(sites))]))
    normal = ydt == 0; bal_normal = float(np.mean([(pred[(yt == k) & normal] == k).mean() for k in range(len(sites)) if ((yt == k) & normal).any()]))
    save_json(target / 'metrics.json', {'sites': sites, 'n_held_out': int(te.sum()), 'balanced_accuracy_site': bal, 'chance': 1 / len(sites),
                                        'balanced_accuracy_site_normal_only': bal_normal,
                                        'confusion_matrix': confusion_matrix(yt, pred, labels=range(len(sites))).tolist(),
                                        'interpretation': 'High site identifiability with site-dependent label mix indicates shortcut risk; reported, not corrected.'})
    log(f'site probe balanced accuracy {bal:.3f} (normal only {bal_normal:.3f}, chance {1/len(sites):.3f})')

def metadata_negative_control(parts):
    name = 'metadata_only_negative_control'; target = WORK / name; target.mkdir(exist_ok=True)
    if (target / 'metrics.json').exists(): return
    import pandas as pd
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.linear_model import LogisticRegression
    def frame(rs):
        df = pd.DataFrame(rs); df['body_weight_numeric'] = pd.to_numeric(df.body_weight, errors='coerce')
        df['body_weight_numeric'] = df.body_weight_numeric.where((df.body_weight_numeric > 0) & (df.body_weight_numeric < 150))
        df['gender'] = df.gender.fillna('unknown').replace('', 'unknown'); return df[['batch', 'gender', 'body_weight_numeric']]
    pre = ColumnTransformer([('numeric', Pipeline([('impute', SimpleImputer(strategy='median', add_indicator=True)), ('scale', StandardScaler())]), ['body_weight_numeric']),
                             ('categorical', OneHotEncoder(handle_unknown='ignore'), ['batch', 'gender'])])
    clf = Pipeline([('preprocess', pre), ('classifier', LogisticRegression(C=1, class_weight='balanced', max_iter=1000, random_state=2026))])
    clf.fit(frame(parts['train']), [int(r['folder_label'][1:]) for r in parts['train']])
    logits = {s: np.log(clf.predict_proba(frame(parts[s])).clip(1e-12, 1)) for s in EVAL_SPLITS + ['turing363'] if len(parts[s])}
    finish_model(name, parts, logits, {'training': 'Logistic regression on weight group, recorded weight and sex only', 'purpose': 'Confounding negative control'})

def turing_rows():
    """Reader-set images present in the curated image source (reader-set IDs not found there are listed and skipped).
    Images excluded from the curated cohort as pixel-label conflicts are still predicted here."""
    stems = json.loads((WORK / 'reader_case_stems.json').read_text())
    with (WORK / 'metadata_manifest.csv').open(encoding='utf-8-sig', newline='') as f: meta = {r['image_id']: r for r in csv.DictReader(f)}
    missing = [s for s in stems if s not in meta]
    save_json(WORK / 'turing_cases_not_in_cleansed_source.json', missing)
    out = []
    for s in stems:
        if s in missing: continue
        r = dict(meta[s]); r['cache_path'] = str(CACHE / (s + '.png')); r['site'] = r['patient_id'][:2]; r['split'] = 'turing363'
        assert Path(r['cache_path']).exists(), r['cache_path']; out.append(r)
    return out

def main():
    with (WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    parts = {s: [r for r in rows if r['split'] == s] for s in ['train'] + EVAL_SPLITS}; parts['turing363'] = turing_rows()
    groups = defaultdict(set); patients = defaultdict(set); hashes = defaultdict(set)
    for r in rows:
        groups[r['split_group']].add(r['split']); patients[r['patient_id']].add(r['split']); hashes[r['pixel_sha256_224']].add(r['split'])
    used = {'train', 'validation', 'internal_test'}
    integrity = {'image_rows': len(rows), 'partition_sizes': {s: len(v) for s, v in parts.items()},
                 'split_groups_in_multiple_partitions': sum(len(v) > 1 for v in groups.values()),
                 'patient_id_strings_in_multiple_partitions': sum(len(v) > 1 for v in patients.values()),
                 'exact_input_pixel_hashes_in_multiple_partitions': sum(len(v) > 1 for v in hashes.values()),
                 'development_sites': sorted({r['site'] for r in rows if r['split'] in used}),
                 'external_sites': sorted({r['site'] for r in rows if r['split'] == 'test'}),
                 'turing_images_in_training_or_validation': sum(r['image_id'] in {t['image_id'] for t in parts['turing363']} for r in parts['train'] + parts['validation'])}
    save_json(WORK / 'split_integrity_check.json', integrity); log(integrity)
    for key in ['split_groups_in_multiple_partitions', 'patient_id_strings_in_multiple_partitions', 'exact_input_pixel_hashes_in_multiple_partitions', 'turing_images_in_training_or_validation']:
        assert integrity[key] == 0, f'Partition integrity failed: {key}'
    assert integrity['development_sites'] == ['01'] and '01' not in integrity['external_sites']
    if DRY_RUN:
        for s, v in parts.items(): assert all(Path(r['cache_path']).exists() for r in v[:50]), s
        log('dry run OK'); return
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is required for this run')
    save_json(WORK / 'environment.json', {'python': sys.version, 'torch': torch.__version__, 'timm': timm.__version__, 'cuda': torch.version.cuda, 'gpu': torch.cuda.get_device_name(0), 'platform': platform.platform()})
    save_json(WORK / 'run_start_provenance.json', {'time': time.strftime('%Y-%m-%d %H:%M:%S'), 'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
              [CODE / 'run_site_split.py', CODE / 'make_site_manifest.py', WORK / 'manifest.csv']}})
    metadata_negative_control(parts)
    linear_probes(parts, rows)
    site_shortcut_probe(rows)
    for seed in SEEDS: train_classifier(parts, True, seed)
    for seed in SEEDS: train_classifier(parts, False, seed)
    save_json(WORK / 'training_complete.json', {'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')})

if __name__ == '__main__':
    try: main()
    except Exception:
        (WORK / 'training_failure.txt').write_text(traceback.format_exc(), encoding='utf-8'); traceback.print_exc(); raise
