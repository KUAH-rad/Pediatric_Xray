"""Prespecified ablations (study protocol addendum). Run after run_site_split.py.
All use DINOv3 ViT-S/16 with the main-run optimizer, augmentation, epochs and early stopping; seed 2026.
A1  leave-one-site-out multi-institution training, evaluated on the held-out external hospital
A1m same as A1 but training set reduced to the single-institution training size (group-wise random)
A2  Development-institution training with pixel-identical conflicting-label images added back (training partition only)
A3  fine-tuning depth: last 4 blocks, full fine-tuning
A4  unweighted cross-entropy
A5  longer training: up to 30 epochs, validation patience 5 (added 2026-10-24 before any test result was opened)
"""
import sys, csv, json, time, hashlib, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from torch import nn
from PIL import Image
import timm
import run_site_split as R

WORK = R.WORK / 'ablations'; WORK.mkdir(exist_ok=True)
EXTERNAL = ['02', '03', '04', '06', '08', '09']
DRY_RUN = '--dry-run' in sys.argv

def train_dinov3(name, parts, blocks=2, weighted=True, seed=2026, max_epochs=8, stop_patience=3):
    target = WORK / name; target.mkdir(exist_ok=True)
    if (target / 'metrics.json').exists(): R.log('skip completed ' + name); return
    R.seed_all(seed); model = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=True, num_classes=11, img_size=224).to(R.DEVICE)
    if blocks != 'all':
        for p in model.parameters(): p.requires_grad = False
        for block in model.blocks[-blocks:]:
            for p in block.parameters(): p.requires_grad = True
        for module in [model.norm, model.get_classifier()]:
            for p in module.parameters(): p.requires_grad = True
    tr, va = parts['train'], parts['validation']
    counts = np.bincount([int(r['folder_label'][1:]) for r in tr], minlength=11)
    w = 1 / np.sqrt(np.maximum(counts, 1)); w = w / w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=R.DEVICE) if weighted else None)
    head_ids = {id(p) for p in model.get_classifier().parameters()}
    groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': 1e-5}, {'params': model.get_classifier().parameters(), 'lr': 1e-4}]
    opt = torch.optim.AdamW(groups, weight_decay=.05); scaler = torch.amp.GradScaler('cuda')
    best = -1; start = 0; patience = 0; history = []
    if (target / 'last.pt').exists():
        ck = torch.load(target / 'last.pt', map_location=R.DEVICE, weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start = ck['epoch'] + 1; best = ck['best']; patience = ck['patience']; history = ck['history']
    for epoch in range(start, max_epochs):
        model.train(); loss_sum = 0; n = 0; t0 = time.time()
        for step, (x, y, _) in enumerate(R.loader(tr, True, batch=16)):
            x = x.to(R.DEVICE, non_blocking=True); y = y.to(R.DEVICE, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16): loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1); scaler.step(opt); scaler.update()
            loss_sum += float(loss.detach()) * len(y); n += len(y)
            if step % 300 == 0: R.log(f'{name} epoch={epoch+1} step={step} loss={loss_sum/n:.4f}')
        yv, zv = R.predict(model, va); m = R.metrics(yv, torch.tensor(zv).softmax(1).numpy())
        history.append({'epoch': epoch + 1, 'loss': loss_sum / n, 'validation_macro_auc': m['macro_auc'], 'validation_macro_recall': m['macro_recall'], 'seconds': time.time() - t0})
        if m['macro_auc'] > best: best = m['macro_auc']; patience = 0; R.save_checkpoint(model.state_dict(), target / 'best.pt')
        else: patience += 1
        R.save_checkpoint({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'patience': patience, 'history': history}, target / 'last.pt')
        R.save_json(target / 'training_history.json', history); R.log(history[-1])
        if patience >= stop_patience: break
    model.load_state_dict(torch.load(target / 'best.pt', map_location=R.DEVICE, weights_only=True))
    logits = {s: R.predict(model, v)[1] for s, v in parts.items() if s != 'train' and len(v)}
    old = R.WORK; R.WORK = WORK
    try:
        R.finish_model(name, parts, logits, {'blocks_trained': blocks, 'weighted_loss': weighted, 'seed': seed, 'n_train': len(tr), 'max_epochs': max_epochs, 'patience': stop_patience, 'epochs_completed': len(history),
                                             'train_sites': dict(sorted({s: sum(r['site'] == s for r in tr) for s in {r['site'] for r in tr}}.items()))})
    finally: R.WORK = old
    del model; torch.cuda.empty_cache()

def group_hash(g, salt):
    return int(hashlib.sha256((salt + '|' + g).encode()).hexdigest()[:8], 16) / 2**32

def loso_parts(rows, held, size_matched_to=None):
    pool = [r for r in rows if r['split'] in ('train', 'validation', 'internal_test', 'test')]
    gsites = defaultdict(set)
    for r in pool: gsites[r['split_group']].add(r['site'])
    test = [r for r in pool if r['split'] == 'test' and r['site'] == held]
    dev = [r for r in pool if held not in gsites[r['split_group']]]
    tr = [r for r in dev if group_hash(r['split_group'], f'loso-{held}') < .85]; va = [r for r in dev if group_hash(r['split_group'], f'loso-{held}') >= .85]
    if size_matched_to:
        g = defaultdict(list)
        for r in tr: g[r['split_group']].append(r)
        keys = sorted(g); random.Random(2026).shuffle(keys); tr = []
        for k in keys:
            if len(tr) >= size_matched_to: break
            tr += g[k]
    return {'train': tr, 'validation': va, 'test': test}

def cleansing_parts(main_parts):
    """Add back site-01 images excluded at curation (pixel-identical, conflicting labels) to training only,
    unless they share an identical 224 image or patient ID with any evaluation image."""
    with (R.WORK / 'metadata_manifest.csv').open(encoding='utf-8-sig', newline='') as f: meta = list(csv.DictReader(f))
    included = {r['image_id'] for v in main_parts.values() for r in v}
    with (R.WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: allrows = list(csv.DictReader(f))
    in_manifest = {r['image_id'] for r in allrows}
    eval_rows = [r for s in ('validation', 'internal_test', 'test') for r in main_parts[s]] + [r for r in allrows if r['split'] == 'reader_reserved' or r['split'].startswith('excluded')]
    turing = main_parts['turing363']; eval_pid_extra = {r['patient_id'] for r in turing}
    eval_hash = {r['pixel_sha256_224'] for r in eval_rows}; eval_pid = {r['patient_id'] for r in eval_rows} | eval_pid_extra
    add = []
    for r in meta:
        if r['image_id'] in in_manifest or r['image_id'] in included or not r['patient_id'].startswith('01') or r['patient_id'] in eval_pid: continue
        p = R.CACHE / (r['image_id'] + '.png')
        if not p.exists(): continue
        with Image.open(p) as im: h = hashlib.sha256(im.tobytes()).hexdigest()
        if h in eval_hash: continue
        r = dict(r); r.update(cache_path=str(p), site='01', split='train', pixel_sha256_224=h, split_group='added_' + r['patient_id']); add.append(r)
    parts = dict(main_parts); parts['train'] = main_parts['train'] + add
    R.save_json(WORK / 'A2_added_images.json', {'n_added': len(add), 'by_label': dict(sorted({l: sum(r['folder_label'] == l for r in add) for l in {r['folder_label'] for r in add}}.items()))})
    return parts

def main():
    with (R.WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    main_parts = {s: [r for r in rows if r['split'] == s] for s in ['train', 'validation', 'internal_test', 'test']}; main_parts['turing363'] = R.turing_rows()
    n_single = len(main_parts['train'])
    plan = []
    for s in EXTERNAL:
        plan.append((f'A1_loso_heldout{s}', loso_parts(rows, s), {}))
        plan.append((f'A1m_loso_sizematched_heldout{s}', loso_parts(rows, s, n_single), {}))
    plan.append(('A2_no_label_cleansing', None, {}))
    plan.append(('A3_last4blocks', main_parts, {'blocks': 4}))
    plan.append(('A3_full_finetune', main_parts, {'blocks': 'all'}))
    plan.append(('A4_unweighted_ce', main_parts, {'weighted': False}))
    plan.append(('A5_longer_training_30ep', main_parts, {'max_epochs': 30, 'stop_patience': 5}))
    summary = {}
    for name, parts, kw in plan:
        if parts is not None:
            summary[name] = {k: len(v) for k, v in parts.items()}
            if name.startswith('A1'):
                test_ids = {r['image_id'] for r in parts['test']}; tr_groups = {r['split_group'] for r in parts['train'] + parts['validation']}
                assert not test_ids & {r['image_id'] for r in parts['train'] + parts['validation']}
                assert not tr_groups & {r['split_group'] for r in parts['test']}
    R.log(summary); R.save_json(WORK / 'ablation_plan_sizes.json', summary)
    if DRY_RUN: R.log('ablation dry run OK'); return
    for name, parts, kw in plan:
        if parts is None: parts = cleansing_parts(main_parts)
        train_dinov3(name, parts, **kw)
    R.save_json(WORK / 'ablations_complete.json', {'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')})

if __name__ == '__main__':
    main()
