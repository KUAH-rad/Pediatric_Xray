"""Input (B) and network (C) ablations, prespecified in the study protocol addendum before any test result was opened.
Same institution split, validation-only selection and temperature scaling as the main run; seed 2026.
Transformer recipe = main DINOv3 recipe (last 2 blocks + norm + head, 8 epochs, patience 3).
CNN recipe = main ConvNeXt recipe (full fine-tuning, lr 1e-4, 12 epochs, patience 4).
B1 DINOv3 ViT-S/16 at 448 x 448        B2 DINOv3 ViT-S/16 with CLAHE (clip 2.0, 8 x 8 tiles)
C1 ResNet-50                           C2 EfficientNet-B0
C3 DINOv2 ViT-S/14 (registers)         C4 DINOv3 ViT-B/16 (larger foundation model)
"""
import sys, csv, json, time
from pathlib import Path
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import timm
from torchvision import transforms as T
import run_site_split as R

WORK = R.WORK / 'ablations_bc'; WORK.mkdir(exist_ok=True)
import config
CACHE448 = config.CACHE448
DRY_RUN = '--dry-run' in sys.argv
PLAN = [  # name, timm id, recipe, image size, clahe
    ('C1_resnet50', 'resnet50.a1_in1k', 'cnn', 224, False),
    ('C2_efficientnet_b0', 'efficientnet_b0.ra_in1k', 'cnn', 224, False),
    ('C3_dinov2_vits14_reg', 'vit_small_patch14_reg4_dinov2.lvd142m', 'vit', 224, False),
    ('C4_dinov3_vitb16', 'vit_base_patch16_dinov3.lvd1689m', 'vit', 224, False),
    ('B2_dinov3_clahe', 'vit_small_patch16_dinov3.lvd1689m', 'vit', 224, True),
    ('B1_dinov3_448', 'vit_small_patch16_dinov3.lvd1689m', 'vit', 448, False),
]

class Images(Dataset):
    def __init__(self, rows, training=False, size=224, clahe=False):
        self.rows, self.size = rows, size; self.clahe = None
        if clahe:
            import cv2; self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        aug = [T.RandomRotation(5), T.ColorJitter(brightness=.1, contrast=.1)] if training else []
        self.transform = T.Compose(aug + [T.ToTensor(), T.Normalize((.485, .456, .406), (.229, .224, .225))])
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]; path = r['cache_path'] if self.size == 224 else str(CACHE448 / (r['image_id'] + '.png'))
        with Image.open(path) as im: g = im.convert('L')
        if self.clahe is not None: g = Image.fromarray(self.clahe.apply(np.asarray(g)))
        return self.transform(g.convert('RGB')), int(r['folder_label'][1:]), i

def loader(rows, cfg, training=False, batch=32):
    return DataLoader(Images(rows, training, cfg['size'], cfg['clahe']), batch_size=batch, shuffle=training, num_workers=0, pin_memory=True)

@torch.inference_mode()
def predict(model, rows, cfg):
    model.eval(); ps = []
    for x, _, _ in loader(rows, cfg, batch=32 if cfg['size'] == 224 else 16):
        with torch.autocast('cuda', dtype=torch.float16): ps.append(model(x.to(R.DEVICE, non_blocking=True)).float().cpu())
    return torch.cat(ps).numpy()

def train(name, model_id, recipe, size, clahe, parts, seed=2026):
    target = WORK / name; target.mkdir(exist_ok=True); cfg = {'size': size, 'clahe': clahe}
    if (target / 'metrics.json').exists(): R.log('skip completed ' + name); return
    R.seed_all(seed); kw = {'img_size': size} if recipe == 'vit' else {}
    model = timm.create_model(model_id, pretrained=True, num_classes=11, **kw).to(R.DEVICE)
    if recipe == 'vit':
        for p in model.parameters(): p.requires_grad = False
        for block in model.blocks[-2:]:
            for p in block.parameters(): p.requires_grad = True
        for module in [model.norm, model.get_classifier()]:
            for p in module.parameters(): p.requires_grad = True
        head_ids = {id(p) for p in model.get_classifier().parameters()}
        groups = [{'params': [p for p in model.parameters() if p.requires_grad and id(p) not in head_ids], 'lr': 1e-5}, {'params': model.get_classifier().parameters(), 'lr': 1e-4}]
        max_epochs, stop = 8, 3
    else:
        groups = [{'params': model.parameters(), 'lr': 1e-4}]; max_epochs, stop = 12, 4
    tr, va = parts['train'], parts['validation']
    counts = np.bincount([int(r['folder_label'][1:]) for r in tr], minlength=11); w = 1 / np.sqrt(counts); w = w / w.mean()
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=R.DEVICE))
    opt = torch.optim.AdamW(groups, weight_decay=.05); scaler = torch.amp.GradScaler('cuda')
    best = -1; start = 0; patience = 0; history = []
    if (target / 'last.pt').exists():
        ck = torch.load(target / 'last.pt', map_location=R.DEVICE, weights_only=False); model.load_state_dict(ck['model']); opt.load_state_dict(ck['optimizer']); scaler.load_state_dict(ck['scaler'])
        start = ck['epoch'] + 1; best = ck['best']; patience = ck['patience']; history = ck['history']
    for epoch in range(start, max_epochs):
        model.train(); loss_sum = 0; n = 0; t0 = time.time()
        for step, (x, y, _) in enumerate(loader(tr, cfg, True, batch=16)):
            x = x.to(R.DEVICE, non_blocking=True); y = y.to(R.DEVICE, non_blocking=True); opt.zero_grad(set_to_none=True)
            with torch.autocast('cuda', dtype=torch.float16): loss = criterion(model(x), y)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1); scaler.step(opt); scaler.update()
            loss_sum += float(loss.detach()) * len(y); n += len(y)
            if step % 300 == 0: R.log(f'{name} epoch={epoch+1} step={step} loss={loss_sum/n:.4f}')
        zv = predict(model, va, cfg); yv = np.array([int(r['folder_label'][1:]) for r in va]); m = R.metrics(yv, torch.tensor(zv).softmax(1).numpy())
        history.append({'epoch': epoch + 1, 'loss': loss_sum / n, 'validation_macro_auc': m['macro_auc'], 'validation_macro_recall': m['macro_recall'], 'seconds': time.time() - t0})
        if m['macro_auc'] > best: best = m['macro_auc']; patience = 0; R.save_checkpoint(model.state_dict(), target / 'best.pt')
        else: patience += 1
        R.save_checkpoint({'model': model.state_dict(), 'optimizer': opt.state_dict(), 'scaler': scaler.state_dict(), 'epoch': epoch, 'best': best, 'patience': patience, 'history': history}, target / 'last.pt')
        R.save_json(target / 'training_history.json', history); R.log(history[-1])
        if patience >= stop: break
    model.load_state_dict(torch.load(target / 'best.pt', map_location=R.DEVICE, weights_only=True))
    logits = {s: predict(model, v, cfg) for s, v in parts.items() if s != 'train' and len(v)}
    old = R.WORK; R.WORK = WORK
    try: R.finish_model(name, parts, logits, {'model': model_id, 'recipe': recipe, 'image_size': size, 'clahe': clahe, 'seed': seed, 'epochs_completed': len(history)})
    finally: R.WORK = old
    del model; torch.cuda.empty_cache()

def main():
    with (R.WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    parts = {s: [r for r in rows if r['split'] == s] for s in ['train', 'validation', 'internal_test', 'test']}; parts['turing363'] = R.turing_rows()
    if DRY_RUN:
        for name, mid, recipe, size, clahe in PLAN:
            m = timm.create_model(mid, pretrained=True, num_classes=11, **({'img_size': size} if recipe == 'vit' else {}))
            x, _, _ = Images(parts['train'][:1], size=224, clahe=clahe)[0]; R.log(f'{name} weights OK, params {sum(p.numel() for p in m.parameters())/1e6:.1f}M, input {tuple(x.shape)}')
        return
    done448 = R.WORK / 'cache448_complete.json'
    for name, mid, recipe, size, clahe in PLAN:
        if size == 448:
            while not done448.exists(): R.log('waiting for 448 cache'); time.sleep(120)
        train(name, mid, recipe, size, clahe, parts)
    R.save_json(WORK / 'ablations_bc_complete.json', {'completed_at': time.strftime('%Y-%m-%d %H:%M:%S')})

if __name__ == '__main__':
    main()
