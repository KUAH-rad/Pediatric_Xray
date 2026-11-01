"""Finding-level explainability and plausibility checks (study analysis plan).
Reads source ROI JSON files read-only. Uses the primary DINOv3 checkpoints from run_site_split.py.
1. Grad-CAM localization against radiologist ROI polygons on external-test images (seed 2026 model)
2. Hepatomegaly/splenomegaly CAM laterality
3. Horizontal-flip test for situs inversus (3-seed ensemble)
4. Patient position code from the source JSON for external-test images
5. One random correctly classified external example per finding for the figure
"""
import sys, csv, json, random
from pathlib import Path
from collections import defaultdict
import numpy as np
import torch
from PIL import Image, ImageDraw
import timm
import run_site_split as R

OUT = R.WORK / 'cam_flip'; OUT.mkdir(exist_ok=True)
import config
FINDINGS = ['정상', '기복증', '종괴', '장중첩증', '공기-액체음영', '이물', '변비', '석회화병변', '비장비대', '간비대', '전내장역위증']
DRY_RUN = '--dry-run' in sys.argv

def load_json(r):
    p = config.annotation_dir(config.group_number(r['batch'])) / f"{r['image_id']}.json"
    if not p.exists(): return None
    return json.loads(p.read_text(encoding='utf-8'))

def roi_mask(js, c, src_size=None):
    """Polygon(s) for finding c mapped to the 224 padded frame (PIL ImageOps.pad, centering .5)."""
    w, h = int(js['images']['width']), int(js['images']['height'])
    sx = sy = 1.0
    if src_size and src_size != (w, h): sx, sy = src_size[0] / w, src_size[1] / h; w, h = src_size
    if w / h > 1: nw, nh = 224, round(h / w * 224)
    else: nw, nh = round(w / h * 224), 224
    ox, oy = int((224 - nw) * .5), int((224 - nh) * .5); s = nw / w
    m = Image.new('L', (224, 224), 0); d = ImageDraw.Draw(m); n = 0
    for a in js.get('annotation') or []:
        if a.get('label') != FINDINGS[c] or not a.get('m_points'): continue
        pts = [(ox + p['x'] * sx * s, oy + p['y'] * sy * s) for p in a['m_points']]
        if len(pts) >= 3: d.polygon(pts, fill=1); n += 1
        elif len(pts) >= 1: d.ellipse([pts[0][0] - 2, pts[0][1] - 2, pts[0][0] + 2, pts[0][1] + 2], fill=1); n += 1
    content = Image.new('L', (224, 224), 0); ImageDraw.Draw(content).rectangle([ox, oy, ox + nw - 1, oy + nh - 1], fill=1)
    return (np.array(m, bool) if n else None), np.array(content, bool)

def load_model(seed):
    m = timm.create_model('vit_small_patch16_dinov3.lvd1689m', pretrained=False, num_classes=11, img_size=224)
    m.load_state_dict(torch.load(R.WORK / f'dinov3_last2blocks_finetune_seed{seed}' / 'best.pt', map_location='cpu', weights_only=True))
    return m.to(R.DEVICE).eval()

def gradcam(model, x, cls):
    store = {}
    h = model.blocks[-1].register_forward_hook(lambda m, i, o: store.__setitem__('t', o))
    x = x.to(R.DEVICE).requires_grad_(False); z = model(x); t = store['t']; h.remove()
    sel = z.gather(1, cls.view(-1, 1).to(R.DEVICE)).sum()
    g, = torch.autograd.grad(sel, t)
    p = model.num_prefix_tokens; tp = t[:, p:]; gp = g[:, p:]
    w = gp.mean(1, keepdim=True); cam = torch.relu((w * tp).sum(-1))
    side = int(cam.shape[1] ** .5); cam = cam.view(-1, 1, side, side)
    cam = torch.nn.functional.interpolate(cam, size=(224, 224), mode='bilinear', align_corners=False)[:, 0]
    cam = cam / cam.flatten(1).amax(1).clamp_min(1e-8).view(-1, 1, 1)
    return cam.detach().cpu().numpy(), torch.softmax(z.detach().float(), 1).cpu().numpy()

def main():
    with (R.WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = list(csv.DictReader(f))
    ext = [r for r in rows if r['split'] == 'test']
    pos = [r for r in ext if r['folder_label'] != 'D00']
    # Source PNG size check (header only) on a sample, to confirm ROI coordinates share the PNG frame.
    rng = random.Random(2026); sample = rng.sample(pos, 40); mism = 0
    for r in sample:
        js = load_json(r)
        if js is None: continue
        with Image.open(config.image_dir(config.group_number(r['batch'])) / f"{r['image_id']}.png") as im: size = im.size
        mism += size != (int(js['images']['width']), int(js['images']['height']))
    R.log(f'ROI frame check: {mism}/40 source PNG sizes differ from JSON size')
    if DRY_RUN:
        r = sample[0]; js = load_json(r); m, content = roi_mask(js, int(r['folder_label'][1:]))
        R.log(f'dry run ROI pixels {None if m is None else int(m.sum())}, content {int(content.sum())}'); return

    # 1-2. Grad-CAM localization, seed 2026, external-test images with a recorded finding.
    model = load_model(2026); res = defaultdict(list); examples = defaultdict(list); position = {}
    for i in range(0, len(pos), 32):
        batch = pos[i:i + 32]; jsons = [load_json(r) for r in batch]
        for r, js in zip(batch, jsons):
            if js: position[r['image_id']] = (js.get('dataInfo') or [{}])[0].get('자세', '')
        x = torch.stack([R.Images([r])[0][0] for r in batch]); cls = torch.tensor([int(r['folder_label'][1:]) for r in batch])
        cams, probs = gradcam(model, x, cls)
        for r, js, cam, p, c in zip(batch, jsons, cams, probs, cls.tolist()):
            if js is None: continue
            m, content = roi_mask(js, c)
            if m is None or not m.any(): continue
            yx = np.unravel_index(np.argmax(np.where(content, cam, -1)), cam.shape)
            cy, cx = np.array(np.nonzero(content & (cam >= .5))).mean(1) if (content & (cam >= .5)).any() else yx
            res[c].append({'hit': bool(m[yx]), 'energy_in_roi': float(cam[m].sum() / max(cam[content].sum(), 1e-8)),
                           'roi_area_fraction': float(m.sum() / content.sum()), 'centroid_x_frac': float(cx / 224),
                           'correct': bool(p.argmax() == c), 'site': r['site']})
            if p.argmax() == c: examples[c].append(r['image_id'])
        if i % 3200 == 0: R.log(f'cam {i}/{len(pos)}')
    summary = {}
    for c in range(1, 11):
        v = res[c]
        if not v: continue
        summary[FINDINGS[c]] = {'n_with_roi': len(v), 'pointing_game_hit_rate': float(np.mean([e['hit'] for e in v])),
                                'chance_mean_roi_area_fraction': float(np.mean([e['roi_area_fraction'] for e in v])),
                                'energy_in_roi_mean': float(np.mean([e['energy_in_roi'] for e in v])),
                                'hit_rate_correct_only': float(np.mean([e['hit'] for e in v if e['correct']])) if any(e['correct'] for e in v) else None,
                                'centroid_x_frac_mean': float(np.mean([e['centroid_x_frac'] for e in v])),
                                'hit_rate_by_site': {s: float(np.mean([e['hit'] for e in v if e['site'] == s])) for s in sorted({e['site'] for e in v})}}
    R.save_json(OUT / 'cam_localization_summary.json', summary)
    R.save_json(OUT / 'cam_localization_per_image.json', {FINDINGS[c]: v for c, v in res.items()})
    R.save_json(OUT / 'position_code_external_positive.json', position)

    # 5. Example panel: one random correct external case per finding (seed 2026), ROI outline + CAM.
    rng = random.Random(2026); byid = {r['image_id']: r for r in pos}; panels = []
    for c in range(1, 11):
        if not examples[c]: continue
        r = byid[rng.choice(sorted(examples[c]))]; js = load_json(r)
        x = R.Images([r])[0][0].unsqueeze(0); cam, _ = gradcam(model, x, torch.tensor([c])); m, _ = roi_mask(js, c)
        base = np.array(Image.open(r['cache_path']).convert('L'), float) / 255
        rgb = np.stack([base] * 3, -1) * .6 + np.stack([cam[0], cam[0] * .35, 1 - cam[0]], -1) * .4 * (cam[0][..., None] > .2)
        edge = m ^ np.array(Image.fromarray(m.astype(np.uint8) * 255).filter(__import__('PIL.ImageFilter', fromlist=['MinFilter']).MinFilter(3)), bool)
        rgb[edge] = [0, 1, 0]
        panels.append({'finding': FINDINGS[c], 'image_id': r['image_id'], 'site': r['site']})
        Image.fromarray((rgb.clip(0, 1) * 255).astype(np.uint8)).resize((448, 448), Image.Resampling.NEAREST).save(OUT / f'cam_example_D{c:02d}.png')
    R.save_json(OUT / 'cam_examples.json', panels)
    del model; torch.cuda.empty_cache()

    # 3. Horizontal-flip test, 3-seed ensemble: all external situs inversus + random 2,000 external normals.
    situs = [r for r in ext if r['folder_label'] == 'D10']; normals = random.Random(2026).sample([r for r in ext if r['folder_label'] == 'D00'], 2000)
    probs = {}
    for seed in R.SEEDS if hasattr(R, 'SEEDS') else [2026]:
        model = load_model(seed)
        for name, rs in [('situs', situs), ('normal', normals)]:
            for flip in (False, True):
                out = []
                with torch.inference_mode():
                    for i in range(0, len(rs), 64):
                        x = torch.stack([R.Images([r])[0][0] for r in rs[i:i + 64]])
                        if flip: x = torch.flip(x, dims=[3])
                        with torch.autocast('cuda', dtype=torch.float16): out.append(torch.softmax(model(x.to(R.DEVICE)).float(), 1).cpu().numpy())
                probs.setdefault((name, flip), []).append(np.concatenate(out))
        del model; torch.cuda.empty_cache()
    flip = {}
    for (name, f), v in probs.items():
        p = np.mean(v, 0); pred = p.argmax(1)
        flip[f'{name}_{"flipped" if f else "original"}'] = {'n': len(pred), 'predicted_situs_inversus': float((pred == 10).mean()), 'predicted_normal': float((pred == 0).mean()),
                                                           'mean_prob_situs_inversus': float(p[:, 10].mean())}
    R.save_json(OUT / 'flip_test.json', flip); R.log(flip)
    R.save_json(OUT / 'cam_flip_complete.json', {'completed_at': __import__('time').strftime('%Y-%m-%d %H:%M:%S')})

if __name__ == '__main__':
    main()
