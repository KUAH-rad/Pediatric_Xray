"""Occlusion-sensitivity localization (sensitivity analysis added 2026-10-25 after the prespecified Grad-CAM maps were found
to be spatially diffuse). Model-agnostic: each 16 x 16 patch of the 224 input is replaced by the image mean and the drop in the
recorded-finding probability is the saliency. DINOv3 seed 2026, external test, up to 100 random images per finding that have a
radiologist ROI (seed 2026). Same pointing-game / energy metrics as run_cam_flip.py."""
import csv, json, random
import numpy as np
import torch
from PIL import Image
import run_site_split as R
import run_cam_flip as C

OUT = C.OUT
model = C.load_model(2026)
with (R.WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: rows = [r for r in csv.DictReader(f) if r['split'] == 'test' and r['folder_label'] != 'D00']
rng = random.Random(2026); by = {}
for r in rows: by.setdefault(int(r['folder_label'][1:]), []).append(r)

@torch.inference_mode()
def occlusion(x, c, k=16):
    g = 224 // k; base = x.clone(); variants = []
    for i in range(g):
        for j in range(g):
            v = base.clone(); v[:, i*k:(i+1)*k, j*k:(j+1)*k] = 0  # 0 = per-channel mean after normalization
            variants.append(v)
    X = torch.stack([base] + variants).to(R.DEVICE)
    with torch.autocast('cuda', dtype=torch.float16): p = torch.softmax(model(X).float(), 1)[:, c].cpu().numpy()
    sal = np.clip(p[0] - p[1:], 0, None).reshape(g, g)
    sal = np.kron(sal, np.ones((k, k)))
    return sal / max(sal.max(), 1e-8), float(p[0])

res = {}; per = {}
for c in range(1, 11):
    pool = by.get(c, [])[:]; rng.shuffle(pool); out = []
    for r in pool:
        if len(out) >= 100: break
        js = C.load_json(r)
        if js is None: continue
        m, content = C.roi_mask(js, c)
        if m is None or not m.any(): continue
        x = R.Images([r])[0][0]; sal, p = occlusion(x, c)
        yx = np.unravel_index(np.argmax(np.where(content, sal, -1)), sal.shape)
        out.append({'image_id': r['image_id'], 'hit': bool(m[yx]), 'energy_in_roi': float(sal[m].sum() / max(sal[content].sum(), 1e-8)),
                    'roi_area_fraction': float(m.sum() / content.sum()), 'prob': p})
        if len(out) <= 1: np.save(OUT / f'occlusion_example_D{c:02d}.npy', sal)
    per[C.FINDINGS[c]] = out
    if out:
        res[C.FINDINGS[c]] = {'n': len(out), 'pointing_game_hit_rate': float(np.mean([e['hit'] for e in out])),
                              'chance_mean_roi_area_fraction': float(np.mean([e['roi_area_fraction'] for e in out])),
                              'energy_in_roi_mean': float(np.mean([e['energy_in_roi'] for e in out]))}
        R.log(f"{C.FINDINGS[c]} n={len(out)} hit={res[C.FINDINGS[c]]['pointing_game_hit_rate']:.2f} chance={res[C.FINDINGS[c]]['chance_mean_roi_area_fraction']:.2f}")
R.save_json(OUT / 'occlusion_localization_summary.json', res); R.save_json(OUT / 'occlusion_per_image.json', per)
