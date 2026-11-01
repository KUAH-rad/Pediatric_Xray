"""Saliency-method comparison (sensitivity analysis, 2026-10-25). Same images as run_occlusion.py (up to 100 per finding, external test).
Methods: Grad-CAM (last block), Grad-CAM++ (last block), Grad-CAM (block 8 of 12), occlusion (from run_occlusion.py).
Outputs metrics per method and a figure grid with a jet colour map (values below 0.3 transparent) and the radiologist ROI outline."""
import csv, json
import numpy as np
import torch
from PIL import Image
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import run_site_split as R
import run_cam_flip as C

OUT = C.OUT
model = C.load_model(2026)
with (R.WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f: byid = {r['image_id']: r for r in csv.DictReader(f)}
occ = json.loads((OUT / 'occlusion_per_image.json').read_text(encoding='utf-8'))

def cam_at(x, c, block, plus=False):
    store = {}; h = model.blocks[block].register_forward_hook(lambda m, i, o: store.__setitem__('t', o))
    z = model(x.unsqueeze(0).to(R.DEVICE)); h.remove(); t = store['t']
    g, = torch.autograd.grad(z[0, c], t)
    p = model.num_prefix_tokens; tp, gp = t[0, p:].detach(), g[0, p:]
    if plus:  # Grad-CAM++ weights (exponential-of-score approximation)
        g2, g3 = gp ** 2, gp ** 3
        a = g2 / (2 * g2 + (tp * g3).sum(0, keepdim=True)).clamp_min(1e-8).where(gp != 0, torch.ones_like(gp))
        w = (a * torch.relu(gp)).sum(0)
    else: w = gp.mean(0)
    cam = torch.relu((tp * w).sum(-1)); s = int(cam.numel() ** .5)
    cam = torch.nn.functional.interpolate(cam.view(1, 1, s, s), size=(224, 224), mode='bilinear', align_corners=False)[0, 0]
    return (cam / cam.max().clamp_min(1e-8)).cpu().numpy()

def occlusion_map(x, c, k=16):
    g = 224 // k; vs = [x]
    for i in range(g):
        for j in range(g):
            v = x.clone(); v[:, i*k:(i+1)*k, j*k:(j+1)*k] = 0; vs.append(v)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        p = torch.softmax(model(torch.stack(vs).to(R.DEVICE)).float(), 1)[:, c].cpu().numpy()
    s = np.kron(np.clip(p[0] - p[1:], 0, None).reshape(g, g), np.ones((k, k))); return s / max(s.max(), 1e-8)

METHODS = {'Grad-CAM (last block)': lambda x, c: cam_at(x, c, -1), 'Grad-CAM++ (last block)': lambda x, c: cam_at(x, c, -1, True),
           'Grad-CAM (block 8)': lambda x, c: cam_at(x, c, 7)}
res = {m: {} for m in list(METHODS) + ['Occlusion']}
examples = {}
for c in range(1, 11):
    name = C.FINDINGS[c]; items = occ.get(name, [])
    if not items: continue
    hits = {m: [] for m in METHODS}; energy = {m: [] for m in METHODS}
    for e in items:
        r = byid[e['image_id']]; js = C.load_json(r); m, content = C.roi_mask(js, c); x = R.Images([r])[0][0]
        for mname, fn in METHODS.items():
            s = fn(x, c); yx = np.unravel_index(np.argmax(np.where(content, s, -1)), s.shape)
            hits[mname].append(bool(m[yx])); energy[mname].append(float(s[m].sum() / max(s[content].sum(), 1e-8)))
    chance = float(np.mean([e['roi_area_fraction'] for e in items]))
    for mname in METHODS: res[mname][name] = {'n': len(items), 'hit': float(np.mean(hits[mname])), 'energy': float(np.mean(energy[mname])), 'chance': chance}
    res['Occlusion'][name] = {'n': len(items), 'hit': float(np.mean([e['hit'] for e in items])), 'energy': float(np.mean([e['energy_in_roi'] for e in items])), 'chance': chance}
    # Figure example: the highest-probability image among the sampled ones (stated in the legend).
    ex = max(items, key=lambda e: e['prob']); examples[name] = ex['image_id']
    R.log(f"{name}: " + ', '.join(f"{k} {v[name]['hit']:.2f}" for k, v in res.items()))
R.save_json(OUT / 'saliency_method_comparison.json', res); R.save_json(OUT / 'saliency_figure_examples.json', examples)

# Figure: rows = findings, columns = image+ROI, three Grad-CAM variants, occlusion. Jet colour map, transparent below 0.3.
names = [n for n in C.FINDINGS[1:] if n in examples]; cols = ['Image + ROI'] + list(METHODS) + ['Occlusion']
fig, axes = plt.subplots(len(names), len(cols), figsize=(2.1 * len(cols), 2.1 * len(names)))
cmap = plt.get_cmap('jet')
for i, name in enumerate(names):
    c = C.FINDINGS.index(name); r = byid[examples[name]]; js = C.load_json(r); m, _ = C.roi_mask(js, c); x = R.Images([r])[0][0]
    base = np.array(Image.open(r['cache_path']).convert('L'))
    maps = [None] + [fn(x, c) for fn in METHODS.values()] + [occlusion_map(x, c)]
    for j, s in enumerate(maps):
        ax = axes[i, j]; ax.imshow(base, cmap='gray'); ax.axis('off')
        if s is not None:
            rgba = cmap(s); rgba[..., 3] = np.where(s >= .3, .25 + .5 * s, 0); ax.imshow(rgba)
        ax.contour(m, levels=[.5], colors='lime', linewidths=.8)
        if i == 0: ax.set_title(cols[j], fontsize=8)
    axes[i, 0].text(-8, 112, name, rotation=90, va='center', ha='right', fontsize=8, fontname='Malgun Gothic')
fig.tight_layout(); fig.savefig(OUT / 'saliency_method_comparison.png', dpi=200); plt.close(fig)
R.log('saved figure')
