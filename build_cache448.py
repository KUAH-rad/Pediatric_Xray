"""448x448 cache for input-resolution ablation B1. Same conversion as prepare_cohort.py (grayscale, aspect-preserving
pad, bilinear), size 448. Reads the source PNGs read-only; writes only to the 448 cache."""
import csv, json
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from PIL import Image, ImageOps

import config
WORK = config.WORK_DIR
CACHE = config.CACHE448; CACHE.mkdir(parents=True, exist_ok=True)

def one(r):
    t = CACHE / (r['image_id'] + '.png')
    if t.exists(): return None
    try:
        with Image.open(r['image_path']) as im:
            if im.mode in ('I', 'I;16', 'I;16B', 'I;16L', 'F'):
                a = np.asarray(im, dtype=np.float32); lo, hi = float(a.min()), float(a.max())
                im = Image.fromarray(np.clip((a - lo) / max(hi - lo, 1e-8) * 255, 0, 255).astype(np.uint8))
            ImageOps.pad(im.convert('L'), (448, 448), method=Image.Resampling.BILINEAR, color=0, centering=(.5, .5)).save(t, compress_level=1)
    except Exception as e: return {'image_id': r['image_id'], 'error': str(e)}

with (WORK / 'manifest.csv').open(encoding='utf-8-sig', newline='') as f:
    rows = [r for r in csv.DictReader(f) if r['split'] in ('train', 'validation', 'internal_test', 'test')]
stems = set(json.loads((WORK / 'reader_case_stems.json').read_text()))
with (WORK / 'metadata_manifest.csv').open(encoding='utf-8-sig', newline='') as f:
    rows += [r for r in csv.DictReader(f) if r['image_id'] in stems]
assert all(Path(r['image_path']).parent.name == 'images' for r in rows), 'unexpected source folder'
with ThreadPoolExecutor(6) as ex: errors = [e for e in ex.map(one, rows) if e]
(WORK / 'cache448_complete.json').write_text(json.dumps({'n': len(rows), 'errors': errors}, indent=1), encoding='utf-8')
print('done', len(rows), 'errors', len(errors))
