"""Build the institution-split manifest from the frozen 2026-10-22 curated manifest.

Input: COHORT_DIR/manifest.csv from prepare_cohort.py (curated, deduplicated, reader groups reserved).
Output: WORK_DIR/manifest.csv and site_split_audit.json.

Development institution: code 01 (institution A). External institutions: codes 02, 03, 04, 06, 08, 09 (B-G).
Institutions 05 and 07 (<25 images) are excluded. Institution codes are the first two characters of the source patient ID.
Patient-image groups that contain images from the development institution and another institution are excluded,
so no identical-pixel or shared-ID group links development and external data.
"""
from pathlib import Path
import csv, json, hashlib, shutil
from collections import defaultdict, Counter

import config
OUT = config.WORK_DIR
SRC = config.COHORT_DIR / 'manifest.csv'
DEV, EXTERNAL, SMALL = '01', {'02', '03', '04', '06', '08', '09'}, {'05', '07'}
SITE_NAMES = {'01': 'A', '02': 'B', '03': 'C', '04': 'D', '06': 'E', '08': 'F', '09': 'G'}  # anonymized institution letters

with SRC.open(encoding='utf-8-sig', newline='') as f:
    rows = list(csv.DictReader(f))
for r in rows:
    r['site'] = r['patient_id'][:2]
    r['split_random_20261022'] = r['split']

group_sites = defaultdict(set)
for r in rows:
    group_sites[r['split_group']].add(r['site'])

for r in rows:
    sites = group_sites[r['split_group']]
    if r['split_random_20261022'] == 'reader_reserved':
        r['split'] = 'reader_reserved'
    elif sites == {DEV}:
        # Label-independent hash of the patient-image group; new salt for this protocol.
        u = int(hashlib.sha256(('site-split-20261024|' + r['split_group']).encode()).hexdigest()[:8], 16) / 2**32
        r['split'] = 'train' if u < .70 else 'validation' if u < .85 else 'internal_test'
    elif DEV in sites:
        r['split'] = 'excluded_shared_with_development_site'
    elif sites <= SMALL or r['site'] in SMALL:
        r['split'] = 'excluded_small_site'
    elif sites <= EXTERNAL:
        r['split'] = 'test'
    else:
        r['split'] = 'excluded_other'

fields = list(rows[0].keys())
OUT.mkdir(parents=True, exist_ok=True)
# Cohort-level files used by the training scripts
for name in ('metadata_manifest.csv', 'reader_case_stems.json', 'stream_features_complete.json'):
    if (config.COHORT_DIR / name).exists() and not (OUT / name).exists(): shutil.copy2(config.COHORT_DIR / name, OUT / name)
with (OUT / 'manifest.csv').open('w', encoding='utf-8-sig', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

splits = ['train', 'validation', 'internal_test', 'test', 'reader_reserved', 'excluded_shared_with_development_site', 'excluded_small_site', 'excluded_other']
audit = {'source_manifest_sha256': hashlib.sha256(SRC.read_bytes()).hexdigest(),
         'site_names': SITE_NAMES, 'development_site': DEV, 'external_sites': sorted(EXTERNAL),
         'partitions': {s: {'images': sum(r['split'] == s for r in rows), 'groups': len({r['split_group'] for r in rows if r['split'] == s}),
                            'sites': dict(Counter(r['site'] for r in rows if r['split'] == s)),
                            'classes': dict(sorted(Counter(r['folder_label'] for r in rows if r['split'] == s).items()))} for s in splits},
         'external_by_site_class': {site: dict(sorted(Counter(r['folder_label'] for r in rows if r['split'] == 'test' and r['site'] == site).items())) for site in sorted(EXTERNAL)},
         'manifest_sha256': hashlib.sha256((OUT / 'manifest.csv').read_bytes()).hexdigest()}
reader_stems = set(json.loads((OUT / 'reader_case_stems.json').read_text()))
audit['turing_cases_in_curated_manifest'] = sum(r['image_id'] in reader_stems for r in rows)
(OUT / 'site_split_audit.json').write_text(json.dumps(audit, indent=2, ensure_ascii=False), encoding='utf-8')
print(json.dumps({s: audit['partitions'][s]['images'] for s in splits}, indent=1))
print('turing cases in curated manifest', audit['turing_cases_in_curated_manifest'])
