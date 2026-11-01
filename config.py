"""Local paths. Edit the defaults below or set the environment variables before running any script."""
import os
from pathlib import Path

# Source dataset root, one folder per body-weight group:
#   DATA_ROOT/data1/images/*.png, DATA_ROOT/data1/annotations/*.json, ..., DATA_ROOT/data4/...
DATA_ROOT = Path(os.environ.get('ABD_DATA_ROOT', '/path/to/dataset'))
N_GROUPS = 4
def group_number(prefix): return int(prefix[-2:])   # the image file-name prefix ends with the group number
def image_dir(n): return DATA_ROOT / f'data{n}' / 'images'
def annotation_dir(n): return DATA_ROOT / f'data{n}' / 'annotations'
# 224 x 224 (and optional 448 x 448) image caches written by prepare_cohort.py and build_cache448.py.
CACHE_DIR = Path(os.environ.get('ABD_CACHE_DIR', '/path/to/cache'))
CACHE224 = CACHE_DIR / 'cache224'
CACHE448 = CACHE_DIR / 'cache448'
# Curated cohort (prepare_cohort.py, stream_features.py): metadata, deduplication, reader-set reservation, fixed features.
COHORT_DIR = Path(os.environ.get('ABD_COHORT_DIR', './cohort'))
# Institution-split experiments (make_site_manifest.py and all training/analysis scripts).
WORK_DIR = Path(os.environ.get('ABD_WORK_DIR', './experiments'))
RESULTS_DIR = Path(os.environ.get('ABD_RESULTS_DIR', './results'))
# Reader study: a text file with one reader-set image ID per line, and the folder with the answer key and reader answer sheets.
READER_IMAGE_LIST = Path(os.environ.get('ABD_READER_LIST', '/path/to/reader_image_ids.txt'))
READER_STUDY_DIR = Path(os.environ.get('ABD_READER_DIR', '/path/to/reader_study'))
READER_KEY_FILE = 'reader_key.xlsx'            # columns: 'Original Filename', 'Renamed Filename'
READER_FILES = [('reader1.xlsx', 'Resident 1'), ('reader2.xlsx', 'Resident 2'), ('reader3.xlsx', 'Pediatric radiologist')]
