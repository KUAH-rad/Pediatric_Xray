# Pediatric Abdominal Radiograph Classification: Institution-Split Evaluation

Code for the methods described in the manuscript: training on one institution, external testing on six institutions,
institutional-association analyses, institution-balanced resampling, saliency-map localization, and statistics.

**No data, images, labels, patient identifiers, reader responses, model weights or checkpoints are included.**
The source dataset is available from its data provider under its own data-use terms.

## Setup

```bash
pip install -r requirements.txt
```

Edit the paths in `config.py`, or set these environment variables:

| Variable | Contents |
|---|---|
| `ABD_DATA_ROOT` | dataset root with one folder per body-weight group: `data1/images`, `data1/annotations` … `data4/images`, `data4/annotations` |
| `ABD_CACHE_DIR` | folder for the 224 × 224 (and 448 × 448) image caches |
| `ABD_COHORT_DIR` | curated cohort outputs (default `./cohort`) |
| `ABD_WORK_DIR` | experiment outputs (default `./experiments`) |
| `ABD_RESULTS_DIR` | final statistics (default `./results`) |
| `ABD_READER_LIST` | text file with one reader-set image ID per line |
| `ABD_READER_DIR` | reader-study answer key and answer sheets (file names set in `config.py`) |

A CUDA GPU is required for training and saliency maps. The statistics scripts run on a CPU.

## Pipeline (run in this order)

| Step | Script | Purpose |
|---|---|---|
| 1 | `prepare_cohort.py` | Read annotation metadata, build the 224 × 224 cache, merge exact-pixel duplicates with patient IDs into patient-image groups, remove cross-label pixel duplicates, reserve reader-set groups |
| 2 | `stream_features.py` | Frozen DINOv3 features (for the linear probes and the institution probe) |
| 3 | `make_site_manifest.py` | Institution split: development institution A (train/validation/internal test by group hash), external institutions B–G (test); groups shared between A and other institutions are excluded |
| 4 | `run_site_split.py` | Main models (DINOv3 ViT-S/16, ConvNeXt V2 Atto; three training runs each), metadata-only logistic regression, frozen-feature linear probes, institution probe |
| 5 | `run_ablations.py` | Leave-one-institution-out multicenter training, size-matched training, label curation, fine-tuning depth, loss weighting, training duration |
| 6 | `build_cache448.py`, `run_ablations_bc.py` | Input resolution, CLAHE and architecture comparisons |
| 7 | `run_site_balanced.py`, `run_seed_replicates.py` | Institution-balanced resampling (weight = min{p(y) / p(y \| institution), 10}) and repeated training runs |
| 8 | `run_cam_flip.py`, `run_occlusion.py`, `run_cam_variants.py` | Grad-CAM pointing game, occlusion sensitivity, intermediate-layer Grad-CAM maps, horizontal-flip test |
| 9 | `shortcut_mechanism.py`, `shortcut_robustness.py` | Correlation of institutional label proportions with predicted probabilities on normal-labeled images; permutation, bootstrap and leave-one-institution-out checks |
| 10 | `analyze_all.py`, `operating_point_ci.py`, `seed_replicate_auroc.py` | Patient-image-group bootstrap CIs, paired model comparison, operating points, calibration, reader comparison |

Notes:
- Institution codes are the first two characters of the source patient ID; manuscript letters A–G are mapped in `make_site_manifest.py` and `analyze_all.py`.
- The seeds in the code (2026, 2027, 2028) only identify the three training runs.
- `shortcut_mechanism.py` and `shortcut_robustness.py` also compare with an earlier mixed-institution random-split model. They read its predictions from `ABD_COHORT_DIR`. The code that trained that comparator is not included.

## License

MIT (see `LICENSE`).
