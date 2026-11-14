# Experiments

Research scripts behind the ablations in the report. **They are not needed to reproduce the
submission** (use `python -m filaments` or `notebooks/pipeline.ipynb` for that). They expect the
original working layout (`data/raw/MAGFiLO_1.0_Kaggle_2026`, `data/splits/`, `output/models/`) and
were run with D-FINE-seg's `dfine train` plus per-experiment configs.

| script | purpose |
|---|---|
| `prepare.py`, `prepare_sem.py` | COCO -> D-FINE-seg instance (YOLO polygons) / semantic (PNG) datasets, one sample per annotator view, date split |
| `prepare_crops.py` | native-resolution 1024 crops (3x3, stride 512) for the tiled instance model |
| `prepare_sem_consensus.py` | consensus (all views agree, disagreement ignored) or union labels |
| `preproc.py`, `prepare_sem_pre.py` | limb-darkening flattening + CLAHE input channels |
| `predict*.py` | cached predictions: instance (full frame, tiles, 8x dihedral TTA), semantic (argmax / probabilities) |
| `score.py`, `sem_from_probs.py` | post-processing sweeps and pooled PQ on validation, submission writing |
| `fuse.py` | instance + semantic fusion (instance groups, semantic pixels) |
| `compare.py` | paired bootstrap over observations: 95% CI of PQ differences |
| `bench.py` | end-to-end latency of every candidate |

## Results (validation: 144 observations / 248 annotator views, pooled PQ)

Inter-annotator agreement on the same scorer: PQ 0.334 on this split (0.340 on all multi-view
training observations). Paired-bootstrap noise between two pipelines is about +-0.013 PQ.

| run | task | input | val PQ | notes |
|---|---|---|---|---|
| inst S 1280 | instance | full frame at 1280 | 0.397 | conf 0.60 |
| sem S 1280 | semantic | full frame at 1280 | 0.360 | argmax, close 4, min area 400 |
| sem S 2048 | semantic | native 2048 | 0.417 | argmax; resolution +0.057 |
| inst S 2048 | instance | native 2048 | 0.414 | conf 0.55 |
| inst S crops -> tiles | instance | native 1024 tiles | 0.421 | trained on crops, inferred on 9 overlapping tiles |
| inst S crops -> full frame | instance | train crops, infer 2048 | 0.286 | confidences collapse: scale mismatch |
| inst M crops -> tiles | instance | native 1024 tiles | 0.418 | M = S |
| inst M 1024 | instance | full frame at 1024 | 0.387 | resolution > capacity |
| sem M 2048 | semantic | native 2048 | 0.418 | M = S |
| fusion (inst + sem) | hybrid | | 0.414-0.423 | no significant gain |
| sem S 2048, p >= 0.4 | semantic | native 2048 | 0.419 | probability threshold instead of argmax |
| **sem S 2048, p >= 0.4 + mean-p >= 0.90** | semantic | native 2048 | **0.429** | **submitted**; cut: +0.010 [+0.005, +0.015] |
| + 8x dihedral TTA | semantic | native 2048 | 0.434 | 7x slower |
| consensus labels | semantic | native 2048 | 0.421 | recall drops |
| union labels | semantic | native 2048 | 0.408 | masks too wide |
| limb-flattened + CLAHE input | semantic | native 2048 | 0.427 | no gain |
| inst S 2048 + TTA | instance | native 2048 | 0.428 | ~12 s/image |
| inst S tiles + TTA | instance | native 1024 tiles | 0.432 | ~17 s/image |
