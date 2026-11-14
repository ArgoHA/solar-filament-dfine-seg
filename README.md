# Solar filament segmentation with D-FINE-seg

Solution to the [Solar Filament Segmentation Challenge 2026](https://www.kaggle.com/competitions/filament-segmentation-2026)
(IEEE BigData 2026 Cup): segment every solar filament in full-disk GONG H-alpha observations
(2048 x 2048), one mask per filament.

**One small real-time model, one forward pass, no ensembles, no test-time augmentation.**
A [D-FINE-seg](https://github.com/ArgoHA/D-FINE-seg)-S semantic segmentation model (8M
parameters) runs on the native-resolution frame. Connected components of its probability map,
filtered by their mean confidence, are the filament instances.

| | validation (144 obs.) | annotators vs each other | public LB |
|---|---|---|---|
| Panoptic Quality | **0.429** | 0.334 (0.340 on all 296 multi-view obs.) | 0.38 |
| SQ / RQ | 0.675 / 0.635 | 0.631 / 0.530 | |
| mean image Dice | 0.668 | 0.633 | |
| mean Dice of matched filaments | 0.803 | 0.771 | |
| one-to-many (GT split, % of GT) | 4.1% | 3.4% | |
| many-to-one (merges, % of predictions) | 3.4% | 3.4% | |
| **end-to-end latency** | **76 ms / image** | | |

Latency: JPEG path -> final non-overlapping RLEs, batch 1, PyTorch, RTX 5070 Ti (16 GB):
decode 11 ms + network 57 ms + post-processing 9 ms. Peak allocated GPU memory 1.5 GB (2.1 GB
reserved). The full test set (180 images) takes about 15 s including model loading.

The model is scored against each annotator separately. It agrees with them better than they
agree with each other on PQ, SQ, RQ and Dice, with fragmentation and merging close to theirs.

## Reproduce

```bash
python3.13 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# competition data (Kaggle CLI, rules accepted on the competition page)
kaggle competitions download -c filament-segmentation-2026 -p data && unzip -q data/filament-segmentation-2026.zip -d data
# -> data/MAGFiLO_1.0_Kaggle_2026/{train,test}

python -m filaments predict  --data data/MAGFiLO_1.0_Kaggle_2026 --out submission.csv   # test -> Kaggle CSV
python -m filaments evaluate --data data/MAGFiLO_1.0_Kaggle_2026                        # validation metrics
```

`submission.csv` is byte-identical to the file submitted to Kaggle
(`submissions/sem_s2048_prob_conf.csv`). The trained weights ship in the repo
(`weights/sem_s_2048.pt`, 32 MB), so no other file is needed.

**Notebook:** [`notebooks/pipeline.ipynb`](notebooks/pipeline.ipynb) walks through the whole
pipeline with outputs: data, dataset preparation, training, step-by-step inference, rubric
metrics with distributions, the submission and latency.

### Training (optional, ~3 h on one 16 GB GPU)

```bash
python -m filaments prepare --data data/MAGFiLO_1.0_Kaggle_2026 --out data/dataset_sem
DFINE_SEG_CONFIG_DIR=configs dfine train      # -> output/models/sem_s_2048_<date>/model.pt
python -m filaments evaluate --data data/MAGFiLO_1.0_Kaggle_2026 --weights output/models/sem_s_2048_<date>/model.pt
```

COCO-pretrained D-FINE-seg weights download automatically from the Hugging Face Hub.

## Method

1. **Data.** Each observation has 1-3 independent annotator views. Every view is a separate
   training sample (the JPEG + that annotator's mask). Training on consensus or union masks
   scored worse (0.421 / 0.408 PQ). Validation holds out 20% of observation *dates*
   (144 observations, 248 views), so frames of the same day never straddle the split.
2. **Model.** D-FINE-seg-S, `task: sem_seg` (HGNetV2 backbone, hybrid encoder, multi-scale
   feature fuser, 2-class head), initialised from COCO instance-segmentation weights. Input is
   the native 2048 x 2048 frame: logits are predicted at 1/4 resolution and filaments are only
   5-15 px wide, so resolution was the largest gain (1280 -> 2048: +0.057 PQ).
   60 epochs, batch 2, flips and 90-degree rotations, EMA, bf16 AMP (`configs/config.yaml`).
3. **Instances.** p(filament) >= 0.4 -> closing (disk r = 4 px) -> 8-connected components
   >= 200 px -> keep a component if its mean probability >= 0.90. An unmatched prediction still
   counts in PQ's denominator, so uncertain components cost more than they bring; the cut adds +0.010 PQ (paired bootstrap 95% CI
   [+0.005, +0.015], confirmed with split-half cross-fitting). Components never overlap.

Everything else we tried (instance segmentation, tiles, fusion, larger models, input
preprocessing, label variants, test-time augmentation) is in [`experiments/`](experiments/README.md).
TTA (+0.005 PQ) and instance-TTA ensembles (+0.003) are within validation noise and cost
7-200x the compute.

## Layout

```
filaments/          data.py (split, datasets), inference.py (model + post-processing),
                    metrics.py (PQ, Dice/IoU distributions, one-to-many / many-to-one), viz.py
configs/config.yaml D-FINE-seg training config of the submitted model
weights/            trained model
notebooks/          end-to-end pipeline notebook
submissions/        the CSV submitted to Kaggle
report/             4-page technical report
experiments/        research scripts and the full ablation table
```

No MAGFiLO data beyond the competition's training annotations was used.

## License

Apache-2.0. Built on D-FINE-seg (Apache-2.0).
