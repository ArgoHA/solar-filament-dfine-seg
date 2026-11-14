"""sem_seg layout with observation-level training targets (same split and samples as
prepare_sem.py). Every train view gets its observation's combined mask:

  consensus: 1 where all annotator views mark filament, 0 where none do, 255 (ignored by
             the loss) where they disagree
  union:     1 where any view marks filament

Observations with a single view keep it unchanged. Val labels stay per-view, so checkpoint
selection matches the per-view runs.

  python prepare_sem_consensus.py consensus|union
"""

import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SRC = ROOT / "data/dataset_sem"
SIZE, IGNORE = 2048, 255


def main(mode: str) -> None:
    OUT = ROOT / {"consensus": "data/dataset_sem_cons", "union": "data/dataset_sem_union"}[mode]
    coco = json.loads(ANN.read_text())
    segs = defaultdict(list)
    for a in coco["annotations"]:
        segs[a["image_id"]].extend(a["segmentation"])
    views = defaultdict(list)  # observation -> view ids
    for im in coco["images"]:
        views[Path(im["file_name"]).stem].append(im["id"])
    train = set(Path(p).stem for p in (SRC / "train.csv").read_text().split())
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    n_ign = n_fg = 0
    for ids in views.values():
        masks = [
            mask_utils.decode(mask_utils.merge(mask_utils.frPyObjects(segs[i], SIZE, SIZE)))
            for i in ids
        ]
        votes = np.sum(masks, 0)
        if mode == "union":
            cons = (votes > 0).astype(np.uint8)
        else:
            cons = np.where(votes == len(ids), 1, np.where(votes == 0, 0, IGNORE)).astype(np.uint8)
        for i, m in zip(ids, masks):
            dst = OUT / "images" / f"{i}.jpeg"
            if not dst.exists():
                dst.symlink_to((SRC / "images" / f"{i}.jpeg").resolve())
            lbl = cons if i in train else m.astype(np.uint8)
            if i in train:
                n_ign += int((lbl == IGNORE).sum())
                n_fg += int((lbl == 1).sum())
            cv2.imwrite(str(OUT / "labels" / f"{i}.png"), lbl)
    for split in ("train", "val"):
        shutil.copy(SRC / f"{split}.csv", OUT / f"{split}.csv")
    print(f"{len(coco['images'])} views -> {OUT}; train fg px {n_fg}, ignored px {n_ign}")


if __name__ == "__main__":
    main(sys.argv[1])
