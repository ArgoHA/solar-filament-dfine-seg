"""Instance dataset (prepare.py output) -> D-FINE-seg sem_seg layout with the same split.

One uint8 PNG per annotator view: 0 = background, 1 = filament (union of that view's
polygons, rasterised with pycocotools exactly like the scorer does).
"""

import json
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SRC, OUT = ROOT / "data/dataset", ROOT / "data/dataset_sem"
SIZE = 2048


def main() -> None:
    coco = json.loads(ANN.read_text())
    segs = defaultdict(list)
    for a in coco["annotations"]:
        segs[a["image_id"]].extend(a["segmentation"])
    (OUT / "labels").mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(exist_ok=True)
    for im in coco["images"]:
        stem = im["id"]
        dst = OUT / "images" / f"{stem}.jpeg"
        if not dst.exists():
            dst.symlink_to((SRC / "images" / f"{stem}.jpeg").resolve())
        rle = mask_utils.merge(mask_utils.frPyObjects(segs[stem], SIZE, SIZE))
        cv2.imwrite(str(OUT / "labels" / f"{stem}.png"), mask_utils.decode(rle).astype(np.uint8))
    for split in ("train", "val"):
        shutil.copy(SRC / f"{split}.csv", OUT / f"{split}.csv")
    print(f"{len(coco['images'])} views -> {OUT}")


if __name__ == "__main__":
    main()
