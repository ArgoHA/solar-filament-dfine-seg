"""Native-resolution crops for instance training (COCO json, since a filament cut by the
crop border can split into several polygons).

train: every annotator view tiled 3x3 (1024 crops, stride 512). val: one view per
observation, 2x2 non-overlapping tiles (only for checkpoint selection; PQ is scored on
full frames by score.py). Same date split as prepare.py.
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026"
ANN = RAW / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SIZE = 2048
MIN_PIECE = 50  # px; slivers left by the crop border are dropped


def crop_polys(full: np.ndarray, x0: int, y0: int, crop: int) -> list[list[float]]:
    m = np.ascontiguousarray(full[y0 : y0 + crop, x0 : x0 + crop])
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    polys = []
    for c in contours:
        if cv2.contourArea(c) >= MIN_PIECE and len(c) >= 3:
            polys.append(c.reshape(-1).astype(float).tolist())
    return polys


def main(crop: int, stride: int, out: Path) -> None:
    coco = json.loads(ANN.read_text())
    val_obs = set((ROOT / "data/splits/val_obs.txt").read_text().split())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)
    (out / "images").mkdir(parents=True, exist_ok=True)
    splits = {
        s: {"images": [], "annotations": [], "categories": [{"id": 1, "name": "filament"}]}
        for s in ("train", "val")
    }
    seen_val = set()
    for im in coco["images"]:
        stem = Path(im["file_name"]).stem
        split = "val" if stem in val_obs else "train"
        if split == "val":
            if stem in seen_val:
                continue
            seen_val.add(stem)
        step = crop if split == "val" else stride
        offs = range(0, SIZE - crop + 1, step)
        img = cv2.imread(str(RAW / "train/train_images" / im["file_name"]), cv2.IMREAD_GRAYSCALE)
        masks = [
            mask_utils.decode(
                mask_utils.merge(mask_utils.frPyObjects(a["segmentation"], SIZE, SIZE))
            )
            for a in anns[im["id"]]
        ]
        for y0 in offs:
            for x0 in offs:
                name = f"{im['id']}_{y0}_{x0}.jpg"
                if not (out / "images" / name).exists():
                    cv2.imwrite(
                        str(out / "images" / name),
                        img[y0 : y0 + crop, x0 : x0 + crop],
                        [cv2.IMWRITE_JPEG_QUALITY, 98],
                    )
                img_id = len(splits[split]["images"])
                splits[split]["images"].append(
                    {"id": img_id, "file_name": name, "width": crop, "height": crop}
                )
                for m in masks:
                    polys = crop_polys(m, x0, y0, crop)
                    if not polys:
                        continue
                    rle = mask_utils.merge(mask_utils.frPyObjects(polys, crop, crop))
                    splits[split]["annotations"].append(
                        {
                            "id": len(splits[split]["annotations"]),
                            "image_id": img_id,
                            "category_id": 1,
                            "iscrowd": 0,
                            "segmentation": polys,
                            "area": float(mask_utils.area(rle)),
                            "bbox": mask_utils.toBbox(rle).tolist(),
                        }
                    )
    for s, d in splits.items():
        (out / f"{s}.json").write_text(json.dumps(d))
        print(s, len(d["images"]), "crops,", len(d["annotations"]), "instances")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--crop", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--out", type=Path, default=ROOT / "data/dataset_crops1024")
    main(**vars(ap.parse_args()))
