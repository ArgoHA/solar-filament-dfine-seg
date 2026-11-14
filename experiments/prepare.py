"""MAGFiLO COCO json -> D-FINE-seg YOLO layout, one sample per annotator view, split by date.

Every annotator view becomes its own image stem (symlink to the shared jpeg), so the
same observation is never split across train/val. Writes:
  data/dataset/{images,labels}/<view_id>.*, train.csv, val.csv
  data/splits/val_obs.txt (observation stems), data/splits/val_gt.json (COCO subset)
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026"
ANN = RAW / "train/MAGFiLO_1.0_Annotations_kaggle2026_train.json"
SIZE = 2048


def main(val_frac: float, seed: int, out: Path) -> None:
    coco = json.loads(ANN.read_text())
    anns = defaultdict(list)
    for a in coco["annotations"]:
        anns[a["image_id"]].append(a)

    dates = sorted({im["file_name"][:8] for im in coco["images"]})
    random.Random(seed).shuffle(dates)
    val_dates = set(dates[: round(len(dates) * val_frac)])

    img_dir, lbl_dir = out / "images", out / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)
    splits = {"train": [], "val": []}
    val_images, val_obs = [], set()
    for im in coco["images"]:
        stem, src = im["id"], RAW / "train/train_images" / im["file_name"]
        dst = img_dir / f"{stem}.jpeg"
        if not dst.exists():
            dst.symlink_to(src)
        lines = []
        for a in anns[stem]:
            poly = a["segmentation"][0]
            xy = " ".join(f"{min(max(v / SIZE, 0), 1):.6f}" for v in poly)
            lines.append(f"0 {xy}")
        (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        split = "val" if im["file_name"][:8] in val_dates else "train"
        splits[split].append(dst.name)
        if split == "val":
            val_images.append(im)
            val_obs.add(Path(im["file_name"]).stem)

    for name, files in splits.items():
        (out / f"{name}.csv").write_text("\n".join(sorted(files)) + "\n")
    sp = ROOT / "data/splits"
    sp.mkdir(parents=True, exist_ok=True)
    (sp / "val_obs.txt").write_text("\n".join(sorted(val_obs)) + "\n")
    ids = {im["id"] for im in val_images}
    sub = {
        **coco,
        "images": val_images,
        "annotations": [a for a in coco["annotations"] if a["image_id"] in ids],
    }
    (sp / "val_gt.json").write_text(json.dumps(sub))
    print({k: len(v) for k, v in splits.items()}, "val obs:", len(val_obs))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--val-frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=ROOT / "data/dataset")
    main(**{k.replace("-", "_"): v for k, v in vars(p.parse_args()).items()})
