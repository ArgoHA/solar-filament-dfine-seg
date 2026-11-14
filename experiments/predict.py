"""Run a D-FINE-seg model on full 2048x2048 frames and cache raw instances.

Low conf threshold here; score.py sweeps the real threshold / min-area / overlap rules.
Output: a pickle {stem: [(score, coco_rle_dict), ...]} sorted by descending score.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from dfine_seg import load_model


def main(model: str, images: Path, stems: Path | None, out: Path, conf: float, size: int) -> None:
    net = load_model(model, conf_thresh=conf, input_height=size, input_width=size)
    paths = sorted(images.glob("*.jpeg"))
    if stems:
        keep = set(stems.read_text().split())
        paths = [p for p in paths if p.stem in keep]
    res = {}
    for p in tqdm(paths):
        o = net(cv2.imread(str(p), cv2.IMREAD_COLOR))[0]
        if "masks" not in o:  # no detections
            res[p.stem] = []
            continue
        masks = o["masks"].bool().cpu().numpy().astype(np.uint8)
        inst = [
            (s, mask_utils.encode(np.asfortranarray(m)))
            for s, m in zip(o["scores"].tolist(), masks)
            if m.any()
        ]
        res[p.stem] = sorted(inst, key=lambda x: -x[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "wb") as f:
        pickle.dump(res, f)
    print(f"{len(res)} images -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--stems", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--size", type=int, default=1280)
    main(**vars(ap.parse_args()))
