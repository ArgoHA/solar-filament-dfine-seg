"""Run a D-FINE-seg sem_seg model and turn the filament mask into instances.

Instances = 8-connected components of the full-res label map (after an optional closing
that bridges small gaps). Same cache format as predict.py, so score.py works unchanged;
every component gets score 1.0 (the wrapper exposes only the argmax map).
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from dfine_seg import load_model


def components(fg: np.ndarray, close: int) -> list[np.ndarray]:
    if close:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * close + 1, 2 * close + 1))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k)
    n, lbl = cv2.connectedComponents(fg, connectivity=8)
    return [(lbl == i).astype(np.uint8) for i in range(1, n)]


def main(model: str, images: Path, stems: Path | None, out: Path, size: int, close: int) -> None:
    net = load_model(model, input_height=size, input_width=size)
    paths = sorted(images.glob("*.jpeg"))
    if stems:
        keep = set(stems.read_text().split())
        paths = [p for p in paths if p.stem in keep]
    res = {}
    for p in tqdm(paths):
        fg = (net(cv2.imread(str(p), cv2.IMREAD_COLOR))[0]["sem_seg"] == 1).cpu().numpy()
        assert fg.shape == (2048, 2048), fg.shape
        res[p.stem] = [
            (1.0, mask_utils.encode(np.asfortranarray(m)))
            for m in components(fg.astype(np.uint8), close)
        ]
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
    ap.add_argument("--size", type=int, default=1280)
    ap.add_argument("--close", type=int, default=0)
    main(**vars(ap.parse_args()))
