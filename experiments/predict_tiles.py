"""Tiled inference for crop-trained models: 1024 tiles at stride 512 over the 2048 frame.

Each instance is kept only by the tile whose core contains its box centre (core = the
central stride-wide band, stretched to the frame edge for border tiles), so tiles partition
the frame and nothing is predicted twice. Same cache format as predict.py.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from dfine_seg import load_model

SIZE = 2048


def core(o: int, tile: int, stride: int) -> tuple[int, int]:
    lo = 0 if o == 0 else o + (tile - stride) // 2
    hi = SIZE if o + tile >= SIZE else o + (tile + stride) // 2
    return lo, hi


def tiled(net, img: np.ndarray, tile: int, stride: int) -> list[tuple]:
    """[(score, rle)] over the full frame, each instance from the tile whose core owns it."""
    offs = range(0, SIZE - tile + 1, stride)
    inst = []
    for y0 in offs:
        for x0 in offs:
            o = net(np.ascontiguousarray(img[y0 : y0 + tile, x0 : x0 + tile]))[0]
            (ylo, yhi), (xlo, xhi) = core(y0, tile, stride), core(x0, tile, stride)
            if "masks" not in o:  # no detections in this tile
                continue
            masks = o["masks"].bool().cpu().numpy()
            for s, b, m in zip(o["scores"].tolist(), o["boxes"].tolist(), masks):
                cx, cy = x0 + (b[0] + b[2]) / 2, y0 + (b[1] + b[3]) / 2
                if not (xlo <= cx < xhi and ylo <= cy < yhi) or not m.any():
                    continue
                full = np.zeros((SIZE, SIZE), dtype=np.uint8)
                full[y0 : y0 + tile, x0 : x0 + tile] = m
                inst.append((s, mask_utils.encode(np.asfortranarray(full))))
    return inst


def main(model, images, stems, out, conf, tile, stride, device) -> None:
    net = load_model(model, conf_thresh=conf, input_height=tile, input_width=tile, device=device)
    paths = sorted(images.glob("*.jpeg"))
    if stems:
        keep = set(stems.read_text().split())
        paths = [p for p in paths if p.stem in keep]
    res = {}
    for p in tqdm(paths):
        inst = tiled(net, cv2.imread(str(p), cv2.IMREAD_COLOR), tile, stride)
        res[p.stem] = sorted(inst, key=lambda x: -x[0])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(res))
    print(f"{len(res)} images -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--stems", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--tile", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--device", default="cuda")
    main(**vars(ap.parse_args()))
