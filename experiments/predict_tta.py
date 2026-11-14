"""Dihedral TTA for instance models (full frame or tiled): 8 flips/rot90s, mask voting.

Predictions from all views are mapped back to the original frame and clustered greedily by
score: a seed takes the best IoU > 0.5 match from every other view. The fused mask keeps
pixels covered by at least --vote of the cluster's members; the fused score is the sum of
member scores / 8, so an instance seen in few views gets a low score. Same cache format as
predict.py, so score.py sweeps the confidence threshold.
"""

import argparse
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils
from tqdm import tqdm

from dfine_seg import load_model
from predict_tiles import tiled

VIEWS = [(k, f) for f in (False, True) for k in range(4)]


def fwd(a: np.ndarray, k: int, flip: bool) -> np.ndarray:
    return np.ascontiguousarray(np.rot90(a[:, ::-1] if flip else a, k))


def inv(a: np.ndarray, k: int, flip: bool) -> np.ndarray:
    a = np.rot90(a, -k)
    return np.ascontiguousarray(a[:, ::-1] if flip else a)


def full_frame(net, img: np.ndarray) -> list[tuple]:
    o = net(img)[0]
    if "masks" not in o:
        return []
    masks = o["masks"].bool().cpu().numpy().astype(np.uint8)
    return [(s, m) for s, m in zip(o["scores"].tolist(), masks) if m.any()]


def merge(preds: list[tuple], vote: float) -> list[tuple]:
    """preds: [(score, rle, view)] -> [(fused score, rle)], best first."""
    preds = sorted(preds, key=lambda x: -x[0])
    if not preds:
        return []
    rles = [p[1] for p in preds]
    iou = np.asarray(mask_utils.iou(rles, rles, [0] * len(rles)))
    views = np.array([p[2] for p in preds])
    free = np.ones(len(preds), dtype=bool)
    out = []
    for i in range(len(preds)):
        if not free[i]:
            continue
        members = [i]
        for v in set(views.tolist()) - {views[i]}:
            cand = np.flatnonzero(free & (views == v) & (iou[i] > 0.5))
            if len(cand):
                members.append(cand[np.argmax(iou[i, cand])])
        free[members] = False
        acc = np.sum([mask_utils.decode(rles[j]) for j in members], 0, dtype=np.int32)
        m = (acc >= max(1, vote * len(members))).astype(np.uint8)
        if m.any():
            score = sum(preds[j][0] for j in members) / len(VIEWS)
            out.append((score, mask_utils.encode(np.asfortranarray(m))))
    return sorted(out, key=lambda x: -x[0])


def predict(net, img: np.ndarray, tile: int, stride: int, vote: float) -> list[tuple]:
    preds = []
    for v, (k, f) in enumerate(VIEWS):
        x = fwd(img, k, f)
        if tile:
            inst = [(s, mask_utils.decode(r)) for s, r in tiled(net, x, tile, stride)]
        else:
            inst = full_frame(net, x)
        preds += [(s, mask_utils.encode(np.asfortranarray(inv(m, k, f))), v) for s, m in inst]
    return merge(preds, vote)


def main(model, images, stems, out, conf, size, tile, stride, vote) -> None:
    net = load_model(model, conf_thresh=conf, input_height=tile or size, input_width=tile or size)
    paths = sorted(images.glob("*.jpeg"))
    if stems:
        keep = set(stems.read_text().split())
        paths = [p for p in paths if p.stem in keep]
    res = {}
    for p in tqdm(paths):
        res[p.stem] = predict(net, cv2.imread(str(p), cv2.IMREAD_COLOR), tile, stride, vote)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(pickle.dumps(res))
    print(f"{len(res)} images -> {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--stems", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--conf", type=float, default=0.1)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--tile", type=int, default=0, help="0 = full frame, else tiled inference")
    ap.add_argument("--stride", type=int, default=512)
    ap.add_argument("--vote", type=float, default=0.5, help="pixel kept if >= vote of members")
    main(**vars(ap.parse_args()))
