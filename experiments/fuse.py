"""Hybrid: instance model decides grouping + confidence, sem_seg model decides pixels.

Most confident instance first, each claims the unclaimed sem_seg foreground inside its own
mask dilated by `grow` px. Optionally keeps its own mask pixels too (`own=1`).
Sweeps on val (pooled PQ) or writes a fused cache for score.py --submission.

  python fuse.py --inst val_inst.pkl --sem val_sem_c0.pkl --conf 0.5 0.6 --grow 0 4 8
"""

import argparse
import itertools
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from score import ROOT, SIZE, load_gt, pq


def sem_fg(inst) -> np.ndarray:
    """Union of the sem_seg cache's components = the foreground map."""
    if not inst:
        return np.zeros((SIZE, SIZE), dtype=bool)
    return mask_utils.decode(mask_utils.merge([r for _, r in inst], intersect=0)).astype(bool)


def fuse(inst, fg, conf: float, grow: int, min_area: int, own: bool) -> list:
    claimed = np.zeros((SIZE, SIZE), dtype=bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * grow + 1, 2 * grow + 1))
    out = []
    for score, rle in inst:
        if score < conf:
            break
        x, y, w, h = (int(v) for v in mask_utils.toBbox(rle))
        y0, y1 = max(y - grow, 0), min(y + h + grow + 1, SIZE)
        x0, x1 = max(x - grow, 0), min(x + w + grow + 1, SIZE)
        m = mask_utils.decode(rle)[y0:y1, x0:x1]
        region = cv2.dilate(m, k).astype(bool) if grow else m.astype(bool)
        new = region & fg[y0:y1, x0:x1]
        if own:
            new |= m.astype(bool)
        new &= ~claimed[y0:y1, x0:x1]
        if new.sum() < min_area:
            continue
        claimed[y0:y1, x0:x1] |= new
        full = np.zeros((SIZE, SIZE), dtype=np.uint8)
        full[y0:y1, x0:x1] = new
        out.append((score, mask_utils.encode(np.asfortranarray(full))))
    return out


def main(a) -> None:
    inst = pickle.loads(a.inst.read_bytes())
    sem = pickle.loads(a.sem.read_bytes())
    fgs = {k: sem_fg(sem.get(k, [])) for k in inst}
    if a.out:
        c, g, m, o = a.conf[0], a.grow[0], a.min_area[0], a.own[0]
        res = {k: fuse(v, fgs[k], c, g, m, bool(o)) for k, v in inst.items()}
        a.out.write_bytes(pickle.dumps(res))
        print(f"fused {len(res)} images -> {a.out}")
        return
    gt = load_gt(a.gt)
    best = None
    for c, g, m, o in itertools.product(a.conf, a.grow, a.min_area, a.own):
        preds = {k: [r for _, r in fuse(inst[k], fgs[k], c, g, m, bool(o))] for k in gt}
        r = pq(gt, preds)
        print(
            f"conf={c:.2f} grow={g} min_area={m} own={o} "
            + " ".join(
                f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
            ),
            flush=True,
        )
        if best is None or r["pq"] > best[0]["pq"]:
            best = (r, c, g, m, o)
    print("best:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--inst", type=Path, required=True)
    ap.add_argument("--sem", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=ROOT / "data/splits/val_gt.json")
    ap.add_argument("--conf", type=float, nargs="+", default=[0.6])
    ap.add_argument("--grow", type=int, nargs="+", default=[4])
    ap.add_argument("--min-area", type=int, nargs="+", default=[200])
    ap.add_argument("--own", type=int, nargs="+", default=[0])
    ap.add_argument("--out", type=Path)
    main(ap.parse_args())
