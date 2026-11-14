"""Probability maps (one or more dirs, averaged) -> threshold -> closing -> components -> PQ.

A component's score is its mean probability; --min-score drops weak ones (mostly FPs).

  python sem_from_probs.py probs/a probs/b --thresh 0.4 0.5 --close 4 --min-area 300 400
  python sem_from_probs.py probs/a_test --thresh 0.5 --close 4 --min-area 300 --submission x.csv
  python sem_from_probs.py probs/a_val --thresh 0.4 --min-area 200 --dump preds/a.pkl  (compare.py)
"""

import argparse
import itertools
import pickle
from pathlib import Path

import cv2
import numpy as np
from pycocotools import mask as mask_utils

from predict_sem import components
from score import ROOT, load_gt, pq, write_submission


def scored(prob: np.ndarray, thresh: float, close: int, min_area: int) -> list[tuple]:
    """[(mean prob, rle)] of the components with at least min_area pixels."""
    fg = (prob >= thresh * 255).astype(np.uint8)
    out = []
    for m in components(fg, close):
        b = m.astype(bool)
        if b.sum() >= min_area:
            out.append((float(prob[b].mean()) / 255, mask_utils.encode(np.asfortranarray(m))))
    return out


def instances(comps: list[tuple], min_score: float) -> list[dict]:
    return [rle for s, rle in comps if s >= min_score]


def load_probs(dirs: list[Path], stem: str) -> np.ndarray:
    return np.mean([cv2.imread(str(d / f"{stem}.png"), 0).astype(np.float32) for d in dirs], 0)


def main(a) -> None:
    stems = sorted(p.stem for p in a.dirs[0].glob("*.png"))
    probs = {s: load_probs(a.dirs, s) for s in stems}
    if a.submission:
        args = a.thresh[0], a.close[0], a.min_area[0]
        preds = {s: instances(scored(probs[s], *args), a.min_score[0]) for s in stems}
        write_submission(preds, a.submission)
        return
    if a.dump:  # predict.py cache format: {stem: [(score, rle)]}, best first
        args = a.thresh[0], a.close[0], a.min_area[0]
        cache = {s: sorted(scored(probs[s], *args), key=lambda x: -x[0]) for s in stems}
        a.dump.write_bytes(pickle.dumps(cache))
        return
    gt = load_gt(a.gt)
    for t, c, m in itertools.product(a.thresh, a.close, a.min_area):
        comps = {s: scored(probs[s], t, c, m) for s in gt}
        for ms in a.min_score:
            r = pq(gt, {s: instances(comps[s], ms) for s in gt})
            print(
                f"thresh={t:.2f} close={c} min_area={m} min_score={ms:.2f} "
                + " ".join(
                    f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()
                ),
                flush=True,
            )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("dirs", type=Path, nargs="+")
    ap.add_argument("--gt", type=Path, default=ROOT / "data/splits/val_gt.json")
    ap.add_argument("--thresh", type=float, nargs="+", default=[0.5])
    ap.add_argument("--close", type=int, nargs="+", default=[4])
    ap.add_argument("--min-area", type=int, nargs="+", default=[300])
    ap.add_argument("--min-score", type=float, nargs="+", default=[0.0])
    ap.add_argument("--submission", type=Path)
    ap.add_argument("--dump", type=Path)
    main(ap.parse_args())
