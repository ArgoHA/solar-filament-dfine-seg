"""Post-process cached predictions, score pooled PQ on val, or write a Kaggle submission.

PQ follows the organisers' self-evaluation notebook: predictions of an observation are
matched (IoU > 0.5) against every annotator view separately, TP/FP/FN and matched IoU
are pooled over all views of all observations.

  score:  python score.py --preds val.pkl --conf 0.4 0.5 0.6 --min-area 0 200
  submit: python score.py --preds test.pkl --conf 0.4 --min-area 200 --submission sub.csv
"""

import argparse
import itertools
import json
import pickle
from collections import defaultdict
from pathlib import Path

import numpy as np
from pycocotools import mask as mask_utils

ROOT = Path(__file__).resolve().parents[1]
SIZE = 2048


def postprocess(inst, conf: float, min_area: int, keep_frac: float) -> list[dict]:
    """Greedy by score: each instance keeps only unclaimed pixels (Kaggle rejects overlaps)."""
    claimed = np.zeros((SIZE, SIZE), dtype=bool)
    out = []
    for score, rle in inst:
        if score < conf:
            break
        m = mask_utils.decode(rle).astype(bool)
        area0 = m.sum()
        m &= ~claimed
        area = m.sum()
        if area < min_area or area < keep_frac * area0:
            continue
        claimed |= m
        out.append(mask_utils.encode(np.asfortranarray(m.astype(np.uint8))))
    return out


def load_gt(path: Path) -> dict[str, list[list[dict]]]:
    """{obs_stem: [view0_rles, view1_rles, ...]}"""
    coco = json.loads(path.read_text())
    by_view = defaultdict(list)
    for a in coco["annotations"]:
        by_view[a["image_id"]].append(a["segmentation"])
    gt = defaultdict(list)
    for im in coco["images"]:
        rles = [mask_utils.merge(mask_utils.frPyObjects(s, SIZE, SIZE)) for s in by_view[im["id"]]]
        gt[Path(im["file_name"]).stem].append(rles)
    return gt


def match(gt: list[dict], pred: list[dict]) -> tuple[int, int, int, float]:
    if not gt or not pred:
        return 0, len(pred), len(gt), 0.0
    iou = np.asarray(mask_utils.iou(pred, gt, [0] * len(gt)))  # (n_pred, n_gt)
    # IoU > 0.5 matches are unique, so no assignment solver is needed
    hits = iou > 0.5
    tp = int(hits.sum())
    return tp, len(pred) - tp, len(gt) - tp, float(iou[hits].sum())


def pq(gt, preds) -> dict:
    tp = fp = fn = 0
    s = 0.0
    for stem, views in gt.items():
        for view in views:
            a, b, c, d = match(view, preds.get(stem, []))
            tp, fp, fn, s = tp + a, fp + b, fn + c, s + d
    return {
        "pq": s / max(tp + 0.5 * fp + 0.5 * fn, 1e-9),
        "sq": s / max(tp, 1),
        "rq": tp / max(tp + 0.5 * fp + 0.5 * fn, 1e-9),
        "tp": tp,
        "fp": fp,
        "fn": fn,
    }


def write_submission(preds: dict, path: Path) -> None:
    rows = ["filament_id,segmentation_rle"]
    for stem in sorted(preds):
        for i, rle in enumerate(preds[stem], 1):
            counts = rle["counts"].decode("ascii")
            assert not set(",\"'\n\r") & set(counts)
            rows.append(f"{stem}_{i},{counts}")
    path.write_text("\n".join(rows) + "\n")
    print(f"{len(rows) - 1} instances over {len(preds)} images -> {path}")


def main(a) -> None:
    raw = pickle.loads(a.preds.read_bytes())
    if a.submission:
        conf, min_area, keep_frac = a.conf[0], a.min_area[0], a.keep_frac[0]
        preds = {k: postprocess(v, conf, min_area, keep_frac) for k, v in raw.items()}
        write_submission(preds, a.submission)
        return
    gt = load_gt(a.gt)
    raw = {k: v for k, v in raw.items() if k in gt}
    grid = itertools.product(a.conf, a.min_area, a.keep_frac)
    best = None
    for conf, min_area, keep_frac in grid:
        preds = {k: postprocess(v, conf, min_area, keep_frac) for k, v in raw.items()}
        r = pq(gt, preds)
        print(
            f"conf={conf:.2f} min_area={min_area:4d} keep_frac={keep_frac:.1f} "
            + " ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items())
        )
        if best is None or r["pq"] > best[0]["pq"]:
            best = (r, conf, min_area, keep_frac)
    print("best:", best)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", type=Path, required=True)
    ap.add_argument("--gt", type=Path, default=ROOT / "data/splits/val_gt.json")
    # several values -> grid sweep on val; a submission uses the first of each
    ap.add_argument("--conf", type=float, nargs="+", default=[0.4])
    ap.add_argument("--min-area", type=int, nargs="+", default=[200])
    ap.add_argument("--keep-frac", type=float, nargs="+", default=[0.5])
    ap.add_argument("--submission", type=Path)
    main(ap.parse_args())
