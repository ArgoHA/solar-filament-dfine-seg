"""Paired bootstrap of pooled val PQ between post-processed prediction caches.

Resamples observations (all views of an observation stay together) and reports each
run's PQ and every pairwise difference with a 95% interval.

  python compare.py tiles=preds/a.pkl:0.5:0 sem=preds/b.pkl:0.5:400   (name=path:conf:min_area)
"""

import argparse
import itertools
import pickle

import numpy as np

from score import ROOT, load_gt, match, postprocess


def totals(gt, raw, conf, min_area) -> np.ndarray:
    """(n_obs, 4): tp, fp, fn, iou_sum per observation."""
    rows = []
    for stem, views in gt.items():
        pred = postprocess(raw.get(stem, []), conf, min_area, 0.5)
        rows.append(np.sum([match(v, pred) for v in views], axis=0))
    return np.asarray(rows, dtype=float)


def pq(t: np.ndarray) -> np.ndarray:
    tp, fp, fn, s = np.moveaxis(t, -1, 0)
    return s / (tp + 0.5 * fp + 0.5 * fn)


def main(runs: list[str], n_boot: int) -> None:
    gt = load_gt(ROOT / "data/splits/val_gt.json")
    tot = {}
    for spec in runs:
        name, rest = spec.split("=", 1)
        path, conf, min_area = rest.rsplit(":", 2)
        raw = pickle.loads((ROOT / path).read_bytes())
        tot[name] = totals(gt, raw, float(conf), int(min_area))
        print(f"{name}: PQ {pq(tot[name].sum(0)):.4f}")
    w = np.random.default_rng(0).multinomial(len(gt), np.full(len(gt), 1 / len(gt)), n_boot)
    boot = {k: pq(w @ v) for k, v in tot.items()}
    for a, b in itertools.combinations(tot, 2):
        d = boot[a] - boot[b]
        lo, hi = np.percentile(d, [2.5, 97.5])
        full = pq(tot[a].sum(0)) - pq(tot[b].sum(0))
        print(f"{a} - {b}: {full:+.4f}  95% CI [{lo:+.4f}, {hi:+.4f}]  P(>0)={np.mean(d > 0):.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--n-boot", type=int, default=2000)
    main(**vars(ap.parse_args()))
