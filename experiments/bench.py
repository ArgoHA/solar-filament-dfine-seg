"""End-to-end latency of each candidate pipeline: JPEG path -> final non-overlapping RLEs.

Torch backend, batch 1, same val images for every variant (first --n of val_obs after
--warmup). Includes image decoding, all forward passes and CPU post-processing.
"""

import argparse
import time

import cv2
import numpy as np
import torch
from pycocotools import mask as mask_utils

from dfine_seg import load_model
from predict_sem import components
from predict_sem_prob import probs
from predict_tiles import tiled
from predict_tta import predict as tta_predict
from score import ROOT, postprocess
from sem_from_probs import instances, scored

MODELS = ROOT / "output/models"
TR = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"


def inst(m, size, conf):
    net = load_model(MODELS / m / "model.pt", conf_thresh=conf, input_height=size, input_width=size)

    def run(img):
        o = net(img)[0]
        if "masks" not in o:
            return []
        masks = o["masks"].bool().cpu().numpy().astype(np.uint8)
        found = [
            (s, mask_utils.encode(np.asfortranarray(x)))
            for s, x in zip(o["scores"].tolist(), masks)
        ]
        return postprocess(sorted(found, key=lambda x: -x[0]), conf, 0, 0.5)

    return run


def tiles(m, conf):
    net = load_model(MODELS / m / "model.pt", conf_thresh=conf, input_height=1024, input_width=1024)
    return lambda img: postprocess(
        sorted(tiled(net, img, 1024, 512), key=lambda x: -x[0]), conf, 0, 0.5
    )


def inst_tta(m, tile, conf):
    size = tile or 2048
    net = load_model(MODELS / m / "model.pt", conf_thresh=0.1, input_height=size, input_width=size)
    return lambda img: postprocess(tta_predict(net, img, tile, 512, 0.5), conf, 0, 0.5)


def sem_argmax(m, close, min_area):
    net = load_model(MODELS / m / "model.pt", input_height=2048, input_width=2048)

    def run(img):
        fg = (net(img)[0]["sem_seg"] == 1).cpu().numpy().astype(np.uint8)
        return [
            mask_utils.encode(np.asfortranarray(c))
            for c in components(fg, close)
            if c.sum() >= min_area
        ]

    return run


def sem_prob(m, tta, thresh, min_area, min_score):
    net = load_model(MODELS / m / "model.pt", input_height=2048, input_width=2048)
    return lambda img: instances(scored(probs(net, img, tta), thresh, 4, min_area), min_score)


VARIANTS = {
    "inst_s1280": lambda: inst("seg_s_1280_2026-10-22", 1280, 0.60),
    "inst_s2048": lambda: inst("seg_s_2048_2026-10-23", 2048, 0.55),
    "inst_s_tiles": lambda: tiles("seg_s_crop1024_2026-10-23", 0.50),
    "inst_s2048_tta": lambda: inst_tta("seg_s_2048_2026-10-23", 0, 0.50),
    "inst_s_tiles_tta": lambda: inst_tta("seg_s_crop1024_2026-10-23", 1024, 0.50),
    "sem_s2048_argmax": lambda: sem_argmax("sem_s_2048_2026-10-23", 4, 400),
    "sem_s2048_prob": lambda: sem_prob("sem_s_2048_2026-10-23", False, 0.4, 300, 0.0),
    "sem_s2048_prob_conf": lambda: sem_prob("sem_s_2048_2026-10-23", False, 0.4, 200, 0.90),
    "sem_s2048_tta": lambda: sem_prob("sem_s_2048_2026-10-23", True, 0.4, 300, 0.0),
    "sem_s2048_tta_conf": lambda: sem_prob("sem_s_2048_2026-10-23", True, 0.4, 200, 0.88),
}


def main(variants, n, warmup) -> None:
    stems = (ROOT / "data/splits/val_obs.txt").read_text().split()[: n + warmup]
    for name in variants:
        run = VARIANTS[name]()
        times = []
        for i, s in enumerate(stems):
            torch.cuda.synchronize()
            t = time.perf_counter()
            run(cv2.imread(str(TR / f"{s}.jpeg"), cv2.IMREAD_COLOR))
            torch.cuda.synchronize()
            if i >= warmup:
                times.append(time.perf_counter() - t)
        print(
            f"{name}: {1000 * np.mean(times):.0f} ms (median {1000 * np.median(times):.0f})",
            flush=True,
        )
        del run
        torch.cuda.empty_cache()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="*", default=list(VARIANTS))
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--warmup", type=int, default=3)
    main(**vars(ap.parse_args()))
