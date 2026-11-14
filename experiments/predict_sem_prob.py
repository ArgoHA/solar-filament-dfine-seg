"""Dump sem_seg filament probabilities (uint8 PNG, p*255) with optional dihedral TTA.

TTA averages softmax over the 8 flips/rot90s of the (square) input, each mapped back.
Probability maps let sem_from_probs.py sweep thresholds and ensemble models on CPU.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm

from dfine_seg import load_model
from preproc import preprocess


def dihedral(x: torch.Tensor, k: int, flip: bool, inverse: bool = False) -> torch.Tensor:
    if not inverse:
        x = x.flip(-1) if flip else x
        return torch.rot90(x, k, (-2, -1))
    x = torch.rot90(x, -k, (-2, -1))
    return x.flip(-1) if flip else x


VIEWS = [(k, f) for f in (False, True) for k in range(4)]


@torch.no_grad()
def probs(net, img: np.ndarray, tta: bool) -> np.ndarray:
    """uint8 p(filament) * 255 at the original resolution."""
    x, _, (orig,) = net._prepare_inputs(img)
    views = VIEWS if tta else VIEWS[:1]
    prob = 0
    for k, f in views:
        logits = net._predict(dihedral(x, k, f))["sem_seg_logits"].float()
        prob = prob + dihedral(logits.softmax(1)[:, 1:2], k, f, inverse=True)
    prob = torch.nn.functional.interpolate(prob / len(views), size=orig, mode="bilinear")
    return (prob[0, 0] * 255).round().byte().cpu().numpy()


def main(
    model: str, images: Path, stems: Path | None, out: Path, size: int, tta: bool, pre: bool
) -> None:
    net = load_model(model, input_height=size, input_width=size)
    paths = sorted(images.glob("*.jpeg"))
    if stems:
        keep = set(stems.read_text().split())
        paths = [p for p in paths if p.stem in keep]
    out.mkdir(parents=True, exist_ok=True)
    for p in tqdm(paths):
        img = cv2.imread(str(p), cv2.IMREAD_COLOR)
        if pre:  # model trained on preproc.py inputs
            img = preprocess(img)
        cv2.imwrite(str(out / f"{p.stem}.png"), probs(net, img, tta))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--images", type=Path, required=True)
    ap.add_argument("--stems", type=Path)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--size", type=int, default=2048)
    ap.add_argument("--tta", action="store_true")
    ap.add_argument("--pre", action="store_true", help="apply preproc.py (model trained on it)")
    main(**vars(ap.parse_args()))
