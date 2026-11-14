"""Numbers and figures of the report, from the shipped weights and the validation split.

  python report/make_figures.py --data data/MAGFiLO_1.0_Kaggle_2026
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from filaments import viz  # noqa: E402
from filaments.data import val_ground_truth  # noqa: E402
from filaments.inference import FilamentSegmenter  # noqa: E402
from filaments.metrics import evaluate, inter_annotator  # noqa: E402

OUT = ROOT / "report/figures"
# validation observations shown in the qualitative figure: (stem, crop y0, y1, x0, x1)
EXAMPLES = [
    ("20111116063134Lh", 434, 1234, 590, 1390),
    ("20140205180154Bh", 712, 1512, 254, 1054),
]


def qualitative(data: Path, gt: dict, preds: dict) -> None:
    fig, axes = plt.subplots(len(EXAMPLES), 4, figsize=(7.0, 1.85 * len(EXAMPLES)))
    for row, (stem, y0, y1, x0, x1) in zip(np.atleast_2d(axes), EXAMPLES):
        img = cv2.imread(str(data / "train/train_images" / f"{stem}.jpeg"))
        views = gt[stem]
        panels = [("input", cv2.cvtColor(img, cv2.COLOR_BGR2RGB))]
        panels += [(f"annotator {i + 1}", viz.overlay(img, v)) for i, v in enumerate(views[:2])]
        panels += [("ours", viz.overlay(img, preds[stem]))]
        for ax, (title, im) in zip(row, panels):
            ax.imshow(im[y0:y1, x0:x1])
            ax.set_title(title, fontsize=8, pad=2)
            ax.axis("off")
    fig.tight_layout(pad=0.2, w_pad=0.2, h_pad=0.4)
    fig.savefig(OUT / "qualitative.jpg", dpi=300, pil_kwargs={"quality": 88})


def main(data: Path) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    gt = val_ground_truth(data)
    seg = FilamentSegmenter(ROOT / "weights/sem_s_2048.pt")
    preds = {s: seg(cv2.imread(str(data / "train/train_images" / f"{s}.jpeg"))) for s in gt}
    model, annot = evaluate(gt, preds), inter_annotator(gt)
    numbers = {"model": model.summary(), "annotators": annot.summary()}
    (OUT / "numbers.json").write_text(json.dumps(numbers, indent=2))
    print(json.dumps(numbers, indent=2))
    viz.distributions(model, annot, figsize=(7.0, 1.75)).savefig(OUT / "distributions.pdf")
    qualitative(data, gt, preds)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data/MAGFiLO_1.0_Kaggle_2026")
    main(ap.parse_args().data)
