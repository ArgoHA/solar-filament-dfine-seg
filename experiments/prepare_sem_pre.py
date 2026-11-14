"""sem_seg dataset with preprocessed inputs (preproc.py): same views, labels and split.

Each observation is preprocessed once into data/pre_images/<obs>.png; every annotator view
(<prefix>-<obs>) links to it. Labels link to data/dataset_sem.
"""

from multiprocessing import Pool
from pathlib import Path

import cv2

from preproc import preprocess

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data/raw/MAGFiLO_1.0_Kaggle_2026/train/train_images"
SRC, PRE, OUT = ROOT / "data/dataset_sem", ROOT / "data/pre_images", ROOT / "data/dataset_sem_pre"


def run(p: Path) -> None:
    dst = PRE / f"{p.stem}.png"
    if not dst.exists():
        cv2.imwrite(str(dst), preprocess(cv2.imread(str(p), cv2.IMREAD_COLOR)))


def main() -> None:
    PRE.mkdir(parents=True, exist_ok=True)
    (OUT / "images").mkdir(parents=True, exist_ok=True)
    with Pool(8) as pool:
        pool.map(run, sorted(RAW.glob("*.jpeg")), chunksize=4)
    if not (OUT / "labels").exists():
        (OUT / "labels").symlink_to(SRC / "labels")
    for split in ("train", "val"):
        views = [Path(v).stem for v in (SRC / f"{split}.csv").read_text().split()]
        for v in views:
            dst = OUT / "images" / f"{v}.png"
            if not dst.exists():
                dst.symlink_to(PRE / f"{v.split('-', 1)[1]}.png")
        (OUT / f"{split}.csv").write_text("".join(f"{v}.png\n" for v in views))
    print(f"{len(list(PRE.glob('*.png')))} observations -> {OUT}")


if __name__ == "__main__":
    main()
