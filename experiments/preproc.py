"""Grayscale H-alpha frame -> 3 channels [raw, limb-flattened, CLAHE(flattened)] (BGR order).

Flattening divides by a smooth disk-only background (masked Gaussian at 1/8 res), which
removes limb darkening and uneven illumination; filaments become dips below 128.
Used both to build the training images and at inference, so the two always match.
"""

import cv2
import numpy as np

LOW, SIGMA, GAIN = 256, 8.0, 256.0  # background at 256², sigma 8 px there (~64 px native)
CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))


def disk_mask(g: np.ndarray) -> np.ndarray:
    """Solar disk: circle fitted (area -> radius) to the largest component brighter than half
    the central median. Otsu also grabs the faint halo some frames have outside the limb."""
    h, w = g.shape
    m = (g > 0.5 * np.median(g[h // 2 - 256 : h // 2 + 256, w // 2 - 256 : w // 2 + 256])).astype(
        np.uint8
    )
    _, _, st, cen = cv2.connectedComponentsWithStats(m)
    i = 1 + st[1:, cv2.CC_STAT_AREA].argmax()
    r = np.sqrt(st[i, cv2.CC_STAT_AREA] / np.pi)
    yy, xx = np.ogrid[: g.shape[0], : g.shape[1]]
    return (xx - cen[i][0]) ** 2 + (yy - cen[i][1]) ** 2 <= r**2


def preprocess(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else img[..., 0]
    disk = disk_mask(g)
    lo = cv2.resize(g.astype(np.float32) * disk, (LOW, LOW), interpolation=cv2.INTER_AREA)
    w = cv2.resize(disk.astype(np.float32), (LOW, LOW), interpolation=cv2.INTER_AREA)
    num, den = cv2.GaussianBlur(lo, (0, 0), SIGMA), cv2.GaussianBlur(w, (0, 0), SIGMA)
    bg = cv2.resize(num / np.maximum(den, 1e-3), g.shape[::-1], interpolation=cv2.INTER_LINEAR)
    flat = 128 + (g / np.maximum(bg, 1.0) - 1) * GAIN
    flat = np.where(disk, np.clip(flat, 0, 255), 0).astype(np.uint8)
    return np.dstack([g, flat, CLAHE.apply(flat)])
