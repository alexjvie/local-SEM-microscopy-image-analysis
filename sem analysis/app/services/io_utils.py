from __future__ import annotations

import base64
from typing import Tuple

import cv2
import numpy as np


def decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image. Try PNG/JPG (TIFF depends on your OpenCV build).")
    return img


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def downscale_to_max(gray: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
    """Return (resized_gray, factor). factor = new/orig."""
    h, w = gray.shape[:2]
    m = max(h, w)
    if m <= max_dim:
        return gray, 1.0
    f = max_dim / float(m)
    new_w = max(1, int(round(w * f)))
    new_h = max(1, int(round(h * f)))
    resized = cv2.resize(gray, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return resized, f


def encode_png_b64(img_bgr_or_gray: np.ndarray, max_dim: int = 1100) -> str:
    """Downscale for transport, encode as PNG base64."""
    if img_bgr_or_gray.ndim == 2:
        small, _ = downscale_to_max(img_bgr_or_gray, max_dim=max_dim)
        ok, buf = cv2.imencode(".png", small)
    else:
        bgr = img_bgr_or_gray
        h, w = bgr.shape[:2]
        m = max(h, w)
        if m > max_dim:
            f = max_dim / float(m)
            new_w = max(1, int(round(w * f)))
            new_h = max(1, int(round(h * f)))
            bgr = cv2.resize(bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", bgr)

    if not ok:
        return ""
    return base64.b64encode(buf.tobytes()).decode("ascii")
