from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from skimage.filters import threshold_otsu


def safe_otsu_threshold(gray_u8: np.ndarray) -> float:
    try:
        return float(threshold_otsu(gray_u8))
    except Exception:
        return float(np.mean(gray_u8))


def detect_bottom_legend_strip_top(gray_u8: np.ndarray) -> Optional[int]:
    """
    Detects top y-coordinate of a bottom legend/overlay strip (black bar) if present.
    """
    h, _w = gray_u8.shape[:2]
    start = int(0.55 * h)
    region = gray_u8[start:, :]

    med = np.median(region, axis=1)
    dark_frac = np.mean(region < 55, axis=1)

    ok = (med < 80) & (dark_frac > 0.40)

    min_run = max(18, int(0.015 * h))
    run = 0
    for i in range(ok.size - 1, -1, -1):
        if ok[i]:
            run += 1
        else:
            if run >= min_run:
                return start + i + 1
            run = 0

    if run >= min_run:
        return start
    return None


def find_scale_bar_bbox(gray_u8: np.ndarray) -> Optional[Dict[str, Any]]:
    """
    Finds a long thin bar in bottom region.
    Returns bbox in FULL image coords: {x,y,w,h,score,polarity,search_y0}
    """
    gray = gray_u8
    h, w = gray.shape[:2]

    strip_top = detect_bottom_legend_strip_top(gray)
    if strip_top is None:
        y0 = int(0.72 * h)
    else:
        y0 = max(strip_top, int(0.65 * h))

    crop = gray[y0:h, :]
    blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
    thr = safe_otsu_threshold(blur)

    masks = [
        ("dark", (blur < thr).astype(np.uint8) * 255),
        ("bright", (blur > thr).astype(np.uint8) * 255),
    ]

    best = None  # (score, x, y, bw, bh, polarity)
    ch, cw = crop.shape[:2]

    for polarity, bwmask in masks:
        k = max(3, int(round(cw * 0.01)))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, 3))
        closed = cv2.morphologyEx(bwmask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            x, y, bw, bh = cv2.boundingRect(c)
            if bw <= 0 or bh <= 0:
                continue

            aspect = bw / float(bh)
            if aspect < 8:
                continue
            if bw < 0.10 * cw:
                continue
            if bh > 0.18 * ch:
                continue

            area = float(cv2.contourArea(c))
            rect_area = float(bw * bh)
            if rect_area <= 0:
                continue
            fill = area / rect_area
            if fill < 0.45:
                continue

            score = bw * fill
            if best is None or score > best[0]:
                best = (score, x, y, bw, bh, polarity)

    if best is None:
        return None

    score, x, y, bw, bh, polarity = best
    return {
        "x": int(x),
        "y": int(y0 + y),
        "w": int(bw),
        "h": int(bh),
        "score": float(score),
        "polarity": polarity,
        "search_y0": int(y0),
    }


def exclude_legend_from_mask(mask: np.ndarray, gray_an: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Excludes:
      (A) bottom legend strip (full width) if detected
      (B) scale bar + label region as extra safety
    """
    h, w = gray_an.shape[:2]
    info: Dict[str, Any] = {"strip_excluded": False, "scale_excluded": False, "excluded_rects": []}

    out = mask.copy()

    strip_top = detect_bottom_legend_strip_top(gray_an)
    if strip_top is not None:
        out[strip_top:h, 0:w] = False
        info["strip_excluded"] = True
        info["strip_top_y"] = int(strip_top)
        info["excluded_rects"].append(
            {"x1": 0, "y1": int(strip_top), "x2": int(w), "y2": int(h), "type": "legend_strip"}
        )

    det = find_scale_bar_bbox(gray_an)
    if det is not None:
        x, y, bw, bh = det["x"], det["y"], det["w"], det["h"]
        x1 = max(0, x - int(round(0.35 * bw)))
        x2 = min(w, x + int(round(2.8 * bw)))
        y1 = max(0, y - int(round(14 * bh)))
        y2 = min(h, y + int(round(12 * bh)))
        out[y1:y2, x1:x2] = False
        info["scale_excluded"] = True
        info["scale_bar_bbox"] = {
            "x": int(x), "y": int(y), "w": int(bw), "h": int(bh),
            "score": float(det["score"]), "polarity": det["polarity"]
        }
        info["excluded_rects"].append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "type": "scale_region"})

    return out, info
