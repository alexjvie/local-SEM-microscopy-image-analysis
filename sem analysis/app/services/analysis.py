from __future__ import annotations

import base64
import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize
from skimage.segmentation import find_boundaries
from skimage.filters import threshold_otsu

from app.services.scale import detect_bottom_legend_strip_top, find_scale_bar_bbox


# -----------------------------
# IO helpers (lokal, damit analysis.py standalone bleibt)
# -----------------------------
def _decode_image(file_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(file_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("Could not decode image. Try PNG/JPG (TIFF depends on your OpenCV build).")
    return img


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        gray = img
    elif img.ndim == 3 and img.shape[2] >= 3:
        gray = cv2.cvtColor(img[:, :, :3], cv2.COLOR_BGR2GRAY)
    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    return gray


def _downscale_to_max(gray: np.ndarray, max_dim: int) -> Tuple[np.ndarray, float]:
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


def _safe_otsu_threshold(gray_u8: np.ndarray) -> float:
    try:
        return float(threshold_otsu(gray_u8))
    except Exception:
        return float(np.mean(gray_u8))


def _encode_png_b64(img_bgr_or_gray: np.ndarray, max_dim: int = 1100) -> str:
    if img_bgr_or_gray.ndim == 2:
        small, _ = _downscale_to_max(img_bgr_or_gray, max_dim=max_dim)
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


# -----------------------------
# Stats helpers
# -----------------------------
def _skeleton_length_pixels(skel: np.ndarray) -> float:
    """Connection-length with diagonal sqrt(2) weighting."""
    s = skel.astype(bool)
    if s.sum() == 0:
        return 0.0

    right = s[:, :-1] & s[:, 1:]
    down = s[:-1, :] & s[1:, :]

    dr = s[:-1, :-1] & s[1:, 1:]
    dl = s[:-1, 1:] & s[1:, :-1]

    hv = float(right.sum() + down.sum())
    diag = float(dr.sum() + dl.sum())
    return hv * 1.0 + diag * math.sqrt(2.0)


def _stats(xs: List[float]) -> Dict[str, float]:
    if not xs:
        return {"n": 0, "mean": 0.0, "median": 0.0, "std": 0.0, "min": 0.0, "max": 0.0, "sum": 0.0}
    arr = np.asarray(xs, dtype=np.float64)
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "median": float(np.median(arr)),
        "std": float(arr.std(ddof=0)),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "sum": float(arr.sum()),
    }


def _hist(xs: List[float], bins: int = 18) -> Optional[Dict[str, Any]]:
    if not xs:
        return None
    x = np.asarray(xs, dtype=np.float64)
    if float(x.max() - x.min()) <= 1e-12:
        return {"bin_edges": [float(x.min()), float(x.max())], "counts": [int(x.size)]}
    counts, edges = np.histogram(x, bins=bins)
    edges_list = [float(round(e, 6)) for e in edges.tolist()]
    counts_list = [int(c) for c in counts.tolist()]
    return {"bin_edges": edges_list, "counts": counts_list}


# -----------------------------
# Legend exclusion from analysis mask
# -----------------------------
def _exclude_legend_from_mask(mask: np.ndarray, gray_an: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Excludes:
      (A) bottom legend strip (full width) if detected
      (B) scale bar + label box region as fallback/extra safety
    """
    h, w = gray_an.shape[:2]
    info: Dict[str, Any] = {"strip_excluded": False, "scale_excluded": False, "excluded_rects": []}

    out = mask.copy()

    # (A) bottom strip
    strip_top = detect_bottom_legend_strip_top(gray_an)
    if strip_top is not None:
        out[strip_top:h, 0:w] = False
        info["strip_excluded"] = True
        info["strip_top_y"] = int(strip_top)
        info["excluded_rects"].append({"x1": 0, "y1": int(strip_top), "x2": int(w), "y2": int(h), "type": "legend_strip"})

    # (B) scale bar region
    det = find_scale_bar_bbox(gray_an)
    if det is not None:
        x, y, bw, bh = det["x"], det["y"], det["w"], det["h"]
        x1 = max(0, x - int(round(0.35 * bw)))
        x2 = min(w, x + int(round(2.8 * bw)))
        y1 = max(0, y - int(round(14 * bh)))
        y2 = min(h, y + int(round(12 * bh)))
        out[y1:y2, x1:x2] = False
        info["scale_excluded"] = True
        info["scale_bar_bbox"] = {"x": int(x), "y": int(y), "w": int(bw), "h": int(bh), "score": float(det["score"]), "polarity": det["polarity"]}
        info["excluded_rects"].append({"x1": int(x1), "y1": int(y1), "x2": int(x2), "y2": int(y2), "type": "scale_region"})

    return out, info


# -----------------------------
# SEM analysis (Porosity + Length)
# -----------------------------
def analyze_sem(
    file_bytes: bytes,
    *,
    nm_per_px: Optional[float],
    invert: bool,
    min_area_px: int,
    hole_area_px: int,
    thr_offset: int,  # ✅ NEU (Slider)
) -> Dict[str, Any]:
    img = _decode_image(file_bytes)
    gray = _to_gray(img)

    # downscale analysis for big images
    gray_an, f = _downscale_to_max(gray, max_dim=2400)

    nm_per_px_eff = None
    if nm_per_px and nm_per_px > 0:
        nm_per_px_eff = float(nm_per_px / f)

    blur = cv2.GaussianBlur(gray_an, (0, 0), sigmaX=1.0)

    # Base threshold (Otsu)
    thr_otsu = float(_safe_otsu_threshold(blur))

    # ✅ User-controlled threshold = Otsu + offset
    thr_offset_i = int(thr_offset)
    thr_used = float(np.clip(thr_otsu + thr_offset_i, 0.0, 255.0))

    # dark foreground by default
    raw_mask = blur < thr_used
    if invert:
        raw_mask = ~raw_mask

    mask = raw_mask
    if min_area_px and min_area_px > 0:
        mask = remove_small_objects(mask, min_size=int(min_area_px))
    if hole_area_px and hole_area_px > 0:
        mask = remove_small_holes(mask, area_threshold=int(hole_area_px))

    # Exclude legend/scale
    mask, legend_info = _exclude_legend_from_mask(mask, gray_an)

    # Previews
    rgb = cv2.cvtColor(gray_an, cv2.COLOR_GRAY2RGB)
    bnd = find_boundaries(mask, mode="outer")
    rgb[bnd] = (255, 0, 0)
    overlay_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    # draw excluded rects in blue
    for r in legend_info.get("excluded_rects", []):
        cv2.rectangle(overlay_bgr, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (255, 0, 0), 2)

    overlay_b64 = _encode_png_b64(overlay_bgr, max_dim=1100)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_b64 = _encode_png_b64(mask_u8, max_dim=1100)

    # Porosity proxy: 2D area fraction of foreground
    total_px = int(mask.size)
    fg_px = int(mask.sum())
    fg_frac = fg_px / total_px if total_px else 0.0

    # Lengths via skeleton per connected component
    lab = label(mask, connectivity=2)
    regs = regionprops(lab)
    if len(regs) > 6000:
        regs = sorted(regs, key=lambda r: r.area, reverse=True)[:6000]

    lengths_px: List[float] = []
    for reg in regs:
        if reg.area < max(1, int(min_area_px)):
            continue
        sl = reg.slice
        obj = (lab[sl] == reg.label)
        skel = skeletonize(obj)
        L = _skeleton_length_pixels(skel)
        if L > 0:
            lengths_px.append(float(L))

    lengths_px.sort()

    stats_px = _stats(lengths_px)
    hist_px = _hist(lengths_px, bins=18)

    stats_um = None
    hist_um = None
    if nm_per_px_eff and nm_per_px_eff > 0:
        um_per_px = nm_per_px_eff / 1000.0
        lengths_um = [x * um_per_px for x in lengths_px]
        stats_um = _stats(lengths_um)
        hist_um = _hist(lengths_um, bins=18)

    summary: Dict[str, Any] = {
        "foreground_area_fraction": float(fg_frac),
        "n_objects": stats_px["n"],
        "mean_length_px": stats_px["mean"],
        "median_length_px": stats_px["median"],
    }
    if stats_um is not None:
        summary["mean_length_um"] = stats_um["mean"]
        summary["median_length_um"] = stats_um["median"]

    return {
        "ok": True,

        # user params
        "invert": bool(invert),
        "min_area_px": int(min_area_px),
        "hole_area_px": int(hole_area_px),
        "nm_per_px_input": float(nm_per_px) if nm_per_px else None,
        "nm_per_px_effective": nm_per_px_eff,
        "resize_factor_used": float(f),

        # ✅ threshold transparency
        "threshold_otsu": float(thr_otsu),
        "threshold_offset": int(thr_offset_i),
        "threshold_used": float(thr_used),

        # legend exclusion info
        "legend_exclusion": legend_info,

        # results
        "area_fraction_foreground": float(fg_frac),
        "area_fraction_background": float(1.0 - fg_frac),
        "stats_px": stats_px,
        "length_hist_px": hist_px,
        "stats_um": stats_um,
        "length_hist_um": hist_um,
        "summary": summary,

        # previews
        "preview_overlay_png_b64": overlay_b64,
        "preview_mask_png_b64": mask_b64,
    }
