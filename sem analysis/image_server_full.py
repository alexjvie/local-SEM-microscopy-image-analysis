from __future__ import annotations

import base64
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_holes, remove_small_objects, skeletonize
from skimage.segmentation import find_boundaries

app = FastAPI(title="SEM Image Analysis", version="0.7")

STATIC_DIR = Path(__file__).resolve().parent / "static"
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# -----------------------------
# Helpers
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
    """Downscale for transport, encode as PNG base64."""
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
# Legend strip detection (bottom black SEM overlay)
# -----------------------------
def _detect_bottom_legend_strip_top(gray_u8: np.ndarray) -> Optional[int]:
    """
    Detects the top y-coordinate of the bottom legend/overlay strip (black bar) if present.
    """
    h, _w = gray_u8.shape[:2]
    start = int(0.55 * h)  # only look in lower half
    region = gray_u8[start:, :]

    med = np.median(region, axis=1)
    dark_frac = np.mean(region < 55, axis=1)  # "black-ish"

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


# -----------------------------
# Scale bar detection + OCR (best effort)
# -----------------------------
def _find_scale_bar_bbox(gray_u8: np.ndarray) -> Optional[Dict[str, Any]]:
    gray = gray_u8
    h, w = gray.shape[:2]

    strip_top = _detect_bottom_legend_strip_top(gray)
    if strip_top is None:
        y0 = int(0.72 * h)
    else:
        y0 = max(strip_top, int(0.65 * h))

    crop = gray[y0:h, :]
    blur = cv2.GaussianBlur(crop, (0, 0), sigmaX=1.0)
    thr = _safe_otsu_threshold(blur)

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


def _try_ocr_text(roi_gray: np.ndarray) -> str:
    """
    OCR best effort:
    - Requires pytesseract + system tesseract.
    """
    try:
        import pytesseract  # type: ignore
    except Exception:
        return ""

    g = roi_gray.copy()
    g = cv2.resize(g, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    g = cv2.GaussianBlur(g, (0, 0), sigmaX=1.0)
    _, bw = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bw_inv = 255 - bw

    cfg = "--psm 7 -c tessedit_char_whitelist=0123456789.,nmuµ "
    txt1 = ""
    txt2 = ""
    try:
        txt1 = pytesseract.image_to_string(bw, config=cfg) or ""
    except Exception:
        txt1 = ""
    try:
        txt2 = pytesseract.image_to_string(bw_inv, config=cfg) or ""
    except Exception:
        txt2 = ""

    t = (txt1 if len(txt1.strip()) >= len(txt2.strip()) else txt2).strip()
    return t


def _parse_length_to_nm(text: str) -> Optional[Tuple[float, str]]:
    if not text:
        return None
    t = text.strip().replace("μ", "µ")
    t = re.sub(r"\s+", " ", t)

    m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(nm|µm|um|mm)", t, flags=re.IGNORECASE)
    if not m:
        t2 = t.replace(" ", "")
        m = re.search(r"([0-9]+(?:[.,][0-9]+)?)\s*(nm|µm|um|mm)", t2, flags=re.IGNORECASE)
        if not m:
            return None

    val = float(m.group(1).replace(",", "."))
    unit = m.group(2).lower().replace("um", "µm")

    if unit == "nm":
        nm = val
        label = f"{val:g} nm"
    elif unit == "µm":
        nm = val * 1000.0
        label = f"{val:g} µm"
    elif unit == "mm":
        nm = val * 1e6
        label = f"{val:g} mm"
    else:
        return None

    return nm, label


def detect_scale_bar_nm_per_px(file_bytes: bytes) -> Dict[str, Any]:
    img = _decode_image(file_bytes)
    gray = _to_gray(img)

    gray_det, f = _downscale_to_max(gray, max_dim=1600)
    det = _find_scale_bar_bbox(gray_det)
    if det is None:
        return {"ok": False, "error": "No scale bar-like rectangle found."}

    x_d, y_d, w_d, h_d = det["x"], det["y"], det["w"], det["h"]
    x = int(round(x_d / f))
    y = int(round(y_d / f))
    bw = max(1, int(round(w_d / f)))
    bh = max(1, int(round(h_d / f)))

    H, W = gray.shape[:2]

    strip_top = _detect_bottom_legend_strip_top(gray)
    if strip_top is None:
        strip_top = int(0.75 * H)

    rois: List[Tuple[str, Tuple[int, int, int, int]]] = []
    rois.append(
        (
            "around_bar",
            (
                max(0, x - int(round(0.30 * bw))),
                max(0, y - int(round(14 * bh))),
                min(W, x + int(round(2.8 * bw))),
                min(H, y + int(round(12 * bh))),
            ),
        )
    )
    rois.append(
        (
            "right_box",
            (
                max(0, x + int(round(0.10 * bw))),
                max(0, y - int(round(10 * bh))),
                min(W, x + int(round(3.0 * bw))),
                min(H, y + int(round(10 * bh))),
            ),
        )
    )
    rois.append(
        (
            "legend_chunk",
            (
                max(0, x - int(round(0.8 * bw))),
                max(0, strip_top),
                min(W, x + int(round(3.2 * bw))),
                min(H, strip_top + int(round(0.25 * (H - strip_top)))),
            ),
        )
    )

    best_parse = None
    for name, (x1, y1, x2, y2) in rois:
        if x2 <= x1 + 10 or y2 <= y1 + 10:
            continue
        roi = gray[y1:y2, x1:x2]
        txt = _try_ocr_text(roi)
        parsed = _parse_length_to_nm(txt)
        if parsed:
            nm, label = parsed
            best_parse = (nm, label, txt, name, (x1, y1, x2, y2))
            break

    rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(rgb, (x, y), (x + bw, y + bh), (0, 0, 255), 2)
    for _name, (x1, y1, x2, y2) in rois:
        cv2.rectangle(rgb, (x1, y1), (x2, y2), (0, 255, 255), 1)
    preview_b64 = _encode_png_b64(rgb, max_dim=1100)

    if not best_parse:
        return {
            "ok": False,
            "error": "Scale bar found, but OCR could not parse a length (e.g., '200 nm', '1 µm').",
            "bar_length_px": bw,
            "preview_png_b64": preview_b64,
            "hint": "Install OCR: pip install pytesseract AND install system tesseract (brew/apt).",
        }

    length_nm, label, ocr_text, roi_name, roi_rect = best_parse
    nm_per_px = float(length_nm / float(bw))

    return {
        "ok": True,
        "nm_per_px": nm_per_px,
        "bar_length_px": bw,
        "scale_label": label,
        "ocr_text": ocr_text,
        "ocr_roi_used": roi_name,
        "ocr_roi_rect": {"x1": roi_rect[0], "y1": roi_rect[1], "x2": roi_rect[2], "y2": roi_rect[3]},
        "preview_png_b64": preview_b64,
    }


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

    strip_top = _detect_bottom_legend_strip_top(gray_an)
    if strip_top is not None:
        out[strip_top:h, 0:w] = False
        info["strip_excluded"] = True
        info["strip_top_y"] = int(strip_top)
        info["excluded_rects"].append({"x1": 0, "y1": int(strip_top), "x2": int(w), "y2": int(h), "type": "legend_strip"})

    det = _find_scale_bar_bbox(gray_an)
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
# SEM analysis (ALWAYS: Porosity + Length) + THRESHOLD SLIDER (Otsu Offset)
# -----------------------------
def analyze_sem(
    file_bytes: bytes,
    *,
    nm_per_px: Optional[float],
    invert: bool,
    min_area_px: int,
    hole_area_px: int,
    thr_offset: int,  # ✅ NEW
) -> Dict[str, Any]:
    img = _decode_image(file_bytes)
    gray = _to_gray(img)

    gray_an, f = _downscale_to_max(gray, max_dim=2400)

    nm_per_px_eff = None
    if nm_per_px and nm_per_px > 0:
        nm_per_px_eff = float(nm_per_px / f)

    blur = cv2.GaussianBlur(gray_an, (0, 0), sigmaX=1.0)

    thr_otsu = float(_safe_otsu_threshold(blur))
    thr_used = float(np.clip(thr_otsu + int(thr_offset), 0.0, 255.0))

    raw_mask = blur < thr_used  # dark foreground by default
    if invert:
        raw_mask = ~raw_mask

    mask = raw_mask
    if min_area_px and min_area_px > 0:
        mask = remove_small_objects(mask, min_size=int(min_area_px))
    if hole_area_px and hole_area_px > 0:
        mask = remove_small_holes(mask, area_threshold=int(hole_area_px))

    mask, legend_info = _exclude_legend_from_mask(mask, gray_an)

    rgb = cv2.cvtColor(gray_an, cv2.COLOR_GRAY2RGB)
    bnd = find_boundaries(mask, mode="outer")
    rgb[bnd] = (255, 0, 0)
    overlay_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    for r in legend_info.get("excluded_rects", []):
        cv2.rectangle(overlay_bgr, (r["x1"], r["y1"]), (r["x2"], r["y2"]), (255, 0, 0), 2)

    overlay_b64 = _encode_png_b64(overlay_bgr, max_dim=1100)
    mask_u8 = (mask.astype(np.uint8) * 255)
    mask_b64 = _encode_png_b64(mask_u8, max_dim=1100)

    total_px = int(mask.size)
    fg_px = int(mask.sum())
    fg_frac = fg_px / total_px if total_px else 0.0

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
        "background_area_fraction": float(1.0 - fg_frac),
        "n_objects": int(stats_px["n"]),
        "mean_length_px": float(stats_px["mean"]),
        "median_length_px": float(stats_px["median"]),
    }
    if stats_um is not None:
        summary["mean_length_um"] = float(stats_um["mean"])
        summary["median_length_um"] = float(stats_um["median"])

    out: Dict[str, Any] = {
        "ok": True,
        "analysis": "porosity+length",

        # ✅ threshold transparency
        "threshold_otsu": float(thr_otsu),
        "threshold_offset": int(thr_offset),
        "threshold_used": float(thr_used),

        "invert": bool(invert),
        "min_area_px": int(min_area_px),
        "hole_area_px": int(hole_area_px),
        "nm_per_px_input": float(nm_per_px) if nm_per_px else None,
        "nm_per_px_effective": nm_per_px_eff,
        "resize_factor_used": float(f),
        "legend_exclusion": legend_info,

        "area_fraction_foreground": float(fg_frac),
        "area_fraction_background": float(1.0 - fg_frac),

        "stats_px": stats_px,
        "length_hist_px": hist_px,
        "stats_um": stats_um,
        "length_hist_um": hist_um,

        "preview_overlay_png_b64": overlay_b64,
        "preview_mask_png_b64": mask_b64,

        "summary": summary,
    }
    return out


# -----------------------------
# FastAPI endpoints + UI
# -----------------------------
@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)


@app.get("/", response_class=HTMLResponse)
def home():
    return HTMLResponse(
        r"""
<!doctype html>
<html lang="de">
<head>
  <meta charset="utf-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1"/>
  <title>SEM Image Analysis</title>
  <style>
    :root{
      --bg:#ffffff; --panel:#ffffff; --text:#111111; --muted:#6b7280;
      --border:#e5e7eb; --shadow:0 10px 30px rgba(0,0,0,0.06);
      --btn:#111111; --btnText:#ffffff;
      --pillBg:#f3f4f6;
      --inputBg:#ffffff;
      --contentMax: 980px;
    }
    [data-theme="dark"]{
      --bg:#0b0f19; --panel:#0f1629; --text:#e5e7eb; --muted:#9ca3af;
      --border:rgba(255,255,255,0.10); --shadow:0 10px 30px rgba(0,0,0,0.35);
      --btn:#e5e7eb; --btnText:#0b0f19;
      --pillBg:rgba(255,255,255,0.08);
      --inputBg:rgba(255,255,255,0.06);
    }
    *{ box-sizing:border-box; }
    body{ margin:0; font-family:-apple-system,system-ui,Segoe UI,Roboto,Helvetica,Arial,sans-serif; background:var(--bg); color:var(--text); }
    .wrap{ min-height:100vh; display:flex; align-items:flex-start; justify-content:center; padding:28px 18px; }
    .shell{ width:100%; max-width:var(--contentMax); display:flex; flex-direction:column; gap:14px; }

    .topbar{ display:flex; align-items:center; justify-content:space-between; gap:12px; }
    .brand{ display:flex; align-items:center; gap:10px; }
    .brand img{ width:34px; height:34px; border-radius:10px; border:1px solid var(--border); background:var(--panel); object-fit:contain; padding:4px; }
    .title{ font-size:14px; font-weight:650; letter-spacing:0.2px; }
    .subtitle{ font-size:12px; color:var(--muted); margin-top:2px; line-height:1.35; }
    .toggleBtn{ border:1px solid var(--border); background:var(--panel); color:var(--text);
      border-radius:12px; padding:8px 10px; cursor:pointer; box-shadow:var(--shadow); font-weight:600; }

    .panel{ border:1px solid var(--border); background:var(--panel); border-radius:18px; box-shadow:var(--shadow); padding:16px; }
    .centerHero{ display:flex; flex-direction:column; align-items:center; text-align:center; gap:10px; padding:6px 0 2px 0; }
    .heroLogo{ width:64px; height:64px; border-radius:18px; border:1px solid var(--border); background:var(--panel);
      object-fit:contain; padding:6px; box-shadow:var(--shadow); }
    .heroH{ font-size:20px; font-weight:800; margin:0; letter-spacing:0.2px; }
    .heroP{ font-size:13px; color:var(--muted); margin:0; max-width:820px; line-height:1.5; }

    .metaRow{ display:flex; align-items:center; justify-content:center; gap:10px; margin-top:6px; flex-wrap:wrap; }
    .btn{ border:0; background:var(--btn); color:var(--btnText); padding:10px 14px; border-radius:14px; cursor:pointer; font-weight:750; }
    .btn.secondary{ border:1px solid var(--border); background:var(--panel); color:var(--text); box-shadow:var(--shadow); font-weight:700; }

    .grid{ display:grid; grid-template-columns:1fr 1fr; gap:14px; margin-top:14px; align-items:start; }
    @media (max-width:980px){ .grid{ grid-template-columns:1fr; } }

    .drop{ border:2px dashed var(--border); border-radius:16px; padding:18px; min-height:200px; display:flex; align-items:center; justify-content:center;
      text-align:center; color:var(--muted); background:var(--inputBg); cursor:pointer; transition:120ms; }
    .drop.dragover{ border-color:var(--btn); color:var(--text); transform:translateY(-1px); }

    .row{ display:flex; gap:10px; flex-wrap:wrap; align-items:end; margin-top:10px; }
    label{ font-size:12px; color:var(--muted); display:block; margin-bottom:4px; }
    input[type="number"]{ border:1px solid var(--border); border-radius:12px; padding:10px 10px; width:160px; background:var(--inputBg); color:var(--text); outline:none; }

    .nmWrap{ display:flex; gap:8px; align-items:end; }
    .btnSmall{ border:1px solid var(--border); background:var(--panel); color:var(--text); padding:10px 12px; border-radius:12px; cursor:pointer; box-shadow:var(--shadow); font-weight:700; }
    .btnSmall:disabled{ opacity:0.5; cursor:not-allowed; box-shadow:none; }

    .chk{ display:flex; align-items:center; gap:8px; padding:8px 10px; border:1px solid var(--border); border-radius:12px; background:var(--inputBg); }
    .status{ font-size:12px; color:var(--muted); margin-top:10px; min-height:18px; }

    .preview img{ width:100%; border-radius:14px; border:1px solid var(--border); background:var(--panel); display:block; }
    .card{ border:1px solid var(--border); border-radius:16px; background:var(--inputBg); padding:12px; }
    .cards{ display:grid; grid-template-columns:1fr 1fr; gap:10px; margin-top:10px; }
    @media (max-width:540px){ .cards{ grid-template-columns:1fr; } }
    .kv{ display:flex; justify-content:space-between; gap:10px; font-size:13px; line-height:1.6; }
    .kv span:first-child{ color:var(--muted); }

    details{ margin-top:10px; border-radius:16px; border:1px solid var(--border); background:var(--panel); padding:10px 12px; }
    summary{ cursor:pointer; color:var(--muted); font-size:12px; font-weight:650; user-select:none; }
    pre{ margin:10px 0 0 0; padding:12px; border:1px solid var(--border); border-radius:14px; background:rgba(127,127,127,0.10);
      overflow:auto; font-size:12px; line-height:1.45; white-space:pre-wrap; }

    .foot{ text-align:center; color:var(--muted); font-size:12px; padding-top:2px; }

    input[type="range"]{ width: 260px; accent-color: var(--btn); }
  </style>
</head>
<body>
<div class="wrap">
  <div class="shell">

    <div class="topbar">
      <div class="brand">
        <img id="logoTop" src="/static/logo_day.png" onerror="this.style.display='none';" alt="logo"/>
        <div>
          <div class="title">SEM Image Analysis</div>
          <div class="subtitle">Enter startet Analyze · Porosity + Length automatisch · Threshold Slider</div>
        </div>
      </div>
      <button class="toggleBtn" id="themeBtn" onclick="toggleTheme()">Dark mode</button>
    </div>

    <div class="panel">
      <div class="centerHero">
        <img id="logoHero" class="heroLogo" src="/static/logo_day.png" onerror="this.style.display='none';" alt="logo"/>
        <div class="heroH">Analyze SEM images locally</div>
        <p class="heroP">
          Dieses Tool berechnet <b>Porosität (2D-Flächenanteil)</b> und <b>Längenstatistik</b> in einem Lauf.
          Die gesamte SEM-Legende unten (schwarzer Balken + Text + Scale-Bar-Box) wird automatisch ausgeschlossen.
        </p>

        <div class="metaRow">
          <button class="btn secondary" onclick="clearAll()">Clear</button>
        </div>
      </div>

      <div class="grid">
        <!-- Left -->
        <div>
          <div id="drop" class="drop" title="Drop image here or click to select.">
            <div>
              <div style="font-weight:800; color: var(--text);">Bild hier ablegen</div>
              <div style="margin-top:6px;">oder klicken zum Auswählen</div>
              <div style="margin-top:10px;font-size:12px;">PNG/JPG (TIFF je nach OpenCV)</div>
            </div>
          </div>
          <input id="file" type="file" accept="image/*" style="display:none"/>

          <div class="row">
            <div>
              <label>nm/px (optional)</label>
              <div class="nmWrap">
                <input id="nm_per_px" type="number" step="0.0001" placeholder="z.B. 2.5"/>
                <button class="btnSmall" id="autoScaleBtn" onclick="autoScale()" disabled>Auto</button>
              </div>
            </div>
            <div>
              <label>min area (px)</label>
              <input id="min_area_px" type="number" step="1" value="80"/>
            </div>
            <div>
              <label>hole area (px)</label>
              <input id="hole_area_px" type="number" step="1" value="0"/>
            </div>
            <div class="chk">
              <input id="invert" type="checkbox"/>
              <label for="invert" style="margin:0;">invert (foreground hell)</label>
            </div>
          </div>

          <!-- ✅ Threshold slider -->
          <div class="row" style="margin-top:12px;">
            <div>
              <label>Threshold (Otsu-Offset)</label>
              <input id="thr_offset" type="range" min="-80" max="80" step="1" value="0">
              <div style="font-size:12px;color:var(--muted);margin-top:4px;">
                Offset: <span id="thr_offset_val">0</span>
              </div>
              <div style="font-size:12px;color:var(--muted);margin-top:4px;max-width:520px;">
                Größerer Offset ⇒ mehr Pixel werden als „Foreground“ (dunkel) klassifiziert (ohne invert).
              </div>
            </div>
          </div>

          <div class="row" style="margin-top:12px;">
            <button class="btn" id="analyzeBtn" onclick="analyze()">Analyze</button>
          </div>

          <div class="status" id="status"></div>

          <div id="scaleBox" style="display:none; margin-top:10px;">
            <details open>
              <summary>Auto-Scale Result (compact)</summary>
              <pre id="scaleOut"></pre>
              <div id="scalePreviewWrap" style="margin-top:10px; display:none;">
                <div style="font-size:12px;color:var(--muted);margin-bottom:4px;">Detection preview</div>
                <img id="scalePreview" alt="scale preview" style="width:100%;border-radius:14px;border:1px solid var(--border);"/>
              </div>
            </details>
          </div>
        </div>

        <!-- Right -->
        <div class="preview">
          <div class="card" id="fileCard" style="display:none;">
            <div class="kv"><span>File</span><span id="fileName"></span></div>
            <div class="kv"><span>Resize factor</span><span id="resizeFactor">-</span></div>
            <div class="kv"><span>Effective nm/px</span><span id="effScale">-</span></div>
            <div class="kv"><span>Threshold used</span><span id="thrUsed">-</span></div>
          </div>

          <div class="cards" id="metricsCards" style="display:none;">
            <div class="card" id="cardArea">
              <div style="font-weight:750; margin-bottom:6px;">Area fractions</div>
              <div class="kv"><span>Foreground</span><span id="mFg">-</span></div>
              <div class="kv"><span>Background</span><span id="mBg">-</span></div>
            </div>
            <div class="card" id="cardLen">
              <div style="font-weight:750; margin-bottom:6px;">Lengths</div>
              <div class="kv"><span>N objects</span><span id="mN">-</span></div>
              <div class="kv"><span>Mean (px)</span><span id="mMeanPx">-</span></div>
              <div class="kv"><span>Median (px)</span><span id="mMedPx">-</span></div>
              <div class="kv"><span>Mean (µm)</span><span id="mMeanUm">-</span></div>
            </div>
          </div>

          <div id="overlayBox" style="display:none;">
            <div style="font-size:12px;color:var(--muted);margin:4px 0;">Overlay (Maske rot, Exclusion blau)</div>
            <img id="overlayImg" alt="overlay"/>
          </div>

          <div id="maskBox" style="display:none;">
            <div style="font-size:12px;color:var(--muted);margin:10px 0 4px 0;">Maske (Legend ausgeschlossen)</div>
            <img id="maskImg" alt="mask"/>
          </div>

          <details id="jsonDetails" style="display:none;">
            <summary>Result JSON (data-only)</summary>
            <pre id="out">{}</pre>
          </details>
        </div>
      </div>
    </div>

    <div class="foot">
      Local-only. Porosity = 2D-Flächenanteil aus Maske.
    </div>

  </div>
</div>

<script>
let selectedFile = null;

function setStatus(s){ document.getElementById("status").textContent = s || ""; }

function setLogoForTheme(t){
  const src = (t === "dark") ? "/static/logo_night.png" : "/static/logo_day.png";
  const top = document.getElementById("logoTop");
  const hero = document.getElementById("logoHero");
  if(top) top.src = src;
  if(hero) hero.src = src;
}

function applyTheme(t){
  document.documentElement.setAttribute("data-theme", t);
  localStorage.setItem("theme", t);
  const btn = document.getElementById("themeBtn");
  if(btn) btn.textContent = (t === "dark") ? "Light mode" : "Dark mode";
  setLogoForTheme(t);
}

function toggleTheme(){
  const cur = localStorage.getItem("theme");
  applyTheme(cur === "dark" ? "light" : "dark");
}

(function initTheme(){
  const saved = localStorage.getItem("theme");
  if(saved){ applyTheme(saved); }
  else{
    const prefersDark = window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches;
    applyTheme(prefersDark ? "dark" : "light");
  }
})();

// slider live value
const thrSlider = document.getElementById("thr_offset");
const thrVal = document.getElementById("thr_offset_val");
thrVal.textContent = thrSlider.value;
thrSlider.addEventListener("input", () => { thrVal.textContent = thrSlider.value; });

function clearAll(){
  selectedFile = null;
  document.getElementById("file").value = "";
  document.getElementById("autoScaleBtn").disabled = true;

  document.getElementById("fileCard").style.display = "none";
  document.getElementById("metricsCards").style.display = "none";
  document.getElementById("overlayBox").style.display = "none";
  document.getElementById("maskBox").style.display = "none";
  document.getElementById("jsonDetails").style.display = "none";

  document.getElementById("scaleBox").style.display = "none";
  document.getElementById("scaleOut").textContent = "";
  document.getElementById("scalePreviewWrap").style.display = "none";
  document.getElementById("scalePreview").src = "";

  document.getElementById("out").textContent = "{}";
  setStatus("");
}

const drop = document.getElementById("drop");
const fileInput = document.getElementById("file");

drop.addEventListener("click", () => fileInput.click());
fileInput.addEventListener("change", (e) => {
  const f = e.target.files && e.target.files[0];
  if(!f) return;
  selectedFile = f;
  document.getElementById("fileName").textContent = f.name;
  document.getElementById("fileCard").style.display = "block";
  document.getElementById("autoScaleBtn").disabled = false;
  setStatus("File selected: " + f.name);
});

drop.addEventListener("dragover", (e) => { e.preventDefault(); drop.classList.add("dragover"); });
drop.addEventListener("dragleave", () => drop.classList.remove("dragover"));
drop.addEventListener("drop", (e) => {
  e.preventDefault();
  drop.classList.remove("dragover");
  const f = e.dataTransfer.files && e.dataTransfer.files[0];
  if(!f) return;
  selectedFile = f;
  document.getElementById("fileName").textContent = f.name;
  document.getElementById("fileCard").style.display = "block";
  document.getElementById("autoScaleBtn").disabled = false;
  setStatus("File dropped: " + f.name);
});

function fmt(x){
  if(x === null || x === undefined) return "-";
  if(typeof x === "number") return x.toFixed(4);
  return String(x);
}

// Enter -> Analyze (global), only if file selected
document.addEventListener("keydown", (e) => {
  if(e.key === "Enter"){
    if(e.shiftKey || e.ctrlKey || e.metaKey || e.altKey) return;
    if(!selectedFile) return;
    e.preventDefault();
    analyze();
  }
});

async function autoScale(){
  if(!selectedFile){ setStatus("Bitte zuerst ein Bild auswählen."); return; }

  setStatus("Auto-Scale: suche Scale-Bar…");
  document.getElementById("scaleBox").style.display = "none";
  document.getElementById("scaleOut").textContent = "";
  document.getElementById("scalePreviewWrap").style.display = "none";
  document.getElementById("scalePreview").src = "";

  const fd = new FormData();
  fd.append("file", selectedFile);

  try{
    const res = await fetch("/api/sem/detect_scale", { method:"POST", body: fd });
    const j = await res.json();

    document.getElementById("scaleBox").style.display = "block";

    const compact = {...j};
    if(compact.preview_png_b64) delete compact.preview_png_b64;
    document.getElementById("scaleOut").textContent = JSON.stringify(compact, null, 2);

    if(j.preview_png_b64){
      document.getElementById("scalePreviewWrap").style.display = "block";
      document.getElementById("scalePreview").src = "data:image/png;base64," + j.preview_png_b64;
    }

    if(j.ok && j.nm_per_px){
      document.getElementById("nm_per_px").value = Number(j.nm_per_px).toFixed(6);
      setStatus(`Auto-Scale ok: ${j.scale_label} / ${j.bar_length_px}px → ${Number(j.nm_per_px).toFixed(6)} nm/px`);
    }else{
      setStatus("Auto-Scale nicht erfolgreich (siehe Details).");
    }
  }catch(e){
    setStatus("Auto-Scale error: " + e);
  }
}

async function analyze(){
  if(!selectedFile){ setStatus("Bitte zuerst ein Bild auswählen."); return; }
  setStatus("Analysiere…");

  const nm_per_px = document.getElementById("nm_per_px").value;
  const invert = document.getElementById("invert").checked ? "1" : "0";
  const min_area_px = document.getElementById("min_area_px").value || "80";
  const hole_area_px = document.getElementById("hole_area_px").value || "0";
  const thr_offset = document.getElementById("thr_offset").value || "0";

  const fd = new FormData();
  fd.append("file", selectedFile);
  fd.append("invert", invert);
  fd.append("min_area_px", min_area_px);
  fd.append("hole_area_px", hole_area_px);
  fd.append("thr_offset", thr_offset);
  if(nm_per_px) fd.append("nm_per_px", nm_per_px);

  try{
    const res = await fetch("/api/sem/analyze", { method:"POST", body: fd });
    const j = await res.json();

    if(!j.ok){
      setStatus("Error: " + (j.error || "unknown"));
      document.getElementById("jsonDetails").style.display = "block";
      document.getElementById("out").textContent = JSON.stringify(j, null, 2);
      return;
    }

    document.getElementById("fileCard").style.display = "block";
    document.getElementById("resizeFactor").textContent = fmt(j.resize_factor_used);
    document.getElementById("effScale").textContent = j.nm_per_px_effective ? (fmt(j.nm_per_px_effective) + " nm/px") : "-";

    if(j.threshold_used !== undefined && j.threshold_used !== null){
      document.getElementById("thrUsed").textContent =
        `${fmt(j.threshold_used)} (Otsu ${fmt(j.threshold_otsu)} + ${j.threshold_offset})`;
    }else{
      document.getElementById("thrUsed").textContent = "-";
    }

    if(j.preview_overlay_png_b64){
      document.getElementById("overlayBox").style.display = "block";
      document.getElementById("overlayImg").src = "data:image/png;base64," + j.preview_overlay_png_b64;
    }
    if(j.preview_mask_png_b64){
      document.getElementById("maskBox").style.display = "block";
      document.getElementById("maskImg").src = "data:image/png;base64," + j.preview_mask_png_b64;
    }

    document.getElementById("metricsCards").style.display = "grid";
    document.getElementById("mFg").textContent = fmt(j.area_fraction_foreground);
    document.getElementById("mBg").textContent = fmt(j.area_fraction_background);

    document.getElementById("mN").textContent = j.stats_px?.n ?? "-";
    document.getElementById("mMeanPx").textContent = fmt(j.stats_px?.mean);
    document.getElementById("mMedPx").textContent = fmt(j.stats_px?.median);
    document.getElementById("mMeanUm").textContent = (j.stats_um && j.stats_um.mean !== undefined) ? fmt(j.stats_um.mean) : "-";

    const compact = {...j};
    if(compact.preview_overlay_png_b64) delete compact.preview_overlay_png_b64;
    if(compact.preview_mask_png_b64) delete compact.preview_mask_png_b64;

    document.getElementById("jsonDetails").style.display = "block";
    document.getElementById("out").textContent = JSON.stringify(compact, null, 2);

    setStatus("Fertig.");
    document.getElementById("metricsCards").scrollIntoView({behavior:"smooth", block:"start"});
  }catch(e){
    setStatus("Error: " + e);
  }
}
</script>
</body>
</html>
"""
    )


@app.post("/api/sem/detect_scale")
async def api_detect_scale(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided.")
        data = await file.read()
        out = detect_scale_bar_nm_per_px(data)
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@app.post("/api/sem/analyze")
async def api_sem_analyze(
    file: UploadFile = File(...),
    nm_per_px: Optional[float] = Form(default=None),
    invert: str = Form(default="0"),
    min_area_px: int = Form(default=80),
    hole_area_px: int = Form(default=0),
    thr_offset: int = Form(default=0),  # ✅ NEW
):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided.")
        data = await file.read()
        inv = invert.strip().lower() in {"1", "true", "yes", "on"}

        out = analyze_sem(
            data,
            nm_per_px=nm_per_px,
            invert=inv,
            min_area_px=int(min_area_px),
            hole_area_px=int(hole_area_px),
            thr_offset=int(thr_offset),  # ✅ NEW
        )
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})
