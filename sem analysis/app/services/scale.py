from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import cv2

from app.services.io_utils import decode_image, downscale_to_max, encode_png_b64, to_gray
from app.services.overlays import detect_bottom_legend_strip_top, find_scale_bar_bbox


def _try_ocr_text(roi_gray):
    """
    OCR best effort:
    - Requires pytesseract + system tesseract
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
        return val, f"{val:g} nm"
    if unit == "µm":
        return val * 1000.0, f"{val:g} µm"
    if unit == "mm":
        return val * 1e6, f"{val:g} mm"
    return None


def detect_scale_bar_nm_per_px(file_bytes: bytes) -> Dict[str, Any]:
    img = decode_image(file_bytes)
    gray = to_gray(img)

    gray_det, f = downscale_to_max(gray, max_dim=1600)
    det = find_scale_bar_bbox(gray_det)
    if det is None:
        return {"ok": False, "error": "No scale bar-like rectangle found."}

    x_d, y_d, w_d, h_d = det["x"], det["y"], det["w"], det["h"]
    x = int(round(x_d / f))
    y = int(round(y_d / f))
    bw = max(1, int(round(w_d / f)))
    bh = max(1, int(round(h_d / f)))

    H, W = gray.shape[:2]
    strip_top = detect_bottom_legend_strip_top(gray)
    if strip_top is None:
        strip_top = int(0.75 * H)

    rois: List[Tuple[str, Tuple[int, int, int, int]]] = [
        ("around_bar", (max(0, x - int(0.30 * bw)), max(0, y - int(14 * bh)), min(W, x + int(2.8 * bw)), min(H, y + int(12 * bh)))),
        ("right_box",  (max(0, x + int(0.10 * bw)), max(0, y - int(10 * bh)), min(W, x + int(3.0 * bw)), min(H, y + int(10 * bh)))),
        ("legend_chunk", (max(0, x - int(0.8 * bw)), max(0, strip_top), min(W, x + int(3.2 * bw)), min(H, strip_top + int(0.25 * (H - strip_top))))),
    ]

    best_parse = None  # (nm,label,text,roi_name,rect)
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
    preview_b64 = encode_png_b64(rgb, max_dim=1100)

    if not best_parse:
        return {
            "ok": False,
            "error": "Scale bar found, but OCR could not parse a length (e.g., '200 nm', '1 µm').",
            "bar_length_px": bw,
            "preview_png_b64": preview_b64,
            "hint": "Install OCR: pip install -r requirements-ocr.txt AND install system tesseract (brew/apt).",
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
