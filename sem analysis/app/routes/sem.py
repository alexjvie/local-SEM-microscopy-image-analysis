from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.services.analysis import analyze_sem
from app.services.scale import detect_scale_bar_nm_per_px

router = APIRouter(prefix="/api/sem", tags=["sem"])


@router.post("/detect_scale")
async def api_detect_scale(file: UploadFile = File(...)):
    try:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided.")
        data = await file.read()
        out = detect_scale_bar_nm_per_px(data)
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})


@router.post("/analyze")
async def api_sem_analyze(
    file: UploadFile = File(...),
    nm_per_px: Optional[float] = Form(default=None),
    invert: str = Form(default="0"),
    min_area_px: int = Form(default=80),
    hole_area_px: int = Form(default=0),
    thr_offset: int = Form(default=0),  # ✅ NEU: Slider-Offset
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
            thr_offset=int(thr_offset),  # ✅ NEU
        )
        return JSONResponse(out)
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)})
