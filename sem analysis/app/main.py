from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles

from app.routes.sem import router as sem_router

app = FastAPI(title="SEM Image Analysis", version="0.6")

# Projekt-Root: .../sem analysis/
ROOT = Path(__file__).resolve().parents[1]

# Bei dir:
# - templates liegen in app/templates/
# - static liegt im Projekt-Root static/
TEMPLATES_DIR = Path(__file__).resolve().parent / "templates"
STATIC_DIR = ROOT / "static"

STATIC_DIR.mkdir(parents=True, exist_ok=True)
TEMPLATES_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# API Routes aktivieren (wichtig!)
app.include_router(sem_router)

@app.get("/favicon.ico")
def favicon():
    return Response(status_code=204)

@app.get("/", response_class=HTMLResponse)
def home():
    html_path = TEMPLATES_DIR / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h3>app/templates/index.html not found</h3>", status_code=500)
    return HTMLResponse(html_path.read_text(encoding="utf-8"))
