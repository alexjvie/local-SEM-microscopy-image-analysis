# SEM Image Analysis (Local)

Lokales Tool zur Analyse von SEM-Bildern (z. B. CNT-SEM):
- **Porosity (2D-Flächenanteil)** aus binärer Maske
- **Längenstatistik** über Skeleton-Längen (px und optional µm)
- **Auto Scale-Bar Erkennung** (best effort, optional mit OCR)
- **SEM-Overlay/Legende** (schwarzer Balken + Text + Scale-Box) wird automatisch **aus der Analyse ausgeschlossen**
- **Drag & Drop UI** + **Dark Mode**

---

## Projektstruktur

Es gibt zwei Startvarianten:

1) **Modular (empfohlen)**  
   - Entry Point: `app/main.py`  
   - Routes: `app/routes/sem.py`  
   - Services/Logik: `app/services/*`  
   - Template: `app/templates/index.html`  
   - Static: `static/` (Projekt-Root)
- Start: `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload`
- Open: http://127.0.0.1:8000

2) **All-in-one (Full Script)**  
   - Eine Datei: `image_server_full.py` (praktisch zum schnellen Testen)
- Start: `python -m uvicorn image_server_full:app --host 127.0.0.1 --port 8000 --reload`
- Open: http://127.0.0.1:8000
---

## Setup

### 1) Virtual Environment (empfohlen)

```bash
cd "/Users/alex.vieweg/Documents/sem analysis"
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
