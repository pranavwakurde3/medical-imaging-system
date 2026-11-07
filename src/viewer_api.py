import os
import sqlite3, time, requests
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel

ORTHANC = os.environ.get("ORTHANC_URL", "http://localhost:8042")
DB = "data/results.db"
app = FastAPI(title="Medical imaging viewer API")


def db():
    c = sqlite3.connect(DB)
    c.row_factory = sqlite3.Row
    c.execute("""CREATE TABLE IF NOT EXISTS audit (
        id INTEGER PRIMARY KEY AUTOINCREMENT, study_id TEXT, action TEXT,
        reviewer TEXT, ai_score REAL, at REAL)""")
    return c


class Decision(BaseModel):
    study_id: str
    decision: str      # "approve" or "reject"
    reviewer: str


@app.get("/")
def home():
    return FileResponse("src/viewer.html")


@app.get("/cases")
def cases():
    c = db()
    rows = c.execute("SELECT study_id, patient_id, score, status, decision FROM results "
                     "ORDER BY (status='pending') DESC, score DESC").fetchall()
    return [dict(r) for r in rows]


@app.get("/overlay/{study_id}")
def overlay(study_id: str):
    r = db().execute("SELECT overlay_path FROM results WHERE study_id=?", (study_id,)).fetchone()
    if not r:
        raise HTTPException(404, "unknown study")
    return FileResponse(r["overlay_path"])


@app.get("/scan/{study_id}")
def scan(study_id: str):
    inst = requests.get(f"{ORTHANC}/studies/{study_id}/instances").json()[0]["ID"]
    img = requests.get(f"{ORTHANC}/instances/{inst}/preview")
    return Response(img.content, media_type="image/png")


@app.post("/decision")
def decide(d: Decision):
    if d.decision not in ("approve", "reject"):
        raise HTTPException(400, "decision must be approve or reject")
    c = db()
    r = c.execute("SELECT score FROM results WHERE study_id=?", (d.study_id,)).fetchone()
    if not r:
        raise HTTPException(404, "unknown study")
    now = time.time()
    c.execute("UPDATE results SET status='reviewed', decision=?, decided_at=? WHERE study_id=?",
              (d.decision, now, d.study_id))
    c.execute("INSERT INTO audit (study_id, action, reviewer, ai_score, at) VALUES (?,?,?,?,?)",
              (d.study_id, d.decision, d.reviewer, r["score"], now))
    c.commit()
    return {"ok": True}


@app.get("/audit")
def audit():
    rows = db().execute("SELECT * FROM audit ORDER BY id DESC LIMIT 50").fetchall()
    return [dict(r) for r in rows]
