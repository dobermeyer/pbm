# /workspaces/pbm/backend/results_api.py
# Step 2, Layer C — Results API & Reporting
# v1.0 — 2026-03-05
#
# Reads from Supabase pbm_results and sessions tables.
# Performs NO calculation. Exposes three REST endpoints:
#   GET /results/{session_id}           → structured biomarker JSON for iOS app
#   GET /results/{session_id}/report    → human-readable report for parents/schools/clinicians
#   GET /sessions/{child_id}            → session history list for a child
#
# Run locally:
#   uvicorn backend.results_api:app --reload --port 8000

from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from fastapi import FastAPI, HTTPException
from supabase import create_client, Client

app = FastAPI(title="PBM Results API", version="1.0.0")

# ---------------------------------------------------------------------------
# Supabase client
# ---------------------------------------------------------------------------

def _get_supabase() -> Client:
    url = os.environ["SUPABASE_URL"]
    key = os.environ["SUPABASE_KEY"]
    return create_client(url, key)


# ---------------------------------------------------------------------------
# Biomarker metadata — labelling scheme (Section 6)
# Ranges are provisional engineering placeholders pending real child data.
# ---------------------------------------------------------------------------

BIOMARKER_META: dict[str, dict[str, Any]] = {
    "saccadic_latency_ms": {
        "label": "Saccadic Latency",
        "unit": "ms",
        "expected_low": 150.0,
        "expected_high": 250.0,
        "flag_condition": lambda v: v > 300.0,
        "flagged_label": "Slower-than-typical visual response",
        "normal_label": "Visual response within typical range",
    },
    "gaze_entropy": {
        "label": "Gaze Entropy",
        "unit": "",
        "expected_low": 1.2,
        "expected_high": 2.0,
        "flag_condition": lambda v: v < 0.8 or v > 2.8,
        "flagged_label": "Unusually focused or unusually scattered gaze",
        "normal_label": "Gaze distribution within typical range",
    },
    "social_gaze_saliency_pct": {
        "label": "Social Gaze Saliency",
        "unit": "",
        "expected_low": 0.4,
        "expected_high": 0.8,
        "flag_condition": lambda v: v < 0.25,
        "flagged_label": "Low attention to faces on screen",
        "normal_label": "Face-directed attention within typical range",
    },
    "pupillary_efficiency_pct": {
        "label": "Pupillary Efficiency",
        "unit": "%",
        "expected_low": 5.0,
        "expected_high": 20.0,
        "flag_condition": lambda v: v < 2.0 or v > 35.0,
        "flagged_label": "Atypical pupil response to visual complexity",
        "normal_label": "Pupil response within typical range",
    },
    "strategic_blink_rate": {
        "label": "Strategic Blink Rate",
        "unit": "blinks/min",
        "expected_low": 12.0,
        "expected_high": 25.0,
        "flag_condition": lambda v: v < 8.0 or v > 40.0,
        "flagged_label": "Atypical blink frequency",
        "normal_label": "Blink rate within typical range",
    },
    "strategic_timing_score": {
        "label": "Strategic Timing Score",
        "unit": "",
        "expected_low": 0.5,
        "expected_high": 0.85,
        "flag_condition": lambda v: v < 0.3,
        "flagged_label": "Blinks not aligned with natural scene pauses",
        "normal_label": "Blink timing within typical range",
    },
    "fixation_stability_px": {
        "label": "Fixation Stability",
        "unit": "px",
        "expected_low": 1.5,
        "expected_high": 4.0,
        "flag_condition": lambda v: v > 6.0,
        "flagged_label": "Gaze drift during still moments",
        "normal_label": "Fixation stability within typical range",
    },
    "info_intake_rate": {
        "label": "Info Intake Rate",
        "unit": "saccades/s",
        "expected_low": 0.02,
        "expected_high": 0.06,
        "flag_condition": lambda v: v < 0.01 or v > 0.10,
        "flagged_label": "Atypical scene-scanning rate",
        "normal_label": "Scene-scanning rate within typical range",
    },
    "saccadic_peak_velocity": {
        "label": "Saccadic Peak Velocity",
        "unit": "deg/s",
        "expected_low": 350.0,
        "expected_high": 550.0,
        "flag_condition": lambda v: v < 250.0,
        "flagged_label": "Low peak eye movement speed",
        "normal_label": "Peak eye movement speed within typical range",
    },
}

REPORT_DISCLAIMER = (
    "This report is a psychometric screening filter, not a clinical diagnosis. "
    "Consult a qualified professional before drawing clinical conclusions."
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _grade(biomarker_key: str, research_flags: dict) -> str:
    """Return 'research' if the biomarker is flagged, else 'clinical'."""
    return "research" if biomarker_key in research_flags else "clinical"


def _plain_english(biomarker_key: str, value: float | None) -> dict[str, Any]:
    """
    Translate a raw float value into an observational plain-English label.
    Returns a dict with 'value', 'flagged', and 'observation' keys.
    Never uses clinical language. Never replaces the raw value.
    """
    if value is None:
        return {"value": None, "flagged": False, "observation": "Not measured in this session"}

    meta = BIOMARKER_META.get(biomarker_key)
    if meta is None:
        return {"value": value, "flagged": False, "observation": "No label available"}

    try:
        flagged = bool(meta["flag_condition"](value))
    except Exception:
        flagged = False

    observation = meta["flagged_label"] if flagged else meta["normal_label"]
    return {
        "value": value,
        "unit": meta["unit"],
        "flagged": flagged,
        "observation": observation,
        "expected_range": f"{meta['expected_low']}–{meta['expected_high']} {meta['unit']}".strip(),
    }


def _quality_warning(quality_score: float | None) -> str | None:
    """Return a quality warning string if required, else None."""
    if quality_score is None:
        return None
    if quality_score >= 0.75:
        return None
    if quality_score >= 0.60:
        return (
            "Session data quality was reduced. "
            "Results should be interpreted with caution."
        )
    return (
        "Session data quality was insufficient for reliable results. "
        "Consider repeating the session."
    )


# ---------------------------------------------------------------------------
# Endpoint 1 — GET /results/{session_id}
# Machine-readable biomarker JSON for the iOS app.
# ---------------------------------------------------------------------------

@app.get("/results/{session_id}")
def get_results(session_id: str) -> dict[str, Any]:
    sb = _get_supabase()

    row = (
        sb.table("pbm_results")
        .select("*")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )

    if row.data is None:
        raise HTTPException(status_code=404, detail=f"No results found for session {session_id}")

    data = row.data
    research_flags: dict = data.get("research_flags") or {}

    biomarkers: dict[str, Any] = {}
    for key in BIOMARKER_META:
        value = data.get(key)
        biomarkers[key] = {
            "value": value,
            "grade": _grade(key, research_flags),
        }

    return {
        "session_id": session_id,
        "quality_score": data.get("quality_score"),
        "biomarkers": biomarkers,
        "research_flags": research_flags,
        "created_at": data.get("created_at"),
    }


# ---------------------------------------------------------------------------
# Endpoint 2 — GET /results/{session_id}/report
# Human-readable structured JSON for parents, schools, and clinicians.
# ---------------------------------------------------------------------------

@app.get("/results/{session_id}/report")
def get_report(session_id: str) -> dict[str, Any]:
    sb = _get_supabase()

    # Fetch biomarker results
    result_row = (
        sb.table("pbm_results")
        .select("*")
        .eq("session_id", session_id)
        .maybe_single()
        .execute()
    )
    if result_row.data is None:
        raise HTTPException(status_code=404, detail=f"No results found for session {session_id}")

    # Fetch session metadata
    session_row = (
        sb.table("sessions")
        .select("child_id, video_id, created_at, sync_valid")
        .eq("id", session_id)
        .maybe_single()
        .execute()
    )
    session_meta = session_row.data or {}

    data = result_row.data
    research_flags: dict = data.get("research_flags") or {}
    quality_score: float | None = data.get("quality_score")

    # Build biomarker section with plain-English labels
    biomarker_report: dict[str, Any] = {}
    for key, meta in BIOMARKER_META.items():
        value = data.get(key)
        entry = _plain_english(key, value)
        entry["label"] = meta["label"]
        entry["grade"] = _grade(key, research_flags)
        # Mandatory RESEARCH GRADE label — never omit if flagged
        if entry["grade"] == "research":
            entry["research_grade_notice"] = (
                "RESEARCH GRADE — this biomarker has not been validated against "
                "a laboratory eye tracker. Interpret with additional caution."
            )
        biomarker_report[key] = entry

    # Quality section
    quality_warning = _quality_warning(quality_score)
    quality_section: dict[str, Any] = {"score": quality_score}
    if quality_warning:
        quality_section["warning"] = quality_warning

    # Session metadata section
    session_date = session_meta.get("created_at") or data.get("created_at")
    session_section = {
        "session_id": session_id,
        "child_id": session_meta.get("child_id"),
        "video_id": session_meta.get("video_id"),
        "date": session_date,
        "sync_valid": session_meta.get("sync_valid"),
    }

    return {
        "report_type": "psychometric_screening_filter",
        "session": session_section,
        "quality": quality_section,
        "biomarkers": biomarker_report,
        "research_flags": research_flags,
        "disclaimer": REPORT_DISCLAIMER,
        "generated_at": datetime.utcnow().isoformat() + "Z",
    }


# ---------------------------------------------------------------------------
# Endpoint 3 — GET /sessions/{child_id}
# Session history list for a child, ordered newest first.
# ---------------------------------------------------------------------------

@app.get("/sessions/{child_id}")
def get_sessions(child_id: str) -> dict[str, Any]:
    sb = _get_supabase()

    # Fetch all sessions for this child
    sessions_resp = (
        sb.table("sessions")
        .select("id, created_at, video_id, sync_valid")
        .eq("child_id", child_id)
        .order("created_at", desc=True)
        .execute()
    )

    if not sessions_resp.data:
        raise HTTPException(status_code=404, detail=f"No sessions found for child {child_id}")

    session_ids = [s["id"] for s in sessions_resp.data]

    # Fetch all pbm_results rows for these sessions in one query
    results_resp = (
        sb.table("pbm_results")
        .select("session_id, quality_score, research_flags, " + ", ".join(BIOMARKER_META.keys()))
        .in_("session_id", session_ids)
        .execute()
    )

    # Index results by session_id
    results_by_session: dict[str, dict] = {
        r["session_id"]: r for r in (results_resp.data or [])
    }

    summaries = []
    for session in sessions_resp.data:
        sid = session["id"]
        result = results_by_session.get(sid)

        if result is None:
            # Session exists but results not yet written (edge case)
            biomarker_grades = {key: None for key in BIOMARKER_META}
            quality_score = None
        else:
            research_flags: dict = result.get("research_flags") or {}
            biomarker_grades = {
                key: _grade(key, research_flags)
                if result.get(key) is not None else None
                for key in BIOMARKER_META
            }
            quality_score = result.get("quality_score")

        summaries.append({
            "session_id": sid,
            "created_at": session["created_at"],
            "video_id": session["video_id"],
            "sync_valid": session.get("sync_valid"),
            "quality_score": quality_score,
            "biomarker_grades": biomarker_grades,
        })

    return {
        "child_id": child_id,
        "session_count": len(summaries),
        "sessions": summaries,
    }