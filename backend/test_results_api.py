# /workspaces/pbm/backend/test_results_api.py
# Step 2, Layer C — Test Suite
# v1.0 — 2026-03-05
#
# Tests all three endpoints using a mocked Supabase client.
# No live Supabase connection required.
#
# Run with:
#   python -m pytest backend/test_results_api.py -v
#   OR
#   python backend/test_results_api.py

from __future__ import annotations

import sys
import os
import uuid
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Minimal inline pytest shim so the file runs without pytest installed
# ---------------------------------------------------------------------------

_tests: list[tuple[str, callable]] = []

def _register(fn):
    _tests.append((fn.__name__, fn))
    return fn


# ---------------------------------------------------------------------------
# Synthetic test data
# ---------------------------------------------------------------------------

SESSION_ID = str(uuid.uuid4())
CHILD_ID = str(uuid.uuid4())
CREATED_AT = datetime.now(timezone.utc).isoformat()

SYNTHETIC_RESULT = {
    "id": str(uuid.uuid4()),
    "session_id": SESSION_ID,
    "saccadic_latency_ms": 187.3,
    "gaze_entropy": 1.42,
    "social_gaze_saliency_pct": 0.61,
    "pupillary_efficiency_pct": 12.4,
    "strategic_blink_rate": 17.2,
    "strategic_timing_score": 0.73,
    "fixation_stability_px": 2.81,
    "info_intake_rate": 0.034,
    "saccadic_peak_velocity": 412.0,
    "research_flags": {"saccadic_peak_velocity": "research-grade"},
    "quality_score": 0.94,
    "created_at": CREATED_AT,
}

# Low-quality variant for threshold tests
LOW_QUALITY_RESULT = dict(SYNTHETIC_RESULT, quality_score=0.65, session_id=str(uuid.uuid4()))
POOR_QUALITY_RESULT = dict(SYNTHETIC_RESULT, quality_score=0.50, session_id=str(uuid.uuid4()))

SYNTHETIC_SESSION = {
    "id": SESSION_ID,
    "child_id": CHILD_ID,
    "video_id": "NinjaGo1.mp4",
    "created_at": CREATED_AT,
    "sync_valid": True,
}


# ---------------------------------------------------------------------------
# Supabase mock factory
# ---------------------------------------------------------------------------

def _make_supabase_mock(result_data, session_data=None, sessions_list=None):
    """
    Build a mock Supabase client that returns controlled data.
    Supports chained .table().select().eq().maybe_single().execute() calls.
    """
    mock_sb = MagicMock()

    def table_side_effect(table_name):
        mock_table = MagicMock()

        def select_side_effect(*args, **kwargs):
            mock_query = MagicMock()

            def eq_side_effect(col, val):
                mock_eq = MagicMock()

                def maybe_single_side_effect():
                    mock_ms = MagicMock()
                    if table_name == "pbm_results":
                        mock_ms.execute.return_value = MagicMock(data=result_data)
                    elif table_name == "sessions":
                        mock_ms.execute.return_value = MagicMock(data=session_data)
                    return mock_ms

                mock_eq.maybe_single = maybe_single_side_effect
                # For .order() chain used in get_sessions
                mock_order = MagicMock()
                mock_order.execute.return_value = MagicMock(
                    data=sessions_list if sessions_list is not None else []
                )
                mock_eq.order = MagicMock(return_value=mock_order)
                return mock_eq

            mock_query.eq = eq_side_effect

            # For .in_() used in get_sessions to fetch results in bulk
            def in_side_effect(col, vals):
                mock_in = MagicMock()
                mock_in.execute.return_value = MagicMock(
                    data=[result_data] if result_data else []
                )
                return mock_in

            mock_query.in_ = in_side_effect
            return mock_query

        mock_table.select = select_side_effect
        return mock_table

    mock_sb.table = table_side_effect
    return mock_sb


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@_register
def test_get_results_returns_all_biomarkers():
    """GET /results/{session_id} returns a value and grade for each biomarker."""
    from results_api import get_results, BIOMARKER_META

    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        response = get_results(SESSION_ID)

    assert response["session_id"] == SESSION_ID
    assert response["quality_score"] == 0.94
    assert "biomarkers" in response

    for key in BIOMARKER_META:
        assert key in response["biomarkers"], f"Missing biomarker: {key}"
        entry = response["biomarkers"][key]
        assert "value" in entry
        assert "grade" in entry


@_register
def test_research_flag_produces_research_grade():
    """
    Any biomarker key present in research_flags must have grade='research'.
    All others must have grade='clinical'.
    """
    from results_api import get_results

    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        response = get_results(SESSION_ID)

    spv = response["biomarkers"]["saccadic_peak_velocity"]
    assert spv["grade"] == "research", (
        "saccadic_peak_velocity must be grade='research' when present in research_flags"
    )

    for key, entry in response["biomarkers"].items():
        if key != "saccadic_peak_velocity":
            assert entry["grade"] == "clinical", (
                f"{key} should be grade='clinical' but got '{entry['grade']}'"
            )


@_register
def test_get_results_404_on_missing_session():
    """GET /results/{session_id} returns 404 when session does not exist."""
    from results_api import get_results
    from fastapi import HTTPException

    mock_sb = _make_supabase_mock(None, None)
    with patch("results_api._get_supabase", return_value=mock_sb):
        try:
            get_results("nonexistent-session-id")
            assert False, "Expected HTTPException 404"
        except HTTPException as e:
            assert e.status_code == 404


@_register
def test_report_contains_disclaimer():
    """GET /results/{session_id}/report always includes the disclaimer footer."""
    from results_api import get_report, REPORT_DISCLAIMER

    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        report = get_report(SESSION_ID)

    assert "disclaimer" in report
    assert report["disclaimer"] == REPORT_DISCLAIMER


@_register
def test_report_research_grade_notice_present():
    """Report must include a RESEARCH GRADE notice on any research-flagged biomarker."""
    from results_api import get_report

    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        report = get_report(SESSION_ID)

    spv = report["biomarkers"]["saccadic_peak_velocity"]
    assert "research_grade_notice" in spv, (
        "RESEARCH GRADE notice must be present for saccadic_peak_velocity"
    )
    assert "RESEARCH GRADE" in spv["research_grade_notice"]


@_register
def test_report_no_research_notice_on_clinical_biomarkers():
    """Clinical-grade biomarkers must not carry a research_grade_notice."""
    from results_api import get_report

    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        report = get_report(SESSION_ID)

    for key, entry in report["biomarkers"].items():
        if entry["grade"] == "clinical":
            assert "research_grade_notice" not in entry, (
                f"{key} is clinical-grade but has a research_grade_notice"
            )


@_register
def test_report_quality_warning_low():
    """quality_score 0.60–0.74 → low-quality warning present."""
    from results_api import get_report

    lq_result = dict(LOW_QUALITY_RESULT, session_id=SESSION_ID)
    mock_sb = _make_supabase_mock(lq_result, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        report = get_report(SESSION_ID)

    assert "warning" in report["quality"], "Low quality score should produce a warning"
    assert "caution" in report["quality"]["warning"].lower()


@_register
def test_report_quality_warning_poor():
    """quality_score < 0.60 → poor-quality warning present."""
    from results_api import get_report

    pq_result = dict(POOR_QUALITY_RESULT, session_id=SESSION_ID)
    mock_sb = _make_supabase_mock(pq_result, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        report = get_report(SESSION_ID)

    assert "warning" in report["quality"], "Poor quality score should produce a warning"
    assert "insufficient" in report["quality"]["warning"].lower()


@_register
def test_report_no_quality_warning_high():
    """quality_score >= 0.75 → no quality warning."""
    from results_api import get_report

    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        report = get_report(SESSION_ID)

    assert "warning" not in report["quality"], (
        "High quality score should not produce a warning"
    )


@_register
def test_report_plain_english_labels_present():
    """Every biomarker in the report must have an 'observation' plain-English label."""
    from results_api import get_report

    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION)
    with patch("results_api._get_supabase", return_value=mock_sb):
        report = get_report(SESSION_ID)

    for key, entry in report["biomarkers"].items():
        assert "observation" in entry, f"Missing plain-English observation for {key}"
        assert isinstance(entry["observation"], str) and len(entry["observation"]) > 0


@_register
def test_report_404_on_missing_session():
    """GET /results/{session_id}/report returns 404 when session does not exist."""
    from results_api import get_report
    from fastapi import HTTPException

    mock_sb = _make_supabase_mock(None, None)
    with patch("results_api._get_supabase", return_value=mock_sb):
        try:
            get_report("nonexistent-id")
            assert False, "Expected HTTPException 404"
        except HTTPException as e:
            assert e.status_code == 404


@_register
def test_get_sessions_returns_list():
    """GET /sessions/{child_id} returns a list of session summaries."""
    from results_api import get_sessions

    sessions_list = [SYNTHETIC_SESSION]
    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION, sessions_list)
    with patch("results_api._get_supabase", return_value=mock_sb):
        response = get_sessions(CHILD_ID)

    assert response["child_id"] == CHILD_ID
    assert response["session_count"] == 1
    assert len(response["sessions"]) == 1

    entry = response["sessions"][0]
    assert "session_id" in entry
    assert "created_at" in entry
    assert "video_id" in entry
    assert "quality_score" in entry
    assert "biomarker_grades" in entry


@_register
def test_get_sessions_biomarker_grades_correct():
    """Session list biomarker_grades must reflect research_flags correctly."""
    from results_api import get_sessions

    sessions_list = [SYNTHETIC_SESSION]
    mock_sb = _make_supabase_mock(SYNTHETIC_RESULT, SYNTHETIC_SESSION, sessions_list)
    with patch("results_api._get_supabase", return_value=mock_sb):
        response = get_sessions(CHILD_ID)

    grades = response["sessions"][0]["biomarker_grades"]
    assert grades["saccadic_peak_velocity"] == "research"
    assert grades["gaze_entropy"] == "clinical"


@_register
def test_get_sessions_404_on_missing_child():
    """GET /sessions/{child_id} returns 404 when child has no sessions."""
    from results_api import get_sessions
    from fastapi import HTTPException

    mock_sb = _make_supabase_mock(None, None, [])
    with patch("results_api._get_supabase", return_value=mock_sb):
        try:
            get_sessions("nonexistent-child-id")
            assert False, "Expected HTTPException 404"
        except HTTPException as e:
            assert e.status_code == 404


@_register
def test_flagged_biomarker_label():
    """A biomarker value outside the flag threshold must produce the flagged observation."""
    from results_api import _plain_english

    # SL > 300 ms → should be flagged
    result = _plain_english("saccadic_latency_ms", 350.0)
    assert result["flagged"] is True
    assert "Slower-than-typical" in result["observation"]

    # SL within range → should not be flagged
    result = _plain_english("saccadic_latency_ms", 200.0)
    assert result["flagged"] is False


@_register
def test_none_value_handled_gracefully():
    """A None biomarker value (not computed) must not raise an exception."""
    from results_api import _plain_english

    result = _plain_english("gaze_entropy", None)
    assert result["value"] is None
    assert result["flagged"] is False
    assert isinstance(result["observation"], str)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all():
    passed = 0
    failed = 0
    errors = []

    # Add backend/ to path so imports work when run from /workspaces/pbm
    sys.path.insert(0, os.path.dirname(__file__))

    print(f"\n{'=' * 60}")
    print("PBM Layer C — Results API Test Suite")
    print(f"{'=' * 60}\n")

    for name, fn in _tests:
        try:
            fn()
            print(f"  ✓  {name}")
            passed += 1
        except Exception as exc:
            print(f"  ✗  {name}")
            print(f"       {type(exc).__name__}: {exc}")
            errors.append((name, exc))
            failed += 1

    print(f"\n{'=' * 60}")
    print(f"Results: {passed} passed, {failed} failed out of {passed + failed} tests")
    print(f"{'=' * 60}\n")

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    run_all()