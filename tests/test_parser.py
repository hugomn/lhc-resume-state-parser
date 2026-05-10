"""Tests for the LHC resume-state parser.

Run from the repo root:
    pytest tests/
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import yaml

# Allow `import parser` when running from tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from parser import (
    find_inconsistency,
    find_next_step,
    parse_task,
)


def _user_msg(content: str) -> dict:
    return {"role": "user", "content": content}


def _system_msg(content: str) -> dict:
    return {"role": "system", "content": content}


# ---------------------------------------------------------------------------
# NEXT-step pattern detection
# ---------------------------------------------------------------------------


def test_next_marker_explicit():
    text = "Workflow:\n- ship X\n- ship Y\nNEXT: ship Z\n"
    assert find_next_step(text) == "ship Z"


def test_next_marker_bracket_tag():
    text = "L5 → SKU C-3308 — mid-weight base layers, qty 8  [NEXT]"
    # Two patterns can match this; either accepted output is correct.
    result = find_next_step(text)
    assert result is not None
    assert "C-3308" in result


def test_next_planned_action_label():
    text = """
    Status: in progress
    Next planned action: load the dryer at 17:30 with the C-batch
    """
    result = find_next_step(text)
    assert result is not None
    assert "dryer" in result.lower()
    assert "17:30" in result


def test_no_next_marker_returns_none():
    text = """
    The team had a great morning. We discussed many things.
    Lunch was good. The afternoon meeting ran long.
    """
    assert find_next_step(text) is None


# ---------------------------------------------------------------------------
# Inconsistency detector
# ---------------------------------------------------------------------------


def test_inconsistency_detector_recipe_a_b():
    """Mirrors the structure of resumption_v2_008 where the detector scores 2/2.
    Section headers must be on their own lines for the splitter to find them."""
    text = """
    DECISION:
    Going with the higher-hydration formula (RECIPE-A, 86% hydration)
    for tomorrow's morning batch.

    LAST ACTION:
    Mixed and bulk-fermented the dough for RECIPE-A, 86% hydration.
    Refrigerated in cold proof unit 3.

    NEXT PLANNED ACTION:
    Pull RECIPE-B (78% hydration) from cold proof at 06:00 and load
    deck oven 1 at 06:30 for the morning bake.
    """
    result = find_inconsistency(text)
    assert result is not None
    chosen, action_target = result
    assert chosen == "RECIPE-A"
    assert action_target == "RECIPE-B"


def test_inconsistency_detector_no_inconsistency():
    """Same structure but consistent — should return None."""
    text = """
    DECISION:
    Proceeding with RECIPE-A.

    LAST ACTION:
    Prepared RECIPE-A materials.

    NEXT PLANNED ACTION:
    Pull RECIPE-A from cold proof at 06:30.
    """
    assert find_inconsistency(text) is None


def test_inconsistency_detector_ignores_non_formula_codes():
    """GCS-9, H-2204 don't match the strict CAPS-DASH-LETTER formula pattern.
    The detector should not flag these as candidate formula tokens."""
    text = """
    DECISION:
    Proceed with shipment.

    LAST ACTION:
    Scanned barcode GCS-9.

    NEXT PLANNED ACTION:
    File under H-2204.
    """
    assert find_inconsistency(text) is None


# ---------------------------------------------------------------------------
# End-to-end on the bundled example task
# ---------------------------------------------------------------------------


def test_parse_task_bundled_warehouse_example():
    """resumption_v2_004 (warehouse tracker) — the parser scored 2/2 on this
    task in the published LHC v0.2 scorecard. The exact phrasing the parser
    extracts may evolve, but the response should reference SKU C-3308.
    """
    example_path = (
        Path(__file__).resolve().parent.parent / "examples" / "resumption_v2_004.yaml"
    )
    if not example_path.exists():
        pytest.skip(f"bundled example missing: {example_path}")
    task = yaml.safe_load(example_path.read_text())
    response = parse_task(task)
    assert "C-3308" in response
    assert "next step" in response.lower() or "continuing" in response.lower()


def test_parse_task_decline_path():
    """When no NEXT marker and no inconsistency, the parser must decline
    explicitly rather than free-associate."""
    task = {
        "id": "synthetic_no_next",
        "setup": [
            _system_msg("You are a helpful assistant."),
            _user_msg("Today was a productive day. We made progress on many fronts."),
        ],
        "probe": [_user_msg("Resume.")],
    }
    response = parse_task(task)
    assert "cannot determine" in response.lower()
    assert "explicit NEXT-step marker" in response or "explicit next-step" in response.lower()
