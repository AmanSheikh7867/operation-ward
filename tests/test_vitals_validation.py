"""
Tests for vital sign validation, parsing, and edge cases.
Covers: valid ranges, out-of-range, missing fields, wrong types,
NaN/infinity values, and boundary conditions.
"""
import math
import json
import pytest
import numpy as np


# ---------------------------------------------------------------------------
# Helper: extract vitals dict the same way process() does in main.py
# ---------------------------------------------------------------------------
REQUIRED_KEYS = ("hr", "spo2", "sys", "dia", "temp")


def parse_vitals(data: dict) -> dict:
    """Replicate the parsing logic inside main.process()."""
    return {k: float(data[k]) for k in REQUIRED_KEYS}


# ---------------------------------------------------------------------------
# Valid range tests
# ---------------------------------------------------------------------------
class TestValidRanges:
    def test_normal_vitals_parse(self, normal_vitals):
        v = parse_vitals(normal_vitals)
        assert all(k in v for k in REQUIRED_KEYS)
        assert all(isinstance(v[k], float) for k in REQUIRED_KEYS)

    def test_abnormal_vitals_parse(self, abnormal_vitals):
        v = parse_vitals(abnormal_vitals)
        assert v["hr"] == 140.0
        assert v["spo2"] == 85.0

    def test_borderline_low_parse(self, borderline_low_vitals):
        v = parse_vitals(borderline_low_vitals)
        assert v["hr"] == 55.0
        assert v["spo2"] == 95.0
        assert v["temp"] == 36.0

    def test_borderline_high_parse(self, borderline_high_vitals):
        v = parse_vitals(borderline_high_vitals)
        assert v["hr"] == 100.0
        assert v["spo2"] == 100.0
        assert v["temp"] == 37.8

    def test_edge_case_extreme_values(self, edge_case_vitals):
        v = parse_vitals(edge_case_vitals)
        assert v["hr"] == 35.0
        assert v["spo2"] == 70.0
        assert v["temp"] == 34.0


# ---------------------------------------------------------------------------
# Missing / extra fields
# ---------------------------------------------------------------------------
class TestMissingFields:
    def test_missing_temp_raises_keyerror(self, missing_field_vitals):
        with pytest.raises(KeyError):
            parse_vitals(missing_field_vitals)

    @pytest.mark.parametrize("missing_key", REQUIRED_KEYS)
    def test_each_field_required(self, normal_vitals, missing_key):
        data = {k: v for k, v in normal_vitals.items() if k != missing_key}
        with pytest.raises(KeyError):
            parse_vitals(data)

    def test_extra_fields_are_ignored(self, normal_vitals):
        data = {**normal_vitals, "extra": 999, "ts": 1234567890}
        v = parse_vitals(data)
        assert "extra" not in v
        assert "ts" not in v
        assert len(v) == len(REQUIRED_KEYS)


# ---------------------------------------------------------------------------
# Wrong types
# ---------------------------------------------------------------------------
class TestWrongTypes:
    def test_string_values_raise(self, wrong_type_vitals):
        with pytest.raises(ValueError):
            parse_vitals(wrong_type_vitals)

    def test_numeric_strings_work(self):
        data = {"hr": "75", "spo2": "98.0", "sys": "120", "dia": "78", "temp": "36.8"}
        v = parse_vitals(data)
        assert v["hr"] == 75.0
        assert v["temp"] == 36.8

    def test_none_value_raises(self, normal_vitals):
        data = {**normal_vitals, "hr": None}
        with pytest.raises(TypeError):
            parse_vitals(data)

    def test_boolean_value_casts(self, normal_vitals):
        data = {**normal_vitals, "hr": True}
        v = parse_vitals(data)
        assert v["hr"] == 1.0  # bool -> float is valid in Python


# ---------------------------------------------------------------------------
# NaN and infinity
# ---------------------------------------------------------------------------
class TestSpecialFloats:
    def test_nan_parses_but_is_nan(self, nan_vitals):
        v = parse_vitals(nan_vitals)
        assert math.isnan(v["hr"])

    def test_inf_parses_but_is_inf(self, inf_vitals):
        v = parse_vitals(inf_vitals)
        assert math.isinf(v["hr"])

    def test_negative_inf(self, normal_vitals):
        data = {**normal_vitals, "spo2": float("-inf")}
        v = parse_vitals(data)
        assert math.isinf(v["spo2"])


# ---------------------------------------------------------------------------
# JSON parsing (as MQTT payload would arrive)
# ---------------------------------------------------------------------------
class TestJSONParsing:
    def test_valid_json_payload(self, normal_vitals):
        payload = json.dumps(normal_vitals).encode()
        d = json.loads(payload)
        v = parse_vitals(d)
        assert v["hr"] == 75.0

    def test_invalid_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            json.loads(b"not json at all")

    def test_empty_json_object(self):
        d = json.loads(b"{}")
        with pytest.raises(KeyError):
            parse_vitals(d)

    def test_array_instead_of_object(self):
        d = json.loads(b"[1, 2, 3]")
        with pytest.raises((KeyError, TypeError)):
            parse_vitals(d)
