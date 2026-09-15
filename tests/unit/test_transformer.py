"""Unit test for transformer predictor determinism (T025, US4).

Kept as a thin re-export module for the task-mapped file path
``tests/unit/test_transformer.py``; the substantive tests live in
tests/integration/test_space_time_cr.py (predictor determinism,
mask-driven behavior, weight serialization).
"""

from tests.integration.test_space_time_cr import (  # noqa: F401
    test_model_version_mismatch_rejected,
    test_model_weights_serialize_roundtrip,
    test_transformer_determinism,
    test_transformer_ignores_field_values,
)
