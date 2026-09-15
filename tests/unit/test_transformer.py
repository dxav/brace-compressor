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


def test_prior_not_constant_across_varied_mask():
    """T036: out_scale must be non-zero so the prior is not a constant field."""
    import numpy as np

    from hoaps_compressor.model.transformer import TransformerPredictor

    pred = TransformerPredictor()
    mask = np.zeros((2, 8, 8), dtype=bool)
    mask[:, :2, :] = True  # "land" strip missing
    prior = pred.base_prior(mask)
    assert prior.shape == mask.shape
    # A collapsed out_scale (tanh(0)=0) yields a constant prior; assert spread.
    assert float(np.ptp(prior)) > 1e-6


def test_causal_block_predict_determinism():
    """T041: same reconstructed state -> same block predictions."""
    import numpy as np

    from hoaps_compressor.model.transformer import TransformerPredictor

    pred = TransformerPredictor()
    mask = np.zeros((2, 4, 4), dtype=bool)
    mask[0, 0, 0] = True
    recon = np.zeros((2 * 4, 4), dtype=np.float64)
    recon[0, 1] = 30.0
    recon[0, 2] = 32.0
    cells = [(0, 0, 1), (0, 0, 2), (0, 0, 3)]
    p1 = pred.causal_block_predict(recon, mask, cells)
    p2 = pred.causal_block_predict(recon, mask, cells)
    np.testing.assert_array_equal(p1, p2)


def test_causal_block_predict_causality():
    """T041: a cell never attends to a not-yet-reconstructed cell.

    The prediction for the first cell in a block must not depend on later
    cells' features (causal mask), even though they are present in the
    block input.
    """
    import numpy as np

    from hoaps_compressor.model.transformer import TransformerPredictor

    pred = TransformerPredictor()
    mask = np.zeros((1, 4, 4), dtype=bool)
    cells = [(0, 0, 0), (0, 0, 1)]
    recon_a = np.zeros((1 * 4, 4), dtype=np.float64)
    recon_b = np.zeros((1 * 4, 4), dtype=np.float64)
    recon_b[0, 1] = 50.0  # later cell's neighbor differs
    pa = pred.causal_block_predict(recon_a, mask, cells)
    pb = pred.causal_block_predict(recon_b, mask, cells)
    # Cell 0 attends only to itself (causal mask), so its prediction is
    # unaffected by changes to cell 1's features.
    assert abs(float(pa[0]) - float(pb[0])) < 1e-6


def test_causal_block_predict_uses_neighbors():
    """T041: predictions change when reconstructed neighbors change."""
    import numpy as np

    from hoaps_compressor.model.transformer import TransformerPredictor

    pred = TransformerPredictor()
    mask = np.zeros((1, 4, 4), dtype=bool)
    cells = [(0, 0, 1), (0, 0, 2)]
    recon_a = np.zeros((1 * 4, 4), dtype=np.float64)
    recon_a[0, 0] = 20.0
    recon_b = np.zeros((1 * 4, 4), dtype=np.float64)
    recon_b[0, 0] = 50.0
    pa = pred.causal_block_predict(recon_a, mask, cells)
    pb = pred.causal_block_predict(recon_b, mask, cells)
    # Different neighbor values should (generally) yield different predictions.
    assert not np.allclose(pa, pb, atol=1e-6)
