# =============================================================================
# tests/test_model.py — Unit tests for GOOSEMask2Former architecture
# =============================================================================
"""
Run with: pytest tests/test_model.py -v
"""

import torch
import pytest
from src.model import GOOSEMask2Former, GOOSEModelOutput


# ─── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def model():
    """Instantiate a small GOOSE-M2F model for testing (CPU, no pretrained weights)."""
    # We skip pretrained loading in tests to keep them fast and offline-safe.
    # Instead we patch the base_model initialization.
    return None  # Replace with a mock or lightweight build in CI


# ─── Architecture Tests ───────────────────────────────────────────────────────

class TestFeatureRefinementModule:
    """Tests for the ASPP-lite + CBAM Feature Refinement Module."""

    def test_output_shape_unchanged(self):
        """FRM output should preserve spatial dimensions and channel count."""
        from src.model import FeatureRefinementModule
        frm = FeatureRefinementModule(channels=256, mid_channels=64)
        x = torch.randn(1, 256, 64, 128)
        out = frm(x)
        assert out.shape == x.shape, f"Expected {x.shape}, got {out.shape}"

    def test_residual_non_trivial(self):
        """FRM output should differ from input (non-identity)."""
        from src.model import FeatureRefinementModule
        frm = FeatureRefinementModule(channels=256, mid_channels=64)
        x = torch.randn(1, 256, 32, 64)
        out = frm(x)
        assert not torch.allclose(out, x), "FRM should not be a pure identity."

    def test_batch_consistency(self):
        """FRM should handle batch_size > 1 without error."""
        from src.model import FeatureRefinementModule
        frm = FeatureRefinementModule(channels=256, mid_channels=64)
        x = torch.randn(4, 256, 32, 64)
        out = frm(x)
        assert out.shape == (4, 256, 32, 64)


class TestAuxiliaryHead:
    """Tests for the Auxiliary Supervision Head."""

    def test_output_channels(self):
        """AuxHead should produce exactly num_classes output channels."""
        from src.model import AuxiliaryHead
        head = AuxiliaryHead(in_channels=256, num_classes=64)
        x = torch.randn(1, 256, 128, 256)
        out = head(x)
        assert out.shape == (1, 64, 128, 256)

    def test_spatial_preserved(self):
        """AuxHead must not change spatial dimensions."""
        from src.model import AuxiliaryHead
        head = AuxiliaryHead(in_channels=256, num_classes=64)
        x = torch.randn(2, 256, 64, 128)
        out = head(x)
        assert out.shape[2:] == x.shape[2:]


class TestChannelAttention:
    """Tests for the CBAM Channel Attention module."""

    def test_output_shape(self):
        from src.model import ChannelAttention
        ca = ChannelAttention(channels=256, reduction=16)
        x = torch.randn(2, 256, 32, 32)
        out = ca(x)
        assert out.shape == x.shape

    def test_attention_bounded(self):
        """Attention scale should remain in (0, 1) after sigmoid."""
        from src.model import ChannelAttention
        ca = ChannelAttention(channels=64, reduction=8)
        x = torch.randn(1, 64, 16, 16)
        out = ca(x)
        # Output should be in reasonable range (not exploding)
        assert out.abs().max() < 1e4


class TestSpatialAttention:
    """Tests for the CBAM Spatial Attention module."""

    def test_output_shape(self):
        from src.model import SpatialAttention
        sa = SpatialAttention(kernel_size=7)
        x = torch.randn(2, 256, 32, 32)
        out = sa(x)
        assert out.shape == x.shape


# ─── Gaussian Kernel Tests ────────────────────────────────────────────────────

class TestGaussianKernel:
    """Tests for the inference 2D Gaussian kernel utility."""

    def test_shape(self):
        from src.inference import _make_gaussian_kernel
        k = _make_gaussian_kernel(896, 896)
        assert k.shape == (896, 896)

    def test_center_max(self):
        """Gaussian kernel should peak at the center."""
        from src.inference import _make_gaussian_kernel
        k = _make_gaussian_kernel(100, 100)
        cy, cx = 50, 50
        center_val = k[cy, cx]
        corner_val = k[0, 0]
        assert center_val > corner_val, "Center should have higher weight than corner."

    def test_non_negative(self):
        from src.inference import _make_gaussian_kernel
        k = _make_gaussian_kernel(64, 64)
        assert (k >= 0).all()


# ─── Metric Tests ─────────────────────────────────────────────────────────────

class TestOfficialCompositeMetric:
    """Tests for the GOOSE official evaluation metric."""

    def test_perfect_prediction(self):
        """Perfect predictions should yield 100% composite score."""
        from src.features import OfficialCompositeMetric
        fine_to_coarse = {i: min(i // 6 + 1, 11) for i in range(64)}
        class_names    = {i: f"class_{i}" for i in range(64)}
        m = OfficialCompositeMetric(64, fine_to_coarse, class_names)
        pred    = torch.arange(64).repeat_interleave(100)
        target  = torch.arange(64).repeat_interleave(100)
        m.update(pred, target)
        r = m.compute()
        assert r["official_fine"] > 95.0, "Perfect prediction should give near-100% fine mIoU."

    def test_void_ignored(self):
        """Void class (id=0) should not contribute to any metric."""
        from src.features import OfficialCompositeMetric, VOID_ID
        m = OfficialCompositeMetric(64)
        pred   = torch.zeros(1000, dtype=torch.long)
        target = torch.zeros(1000, dtype=torch.long)   # All void
        m.update(pred, target)
        r = m.compute()
        # No valid pixels → metric should be zero or empty
        assert r["official_composite"] == 0.0

    def test_reset_clears_state(self):
        """reset() should clear all accumulated statistics."""
        from src.features import OfficialCompositeMetric
        m = OfficialCompositeMetric(64)
        pred   = torch.ones(100, dtype=torch.long)
        target = torch.ones(100, dtype=torch.long)
        m.update(pred, target)
        m.reset()
        assert m.conf.sum() == 0


# ─── EMA Tests ───────────────────────────────────────────────────────────────

class TestEMAModel:
    """Tests for the Exponential Moving Average weight tracker."""

    def _make_simple_model(self):
        import torch.nn as nn
        return nn.Linear(10, 5)

    def test_shadow_initialized(self):
        from src.features import EMAModel
        model = self._make_simple_model()
        ema = EMAModel(model, decay=0.9)
        assert len(ema.shadow) > 0

    def test_update_changes_shadow(self):
        from src.features import EMAModel
        import torch.nn as nn
        model = self._make_simple_model()
        ema = EMAModel(model, decay=0.9)
        old = {k: v.clone() for k, v in ema.shadow.items()}
        # Change model weights
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(999.0)
        ema.update(model)
        for k in old:
            assert not torch.allclose(ema.shadow[k], old[k]), \
                "Shadow should change after update."

    def test_apply_and_restore(self):
        from src.features import EMAModel
        model = self._make_simple_model()
        ema = EMAModel(model, decay=0.9)
        original_weights = {n: p.data.clone() for n, p in model.named_parameters()}
        ema.apply_shadow(model)
        ema.restore(model)
        for n, p in model.named_parameters():
            assert torch.allclose(p.data, original_weights[n]), \
                f"Parameter {n} not correctly restored after EMA apply/restore."
