# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Tests for quantization_ready.py core functionality."""

import logging

import pytest

from quant_tools.quantization_ready import (
    _REGISTRY,
    get_quant_model,
    register_quantization_provider,
    register_quantization_providers_from_package,
)

from .fixtures import mock_quantized_models

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def clean_registry():
    """Clean the provider registry before and after tests."""
    original_registry = _REGISTRY.copy()
    _REGISTRY.clear()
    yield
    _REGISTRY.clear()
    _REGISTRY.update(original_registry)


@pytest.fixture
def logger():
    """Provide a test logger."""
    return logging.getLogger("test")


# ============================================================================
# CORE FUNCTIONALITY TESTS
# ============================================================================


class TestRegisterQuantizationProvider:
    """Tests for register_quantization_provider function."""

    def test_register_single_provider(self, clean_registry):
        """Test registering a single provider."""

        def init_func():
            return []

        register_quantization_provider(
            model_type="test_model", initializers=[init_func], prefix_modules=[], priority=100
        )

        assert "test_model" in _REGISTRY
        assert _REGISTRY["test_model"][0].priority == 100
        assert len(_REGISTRY["test_model"][0].initializers) == 1

    def test_register_multiple_providers(self, clean_registry):
        """Test registering providers for different model types."""

        def init_a():
            return []

        def init_b():
            return []

        register_quantization_provider(model_type="model_a", initializers=[init_a], prefix_modules=[], priority=100)

        register_quantization_provider(model_type="model_b", initializers=[init_b], prefix_modules=[], priority=100)

        assert "model_a" in _REGISTRY
        assert "model_b" in _REGISTRY
        assert len(_REGISTRY) == 2

    def test_provider_priority(self, clean_registry):
        """Test that provider priority is correctly stored."""

        def init_func():
            return []

        register_quantization_provider(
            model_type="test_model", initializers=[init_func], prefix_modules=[], priority=200
        )

        assert _REGISTRY["test_model"][0].priority == 200


class TestRegisterQuantizationProvidersFromPackage:
    """Tests for register_quantization_providers_from_package function."""

    def test_auto_register_from_package(self, clean_registry):
        """Test automatic registration from package structure."""
        register_quantization_providers_from_package(
            package=mock_quantized_models, skip=("common", "__pycache__"), default_priority=100
        )

        assert "model_a" in _REGISTRY
        assert "model_b" in _REGISTRY
        assert "model_c" in _REGISTRY
        assert "common" not in _REGISTRY

    def test_skip_parameter(self, clean_registry):
        """Test that skip parameter excludes specified directories."""
        register_quantization_providers_from_package(
            package=mock_quantized_models, skip=("common", "model_b", "__pycache__"), default_priority=100
        )

        assert "model_a" in _REGISTRY
        assert "model_b" not in _REGISTRY
        assert "model_c" in _REGISTRY
        assert "common" not in _REGISTRY

    def test_meta_file_multiple_model_types(self, clean_registry):
        """Test that meta.py MODEL_TYPES registers multiple model types."""
        register_quantization_providers_from_package(
            package=mock_quantized_models, skip=("common", "__pycache__"), default_priority=100
        )

        # model_a has MODEL_TYPES = ("model_a", "model_a_variant") in meta.py
        assert "model_a" in _REGISTRY
        assert "model_a_variant" in _REGISTRY
        # Both should reference the same provider
        assert _REGISTRY["model_a"][0] == _REGISTRY["model_a_variant"][0]


class TestGetQuantModel:
    """Tests for get_quant_model function."""

    def test_get_quant_model_with_registered_provider(self, clean_registry, logger):
        """Test get_quant_model with a registered provider."""
        from tests.quant_tools.fixtures.mock_base_models.model_a import ModelA

        # Register providers
        register_quantization_providers_from_package(
            package=mock_quantized_models, skip=("common", "__pycache__"), default_priority=100
        )

        # Create Model A instance
        model = ModelA()

        # Quantize the model
        quant_model = get_quant_model(model, logger)

        assert quant_model is not None
        # Verify components were quantized
        assert hasattr(quant_model, "attention")
        assert hasattr(quant_model, "decoder")

        assert quant_model.attention.name == "quantized_model_a_attention"
        assert quant_model.decoder.name == "quantized_shared_decoder_model_a"

    def test_get_quant_model_variant_from_meta(self, clean_registry, logger):
        """Test get_quant_model with a registered provider."""
        from tests.quant_tools.fixtures.mock_base_models.model_a import ModelAVariant

        # Register providers, since meta.py defines model_a_variant
        register_quantization_providers_from_package(
            package=mock_quantized_models, skip=("common", "__pycache__"), default_priority=100
        )

        # Create Model A Variant instance
        model = ModelAVariant()

        # Quantize the model
        quant_model = get_quant_model(model, logger)

        assert quant_model is not None
        # Verify components were quantized
        assert hasattr(quant_model, "attention")
        assert hasattr(quant_model, "decoder")

        assert quant_model.attention.name == "quantized_model_a_attention"
        assert quant_model.decoder.name == "quantized_shared_decoder_model_a"

    def test_get_quant_model_no_provider_raises(self, clean_registry, logger):
        """Test that get_quant_model raises when no provider is registered."""
        import torch.nn as nn

        class MockConfig:
            model_type = "nonexistent_model"

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()

        model = MockModel()

        with pytest.raises(NotImplementedError, match="No quantization provider"):
            get_quant_model(model, logger, default_fallback=False)

    def test_get_quant_model_no_model_type_raises(self, clean_registry, logger):
        """Test that get_quant_model raises when model_type cannot be determined."""
        import torch.nn as nn

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()

        model = MockModel()

        with pytest.raises(NotImplementedError, match="Cannot determine model_type"):
            get_quant_model(model, logger)

    def test_get_quant_model_with_fallback(self, clean_registry, logger):
        """Test get_quant_model with default_fallback=True."""
        import fastforward as ff
        import torch.nn as nn

        class MockConfig:
            model_type = "nonexistent_model"

        class MockModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.config = MockConfig()
                self.linear = nn.Linear(10, 10)

        model = MockModel()

        # Should not raise with default_fallback=True
        quant_model = get_quant_model(model, logger, default_fallback=True)

        assert quant_model is not None
        # Check that we quantized the linear layer correctly
        assert isinstance(quant_model.linear, ff.nn.QuantizedLinear)


class TestNameCollisionHandling:
    """Test that same-named quantized modules are correctly isolated by model_type."""

    def test_same_named_modules_different_models(self, clean_registry, logger):
        """Test that model_a and model_b both have SharedDecoder quantized differently."""
        from tests.quant_tools.fixtures.mock_base_models.model_a import ModelA
        from tests.quant_tools.fixtures.mock_base_models.model_b import ModelB

        # Register both providers
        register_quantization_providers_from_package(
            package=mock_quantized_models, skip=("common", "__pycache__"), default_priority=100
        )

        # Create Model A and Model B instances
        model_a = ModelA()
        model_b = ModelB()

        # Quantize model_a - should use model_a's QuantizedSharedDecoder
        quant_a = get_quant_model(model_a, logger)

        assert quant_a is not None
        assert hasattr(quant_a, "decoder")
        # Check it's the model_a version
        assert quant_a.decoder.name == "quantized_shared_decoder_model_a"

        # Quantize model_b - should use model_b's QuantizedSharedDecoder (different impl)
        quant_b = get_quant_model(model_b, logger)
        assert quant_b is not None
        assert hasattr(quant_b, "decoder")
        # Check it's the model_b version
        assert quant_b.decoder.name == "quantized_shared_decoder_model_b"

        # Both models have SharedDecoder but each uses its own quantized version
        # The whitelist mechanism ensures correct isolation by model_type
