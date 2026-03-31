"""
Unit tests for Polar-coordinate components.

Tests:
- Polar pseudo-query management
- Spherical gradient descent
- Polar qTTT adaptation
- Depth-priority gating
"""

import pytest
import torch
import torch.nn.functional as F

from src.attnres.polar_pseudo_query import (
    PolarPseudoQuery, PolarPseudoQueryManager,
    PolarQueryAdapter, create_pseudo_query_manager
)
from src.qttt.polar_adaptation import (
    SphericalSGD, PolarQueryAdapter as QTTTAdapter,
    PolarQTTT, DepthPriorityController
)
from src.gating.depth_priority import (
    DepthPriorityGatingController, AdaptiveThresholdWithDepthPriority,
    create_depth_priority_controller
)


class TestPolarPseudoQuery:
    """Tests for polar pseudo-query representation."""
    
    def test_polar_decomposition(self):
        """Test that w = r * u(θ)."""
        dim = 64
        pq = PolarPseudoQuery(dim)
        
        w = pq.forward()
        r = pq.r
        u = pq.get_direction()
        
        # Check decomposition
        expected_w = r * u
        assert torch.allclose(w, expected_w, atol=1e-5)
        
        # Check unit direction
        u_norm = torch.norm(u)
        assert torch.allclose(u_norm, torch.tensor(1.0), atol=1e-6)
    
    def test_freeze_magnitude(self):
        """Test magnitude freezing for qTTT."""
        dim = 64
        pq = PolarPseudoQuery(dim)
        
        # Initially unfrozen
        assert pq.r.requires_grad
        
        # Freeze
        pq.freeze_magnitude()
        assert not pq.r.requires_grad
        assert pq._freeze_r
        
        # Unfreeze
        pq.unfreeze_magnitude()
        assert pq.r.requires_grad
        assert not pq._freeze_r
    
    def test_angles_to_unit_vector(self):
        """Test angle-to-vector conversion."""
        dim = 4
        pq = PolarPseudoQuery(dim)
        
        # Test with specific angles
        theta = torch.tensor([0.0, 0.0, 0.0])  # Should point along first axis
        u = pq.angles_to_unit_vector(theta)
        
        # First component should be cos(0) = 1, others near 0
        assert torch.allclose(u[0], torch.tensor(1.0), atol=1e-5)
        assert torch.allclose(u[1:], torch.zeros(dim-1), atol=1e-5)


class TestPolarPseudoQueryManager:
    """Tests for polar pseudo-query manager."""
    
    def test_parameter_count_reduction(self):
        """Test that polar mode reduces trainable parameters in qTTT."""
        num_layers = 8
        dim = 64
        num_blocks = 4
        
        # Cartesian mode
        cart_manager = PolarPseudoQueryManager(
            num_layers, dim, num_blocks, use_polar=False
        )
        cart_params = cart_manager.get_parameter_count()
        
        # Polar mode with qTTT
        polar_manager = PolarPseudoQueryManager(
            num_layers, dim, num_blocks, use_polar=True
        )
        polar_manager.enable_qttt_mode()
        polar_params = polar_manager.get_parameter_count()
        
        # Polar qTTT should have ~50% trainable params
        reduction = 1 - polar_params['trainable'] / cart_params['total']
        assert 0.4 < reduction < 0.6, f"Parameter reduction {reduction} not in expected range"
    
    def test_qttt_mode_toggle(self):
        """Test enabling/disabling qTTT mode."""
        manager = PolarPseudoQueryManager(4, 64, use_polar=True)
        
        # Initially both trainable
        assert manager.magnitudes.requires_grad
        assert manager.directions.requires_grad
        
        # Enable qTTT
        manager.enable_qttt_mode()
        assert not manager.magnitudes.requires_grad
        assert manager.directions.requires_grad
        
        # Disable qTTT
        manager.disable_qttt_mode()
        assert manager.magnitudes.requires_grad
        assert manager.directions.requires_grad
    
    def test_get_pseudo_query(self):
        """Test getting pseudo-query in different modes."""
        manager = PolarPseudoQueryManager(4, 64, use_polar=True)
        
        # Get for layer 0, attention
        w = manager.get_pseudo_query(0, is_mlp=False)
        assert w.shape == (64,)
        
        # Get for layer 0, MLP
        w_mlp = manager.get_pseudo_query(0, is_mlp=True)
        assert w_mlp.shape == (64,)
    
    def test_direction_only_access(self):
        """Test accessing only direction component."""
        manager = PolarPseudoQueryManager(4, 64, use_polar=True)
        
        direction = manager.get_direction_only(0, is_mlp=False)
        
        # Should be unit vector
        assert torch.allclose(torch.norm(direction), torch.tensor(1.0), atol=1e-5)


class TestSphericalSGD:
    """Tests for spherical gradient descent."""
    
    def test_unit_norm_preservation(self):
        """Test that updates preserve unit norm."""
        opt = SphericalSGD(learning_rate=0.1)
        
        direction = torch.tensor([1.0, 0.0, 0.0])
        gradient = torch.tensor([0.0, 1.0, 0.0])
        
        # Multiple updates
        for _ in range(10):
            direction = opt.step(direction, gradient)
            norm = torch.norm(direction)
            assert torch.allclose(norm, torch.tensor(1.0), atol=1e-5)
    
    def test_gradient_following(self):
        """Test that direction moves with gradient."""
        opt = SphericalSGD(learning_rate=0.5)
        
        # Start at north pole
        direction = torch.tensor([0.0, 0.0, 1.0])
        # Gradient points in x direction
        gradient = torch.tensor([1.0, 0.0, 0.0])
        
        # After update, z should decrease, x should increase
        new_direction = opt.step(direction, gradient)
        
        assert new_direction[2] < direction[2]  # z decreased
        assert abs(new_direction[0]) > abs(direction[0])  # x increased
    
    def test_tangent_projection(self):
        """Test that radial gradient component is removed."""
        opt = SphericalSGD(learning_rate=0.1)
        
        direction = torch.tensor([1.0, 0.0, 0.0])
        # Gradient with parallel component
        gradient = torch.tensor([1.0, 1.0, 0.0])
        
        # Should still preserve norm
        new_direction = opt.step(direction, gradient)
        assert torch.allclose(torch.norm(new_direction), torch.tensor(1.0), atol=1e-5)


class TestPolarQTTTAdapter:
    """Tests for polar qTTT adapter."""
    
    def test_parameter_selection(self):
        """Test that only direction is adapted when configured."""
        from src.qttt.polar_adaptation import PolarQTTTConfig
        
        config = PolarQTTTConfig(adapt_magnitude=False, adapt_direction=True)
        
        magnitude = torch.tensor(2.0)
        direction = torch.tensor([1.0, 0.0, 0.0])
        
        adapter = QTTTAdapter(magnitude, direction, config)
        
        assert not adapter.r_adapt.requires_grad  # Frozen
        assert adapter.u_adapt.requires_grad     # Trainable
    
    def test_spherical_update(self):
        """Test spherical optimization updates."""
        from src.qttt.polar_adaptation import PolarQTTTConfig
        
        config = PolarQTTTConfig(
            adapt_magnitude=False,
            use_spherical_sgd=True,
            learning_rate=0.1
        )
        
        magnitude = torch.tensor(1.0)
        direction = torch.tensor([1.0, 0.0, 0.0])
        
        adapter = QTTTAdapter(magnitude, direction, config)
        
        # Create dummy loss
        query = adapter.get_query()
        loss = (query - torch.tensor([0.0, 1.0, 0.0])).pow(2).sum()
        
        # Update
        adapter.update(loss)
        
        # Direction should change but remain unit norm
        new_direction = adapter.get_direction()
        assert torch.allclose(torch.norm(new_direction), torch.tensor(1.0), atol=1e-5)


class TestPolarQTTT:
    """Tests for polar qTTT module."""
    
    def test_config_defaults(self):
        """Test default configuration."""
        from src.qttt.polar_adaptation import PolarQTTTConfig
        
        config = PolarQTTTConfig()
        
        assert config.adapt_magnitude == False
        assert config.adapt_direction == True
        assert config.use_turboquant == True
        assert config.turboquant_bits == 4
    
    def test_effective_cost_calculation(self):
        """Test cost calculation with TurboQuant."""
        from src.qttt.polar_adaptation import PolarQTTTConfig
        
        config = PolarQTTTConfig(use_turboquant=True, num_steps=16)
        qttt = PolarQTTT(config, hidden_dim=512, num_heads=8)
        
        cost_with_turbo = qttt.compute_effective_cost(
            batch_size=1, seq_len=1000, use_turboquant=True
        )
        cost_without = qttt.compute_effective_cost(
            batch_size=1, seq_len=1000, use_turboquant=False
        )
        
        # With TurboQuant should be cheaper
        assert cost_with_turbo['turboquant_discount'] == 8.0
        assert cost_with_turbo['total_flops'] < cost_without['total_flops']


class TestDepthPriorityController:
    """Tests for depth-priority gating controller."""
    
    def test_strict_depth_priority(self):
        """Test that depth is strictly prioritized under TurboQuant."""
        controller = DepthPriorityController(
            max_qttt_steps=32,
            think_tokens_if_forced=0,
            turboquant_enabled=True
        )
        
        # When gating active
        qttt_steps, think_tokens = controller.allocate(
            gating_active=True, budget_constraint='constrained'
        )
        
        assert qttt_steps == 32
        assert think_tokens == 0  # Strict depth priority
    
    def test_cost_factor(self):
        """Test cost factor with/without TurboQuant."""
        controller_with = DepthPriorityController(turboquant_enabled=True)
        controller_without = DepthPriorityController(turboquant_enabled=False)
        
        assert controller_with.depth_cost_factor == 1/8
        assert controller_without.depth_cost_factor == 1.0
        
        assert controller_with.flop_equivalence_multiplier == 16
        assert controller_without.flop_equivalence_multiplier == 2


class TestDepthPriorityGating:
    """Tests for depth-priority gating controller."""
    
    def test_decision_structure(self):
        """Test that decisions include depth/width allocation."""
        from src.gating.threshold import TargetRateThreshold
        
        threshold = TargetRateThreshold(target_rate=0.3)
        controller = DepthPriorityGatingController(
            threshold_calibrator=threshold,
            max_qttt_steps=32,
            turboquant_enabled=True
        )
        
        # High loss should trigger adaptation
        should_adapt, qttt_steps, think_tokens, threshold_val = controller.decide(
            reconstruction_loss=5.0
        )
        
        assert should_adapt == True
        assert qttt_steps > 0
        # With TurboQuant, think tokens should be 0
        assert think_tokens == 0
    
    def test_allocation_report(self):
        """Test allocation reporting."""
        from src.gating.threshold import TargetRateThreshold
        
        threshold = TargetRateThreshold(target_rate=0.3)
        controller = DepthPriorityGatingController(
            threshold_calibrator=threshold,
            turboquant_enabled=True
        )
        
        # Make some decisions
        for loss in [1.0, 3.0, 5.0, 2.0, 4.0]:
            controller.decide(loss)
        
        report = controller.get_allocation_report()
        
        assert 'adaptation_rate' in report
        assert 'depth_priority_ratio' in report
        assert 'turboquant_savings' in report
        assert report['turboquant_savings'] == 8.0
    
    def test_policy_comparison(self):
        """Test policy comparison shows TurboQuant advantage."""
        from src.gating.threshold import TargetRateThreshold
        
        threshold = TargetRateThreshold(target_rate=0.3)
        controller = DepthPriorityGatingController(
            threshold_calibrator=threshold,
            turboquant_enabled=True
        )
        
        comparison = controller.get_policy_comparison()
        
        assert 'standard_policy' in comparison
        assert 'turboquant_policy' in comparison
        # TurboQuant should show higher equivalence multiplier
        assert '8x' in comparison['turboquant_policy']['savings_vs_standard'] or \
               '8.0' in str(comparison['turboquant_policy']['savings_vs_standard'])


class TestAdaptiveThreshold:
    """Tests for adaptive threshold with depth adjustment."""
    
    def test_depth_adjustment(self):
        """Test that depth steps adjust based on adaptation rate."""
        threshold = AdaptiveThresholdWithDepthPriority(
            target_rate=0.3,
            max_depth_steps=32
        )
        
        # Simulate high adaptation rate
        for _ in range(60):
            threshold.update(5.0)  # High loss -> always adapt
            threshold.should_adapt(5.0)
        
        # Should recommend fewer steps
        recommended = threshold.get_recommended_depth_steps()
        assert recommended < 32
        
        # Reset and simulate low adaptation rate
        threshold = AdaptiveThresholdWithDepthPriority(
            target_rate=0.3,
            max_depth_steps=32
        )
        for _ in range(60):
            threshold.update(1.0)  # Low loss -> rarely adapt
            threshold.should_adapt(1.0)
        
        recommended = threshold.get_recommended_depth_steps()
        # With low rate, could stay at default or increase


class TestIntegration:
    """Integration tests for polar components."""
    
    def test_end_to_end_polar_qttt(self):
        """Test complete polar qTTT workflow."""
        from src.qttt.polar_adaptation import PolarQTTTConfig
        from src.qttt.adaptation import KVCache
        
        # Setup
        config = PolarQTTTConfig(num_steps=4, learning_rate=0.01)
        qttt = PolarQTTT(config, hidden_dim=256, num_heads=8)
        
        # Create dummy data
        magnitude = torch.tensor(1.5)
        direction = F.normalize(torch.randn(256), dim=0)
        
        # Create dummy KV cache
        keys = torch.randn(1, 8, 100, 32)
        values = torch.randn(1, 8, 100, 32)
        kv_cache = KVCache(keys, values)
        
        # Adapt
        seq_positions = torch.tensor([50, 60, 70, 80])
        adapted_direction, loss_history = qttt.adapt_pseudo_query(
            magnitude, direction, kv_cache, seq_positions
        )
        
        # Verify
        assert adapted_direction.shape == direction.shape
        assert torch.allclose(torch.norm(adapted_direction), torch.tensor(1.0), atol=1e-4)
        assert len(loss_history) == config.num_steps
    
    def test_factory_functions(self):
        """Test factory function for creating components."""
        # Create manager
        manager = create_pseudo_query_manager(
            num_layers=8, dim=256, use_polar=True, enable_qttt=True
        )
        assert manager._qttt_mode
        
        # Create controller
        controller = create_depth_priority_controller(
            target_rate=0.3,
            turboquant_enabled=True
        )
        assert controller.turboquant_enabled


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
