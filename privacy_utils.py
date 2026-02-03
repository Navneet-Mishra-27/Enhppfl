"""
EnhPPFL Privacy Utilities - Thesis Defense Version
===================================================
Core privacy-preserving mechanisms with quantitative guarantees:
- Fisher Information computation with Opacus
- Posterior-Inspired Orthogonal Gaussian Sampling (Defense mechanism)
- Adaptive Top-k Sparse Compression (73% reduction target)
- Rényi Differential Privacy Accounting

Author: Navneet Mishra
Supervisor: Prof. (Dr.) Prachet Bhuyan
Date: November 30, 2025
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict
import warnings

from opacus import GradSampleModule
from opacus.grad_sample import GradSampleModule as GSM


# ============================================================================
# FISHER INFORMATION COMPUTATION
# ============================================================================

class FisherInformationComputer:
    """
    Computes diagonal Fisher Information Matrix using Opacus.
    Critical for adaptive layer-wise privacy budgeting.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.cached_fisher = None
        self.cache_counter = 0
        self.cache_refresh_interval = 5
    
    def compute_fisher_diagonal(
        self, 
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        max_batches: int = 5
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diagonal Fisher Information using per-sample gradients.
        
        F_ii = E[(∂log p(y|x;θ_i) / ∂θ_i)²]
        
        Args:
            model: Neural network
            data_loader: Training data
            criterion: Loss function
            max_batches: Maximum batches to use (for efficiency)
        
        Returns:
            Dictionary mapping parameter names to diagonal Fisher tensors
        """
        # Wrap model for per-sample gradients
        if not isinstance(model, GradSampleModule):
            model = GradSampleModule(model)
        
        model.train()
        model.to(self.device)
        
        fisher_dict = {}
        num_samples = 0
        
        with torch.enable_grad():
            for batch_idx, (inputs, targets) in enumerate(data_loader):
                if batch_idx >= max_batches:
                    break
                
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                batch_size = inputs.size(0)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Backward pass populates .grad_sample
                model.zero_grad()
                loss.backward()
                
                # Accumulate squared per-sample gradients
                for name, param in model.named_parameters():
                    if param.requires_grad and hasattr(param, 'grad_sample'):
                        # grad_sample: [batch_size, *param.shape]
                        per_sample_grads = param.grad_sample
                        squared_grads = (per_sample_grads ** 2).sum(dim=0)
                        
                        if name not in fisher_dict:
                            fisher_dict[name] = squared_grads
                        else:
                            fisher_dict[name] += squared_grads
                
                num_samples += batch_size
        
        # Average over samples
        for name in fisher_dict:
            fisher_dict[name] /= num_samples
        
        # Remove wrapper
        if isinstance(model, GradSampleModule):
            model = model._module
        
        return fisher_dict
    
    def get_fisher_traces(self, fisher_dict: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compute trace (sum of diagonal) for each layer."""
        return {name: fisher.sum().item() + 1e-8 for name, fisher in fisher_dict.items()}
    
    def get_cached_or_compute(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        force_refresh: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Get cached Fisher or compute if expired."""
        if self.cached_fisher is None or force_refresh or \
           self.cache_counter >= self.cache_refresh_interval:
            self.cached_fisher = self.compute_fisher_diagonal(model, data_loader, criterion)
            self.cache_counter = 0
        else:
            self.cache_counter += 1
        
        return self.cached_fisher


# ============================================================================
# ORTHOGONAL PROJECTION (DEFENSE MECHANISM)
# ============================================================================

class PosteriorInspiredProjection:
    """
    Posterior-Inspired Orthogonal Gaussian Sampling.
    
    This is the CORE DEFENSE against gradient inversion attacks.
    Projects DP noise into the null space of the gradient to prevent
    attackers from matching gradient patterns.
    
    Mathematical Formulation:
    1. Compute orthogonal projection: P^⊥ = I - (gg^T)/(||g||²)
    2. Sample noise: z ~ N(0, σ²C²I)
    3. Project noise: z_perp = P^⊥ @ z
    4. Perturbed gradient: g' = g + z_perp
    
    This ensures: <g, z_perp> ≈ 0 (orthogonality)
    """
    
    def __init__(
        self,
        base_noise_multiplier: float = 2.0,
        base_clipping_threshold: float = 1.0,
        adaptive: bool = True
    ):
        self.base_noise_multiplier = base_noise_multiplier
        self.base_clipping_threshold = base_clipping_threshold
        self.adaptive = adaptive
    
    def compute_orthogonal_projection_matrix(
        self, 
        gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute projection matrix onto orthogonal complement.
        
        P^⊥ = I - (g⊗g) / ||g||²
        
        This is the key to defense: noise in the null space doesn't
        leak information about the original gradient direction.
        """
        g = gradient.flatten()
        d = len(g)
        
        # Normalize for numerical stability
        g_norm = torch.norm(g).item() + 1e-8
        g_normalized = g / g_norm
        
        # Outer product: g⊗g / ||g||²
        outer = torch.outer(g_normalized, g_normalized)
        
        # P^⊥ = I - outer
        identity = torch.eye(d, device=gradient.device, dtype=gradient.dtype)
        P_perp = identity - outer
        
        return P_perp
    
    def sample_orthogonal_noise(
        self,
        gradient: torch.Tensor,
        noise_multiplier: float,
        clipping_threshold: float
    ) -> torch.Tensor:
        """
        Sample Gaussian noise in orthogonal subspace.
        
        This is what prevents gradient inversion attacks:
        - Noise magnitude: σ·C (provides DP)
        - Noise direction: orthogonal to g (breaks inversion)
        
        Returns:
            Orthogonal noise tensor (same shape as gradient)
        """
        original_shape = gradient.shape
        g_flat = gradient.flatten()
        d = len(g_flat)
        
        # Compute orthogonal projection matrix
        P_perp = self.compute_orthogonal_projection_matrix(gradient)
        
        # Sample noise in full space
        noise_full = torch.randn(d, device=gradient.device, dtype=gradient.dtype)
        noise_full *= noise_multiplier * clipping_threshold
        
        # Project to orthogonal subspace
        noise_orthogonal = torch.matmul(P_perp, noise_full)
        
        return noise_orthogonal.reshape(original_shape)
    
    def apply_adaptive_clipping(
        self,
        gradients: Dict[str, torch.Tensor],
        fisher_traces: Dict[str, float]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Adaptive layer-wise gradient clipping based on Fisher information.
        
        C_l = C̄ · √(Trace(F_l) + ε)
        
        High Fisher info → more important layer → higher clipping threshold
        """
        clipped_gradients = {}
        clipping_thresholds = {}
        
        for name, grad in gradients.items():
            if name in fisher_traces:
                # Adaptive threshold based on Fisher information
                fisher_scale = np.sqrt(fisher_traces[name])
                C_l = self.base_clipping_threshold * fisher_scale
            else:
                C_l = self.base_clipping_threshold
            
            # Clip gradient: g̃ = g · min(1, C_l / ||g||)
            grad_norm = torch.norm(grad).item() + 1e-8
            clip_factor = min(1.0, C_l / grad_norm)
            
            clipped_gradients[name] = grad * clip_factor
            clipping_thresholds[name] = C_l
        
        return clipped_gradients, clipping_thresholds
    
    def compute_adaptive_noise_multipliers(
        self,
        fisher_traces: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adaptive noise multipliers based on Fisher information.
        
        σ_l = σ_base · √(Trace(F_l) / F̄)
        
        High-sensitivity layers get proportionally MORE noise.
        """
        if not fisher_traces:
            return {}
        
        mean_fisher = np.mean(list(fisher_traces.values()))
        noise_multipliers = {}
        
        for name, trace in fisher_traces.items():
            # Scale noise by Fisher information (square root for stability)
            noise_multipliers[name] = self.base_noise_multiplier * np.sqrt(trace / mean_fisher)
        
        return noise_multipliers
    
    def add_orthogonal_noise(
        self,
        gradients: Dict[str, torch.Tensor],
        fisher_traces: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Apply complete defense mechanism:
        1. Adaptive clipping (based on Fisher)
        2. Adaptive noise computation (based on Fisher)
        3. Orthogonal projection (defense core)
        
        Returns:
            (perturbed_gradients, average_noise_multiplier)
        """
        # Step 1: Adaptive clipping
        if self.adaptive and fisher_traces:
            clipped_gradients, clipping_thresholds = self.apply_adaptive_clipping(
                gradients, fisher_traces
            )
            noise_multipliers = self.compute_adaptive_noise_multipliers(fisher_traces)
        else:
            # Uniform clipping and noise
            clipped_gradients = {}
            clipping_thresholds = {}
            noise_multipliers = {}
            
            for name, grad in gradients.items():
                grad_norm = torch.norm(grad).item() + 1e-8
                clip_factor = min(1.0, self.base_clipping_threshold / grad_norm)
                clipped_gradients[name] = grad * clip_factor
                clipping_thresholds[name] = self.base_clipping_threshold
                noise_multipliers[name] = self.base_noise_multiplier
        
        # Step 2: Add orthogonal noise (DEFENSE MECHANISM)
        perturbed_gradients = {}
        
        for name, grad in clipped_gradients.items():
            sigma = noise_multipliers[name]
            C = clipping_thresholds[name]
            
            # Sample and add orthogonal noise
            noise = self.sample_orthogonal_noise(grad, sigma, C)
            perturbed_gradients[name] = grad + noise
        
        # Compute average noise multiplier for privacy accounting
        avg_noise_multiplier = np.mean(list(noise_multipliers.values()))
        
        return perturbed_gradients, avg_noise_multiplier


# ============================================================================
# TOP-K SPARSE COMPRESSION (73% REDUCTION TARGET)
# ============================================================================

class TopkCompressor:
    """
    Adaptive top-k gradient sparsification with error compensation.
    
    Target: 73% communication reduction as stated in thesis.
    
    Algorithm:
    1. Compute adaptive k(t) = ⌈d · (1 - e^(-λt/T))⌉
    2. Select top-k largest gradient components
    3. Maintain residual for error compensation
    4. Transmit only (indices, values) - sparse representation
    """
    
    def __init__(self, lambda_param: float = 0.5):
        self.lambda_param = lambda_param
        self.residuals = {}
    
    def compute_k(self, round_num: int, total_rounds: int, dimension: int) -> int:
        """
        Compute adaptive sparsity level.
        
        k(t) = ⌈d · (1 - e^(-λt/T))⌉
        
        Early rounds: very sparse (low communication)
        Later rounds: denser (better convergence)
        """
        t = round_num + 1  # 1-indexed
        fraction = 1 - np.exp(-self.lambda_param * t / total_rounds)
        k = int(np.ceil(dimension * fraction))
        return max(k, 1)
    
    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
        k: int,
        client_id: str
    ) -> Tuple[Dict[str, Tuple], int]:
        """
        Compress gradients to sparse format.
        
        Returns:
            (sparse_dict, num_transmitted_params)
            
        sparse_dict format: {layer_name: (indices, values, shape)}
        """
        # Initialize residuals
        if client_id not in self.residuals:
            self.residuals[client_id] = {
                name: torch.zeros_like(grad) 
                for name, grad in gradients.items()
            }
        
        # Flatten all gradients with error compensation
        all_grads = []
        layer_info = []
        current_offset = 0
        
        for name, grad in gradients.items():
            # Add previous residual (error feedback)
            compensated = grad + self.residuals[client_id][name]
            flat_grad = compensated.flatten()
            
            all_grads.append(flat_grad)
            layer_info.append({
                'name': name,
                'shape': grad.shape,
                'offset': current_offset,
                'size': len(flat_grad)
            })
            current_offset += len(flat_grad)
        
        # Concatenate
        all_grads_flat = torch.cat(all_grads)
        total_dim = len(all_grads_flat)
        k = min(k, total_dim)
        
        # Top-k selection
        abs_grads = torch.abs(all_grads_flat)
        topk_values, topk_indices = torch.topk(abs_grads, k)
        
        # Keep original values (with sign)
        topk_values = all_grads_flat[topk_indices]
        
        # Create sparse representation
        sparse_flat = torch.zeros_like(all_grads_flat)
        sparse_flat[topk_indices] = topk_values
        
        # Compute residual
        residual_flat = all_grads_flat - sparse_flat
        
        # Decompose back into layers
        sparse_dict = {}
        current_pos = 0
        
        for info in layer_info:
            name = info['name']
            shape = info['shape']
            size = info['size']
            
            # Extract layer's portion
            layer_sparse = sparse_flat[current_pos:current_pos + size]
            layer_residual = residual_flat[current_pos:current_pos + size]
            
            # Sparse representation: (indices, values, shape)
            layer_indices = torch.nonzero(layer_sparse).flatten()
            layer_values = layer_sparse[layer_indices]
            
            sparse_dict[name] = (layer_indices, layer_values, shape)
            
            # Update residual
            self.residuals[client_id][name] = layer_residual.reshape(shape)
            
            current_pos += size
        
        return sparse_dict, k
    
    def decompress(
        self,
        sparse_dict: Dict[str, Tuple]
    ) -> Dict[str, torch.Tensor]:
        """Decompress sparse representation to dense format."""
        dense_dict = {}
        
        for name, (indices, values, shape) in sparse_dict.items():
            total_size = int(np.prod(shape))
            dense_flat = torch.zeros(total_size, device=values.device, dtype=values.dtype)
            dense_flat[indices] = values
            dense_dict[name] = dense_flat.reshape(shape)
        
        return dense_dict
    
    def compute_bandwidth_savings(
        self,
        total_params: int,
        transmitted_params: int,
        bytes_per_param: int = 4
    ) -> Tuple[int, int, float]:
        """
        Compute exact bandwidth usage.
        
        Returns:
            (dense_bytes, sparse_bytes, reduction_percentage)
        """
        # Dense transmission
        dense_bytes = total_params * bytes_per_param
        
        # Sparse transmission: indices (4 bytes) + values (4 bytes)
        sparse_bytes = transmitted_params * (4 + bytes_per_param)
        
        # Reduction percentage
        reduction = (1 - sparse_bytes / dense_bytes) * 100
        
        return dense_bytes, sparse_bytes, reduction


# ============================================================================
# CRYPTOGRAPHIC UTILITIES
# ============================================================================

class CryptoUtils:
    """ECDH-based secure aggregation for coordinate sharing."""
    
    @staticmethod
    def generate_key_pair():
        """Generate ECDH key pair."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization
        
        private_key = ec.generate_private_key(ec.SECP256R1())
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        return private_key, public_bytes
    
    @staticmethod
    def derive_shared_secret(private_key, peer_public_bytes):
        """Derive shared secret from ECDH."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import hashes
        from cryptography.hazmat.primitives.kdf.hkdf import HKDF
        
        peer_public_key = ec.EllipticCurvePublicKey.from_encoded_point(
            ec.SECP256R1(), peer_public_bytes
        )
        
        shared_key = private_key.exchange(ec.ECDH(), peer_public_key)
        
        derived_key = HKDF(
            algorithm=hashes.SHA256(),
            length=32,
            salt=None,
            info=b'enhppfl_secagg'
        ).derive(shared_key)
        
        return derived_key


# ============================================================================
# PRIVACY ACCOUNTING
# ============================================================================

class RenyiDPAccountant:
    """
    Rényi Differential Privacy accountant with moments method.
    Provides tight composition bounds.
    """
    
    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        orders: List[float] = None
    ):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.orders = orders or [2.0, 4.0, 8.0, 16.0, 32.0, 64.0]
        self.rdp_budget = {order: 0.0 for order in self.orders}
    
    def compute_rdp_single_round(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        order: float
    ) -> float:
        """Compute RDP for single round with Gaussian mechanism."""
        if order == 1.0:
            return 0.0
        return (order * sampling_rate) / (2 * noise_multiplier ** 2)
    
    def add_round(self, noise_multiplier: float, sampling_rate: float = 0.1):
        """Add one training round."""
        for order in self.orders:
            self.rdp_budget[order] += self.compute_rdp_single_round(
                noise_multiplier, sampling_rate, order
            )
    
    def get_epsilon(self, delta: float = None) -> float:
        """Convert RDP to (ε, δ)-DP."""
        delta = delta or self.target_delta
        epsilon_values = []
        for order in self.orders:
            if order > 1.0:
                eps = self.rdp_budget[order] + np.log(1.0 / delta) / (order - 1.0)
                epsilon_values.append(eps)
        return min(epsilon_values) if epsilon_values else float('inf')
    
    def reset(self):
        """Reset privacy budget."""
        self.rdp_budget = {order: 0.0 for order in self.orders}
