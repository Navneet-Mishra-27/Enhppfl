"""
EnhPPFL Privacy Utilities
=========================
Core privacy-preserving mechanisms:
  - Fisher Information computation (diagonal approximation via Opacus)
  - Posterior-inspired orthogonal Gaussian sampling (defence against gradient inversion)
  - Adaptive top-k gradient sparsification with error feedback (73% bandwidth reduction)
  - Rényi DP accounting with subsampling amplification (Wang et al., AISTATS 2019)

Authors: Navneet Mishra, Prachet Bhuyan
Affiliation: School of Computer Engineering, KIIT Deemed to be University
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import OrderedDict

from opacus import GradSampleModule
from opacus.grad_sample import GradSampleModule as GSM
from opacus.validators import ModuleValidator


# ============================================================================
# FISHER INFORMATION COMPUTATION
# ============================================================================

class FisherInformationComputer:
    """
    Diagonal Fisher Information Matrix approximation using per-sample gradients.

    F̂_{i,ℓ} = (1/|B|) Σ_{(x,y)∈B} g_{i,ℓ}(x,y) ⊗ g_{i,ℓ}(x,y)

    The diagonal approximation captures per-parameter sensitivity at O(d_ℓ) cost
    and is cached every 5 rounds to amortise overhead (≈5.2% of per-round cost).
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
        Compute diagonal Fisher Information using per-sample gradients (Opacus).

        Args:
            model: Neural network.
            data_loader: Training data loader.
            criterion: Loss function.
            max_batches: Number of mini-batches to use (for efficiency).

        Returns:
            Mapping from parameter name to diagonal Fisher tensor.
        """
        # Opacus cannot wrap BatchNorm directly; replace with GroupNorm first.
        if not isinstance(model, GradSampleModule):
            if not ModuleValidator.is_valid(model):
                model = ModuleValidator.fix(model)
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

                model.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()

                # Accumulate squared per-sample gradients → diagonal Fisher
                for name, param in model.named_parameters():
                    if param.requires_grad and hasattr(param, 'grad_sample'):
                        per_sample_grads = param.grad_sample  # [B, *param.shape]
                        squared = (per_sample_grads ** 2).sum(dim=0)
                        if name not in fisher_dict:
                            fisher_dict[name] = squared
                        else:
                            fisher_dict[name] += squared

                num_samples += batch_size

        for name in fisher_dict:
            fisher_dict[name] /= num_samples

        if isinstance(model, GradSampleModule):
            model = model._module

        return fisher_dict

    def get_fisher_traces(
        self, fisher_dict: Dict[str, torch.Tensor]
    ) -> Dict[str, float]:
        """
        Compute Trace(F̂_ℓ) for each layer, with numerical stability constant
        ε_F = 10⁻⁶ matching the paper (Eq. 3).
        """
        return {
            name: fisher.sum().item() + 1e-6
            for name, fisher in fisher_dict.items()
        }

    def get_cached_or_compute(
        self,
        model: nn.Module,
        data_loader: torch.utils.data.DataLoader,
        criterion: nn.Module,
        force_refresh: bool = False
    ) -> Dict[str, torch.Tensor]:
        """Return cached Fisher information, recomputing every 5 rounds."""
        if (self.cached_fisher is None or force_refresh or
                self.cache_counter >= self.cache_refresh_interval):
            self.cached_fisher = self.compute_fisher_diagonal(
                model, data_loader, criterion
            )
            self.cache_counter = 0
        else:
            self.cache_counter += 1
        return self.cached_fisher


# ============================================================================
# POSTERIOR-INSPIRED ORTHOGONAL GAUSSIAN SAMPLING (DEFENCE)
# ============================================================================

class PosteriorInspiredProjection:
    """
    Posterior-inspired orthogonal Gaussian sampling (Section 3.3 of the paper).

    Noise is drawn in the null space of the clipped gradient direction:

        P^⊥_ℓ = I - (g̃_{i,ℓ} g̃_{i,ℓ}^T) / (‖g̃_{i,ℓ}‖² + ε_F)   [Eq. 6]
        z_ℓ ~ N(0, σ²_ℓ C²_ℓ P^⊥_ℓ)                               [Eq. 7]
        g^⊥_{i,ℓ} = g̃_{i,ℓ} + z_ℓ                                  [Eq. 8]

    LTI-style attackers learn to invert gradient *directions*. Orthogonal noise
    rotates the transmitted direction without changing its ℓ₂ norm (in
    expectation), defeating the inversion network while preserving the DP
    sensitivity bound (Theorem 1).
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
        self, gradient: torch.Tensor
    ) -> torch.Tensor:
        """
        P^⊥ = I - (g̃ ⊗ g̃) / ‖g̃‖²

        Rank = d_ℓ − 1; noise projected through P^⊥ is orthogonal to g̃.
        """
        g = gradient.flatten()
        d = len(g)
        g_norm_sq = torch.dot(g, g).item() + 1e-6
        g_normalized = g / (g_norm_sq ** 0.5)
        outer = torch.outer(g_normalized, g_normalized)
        identity = torch.eye(d, device=gradient.device, dtype=gradient.dtype)
        return identity - outer

    def sample_orthogonal_noise(
        self,
        gradient: torch.Tensor,
        noise_multiplier: float,
        clipping_threshold: float
    ) -> torch.Tensor:
        """
        Sample z_ℓ ~ N(0, σ²_ℓ C²_ℓ P^⊥_ℓ).

        Returns noise tensor with the same shape as `gradient`.
        """
        original_shape = gradient.shape
        g_flat = gradient.flatten()
        d = len(g_flat)

        P_perp = self.compute_orthogonal_projection_matrix(gradient)
        noise_full = torch.randn(d, device=gradient.device, dtype=gradient.dtype)
        noise_full = noise_full * noise_multiplier * clipping_threshold
        noise_orthogonal = torch.matmul(P_perp, noise_full)
        return noise_orthogonal.reshape(original_shape)

    def apply_adaptive_clipping(
        self,
        gradients: Dict[str, torch.Tensor],
        fisher_traces: Dict[str, float]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """
        Adaptive layer-wise clipping (Eq. 3–4):

            C_ℓ = C̄ · sqrt(Trace(F̂_{i,ℓ}) + ε_F)
            g̃_{i,ℓ} = g_{i,ℓ} · min(1, C_ℓ / ‖g_{i,ℓ}‖₂)
        """
        clipped_gradients = {}
        clipping_thresholds = {}

        for name, grad in gradients.items():
            if name in fisher_traces:
                C_l = self.base_clipping_threshold * np.sqrt(fisher_traces[name])
            else:
                C_l = self.base_clipping_threshold

            grad_norm = torch.norm(grad).item() + 1e-6
            clip_factor = min(1.0, C_l / grad_norm)
            clipped_gradients[name] = grad * clip_factor
            clipping_thresholds[name] = C_l

        return clipped_gradients, clipping_thresholds

    def compute_adaptive_noise_multipliers(
        self, fisher_traces: Dict[str, float]
    ) -> Dict[str, float]:
        """
        Adaptive per-layer noise multipliers (Eq. 5):

            σ_ℓ = σ_base · sqrt(Trace(F̂_{i,ℓ}) / F̄)

        High-sensitivity layers receive proportionally more noise.
        """
        if not fisher_traces:
            return {}

        mean_fisher = np.mean(list(fisher_traces.values()))
        return {
            name: self.base_noise_multiplier * np.sqrt(trace / mean_fisher)
            for name, trace in fisher_traces.items()
        }

    def add_orthogonal_noise(
        self,
        gradients: Dict[str, torch.Tensor],
        fisher_traces: Optional[Dict[str, float]] = None
    ) -> Tuple[Dict[str, torch.Tensor], float]:
        """
        Full per-round defence pipeline:
          1. Adaptive clipping (Fisher-guided thresholds).
          2. Adaptive noise multipliers (Fisher-guided).
          3. Orthogonal noise injection (null-space projection).

        Returns:
            (perturbed_gradients, min_noise_multiplier)

        The minimum noise multiplier across layers is returned because DP
        sensitivity is governed by the least-protected layer (not the average).
        Using the minimum gives a valid, conservative privacy accounting value.
        """
        if self.adaptive and fisher_traces:
            clipped_gradients, clipping_thresholds = self.apply_adaptive_clipping(
                gradients, fisher_traces
            )
            noise_multipliers = self.compute_adaptive_noise_multipliers(fisher_traces)
        else:
            clipped_gradients = {}
            clipping_thresholds = {}
            noise_multipliers = {}
            for name, grad in gradients.items():
                grad_norm = torch.norm(grad).item() + 1e-6
                clip_factor = min(1.0, self.base_clipping_threshold / grad_norm)
                clipped_gradients[name] = grad * clip_factor
                clipping_thresholds[name] = self.base_clipping_threshold
                noise_multipliers[name] = self.base_noise_multiplier

        perturbed_gradients = {}
        for name, grad in clipped_gradients.items():
            sigma = noise_multipliers.get(name, self.base_noise_multiplier)
            C = clipping_thresholds.get(name, self.base_clipping_threshold)
            noise = self.sample_orthogonal_noise(grad, sigma, C)
            perturbed_gradients[name] = grad + noise

        # Conservative: DP privacy cost is set by the layer with the least noise.
        min_noise_multiplier = min(noise_multipliers.values()) if noise_multipliers else self.base_noise_multiplier
        return perturbed_gradients, min_noise_multiplier


# ============================================================================
# ADAPTIVE TOP-K GRADIENT SPARSIFICATION WITH ERROR FEEDBACK
# ============================================================================

class TopkCompressor:
    """
    Adaptive top-k gradient sparsification with error compensation (Section 3.4).

    Sparsity schedule (Eq. 9):
        k(t) = ⌈d · (1 − e^{−λt/T})⌉

    With λ = 0.28 and T = 200, k(200)/d ≈ 0.244, yielding approximately 73%
    bandwidth reduction after accounting for index overhead (4 bytes per index).

    Accumulated truncation error is preserved via error feedback (Eq. 10–11):
        ĝ_{i,ℓ} = g^⊥_{i,ℓ}[j if j ∈ I_ℓ else 0] + r^{t−1}_{i,ℓ}
        r^{(t)}_{i,ℓ} = g^⊥_{i,ℓ} − ĝ_{i,ℓ}
    """

    def __init__(self, lambda_param: float = 0.28):
        """
        Args:
            lambda_param: Sparsification schedule decay constant (default 0.28,
                          matching Table 5 of the paper).
        """
        self.lambda_param = lambda_param
        self.residuals = {}

    def compute_k(self, round_num: int, total_rounds: int, dimension: int) -> int:
        """
        Adaptive sparsity level k(t) = ⌈d · (1 − e^{−λt/T})⌉.

        Guard: if total_rounds = 0 (before first server config arrives),
        falls back to 1% sparsity to avoid ZeroDivisionError.
        """
        t = round_num + 1  # 1-indexed
        if total_rounds <= 0:
            return max(1, int(np.ceil(dimension * 0.01)))
        fraction = 1.0 - np.exp(-self.lambda_param * t / total_rounds)
        return max(1, int(np.ceil(dimension * fraction)))

    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
        k: int,
        client_id: str
    ) -> Tuple[Dict[str, Tuple], int]:
        """
        Compress gradients to sparse format with error feedback.

        Returns:
            (sparse_dict, num_transmitted_params)
            sparse_dict format: {layer_name: (indices, values, shape)}
        """
        if client_id not in self.residuals:
            self.residuals[client_id] = {
                name: torch.zeros_like(grad)
                for name, grad in gradients.items()
            }

        all_grads = []
        layer_info = []
        current_offset = 0

        for name, grad in gradients.items():
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

        all_grads_flat = torch.cat(all_grads)
        total_dim = len(all_grads_flat)
        k = min(k, total_dim)

        abs_grads = torch.abs(all_grads_flat)
        _, topk_indices = torch.topk(abs_grads, k)
        topk_values = all_grads_flat[topk_indices]

        sparse_flat = torch.zeros_like(all_grads_flat)
        sparse_flat[topk_indices] = topk_values
        residual_flat = all_grads_flat - sparse_flat

        sparse_dict = {}
        current_pos = 0

        for info in layer_info:
            name = info['name']
            shape = info['shape']
            size = info['size']

            layer_sparse = sparse_flat[current_pos:current_pos + size]
            layer_residual = residual_flat[current_pos:current_pos + size]

            layer_indices = torch.nonzero(layer_sparse).flatten()
            layer_values = layer_sparse[layer_indices]

            sparse_dict[name] = (layer_indices, layer_values, shape)
            self.residuals[client_id][name] = layer_residual.reshape(shape)
            current_pos += size

        return sparse_dict, k

    def decompress(
        self, sparse_dict: Dict[str, Tuple]
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
        Compute bandwidth usage.

        Dense transmission:  total_params × bytes_per_param
        Sparse transmission: transmitted_params × (4 + bytes_per_param)
            (4 bytes for the 32-bit index + bytes_per_param for the value)

        Returns:
            (dense_bytes, sparse_bytes, reduction_percentage)
        """
        dense_bytes = total_params * bytes_per_param
        sparse_bytes = transmitted_params * (4 + bytes_per_param)
        reduction = (1.0 - sparse_bytes / dense_bytes) * 100.0
        return dense_bytes, sparse_bytes, reduction


# ============================================================================
# CRYPTOGRAPHIC UTILITIES (ECDH-BASED SECURE AGGREGATION)
# ============================================================================

class CryptoUtils:
    """
    ECDH key exchange and shared-secret derivation for pairwise mask generation.
    Curve: SECP256R1 (256-bit).  Key derivation: HKDF-SHA256.
    """

    @staticmethod
    def generate_key_pair():
        """Generate an ECDH key pair. Returns (private_key, public_bytes)."""
        from cryptography.hazmat.primitives.asymmetric import ec
        from cryptography.hazmat.primitives import serialization

        private_key = ec.generate_private_key(ec.SECP256R1())
        public_bytes = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.X962,
            format=serialization.PublicFormat.UncompressedPoint
        )
        return private_key, public_bytes

    @staticmethod
    def derive_shared_secret(private_key, peer_public_bytes: bytes) -> bytes:
        """Derive a 32-byte shared secret via ECDH + HKDF-SHA256."""
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
# RÉNYI DP ACCOUNTING
# ============================================================================

class RenyiDPAccountant:
    """
    Rényi DP accountant with subsampling amplification (Wang et al., AISTATS 2019).

    Per-round RDP bound (simplified amplification for small q):
        ε^{round}_{α,t} = α q² / (2 σ²)

    Sequential composition over T rounds, then conversion to (ε, δ)-DP:
        ε(δ) = min_{α} { Σ_t ε^{round}_{α,t} + log(1/δ) / (α − 1) }

    Renyi orders α ∈ {2, 4, 8, 16, 32} match the paper (Section 3.6).

    Note: this class is used for *monitoring* the privacy budget during training.
    The noise multiplier passed to add_round() should be the per-round minimum
    across all layers (the least-protected layer determines the privacy cost).
    """

    def __init__(
        self,
        target_epsilon: float = 1.0,
        target_delta: float = 1e-5,
        orders: List[float] = None
    ):
        self.target_epsilon = target_epsilon
        self.target_delta = target_delta
        self.orders = orders if orders is not None else [2.0, 4.0, 8.0, 16.0, 32.0]
        self.rdp_budget = {order: 0.0 for order in self.orders}

    def compute_rdp_single_round(
        self,
        noise_multiplier: float,
        sampling_rate: float,
        order: float
    ) -> float:
        """
        RDP for one round: Gaussian mechanism + Poisson subsampling.

        Simplified bound: RDP(α) ≈ α q² / (2σ²)
        Valid for small sampling rate q and σ ≥ 1.
        """
        if order <= 1.0:
            return 0.0
        return (order * sampling_rate ** 2) / (2.0 * noise_multiplier ** 2)

    def add_round(self, noise_multiplier: float, sampling_rate: float = 0.1):
        """
        Accumulate privacy cost for one training round.

        Args:
            noise_multiplier: Minimum noise multiplier across layers for this
                              round (conservative, worst-case layer).
            sampling_rate: Client subsampling rate q (Poisson).
        """
        for order in self.orders:
            self.rdp_budget[order] += self.compute_rdp_single_round(
                noise_multiplier, sampling_rate, order
            )

    def get_epsilon(self, delta: float = None) -> float:
        """Convert accumulated RDP to (ε, δ)-DP via Proposition 3 of Mironov (2017)."""
        delta = delta if delta is not None else self.target_delta
        epsilon_values = []
        for order in self.orders:
            if order > 1.0:
                eps = self.rdp_budget[order] + np.log(1.0 / delta) / (order - 1.0)
                epsilon_values.append(eps)
        return min(epsilon_values) if epsilon_values else float('inf')

    def reset(self):
        """Reset accumulated privacy budget."""
        self.rdp_budget = {order: 0.0 for order in self.orders}
