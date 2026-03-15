"""
EnhPPFL Defence Verification
=============================
Verifies the orthogonal Gaussian sampling defence by measuring gradient
inversion attack success rates with and without the defence mechanism.

Methodology:
  1. Draw a test sample and compute its gradient (clean and defended).
  2. Run a DLG-style L-BFGS gradient inversion attack on each gradient.
  3. Measure reconstruction quality with SSIM.
  4. Attack succeeds if SSIM > 0.7.

Reported claims (paper Table 6, adaptive LTI attack):
  Without defence:  ~91% success rate
  With EnhPPFL:     ~8% success rate

Note: image-based inversion is only meaningful for CIFAR-10 / ResNet-18.
Tabular data (NSL-KDD) requires a different inversion protocol.

Usage:
    python verify_defense.py --model-type resnet18 --num-samples 50
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from typing import Dict, Tuple, List
import sys

from privacy_utils import PosteriorInspiredProjection
from models import create_model


# ============================================================================
# GRADIENT INVERSION ATTACK (DLG-STYLE, L-BFGS)
# ============================================================================

class GradientInversionAttack:
    """
    DLG / iDLG-style gradient inversion attack (Zhu et al., 2019).

    Minimises ‖∇_θ L(x̂, ŷ; θ) − g‖² over a dummy input x̂ using L-BFGS.
    """

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        iterations: int = 300,
        learning_rate: float = 0.1
    ):
        self.model         = model
        self.device        = device
        self.iterations    = iterations
        self.learning_rate = learning_rate
        self.criterion     = nn.CrossEntropyLoss()

    def invert_gradient(
        self,
        target_gradient: Dict[str, torch.Tensor],
        original_image: torch.Tensor,
        true_label: int,
        input_shape: tuple
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """
        Attempt to reconstruct the input from a gradient.

        Args:
            target_gradient: Per-parameter gradient dict to invert.
            original_image:  Ground-truth image (used only to compute SSIM).
            true_label:      Known label (iDLG assumption).
            input_shape:     Shape of the input tensor (batch_size, C, H, W).

        Returns:
            (reconstructed_tensor, ssim_score, loss_history)
        """
        dummy_input = torch.randn(input_shape, device=self.device, requires_grad=True)
        dummy_label = torch.tensor([true_label], device=self.device)
        optimizer   = optim.LBFGS([dummy_input], lr=self.learning_rate)
        loss_history: List[float] = []

        for iteration in range(self.iterations):
            def closure():
                optimizer.zero_grad()
                dummy_output = self.model(dummy_input)
                dummy_loss   = self.criterion(dummy_output, dummy_label)
                dummy_gradient = torch.autograd.grad(
                    dummy_loss, self.model.parameters(), create_graph=True
                )
                grad_diff = sum(
                    ((dg - tg) ** 2).sum()
                    for dg, tg in zip(dummy_gradient, target_gradient.values())
                )
                grad_diff.backward()
                return grad_diff

            loss = optimizer.step(closure)
            loss_history.append(loss.item())

            if (iteration + 1) % 100 == 0:
                print(f"    iter {iteration + 1}/{self.iterations}  loss={loss.item():.6f}")

        # Compute SSIM
        reconstructed = dummy_input.detach().cpu().numpy()[0]
        original      = original_image.cpu().numpy()

        def _norm(arr):
            lo, hi = arr.min(), arr.max()
            return (arr - lo) / (hi - lo + 1e-8)

        reconstructed = _norm(reconstructed)
        original      = _norm(original)

        try:
            from skimage.metrics import structural_similarity as ssim_fn
            if reconstructed.shape[0] > 1:
                ssim_score = float(np.mean([
                    ssim_fn(original[c], reconstructed[c], data_range=1.0)
                    for c in range(reconstructed.shape[0])
                ]))
            else:
                ssim_score = float(ssim_fn(original[0], reconstructed[0], data_range=1.0))
        except ImportError:
            # Fall back to pixel-correlation if scikit-image is unavailable
            ssim_score = float(np.corrcoef(original.flatten(), reconstructed.flatten())[0, 1])
            ssim_score = max(0.0, ssim_score)

        return dummy_input.detach(), ssim_score, loss_history


# ============================================================================
# DEFENCE VERIFIER
# ============================================================================

class DefenceVerifier:
    """Compares attack success rates with and without EnhPPFL defence."""

    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        noise_multiplier: float = 2.0,
        clipping_threshold: float = 1.0
    ):
        self.model   = model.to(device)
        self.device  = device
        self.defence = PosteriorInspiredProjection(
            base_noise_multiplier=noise_multiplier,
            base_clipping_threshold=clipping_threshold,
            adaptive=False   # fixed noise for controlled comparison
        )
        self.attacker  = GradientInversionAttack(model, device)
        self.criterion = nn.CrossEntropyLoss()

    def _compute_gradient(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        apply_defence: bool = False
    ) -> Dict[str, torch.Tensor]:
        self.model.zero_grad()
        loss = self.criterion(self.model(image), label)
        loss.backward()

        gradients = {
            name: param.grad.data.clone()
            for name, param in self.model.named_parameters()
            if param.grad is not None
        }

        if apply_defence:
            perturbed, _ = self.defence.add_orthogonal_noise(gradients)
            return perturbed
        return gradients

    def verify_single_sample(
        self,
        image: torch.Tensor,
        label: int,
        sample_id: int
    ) -> Dict:
        print(f"\n[Sample {sample_id}]")
        label_tensor = torch.tensor([label], device=self.device)

        print(f"  Attack without defence...")
        grad_clean       = self._compute_gradient(image, label_tensor, apply_defence=False)
        _, ssim_clean, _ = self.attacker.invert_gradient(
            grad_clean, image[0], label, image.shape
        )

        print(f"  Attack with EnhPPFL defence...")
        grad_defended       = self._compute_gradient(image, label_tensor, apply_defence=True)
        _, ssim_defended, _ = self.attacker.invert_gradient(
            grad_defended, image[0], label, image.shape
        )

        success_clean    = ssim_clean    > 0.7
        success_defended = ssim_defended > 0.7

        print(f"  Without defence: SSIM={ssim_clean:.4f}    ({'SUCCESS' if success_clean else 'FAIL'})")
        print(f"  With defence:    SSIM={ssim_defended:.4f}  ({'SUCCESS' if success_defended else 'FAIL'})")

        return {
            'sample_id':                  sample_id,
            'label':                      label,
            'ssim_no_defence':            ssim_clean,
            'ssim_with_defence':          ssim_defended,
            'attack_success_no_defence':  success_clean,
            'attack_success_with_defence': success_defended,
        }

    def verify_multiple_samples(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 50
    ) -> Dict:
        print(f"\n{'='*70}")
        print(f"Defence Verification — {num_samples} samples")
        print(f"{'='*70}")

        results = []
        for i, (images, labels) in enumerate(data_loader):
            if i >= num_samples:
                break
            image = images[0:1].to(self.device)
            label = int(labels[0].item())
            results.append(self.verify_single_sample(image, label, i))

        ssim_nd   = [r['ssim_no_defence']   for r in results]
        ssim_wd   = [r['ssim_with_defence']  for r in results]
        success_nd = [r['attack_success_no_defence']   for r in results]
        success_wd = [r['attack_success_with_defence']  for r in results]

        rate_nd = float(np.mean(success_nd)) * 100.0
        rate_wd = float(np.mean(success_wd)) * 100.0

        print(f"\n{'='*70}")
        print(f"Results ({len(results)} samples)")
        print(f"{'='*70}")
        print(f"\nWithout defence:  avg SSIM={np.mean(ssim_nd):.4f},  success rate={rate_nd:.1f}%")
        print(f"With defence:     avg SSIM={np.mean(ssim_wd):.4f},  success rate={rate_wd:.1f}%")
        print(f"\nDefence reduces attack success by {rate_nd - rate_wd:.1f} pp "
              f"({(1 - rate_wd / rate_nd) * 100:.1f}% relative).")

        target_met = rate_wd <= 8.0
        print(f"\nClaim (≤8% success with defence): {'PASS' if target_met else 'FAIL'}")
        print(f"{'='*70}")

        return {
            'num_samples':              len(results),
            'success_rate_no_defence':  rate_nd,
            'success_rate_with_defence': rate_wd,
            'ssim_no_defence':          ssim_nd,
            'ssim_with_defence':        ssim_wd,
            'results':                  results
        }


# ============================================================================
# OPTIONAL VISUALISATION
# ============================================================================

def plot_verification_results(stats: Dict, output_file: str = 'defence_verification.png'):
    """Save a grid of original vs. reconstructed images (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed — skipping plot.")
        return

    n = min(3, len(stats['results']))
    fig, axes = plt.subplots(2, n, figsize=(5 * n, 8))

    for i in range(n):
        r = stats['results'][i]
        orig = np.transpose(r.get('original_image', np.zeros((3, 32, 32))), (1, 2, 0))
        axes[0, i].imshow(np.clip(orig, 0, 1))
        axes[0, i].set_title(f"Original (label {r['label']})")
        axes[0, i].axis('off')

        recon = r.get('reconstructed_no_defence', np.zeros((3, 32, 32)))
        recon = np.transpose(recon, (1, 2, 0))
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
        axes[1, i].imshow(np.clip(recon, 0, 1))
        axes[1, i].set_title(f"Reconstructed (no defence)\nSSIM={r['ssim_no_defence']:.3f}")
        axes[1, i].axis('off')

    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"Plot saved to {output_file}")
    plt.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EnhPPFL Defence Verification')
    parser.add_argument('--model-type',         type=str,   default='resnet18',
                        choices=['resnet18', 'mlp'])
    parser.add_argument('--dataset',            type=str,   default='cifar10',
                        choices=['cifar10'])
    parser.add_argument('--num-samples',        type=int,   default=50)
    parser.add_argument('--noise-multiplier',   type=float, default=2.0)
    parser.add_argument('--clipping-threshold', type=float, default=1.0)
    parser.add_argument('--device',             type=str,   default='cpu')
    parser.add_argument('--data-dir',           type=str,   default='./data')
    parser.add_argument('--save-plot',          action='store_true')
    args = parser.parse_args()

    # Only CIFAR-10 supports image-based inversion verification
    if args.dataset != 'cifar10':
        print("Gradient inversion verification is only implemented for CIFAR-10.")
        sys.exit(1)

    print("=" * 70)
    print("EnhPPFL Defence Verification")
    print("=" * 70)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    testset     = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    test_loader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True)

    if args.model_type == 'resnet18':
        model = create_model('resnet18', num_classes=10)
    else:
        # MLP inversion not meaningful for tabular data — exit cleanly
        print("MLP gradient inversion is not implemented (tabular data requires a "
              "different attack protocol). Use --model-type resnet18.")
        sys.exit(1)

    model.eval()

    verifier = DefenceVerifier(
        model=model,
        device=args.device,
        noise_multiplier=args.noise_multiplier,
        clipping_threshold=args.clipping_threshold
    )

    stats = verifier.verify_multiple_samples(test_loader, args.num_samples)

    if args.save_plot:
        plot_verification_results(stats)


if __name__ == '__main__':
    main()
