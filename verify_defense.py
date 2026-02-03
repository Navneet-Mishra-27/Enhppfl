"""
EnhPPFL Defense Verification - Thesis Defense Version
======================================================
This script verifies the defense mechanism by performing gradient inversion attacks.

THESIS CLAIM: Attack Success Rate ≤ 8%

Methodology:
1. Take real training samples
2. Compute gradients (with and without defense)
3. Perform gradient inversion attack (L-BFGS optimization)
4. Measure reconstruction quality using SSIM
5. Attack succeeds if SSIM > 0.7 (high similarity)

Expected Results:
- Without defense: ~90% attack success (SSIM > 0.7)
- With EnhPPFL defense: ~8% attack success (SSIM < 0.2)

Usage:
    python verify_defense.py --model-type resnet18 --num-samples 50
    python verify_defense.py --model-type mlp --dataset nslkdd --num-samples 50
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from typing import Dict, Tuple, List
import sys

from privacy_utils import (
    FisherInformationComputer,
    PosteriorInspiredProjection
)
from models import create_model


# ============================================================================
# GRADIENT INVERSION ATTACK
# ============================================================================

class GradientInversionAttack:
    """
    Simulates DLG/iDLG/FedLeak gradient inversion attack.
    
    Attack mechanism:
    1. Start with random noise image
    2. Compute its gradient
    3. Minimize ||gradient - target_gradient||²
    4. Use L-BFGS optimizer for fast convergence
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        iterations: int = 300,
        learning_rate: float = 0.1
    ):
        self.model = model
        self.device = device
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.criterion = nn.CrossEntropyLoss()
    
    def invert_gradient(
        self,
        target_gradient: Dict[str, torch.Tensor],
        original_image: torch.Tensor,
        true_label: int,
        input_shape: tuple
    ) -> Tuple[torch.Tensor, float, List[float]]:
        """
        Perform gradient inversion attack.
        
        Args:
            target_gradient: The gradient to invert
            original_image: Original image (for SSIM comparison)
            true_label: True label of the image
            input_shape: Shape of input tensor
        
        Returns:
            (reconstructed_image, final_ssim, loss_history)
        """
        # Initialize dummy input (random noise)
        dummy_input = torch.randn(input_shape, device=self.device, requires_grad=True)
        dummy_label = torch.tensor([true_label], device=self.device)
        
        # Optimizer for dummy input
        optimizer = optim.LBFGS([dummy_input], lr=self.learning_rate)
        
        loss_history = []
        
        for iteration in range(self.iterations):
            def closure():
                optimizer.zero_grad()
                
                # Forward pass on dummy input
                dummy_output = self.model(dummy_input)
                dummy_loss = self.criterion(dummy_output, dummy_label)
                
                # Compute gradient of dummy input
                dummy_gradient = torch.autograd.grad(
                    dummy_loss, self.model.parameters(), create_graph=True
                )
                
                # Compute distance to target gradient
                grad_diff = 0.0
                for dummy_g, target_g in zip(dummy_gradient, target_gradient.values()):
                    grad_diff += ((dummy_g - target_g) ** 2).sum()
                
                grad_diff.backward()
                return grad_diff
            
            loss = optimizer.step(closure)
            loss_history.append(loss.item())
            
            if (iteration + 1) % 50 == 0:
                print(f"    Iteration {iteration + 1}/{self.iterations}, Loss: {loss.item():.6f}")
        
        # Compute SSIM
        reconstructed = dummy_input.detach().cpu().numpy()[0]
        original = original_image.cpu().numpy()
        
        # Normalize to [0, 1] for SSIM
        reconstructed = (reconstructed - reconstructed.min()) / (reconstructed.max() - reconstructed.min() + 1e-8)
        original = (original - original.min()) / (original.max() - original.min() + 1e-8)
        
        # For multi-channel images, compute SSIM per channel
        if reconstructed.shape[0] > 1:
            ssim_score = np.mean([
                ssim(original[i], reconstructed[i], data_range=1.0)
                for i in range(reconstructed.shape[0])
            ])
        else:
            ssim_score = ssim(original[0], reconstructed[0], data_range=1.0)
        
        return dummy_input.detach(), ssim_score, loss_history


# ============================================================================
# DEFENSE VERIFICATION
# ============================================================================

class DefenseVerifier:
    """
    Verifies the defense mechanism by comparing attack success rates.
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: str = 'cpu',
        noise_multiplier: float = 2.0,
        clipping_threshold: float = 1.0
    ):
        self.model = model.to(device)
        self.device = device
        
        # Defense mechanism
        self.defense = PosteriorInspiredProjection(
            base_noise_multiplier=noise_multiplier,
            base_clipping_threshold=clipping_threshold,
            adaptive=False  # For consistency in verification
        )
        
        # Attack simulator
        self.attacker = GradientInversionAttack(model, device)
        
        self.criterion = nn.CrossEntropyLoss()
    
    def compute_gradient(
        self,
        image: torch.Tensor,
        label: torch.Tensor,
        apply_defense: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        Compute gradient for a single sample.
        
        Args:
            image: Input image
            label: True label
            apply_defense: Whether to apply orthogonal projection defense
        
        Returns:
            Dictionary of gradients per parameter
        """
        self.model.zero_grad()
        
        output = self.model(image)
        loss = self.criterion(output, label)
        loss.backward()
        
        # Extract gradients
        gradients = {}
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                gradients[name] = param.grad.data.clone()
        
        # Apply defense if requested
        if apply_defense:
            # Apply orthogonal projection
            perturbed_gradients, _ = self.defense.add_orthogonal_noise(
                gradients, fisher_traces=None
            )
            return perturbed_gradients
        else:
            return gradients
    
    def verify_single_sample(
        self,
        image: torch.Tensor,
        label: int,
        sample_id: int
    ) -> Dict:
        """
        Verify defense on a single sample.
        
        Returns:
            Dictionary with attack results (with and without defense)
        """
        print(f"\n[Sample {sample_id}] Starting verification...")
        
        label_tensor = torch.tensor([label], device=self.device)
        
        # 1. Attack WITHOUT defense (baseline)
        print(f"  [Attack 1] Without defense (baseline):")
        gradient_no_defense = self.compute_gradient(image, label_tensor, apply_defense=False)
        reconstructed_no_defense, ssim_no_defense, _ = self.attacker.invert_gradient(
            gradient_no_defense, image[0], label, image.shape
        )
        print(f"    SSIM: {ssim_no_defense:.4f}")
        
        # 2. Attack WITH defense (EnhPPFL)
        print(f"  [Attack 2] With EnhPPFL defense:")
        gradient_with_defense = self.compute_gradient(image, label_tensor, apply_defense=True)
        reconstructed_with_defense, ssim_with_defense, _ = self.attacker.invert_gradient(
            gradient_with_defense, image[0], label, image.shape
        )
        print(f"    SSIM: {ssim_with_defense:.4f}")
        
        # Determine success (SSIM > 0.7 = attack succeeds)
        success_no_defense = ssim_no_defense > 0.7
        success_with_defense = ssim_with_defense > 0.7
        
        print(f"  [Result] Attack success:")
        print(f"    Without defense: {'✗ SUCCESS' if success_no_defense else '✓ FAILED'} (SSIM={ssim_no_defense:.4f})")
        print(f"    With defense: {'✗ SUCCESS' if success_with_defense else '✓ FAILED'} (SSIM={ssim_with_defense:.4f})")
        
        return {
            'sample_id': sample_id,
            'label': label,
            'ssim_no_defense': ssim_no_defense,
            'ssim_with_defense': ssim_with_defense,
            'attack_success_no_defense': success_no_defense,
            'attack_success_with_defense': success_with_defense,
            'original_image': image[0].cpu().numpy(),
            'reconstructed_no_defense': reconstructed_no_defense[0].cpu().numpy(),
            'reconstructed_with_defense': reconstructed_with_defense[0].cpu().numpy()
        }
    
    def verify_multiple_samples(
        self,
        data_loader: torch.utils.data.DataLoader,
        num_samples: int = 50
    ) -> Dict:
        """
        Verify defense on multiple samples and compute statistics.
        
        Returns:
            Dictionary with aggregate statistics
        """
        print(f"\n{'='*70}")
        print(f"DEFENSE VERIFICATION - Testing {num_samples} samples")
        print(f"{'='*70}")
        
        results = []
        
        for i, (images, labels) in enumerate(data_loader):
            if i >= num_samples:
                break
            
            image = images[0:1].to(self.device)  # Take first image from batch
            label = labels[0].item()
            
            result = self.verify_single_sample(image, label, i)
            results.append(result)
        
        # Compute aggregate statistics
        ssim_no_defense = [r['ssim_no_defense'] for r in results]
        ssim_with_defense = [r['ssim_with_defense'] for r in results]
        success_no_defense = [r['attack_success_no_defense'] for r in results]
        success_with_defense = [r['attack_success_with_defense'] for r in results]
        
        success_rate_no_defense = np.mean(success_no_defense) * 100
        success_rate_with_defense = np.mean(success_with_defense) * 100
        
        print(f"\n{'='*70}")
        print(f"FINAL RESULTS")
        print(f"{'='*70}")
        print(f"\nSamples tested: {len(results)}")
        
        print(f"\n[1] BASELINE (No Defense):")
        print(f"    Average SSIM: {np.mean(ssim_no_defense):.4f}")
        print(f"    Attack Success Rate: {success_rate_no_defense:.1f}%")
        
        print(f"\n[2] EnhPPFL (With Defense):")
        print(f"    Average SSIM: {np.mean(ssim_with_defense):.4f}")
        print(f"    **Attack Success Rate: {success_rate_with_defense:.1f}%**")
        print(f"    Target: ≤ 8%")
        print(f"    Status: {'✓ PASS' if success_rate_with_defense <= 8.0 else '✗ FAIL'}")
        
        print(f"\n[3] Defense Effectiveness:")
        print(f"    Reduction in attack success: {success_rate_no_defense - success_rate_with_defense:.1f}%")
        print(f"    Relative improvement: {(1 - success_rate_with_defense/success_rate_no_defense)*100:.1f}%")
        
        print(f"\n{'='*70}")
        
        return {
            'num_samples': len(results),
            'ssim_no_defense': ssim_no_defense,
            'ssim_with_defense': ssim_with_defense,
            'success_rate_no_defense': success_rate_no_defense,
            'success_rate_with_defense': success_rate_with_defense,
            'results': results
        }


# ============================================================================
# VISUALIZATION
# ============================================================================

def plot_verification_results(stats: Dict, output_file: str = 'defense_verification.png'):
    """Plot verification results."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Show 3 example reconstructions
    for i in range(min(3, len(stats['results']))):
        result = stats['results'][i]
        
        # Original
        axes[0, i].imshow(np.transpose(result['original_image'], (1, 2, 0)))
        axes[0, i].set_title(f"Original (Label: {result['label']})")
        axes[0, i].axis('off')
        
        # Without defense
        recon_no_def = np.transpose(result['reconstructed_no_defense'], (1, 2, 0))
        recon_no_def = (recon_no_def - recon_no_def.min()) / (recon_no_def.max() - recon_no_def.min())
        axes[1, i].imshow(recon_no_def)
        axes[1, i].set_title(f"Attack w/o Defense\nSSIM: {result['ssim_no_defense']:.3f}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n[Visualization] Saved to: {output_file}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EnhPPFL Defense Verification')
    parser.add_argument('--model-type', type=str, default='resnet18',
                       choices=['resnet18', 'mlp'])
    parser.add_argument('--dataset', type=str, default='cifar10',
                       choices=['cifar10', 'nslkdd'])
    parser.add_argument('--num-samples', type=int, default=50,
                       help='Number of samples to test')
    parser.add_argument('--noise-multiplier', type=float, default=2.0)
    parser.add_argument('--clipping-threshold', type=float, default=1.0)
    parser.add_argument('--device', type=str, default='cpu')
    parser.add_argument('--data-dir', type=str, default='./data')
    parser.add_argument('--save-plot', action='store_true')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EnhPPFL Defense Verification - THESIS CLAIM: Attack Success ≤ 8%")
    print("=" * 70)
    
    # Load data
    print(f"\nLoading {args.dataset.upper()} test data...")
    if args.dataset == 'cifar10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])
        testset = torchvision.datasets.CIFAR10(
            root=args.data_dir, train=False, download=True, transform=transform
        )
        test_loader = torch.utils.data.DataLoader(
            testset, batch_size=1, shuffle=True
        )
    else:
        print("ERROR: NSL-KDD defense verification not yet implemented (requires tabular attack)")
        print("Use CIFAR-10 for image-based gradient inversion verification")
        sys.exit(1)
    
    # Create model
    print(f"Creating {args.model_type.upper()} model...")
    if args.model_type == 'resnet18':
        model = create_model('resnet18', num_classes=10)
    else:
        model = create_model('mlp', input_dim=41, num_classes=2)
    
    model.eval()  # Set to eval mode for consistent results
    
    # Create verifier
    verifier = DefenseVerifier(
        model=model,
        device=args.device,
        noise_multiplier=args.noise_multiplier,
        clipping_threshold=args.clipping_threshold
    )
    
    # Run verification
    stats = verifier.verify_multiple_samples(test_loader, args.num_samples)
    
    # Plot results
    if args.save_plot and args.dataset == 'cifar10':
        plot_verification_results(stats, 'defense_verification.png')
    
    print("\n[Verification Complete]")


if __name__ == '__main__':
    main()
