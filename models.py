"""
EnhPPFL Model Definitions
=========================
Neural network architectures used in EnhPPFL experiments:
  1. ResNet-18 (CIFAR-10) — image classification, ~11.2M parameters
  2. 4-layer MLP (NSL-KDD) — cyber threat detection, ~10,738 parameters

Input dimension for NSL-KDD is 122 (41 raw features after one-hot encoding of
protocol_type, service, and flag columns).

Authors: Navneet Mishra, Prachet Bhuyan
Affiliation: School of Computer Engineering, KIIT Deemed to be University
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# RESNET-18 FOR CIFAR-10
# ============================================================================

class BasicBlock(nn.Module):
    """Basic residual block for ResNet-18."""
    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes, self.expansion * planes,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32×32 images).

    Architecture:
        Conv1: 3 → 64, kernel=3, stride=1
        Layer1: 64 → 64,  2 blocks, stride=1
        Layer2: 64 → 128, 2 blocks, stride=2
        Layer3: 128 → 256, 2 blocks, stride=2
        Layer4: 256 → 512, 2 blocks, stride=2
        AvgPool (adaptive 1×1), FC 512 → num_classes

    Total trainable parameters: ~11.2M
    """

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(BasicBlock, 64,  2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for s in strides:
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = out.view(out.size(0), -1)
        return self.fc(out)


# ============================================================================
# 4-LAYER MLP FOR NSL-KDD CYBER THREAT DETECTION
# ============================================================================

class NSLKDD_MLP(nn.Module):
    """
    4-layer MLP for binary classification on NSL-KDD.

    Architecture (paper Section 3.9):
        Input:   122 features  (41 raw + one-hot encoding of 3 categorical columns)
        Hidden1: 64  neurons, BatchNorm1d, ReLU, Dropout(0.3)
        Hidden2: 32  neurons, BatchNorm1d, ReLU, Dropout(0.3)
        Hidden3: 16  neurons, BatchNorm1d, ReLU, Dropout(0.2)
        Output:  2   classes (normal vs. attack)

    Trainable parameters with input_dim=122:
        fc1:  122×64 + 64  = 7,872   bn1: 128
        fc2:  64×32  + 32  = 2,080   bn2:  64
        fc3:  32×16  + 16  =   528   bn3:  32
        fc4:  16×2   + 2   =    34
        Total: 10,738 parameters
    """

    def __init__(self, input_dim: int = 122, num_classes: int = 2):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)

        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)

        self.fc3 = nn.Linear(32, 16)
        self.bn3 = nn.BatchNorm1d(16)
        self.dropout3 = nn.Dropout(0.2)

        self.fc4 = nn.Linear(16, num_classes)

    def forward(self, x):
        x = self.dropout1(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout2(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout3(F.relu(self.bn3(self.fc3(x))))
        return self.fc4(x)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str, **kwargs) -> nn.Module:
    """
    Instantiate a model by name.

    Args:
        model_type: 'resnet18' or 'mlp'.
        **kwargs:
            resnet18 — num_classes (default 10)
            mlp      — input_dim (default 122), num_classes (default 2)

    Returns:
        Instantiated nn.Module.
    """
    if model_type == 'resnet18':
        return ResNet18(num_classes=kwargs.get('num_classes', 10))

    if model_type == 'mlp':
        return NSLKDD_MLP(
            input_dim=kwargs.get('input_dim', 122),
            num_classes=kwargs.get('num_classes', 2)
        )

    raise ValueError(f"Unknown model_type '{model_type}'. Choose 'resnet18' or 'mlp'.")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """Print per-layer parameter counts."""
    total = count_parameters(model)
    print(f"\n{'='*60}")
    print(f"{model_name}  —  {total:,} trainable parameters")
    print(f"{'='*60}")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:<40s}: {param.numel():>10,}")
    print(f"{'='*60}\n")


# ============================================================================
# SMOKE TESTS
# ============================================================================

if __name__ == '__main__':
    # ResNet-18
    resnet = create_model('resnet18', num_classes=10)
    print_model_summary(resnet, "ResNet-18 (CIFAR-10)")
    x_img = torch.randn(4, 3, 32, 32)
    out = resnet(x_img)
    assert out.shape == (4, 10), f"Unexpected output shape: {out.shape}"
    print("ResNet-18 forward pass OK\n")

    # NSL-KDD MLP with correct input dim
    mlp = create_model('mlp', input_dim=122, num_classes=2)
    print_model_summary(mlp, "NSL-KDD MLP (input_dim=122)")
    x_tab = torch.randn(4, 122)
    out = mlp(x_tab)
    assert out.shape == (4, 2), f"Unexpected output shape: {out.shape}"
    print("NSL-KDD MLP forward pass OK")
