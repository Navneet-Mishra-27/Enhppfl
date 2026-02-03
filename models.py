"""
EnhPPFL Model Definitions - Thesis Defense Version
===================================================
Neural network architectures used in thesis experiments:
1. ResNet-18 (CIFAR-10) - for image classification experiments
2. 4-Layer MLP (NSL-KDD) - for cyber threat detection (91.9% F1 target)

Author: Navneet Mishra
Supervisor: Prof. (Dr.) Prachet Bhuyan
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

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
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
                    in_planes, self.expansion * planes, kernel_size=1, 
                    stride=stride, bias=False
                ),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet18(nn.Module):
    """
    ResNet-18 adapted for CIFAR-10 (32x32 images).
    
    Architecture:
    - Conv1: 3->64, kernel=3
    - Layer1: 64->64, 2 blocks
    - Layer2: 64->128, 2 blocks
    - Layer3: 128->256, 2 blocks
    - Layer4: 256->512, 2 blocks
    - AvgPool, FC: 512->10
    
    Total parameters: ~11.2M
    """

    def __init__(self, num_classes=10):
        super(ResNet18, self).__init__()
        self.in_planes = 64

        # Initial convolution (adapted for CIFAR-10)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        
        # Residual layers
        self.layer1 = self._make_layer(BasicBlock, 64, 2, stride=1)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)
        
        # Classification head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
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
        out = self.fc(out)
        return out


# ============================================================================
# MLP FOR NSL-KDD CYBER THREAT DETECTION
# ============================================================================

class NSLKDD_MLP(nn.Module):
    """
    4-Layer MLP for NSL-KDD cyber threat detection.
    
    Architecture (as per thesis):
    - Input: 41 features (NSL-KDD dataset)
    - Hidden1: 64 neurons + ReLU + Dropout(0.3)
    - Hidden2: 32 neurons + ReLU + Dropout(0.3)
    - Hidden3: 16 neurons + ReLU + Dropout(0.2)
    - Output: 2 classes (Normal vs Attack)
    
    This is the model that achieves 91.9% F1 score in thesis.
    
    Total parameters: ~3,500
    """

    def __init__(self, input_dim=41, num_classes=2):
        super(NSLKDD_MLP, self).__init__()
        
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
        # Layer 1
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        
        # Layer 2
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        
        # Layer 3
        x = self.fc3(x)
        x = self.bn3(x)
        x = F.relu(x)
        x = self.dropout3(x)
        
        # Output layer
        x = self.fc4(x)
        return x


# ============================================================================
# MODEL FACTORY
# ============================================================================

def create_model(model_type: str, **kwargs):
    """
    Factory function to create models.
    
    Args:
        model_type: 'resnet18' or 'mlp'
        **kwargs: Additional arguments for model initialization
    
    Returns:
        Instantiated model
    """
    if model_type == 'resnet18':
        num_classes = kwargs.get('num_classes', 10)
        return ResNet18(num_classes=num_classes)
    
    elif model_type == 'mlp':
        input_dim = kwargs.get('input_dim', 41)
        num_classes = kwargs.get('num_classes', 2)
        return NSLKDD_MLP(input_dim=input_dim, num_classes=num_classes)
    
    else:
        raise ValueError(f"Unknown model type: {model_type}. Use 'resnet18' or 'mlp'.")


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_summary(model: nn.Module, model_name: str = "Model"):
    """Print detailed model summary."""
    total_params = count_parameters(model)
    
    print(f"\n{'='*70}")
    print(f"{model_name} Summary")
    print(f"{'='*70}")
    print(f"Total trainable parameters: {total_params:,}")
    
    print(f"\nLayer-wise parameters:")
    for name, param in model.named_parameters():
        if param.requires_grad:
            print(f"  {name:40s}: {param.numel():>10,} params")
    
    print(f"{'='*70}\n")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    # Test ResNet-18
    print("Testing ResNet-18 for CIFAR-10:")
    resnet = create_model('resnet18', num_classes=10)
    print_model_summary(resnet, "ResNet-18 (CIFAR-10)")
    
    # Test forward pass
    x_img = torch.randn(4, 3, 32, 32)
    out_img = resnet(x_img)
    print(f"Input shape: {x_img.shape}")
    print(f"Output shape: {out_img.shape}")
    print(f"✓ ResNet-18 forward pass successful\n")
    
    # Test NSL-KDD MLP
    print("Testing MLP for NSL-KDD:")
    mlp = create_model('mlp', input_dim=41, num_classes=2)
    print_model_summary(mlp, "NSL-KDD MLP")
    
    # Test forward pass
    x_tabular = torch.randn(4, 41)
    out_tabular = mlp(x_tabular)
    print(f"Input shape: {x_tabular.shape}")
    print(f"Output shape: {out_tabular.shape}")
    print(f"✓ MLP forward pass successful\n")
