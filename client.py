"""
EnhPPFL Flower Client
=====================
Federated learning client implementing the EnhPPFL framework:
  - Adaptive layer-wise Rényi DP with Fisher-guided noise allocation
  - Posterior-inspired orthogonal Gaussian sampling (defence against gradient inversion)
  - Adaptive top-k gradient sparsification with error feedback (73% bandwidth target)
  - ECDH-based pairwise mask generation for single-round secure aggregation

Usage:
    # CIFAR-10
    python client.py --client-id 0 --total-clients 10 --model-type resnet18

    # NSL-KDD
    python client.py --client-id 0 --total-clients 10 --model-type mlp --dataset nslkdd

Authors: Navneet Mishra, Prachet Bhuyan
Affiliation: School of Computer Engineering, KIIT Deemed to be University
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, TensorDataset
import torchvision
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from collections import OrderedDict
import pickle
import sys
import flwr as fl

from privacy_utils import (
    FisherInformationComputer,
    PosteriorInspiredProjection,
    TopkCompressor,
    CryptoUtils,
    RenyiDPAccountant
)
from models import create_model


# ============================================================================
# DATASET LOADING
# ============================================================================

def load_cifar10_data(
    client_id: int,
    total_clients: int,
    batch_size: int = 32,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """Load CIFAR-10 with IID partitioning across clients."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    trainset = torchvision.datasets.CIFAR10(
        root=data_dir, train=True, download=True, transform=transform
    )
    testset = torchvision.datasets.CIFAR10(
        root=data_dir, train=False, download=True, transform=transform
    )

    total_train = len(trainset)
    samples_per_client = total_train // total_clients
    start_idx = client_id * samples_per_client
    end_idx = (
        start_idx + samples_per_client
        if client_id < total_clients - 1
        else total_train
    )

    client_trainset = Subset(trainset, list(range(start_idx, end_idx)))
    train_loader = DataLoader(client_trainset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader  = DataLoader(testset,         batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, test_loader


def load_nslkdd_data(
    client_id: int,
    total_clients: int,
    batch_size: int = 32,
    data_dir: str = './data'
) -> Tuple[DataLoader, DataLoader]:
    """
    Load and preprocess NSL-KDD for federated cyber threat detection.

    Preprocessing pipeline:
      1. Binary label: 0 = normal, 1 = attack.
      2. One-hot encoding of categorical columns (protocol_type, service, flag).
      3. Column alignment between train and test splits.
      4. StandardScaler normalisation (fit on training partition only).

    After one-hot encoding, the feature dimension is approximately 122.

    Download: https://www.unb.ca/cic/datasets/nsl.html
    Place KDDTrain+.txt and KDDTest+.txt in `data_dir`.
    """
    import os

    train_file = os.path.join(data_dir, 'KDDTrain+.txt')
    test_file  = os.path.join(data_dir, 'KDDTest+.txt')

    if not os.path.exists(train_file):
        print(f"NSL-KDD dataset not found at {train_file}")
        print("Download from: https://www.unb.ca/cic/datasets/nsl.html")
        print("Place KDDTrain+.txt and KDDTest+.txt in the data directory.")
        sys.exit(1)

    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
        'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
        'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
        'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
        'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
        'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
        'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
        'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate',
        'dst_host_serror_rate', 'dst_host_srv_serror_rate',
        'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    train_df = pd.read_csv(train_file, names=columns)
    test_df  = pd.read_csv(test_file,  names=columns)

    def preprocess(df):
        df = df.copy()
        df['label'] = df['label'].apply(lambda x: 0 if x == 'normal' else 1)
        df = pd.get_dummies(df, columns=['protocol_type', 'service', 'flag'])
        df = df.drop('difficulty', axis=1, errors='ignore')
        return df

    train_df = preprocess(train_df)
    test_df  = preprocess(test_df)

    # Align columns so train and test have the same feature set
    train_cols = set(train_df.columns) - {'label'}
    test_cols  = set(test_df.columns)  - {'label'}
    all_cols   = sorted(train_cols | test_cols)

    for col in all_cols:
        if col not in train_df.columns:
            train_df[col] = 0
        if col not in test_df.columns:
            test_df[col] = 0

    X_train = train_df[all_cols].values.astype(np.float32)
    y_train = train_df['label'].values.astype(np.int64)
    X_test  = test_df[all_cols].values.astype(np.float32)
    y_test  = test_df['label'].values.astype(np.int64)

    from sklearn.preprocessing import StandardScaler
    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # IID partition
    total_train        = len(X_train)
    samples_per_client = total_train // total_clients
    start_idx          = client_id * samples_per_client
    end_idx            = (
        start_idx + samples_per_client
        if client_id < total_clients - 1
        else total_train
    )

    train_dataset = TensorDataset(
        torch.from_numpy(X_train[start_idx:end_idx]),
        torch.from_numpy(y_train[start_idx:end_idx])
    )
    test_dataset = TensorDataset(
        torch.from_numpy(X_test),
        torch.from_numpy(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(test_dataset,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader


# ============================================================================
# ENHPPFL FLOWER CLIENT
# ============================================================================

class EnhPPFLClient(fl.client.NumPyClient):
    """
    Flower NumPyClient implementing the full EnhPPFL per-round pipeline:

        1. Local gradient computation (1 epoch SGD).
        2. Fisher information computation (cached every 5 rounds).
        3. Adaptive clipping + orthogonal Gaussian noise injection.
        4. Adaptive top-k sparsification with error feedback.
        5. ECDH pairwise mask application (single-round SecAgg).
    """

    def __init__(
        self,
        client_id: str,
        model: nn.Module,
        train_loader: DataLoader,
        test_loader: DataLoader,
        device: str = 'cpu',
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        base_noise_multiplier: float = 2.0,
        base_clipping: float = 1.0,
        lambda_sparsification: float = 0.28,
        learning_rate: float = 0.01,
        local_epochs: int = 1,
        model_type: str = 'resnet18'
    ):
        self.client_id  = client_id
        self.model      = model.to(device)
        self.train_loader = train_loader
        self.test_loader  = test_loader
        self.device     = device
        self.model_type = model_type

        self.fisher_computer     = FisherInformationComputer(device)
        self.orthogonal_projection = PosteriorInspiredProjection(
            base_noise_multiplier=base_noise_multiplier,
            base_clipping_threshold=base_clipping,
            adaptive=True
        )
        self.compressor = TopkCompressor(lambda_param=lambda_sparsification)
        self.privacy_accountant = RenyiDPAccountant(
            target_epsilon=privacy_epsilon,
            target_delta=privacy_delta
        )

        self.criterion  = nn.CrossEntropyLoss()
        self.optimizer  = optim.SGD(
            self.model.parameters(),
            lr=learning_rate, momentum=0.9, weight_decay=5e-4
        )
        self.local_epochs = local_epochs

        self.crypto      = CryptoUtils()
        self.private_key = None
        self.public_key  = None
        self.peer_public_keys: Dict = {}

        self.current_round = 0
        self.total_rounds  = 0

        print(f"[Client {client_id}] Initialised ({model_type} on {device})")
        print(f"  Privacy:  ε={privacy_epsilon}, δ={privacy_delta}")
        print(f"  Noise:    σ_base={base_noise_multiplier}, C̄={base_clipping}")
        print(f"  Sparsity: λ={lambda_sparsification}")
        print(f"  Samples:  {len(train_loader.dataset)}")

    # ------------------------------------------------------------------
    # Flower interface
    # ------------------------------------------------------------------

    def get_parameters(self, config: Dict) -> List[np.ndarray]:
        return [val.cpu().numpy() for val in self.model.state_dict().values()]

    def set_parameters(self, parameters: List[np.ndarray]):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict  = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)

    def fit(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[List[np.ndarray], int, Dict]:
        """
        Execute one federated round locally and return a private sparse update.
        """
        self.set_parameters(parameters)
        self.current_round = config.get('round_num', 0)
        self.total_rounds  = config.get('total_rounds', 200)

        # Deserialise peer public keys (Flower config only allows Scalar values;
        # the dict is sent as pickle-serialised bytes).
        raw_keys = config.get('peer_public_keys', b'{}')
        if isinstance(raw_keys, bytes) and raw_keys:
            self.peer_public_keys = pickle.loads(raw_keys)
        else:
            self.peer_public_keys = raw_keys if isinstance(raw_keys, dict) else {}

        self.private_key, self.public_key = self.crypto.generate_key_pair()

        # Snapshot initial parameters to compute the pseudo-gradient Δθ = θ_0 − θ_t
        initial_params = {
            name: param.clone().detach()
            for name, param in self.model.named_parameters()
        }

        # Local training
        self.model.train()
        num_samples = 0
        epoch_loss  = 0.0

        for _ in range(self.local_epochs):
            for inputs, targets in self.train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                loss = self.criterion(self.model(inputs), targets)
                loss.backward()
                self.optimizer.step()
                num_samples += len(inputs)
                epoch_loss  += loss.item()

        avg_loss = epoch_loss / (len(self.train_loader) * self.local_epochs)
        print(f"\n[Client {self.client_id}] Round {self.current_round} — "
              f"samples={num_samples}, loss={avg_loss:.4f}")

        # Pseudo-gradients: Δθ = θ_before − θ_after
        gradients = {
            name: initial_params[name] - param.data
            for name, param in self.model.named_parameters()
        }

        # Fisher information (cached every 5 rounds)
        fisher_dict   = self.fisher_computer.get_cached_or_compute(
            self.model, self.train_loader, self.criterion
        )
        fisher_traces = self.fisher_computer.get_fisher_traces(fisher_dict)

        # Orthogonal noise injection (returns min noise for conservative DP tracking)
        perturbed_gradients, min_noise = self.orthogonal_projection.add_orthogonal_noise(
            gradients, fisher_traces
        )
        print(f"  Noise: min σ_ℓ={min_noise:.3f}")

        # Privacy accounting — use min noise (worst-case layer)
        sampling_rate = 0.1  # matches fraction_fit = 0.1 in server config
        self.privacy_accountant.add_round(min_noise, sampling_rate)
        current_epsilon = self.privacy_accountant.get_epsilon()
        print(f"  Privacy: ε={current_epsilon:.3f}")

        # Top-k sparsification
        total_dim = sum(g.numel() for g in perturbed_gradients.values())
        k         = self.compressor.compute_k(
            self.current_round, self.total_rounds, total_dim
        )
        sparse_gradients, actual_k = self.compressor.compress(
            perturbed_gradients, k, self.client_id
        )

        dense_bytes, sparse_bytes, reduction = self.compressor.compute_bandwidth_savings(
            total_dim, actual_k
        )
        print(f"  Compression: {actual_k}/{total_dim} params, reduction={reduction:.1f}%")

        # Pairwise cryptographic masking
        masked_sparse     = self._apply_secagg_masks(sparse_gradients)
        serialized_update = self._serialize_sparse_update(masked_sparse)

        metrics = {
            'epsilon':              current_epsilon,
            'noise_multiplier':     min_noise,       # minimum across layers
            'sparsity':             actual_k / total_dim,
            'num_params':           actual_k,
            'communication_reduction': reduction,
            'dense_bytes':          dense_bytes,
            'sparse_bytes':         sparse_bytes,
            'training_loss':        avg_loss,
        }

        return [serialized_update], num_samples, metrics

    def evaluate(
        self,
        parameters: List[np.ndarray],
        config: Dict
    ) -> Tuple[float, int, Dict]:
        """Evaluate model on local test set."""
        self.set_parameters(parameters)
        self.model.eval()

        loss    = 0.0
        correct = 0
        total   = 0
        all_preds   = []
        all_targets = []

        with torch.no_grad():
            for inputs, targets in self.test_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss   += self.criterion(outputs, targets).item()
                _, predicted = torch.max(outputs, 1)
                total   += targets.size(0)
                correct += (predicted == targets).sum().item()
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())

        accuracy = correct / total
        avg_loss = loss / len(self.test_loader)

        if self.model_type == 'mlp':
            from sklearn.metrics import f1_score
            # Weighted F1 matches the evaluation protocol in the paper.
            f1 = f1_score(all_targets, all_preds, average='weighted')
            print(f"[Client {self.client_id}] Eval: loss={avg_loss:.4f}, "
                  f"acc={accuracy:.4f}, F1(weighted)={f1:.4f}")
            return avg_loss, total, {'accuracy': accuracy, 'f1_score': f1}
        else:
            print(f"[Client {self.client_id}] Eval: loss={avg_loss:.4f}, "
                  f"acc={accuracy:.4f}")
            return avg_loss, total, {'accuracy': accuracy}

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _apply_secagg_masks(self, sparse_gradients: Dict) -> Dict:
        """
        Apply pairwise cancelling ECDH masks (single-round SecAgg).

        Client i adds mask_{ij} if i < j, subtracts if i > j.
        Masks cancel exactly upon summation at the server.
        """
        masked_sparse = {}
        for name, (indices, values, shape) in sparse_gradients.items():
            masked_values = values.clone()
            for peer_id, peer_public_key in self.peer_public_keys.items():
                if peer_id == self.client_id:
                    continue
                shared_secret = self.crypto.derive_shared_secret(
                    self.private_key, peer_public_key
                )
                if len(masked_values) > 0:
                    mask_seed = shared_secret + str(self.current_round).encode()
                    np.random.seed(int.from_bytes(mask_seed[:4], byteorder='big'))
                    mask = torch.from_numpy(
                        np.random.randn(len(masked_values)).astype(np.float32)
                    ).to(values.device)
                    if self.client_id < peer_id:
                        masked_values += mask
                    else:
                        masked_values -= mask
            masked_sparse[name] = (indices, masked_values, shape)
        return masked_sparse

    def _serialize_sparse_update(self, sparse_dict: Dict) -> np.ndarray:
        """Serialise sparse update for Flower transmission."""
        serializable = {}
        for name, (indices, values, shape) in sparse_dict.items():
            serializable[name] = {
                'indices': indices.cpu().numpy(),
                'values':  values.cpu().numpy(),
                'shape':   shape
            }
        serializable['_public_key'] = self.public_key
        serializable['_client_id']  = self.client_id
        return pickle.dumps(serializable)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EnhPPFL Flower Client')
    parser.add_argument('--client-id',        type=int,   required=True)
    parser.add_argument('--total-clients',    type=int,   default=10)
    parser.add_argument('--server-address',   type=str,   default='localhost:8080')
    parser.add_argument('--model-type',       type=str,   default='resnet18',
                        choices=['resnet18', 'mlp'])
    parser.add_argument('--dataset',          type=str,   default='cifar10',
                        choices=['cifar10', 'nslkdd'])
    parser.add_argument('--device',           type=str,   default='cpu')
    parser.add_argument('--privacy-epsilon',  type=float, default=1.0)
    parser.add_argument('--privacy-delta',    type=float, default=1e-5)
    parser.add_argument('--base-noise',       type=float, default=2.0)
    parser.add_argument('--base-clipping',    type=float, default=1.0)
    parser.add_argument('--lambda-sparse',    type=float, default=0.28,
                        help='Sparsification schedule decay constant (default 0.28)')
    parser.add_argument('--learning-rate',    type=float, default=0.01)
    parser.add_argument('--local-epochs',     type=int,   default=1)
    parser.add_argument('--batch-size',       type=int,   default=32)
    parser.add_argument('--data-dir',         type=str,   default='./data')

    args = parser.parse_args()

    print("=" * 70)
    print(f"EnhPPFL Client {args.client_id}")
    print("=" * 70)

    # Load data
    print(f"\nLoading {args.dataset.upper()} data...")
    if args.dataset == 'cifar10':
        train_loader, test_loader = load_cifar10_data(
            args.client_id, args.total_clients, args.batch_size, args.data_dir
        )
    else:
        train_loader, test_loader = load_nslkdd_data(
            args.client_id, args.total_clients, args.batch_size, args.data_dir
        )

    # Build model — for MLP, infer input_dim from the actual data
    print(f"Creating {args.model_type.upper()} model...")
    if args.model_type == 'resnet18':
        model = create_model('resnet18', num_classes=10)
    else:
        sample_batch = next(iter(train_loader))
        input_dim    = sample_batch[0].shape[1]
        model        = create_model('mlp', input_dim=input_dim, num_classes=2)
        print(f"  MLP input_dim={input_dim} (derived from data)")

    client = EnhPPFLClient(
        client_id=str(args.client_id),
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=args.device,
        privacy_epsilon=args.privacy_epsilon,
        privacy_delta=args.privacy_delta,
        base_noise_multiplier=args.base_noise,
        base_clipping=args.base_clipping,
        lambda_sparsification=args.lambda_sparse,
        learning_rate=args.learning_rate,
        local_epochs=args.local_epochs,
        model_type=args.model_type
    )

    print(f"\nConnecting to server at {args.server_address}...")
    fl.client.start_numpy_client(
        server_address=args.server_address,
        client=client
    )


if __name__ == '__main__':
    main()
