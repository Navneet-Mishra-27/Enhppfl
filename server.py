"""
EnhPPFL Flower Server
=====================
Federated learning server implementing the EnhPPFL aggregation strategy:
  - Sparse gradient aggregation with mask cancellation (single-round SecAgg)
  - ECDH public-key distribution
  - Conservative Rényi DP budget tracking (min noise across layers and clients)
  - Communication bandwidth monitoring

Usage:
    # ResNet-18 / CIFAR-10
    python server.py --model-type resnet18 --total-rounds 200

    # MLP / NSL-KDD
    python server.py --model-type mlp --total-rounds 200 --input-dim 122

Authors: Navneet Mishra, Prachet Bhuyan
Affiliation: School of Computer Engineering, KIIT Deemed to be University
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import flwr as fl
import logging

logging.basicConfig(level=logging.INFO)

from flwr.common import (
    FitRes, Parameters, Scalar,
    ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy

from privacy_utils import RenyiDPAccountant
from models import create_model


# ============================================================================
# ENHPPFL AGGREGATION STRATEGY
# ============================================================================

class EnhPPFLStrategy(fl.server.strategy.Strategy):
    """
    Custom Flower strategy for EnhPPFL.

    Responsibilities per round:
      - Distribute ECDH public keys to participating clients.
      - Aggregate sparse, masked client updates (masks cancel in the sum).
      - Track privacy budget (conservative: minimum noise across layers/clients).
      - Monitor cumulative communication savings.
    """

    def __init__(
        self,
        model: nn.Module,
        total_rounds: int = 200,
        min_fit_clients: int = 5,
        min_available_clients: int = 5,
        fraction_fit: float = 0.1,
        fraction_evaluate: float = 0.1,
        privacy_epsilon: float = 1.0,
        privacy_delta: float = 1e-5,
        learning_rate: float = 0.01,
        device: str = 'cpu',
        model_type: str = 'resnet18'
    ):
        self.model      = model.to(device)
        self.device     = device
        self.model_type = model_type
        self.total_rounds          = total_rounds
        self.min_fit_clients       = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_fit          = fraction_fit
        self.fraction_evaluate     = fraction_evaluate
        self.learning_rate         = learning_rate

        self.privacy_accountant = RenyiDPAccountant(
            target_epsilon=privacy_epsilon,
            target_delta=privacy_delta
        )

        self.current_round      = 0
        self.client_public_keys = {}

        self.total_dense_bytes  = 0
        self.total_sparse_bytes = 0
        self.round_bandwidth_stats: List[Dict] = []

        print(f"[Server] EnhPPFL strategy initialised")
        print(f"  Model:   {model_type.upper()}")
        print(f"  Rounds:  {total_rounds}")
        print(f"  Privacy: ε={privacy_epsilon}, δ={privacy_delta}")

    # ------------------------------------------------------------------
    # Flower strategy interface
    # ------------------------------------------------------------------

    def initialize_parameters(
        self,
        client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        print("[Server] Initialising global model parameters")
        initial_params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        return ndarrays_to_parameters(initial_params)

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        self.current_round = server_round
        print(f"\n{'='*70}")
        print(f"[Server] Round {server_round}/{self.total_rounds}")
        print(f"{'='*70}")

        sample_size = max(
            int(self.fraction_fit * client_manager.num_available()),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        print(f"[Server] Selected {len(clients)} clients")

        # peer_public_keys must be Scalar — serialise as bytes.
        config = {
            'round_num':        server_round,
            'total_rounds':     self.total_rounds,
            'peer_public_keys': pickle.dumps(self.client_public_keys)
        }
        fit_ins = fl.common.FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate sparse, masked client updates."""
        if not results:
            print("[Server] WARNING: No results received.")
            return None, {}

        if failures:
            print(f"[Server] {len(failures)} client failure(s) this round")

        print(f"[Server] Aggregating {len(results)} updates...")

        client_updates  = []
        client_metrics  = []

        for _, fit_res in results:
            update = pickle.loads(fit_res.parameters.tensors[0])

            # Cache public key for next round's config
            client_id  = update.pop('_client_id')
            public_key = update.pop('_public_key')
            self.client_public_keys[client_id] = public_key

            client_updates.append(update)
            client_metrics.append(fit_res.metrics)

        # Aggregate and apply
        aggregated_sparse = self._aggregate_sparse_updates(client_updates)
        aggregated_dense  = self._decompress_update(aggregated_sparse)
        self._update_global_model(aggregated_dense)

        # Privacy accounting — use the minimum noise multiplier reported
        # across all participating clients (conservative: worst-case client).
        noise_values  = [m.get('noise_multiplier', 0.0) for m in client_metrics]
        min_noise     = float(np.min(noise_values)) if noise_values else 0.0
        self.privacy_accountant.add_round(min_noise, self.fraction_fit)
        server_epsilon = self.privacy_accountant.get_epsilon()

        # Bandwidth statistics
        round_dense_bytes  = float(np.sum([m.get('dense_bytes',  0) for m in client_metrics]))
        round_sparse_bytes = float(np.sum([m.get('sparse_bytes', 0) for m in client_metrics]))
        round_reduction    = float(np.mean([m.get('communication_reduction', 0) for m in client_metrics]))

        self.total_dense_bytes  += round_dense_bytes
        self.total_sparse_bytes += round_sparse_bytes
        cumulative_reduction = (
            (1.0 - self.total_sparse_bytes / self.total_dense_bytes) * 100.0
            if self.total_dense_bytes > 0 else 0.0
        )

        self.round_bandwidth_stats.append({
            'round':                server_round,
            'dense_bytes':          round_dense_bytes,
            'sparse_bytes':         round_sparse_bytes,
            'reduction':            round_reduction,
            'cumulative_reduction': cumulative_reduction
        })

        avg_sparsity = float(np.mean([m.get('sparsity', 0) for m in client_metrics]))

        print(f"[Server] Round complete")
        print(f"  Privacy (server):    ε={server_epsilon:.3f}")
        print(f"  Min noise (clients): σ_min={min_noise:.3f}")
        print(f"  Sparsity:            {avg_sparsity:.2%}")
        print(f"  Bandwidth (round):   {round_sparse_bytes:,.0f} / {round_dense_bytes:,.0f} bytes "
              f"(reduction {round_reduction:.1f}%)")
        print(f"  Bandwidth (cumul.):  reduction {cumulative_reduction:.1f}%")

        updated_params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        metrics = {
            'epsilon':              server_epsilon,
            'min_client_noise':     min_noise,
            'avg_sparsity':         avg_sparsity,
            'round_reduction':      round_reduction,
            'cumulative_reduction': cumulative_reduction,
            'num_clients':          len(results)
        }
        return ndarrays_to_parameters(updated_params), metrics

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        if server_round % 10 != 0:
            return []

        sample_size = max(
            int(self.fraction_evaluate * client_manager.num_available()),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        config   = {'round_num': server_round}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        return [(client, eval_ins) for client in clients]

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        total_samples  = sum(r.num_examples for _, r in results)
        weighted_loss  = sum(r.loss * r.num_examples for _, r in results) / total_samples
        avg_accuracy   = float(np.mean([r.metrics.get('accuracy', 0) for _, r in results]))

        if self.model_type == 'mlp':
            avg_f1 = float(np.mean([r.metrics.get('f1_score', 0) for _, r in results]))
            print(f"\n[Server] Round {server_round} evaluation")
            print(f"  Loss:         {weighted_loss:.4f}")
            print(f"  Accuracy:     {avg_accuracy:.4f}")
            print(f"  F1 (weighted): {avg_f1:.4f}")
            return weighted_loss, {
                'accuracy':               avg_accuracy,
                'f1_score':               avg_f1,
                'num_clients_evaluated':  len(results)
            }
        else:
            print(f"\n[Server] Round {server_round} evaluation")
            print(f"  Loss:     {weighted_loss:.4f}")
            print(f"  Accuracy: {avg_accuracy:.4f}")
            return weighted_loss, {
                'accuracy':              avg_accuracy,
                'num_clients_evaluated': len(results)
            }

    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        return None  # server-side evaluation not used

    # ------------------------------------------------------------------
    # Aggregation helpers
    # ------------------------------------------------------------------

    def _aggregate_sparse_updates(self, client_updates: List[Dict]) -> Dict:
        """Sum sparse updates across clients, then divide by client count."""
        num_clients = len(client_updates)
        aggregated  = {}

        layer_names = list(client_updates[0].keys())

        for layer_name in layer_names:
            all_indices = []
            all_values  = []
            shape       = None

            for update in client_updates:
                if layer_name in update:
                    all_indices.append(torch.from_numpy(update[layer_name]['indices']))
                    all_values.append(torch.from_numpy(update[layer_name]['values']))
                    shape = update[layer_name]['shape']

            if not all_indices:
                continue

            combined_indices = torch.cat(all_indices)
            combined_values  = torch.cat(all_values)

            unique_indices    = torch.unique(combined_indices)
            aggregated_values = torch.zeros(len(unique_indices), dtype=torch.float32)

            for idx, unique_idx in enumerate(unique_indices):
                mask = combined_indices == unique_idx
                aggregated_values[idx] = combined_values[mask].sum()

            aggregated_values /= num_clients

            aggregated[layer_name] = {
                'indices': unique_indices.numpy(),
                'values':  aggregated_values.numpy(),
                'shape':   shape
            }

        return aggregated

    def _decompress_update(self, sparse_update: Dict) -> Dict[str, torch.Tensor]:
        """Decompress sparse update dict to dense gradient tensors."""
        dense_update = {}
        for layer_name, sparse_data in sparse_update.items():
            indices    = torch.from_numpy(sparse_data['indices'])
            values     = torch.from_numpy(sparse_data['values'])
            shape      = sparse_data['shape']
            total_size = int(np.prod(shape))
            dense_flat = torch.zeros(total_size, dtype=torch.float32)
            dense_flat[indices] = values
            dense_update[layer_name] = dense_flat.reshape(shape)
        return dense_update

    def _update_global_model(self, aggregated_gradients: Dict[str, torch.Tensor]):
        """Apply gradient update: θ ← θ − η·∇L"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_gradients:
                    param.data -= self.learning_rate * aggregated_gradients[name].to(self.device)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def print_final_summary(self):
        final_epsilon        = self.privacy_accountant.get_epsilon()
        cumulative_reduction = (
            (1.0 - self.total_sparse_bytes / self.total_dense_bytes) * 100.0
            if self.total_dense_bytes > 0 else 0.0
        )

        print("\n" + "=" * 70)
        print("EnhPPFL — Training Complete")
        print("=" * 70)

        print(f"\n[Privacy]")
        print(f"  Final ε = {final_epsilon:.3f}  (target ≤ 1.0)")
        status = "PASS" if final_epsilon <= 1.0 else "FAIL"
        print(f"  Status:  {status}")

        print(f"\n[Communication]")
        print(f"  Total dense bytes:  {self.total_dense_bytes:,.0f}")
        print(f"  Total sparse bytes: {self.total_sparse_bytes:,.0f}")
        print(f"  Cumulative reduction: {cumulative_reduction:.1f}%  (target ≥ 73%)")
        status = "PASS" if cumulative_reduction >= 73.0 else "FAIL"
        print(f"  Status: {status}")

        print(f"\n[Utility] See evaluation logs above.")
        print(f"  Target (NSL-KDD F1):     0.919")
        print(f"  Target (CIFAR-10 Acc):   0.886")
        print("=" * 70)


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EnhPPFL Flower Server')
    parser.add_argument('--server-address',        type=str,   default='0.0.0.0:8080')
    parser.add_argument('--model-type',            type=str,   default='resnet18',
                        choices=['resnet18', 'mlp'])
    parser.add_argument('--total-rounds',          type=int,   default=200)
    parser.add_argument('--min-clients',           type=int,   default=5)
    parser.add_argument('--min-available-clients', type=int,   default=5)
    parser.add_argument('--fraction-fit',          type=float, default=0.1)
    parser.add_argument('--fraction-evaluate',     type=float, default=0.1)
    parser.add_argument('--privacy-epsilon',       type=float, default=1.0)
    parser.add_argument('--privacy-delta',         type=float, default=1e-5)
    parser.add_argument('--learning-rate',         type=float, default=0.01)
    parser.add_argument('--device',                type=str,   default='cpu')
    parser.add_argument('--input-dim',             type=int,   default=122,
                        help='MLP input dimension for NSL-KDD after one-hot encoding (~122)')

    args = parser.parse_args()

    print("=" * 70)
    print("EnhPPFL Federated Learning Server")
    print("=" * 70)
    print(f"\n  Model:   {args.model_type.upper()}")
    print(f"  Server:  {args.server_address}")
    print(f"  Rounds:  {args.total_rounds}")
    print(f"  Privacy: ε={args.privacy_epsilon}, δ={args.privacy_delta}")

    print(f"\n[Server] Creating {args.model_type.upper()} model...")
    if args.model_type == 'resnet18':
        model = create_model('resnet18', num_classes=10)
    else:
        # Use --input-dim 122 (or whatever the client data resolves to)
        model = create_model('mlp', input_dim=args.input_dim, num_classes=2)
        print(f"  MLP input_dim={args.input_dim}")

    strategy = EnhPPFLStrategy(
        model=model,
        total_rounds=args.total_rounds,
        min_fit_clients=args.min_clients,
        min_available_clients=args.min_available_clients,
        fraction_fit=args.fraction_fit,
        fraction_evaluate=args.fraction_evaluate,
        privacy_epsilon=args.privacy_epsilon,
        privacy_delta=args.privacy_delta,
        learning_rate=args.learning_rate,
        device=args.device,
        model_type=args.model_type
    )

    print(f"\n[Server] Starting on {args.server_address} — waiting for clients...\n")
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.total_rounds),
        strategy=strategy
    )

    strategy.print_final_summary()

    output_path = f'enhppfl_{args.model_type}_final.pt'
    torch.save(model.state_dict(), output_path)
    print(f"\n[Server] Model saved to {output_path}")


if __name__ == '__main__':
    main()
