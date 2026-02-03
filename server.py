"""
EnhPPFL Flower Server - Thesis Defense Version
===============================================
Distributed federated learning server with:
- Sparse gradient aggregation
- ECDH key distribution
- Privacy budget tracking
- Explicit bandwidth monitoring (73% reduction target)

Usage:
    python server.py --model-type resnet18 --total-rounds 200
    python server.py --model-type mlp --total-rounds 100 --dataset nslkdd
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
import pickle
from typing import Dict, List, Tuple, Optional, Union
from collections import OrderedDict
import flwr as fl
from flwr.common import (
    FitRes, Parameters, Scalar, ndarrays_to_parameters, parameters_to_ndarrays
)
from flwr.server.client_proxy import ClientProxy

from privacy_utils import RenyiDPAccountant
from models import create_model


# ============================================================================
# ENHPPFL STRATEGY
# ============================================================================

class EnhPPFLStrategy(fl.server.strategy.Strategy):
    """
    Custom Flower strategy implementing EnhPPFL secure aggregation.
    
    Key responsibilities:
    - Distribute ECDH public keys
    - Aggregate sparse gradients
    - Track privacy budget
    - Monitor communication bandwidth (THESIS METRIC)
    """
    
    def __init__(
        self,
        model: nn.Module,
        total_rounds: int = 100,
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
        self.model = model.to(device)
        self.device = device
        self.model_type = model_type
        self.total_rounds = total_rounds
        self.min_fit_clients = min_fit_clients
        self.min_available_clients = min_available_clients
        self.fraction_fit = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.learning_rate = learning_rate
        
        # Privacy accounting
        self.privacy_accountant = RenyiDPAccountant(
            target_epsilon=privacy_epsilon,
            target_delta=privacy_delta
        )
        
        # Tracking
        self.current_round = 0
        self.client_public_keys = {}
        
        # Bandwidth tracking (THESIS REQUIREMENT)
        self.total_dense_bytes = 0
        self.total_sparse_bytes = 0
        self.round_bandwidth_stats = []
        
        print(f"[Server] EnhPPFL Strategy initialized")
        print(f"  Model: {model_type.upper()}")
        print(f"  Total rounds: {total_rounds}")
        print(f"  Privacy: epsilon={privacy_epsilon}, delta={privacy_delta}")

    
    def initialize_parameters(
        self, 
        client_manager: fl.server.client_manager.ClientManager
    ) -> Optional[Parameters]:
        """Initialize global model parameters."""
        print("[Server] Initializing global model")
        initial_params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        return ndarrays_to_parameters(initial_params)
    
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.FitIns]]:
        """Configure clients for training round."""
        self.current_round = server_round
        print(f"\n{'='*70}")
        print(f"[Server] Round {server_round}/{self.total_rounds}")
        print(f"{'='*70}")
        
        # Sample clients
        sample_size = max(
            int(self.fraction_fit * client_manager.num_available()),
            self.min_fit_clients
        )
        clients = client_manager.sample(
            num_clients=sample_size,
            min_num_clients=self.min_fit_clients
        )
        
        print(f"[Server] Selected {len(clients)} clients")
        
        # Config with peer public keys for SecAgg
        config = {
            'round_num': server_round,
            'total_rounds': self.total_rounds,
            'peer_public_keys': self.client_public_keys
        }
        
        fit_ins = fl.common.FitIns(parameters, config)
        return [(client, fit_ins) for client in clients]
    
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Aggregate sparse, masked client updates.
        
        This is where the 73% communication reduction is verified.
        """
        if not results:
            print("[Server] WARNING: No results!")
            return None, {}
        
        if failures:
            print(f"[Server] {len(failures)} failures")
        
        print(f"[Server] Aggregating {len(results)} updates...")
        
        # Extract updates and metrics
        client_updates = []
        client_metrics = []
        
        for client_proxy, fit_res in results:
            serialized = fit_res.parameters.tensors[0]
            update = pickle.loads(serialized)
            
            # Store public key
            client_id = update['_client_id']
            public_key = update['_public_key']
            self.client_public_keys[client_id] = public_key
            
            del update['_public_key']
            del update['_client_id']
            
            client_updates.append(update)
            client_metrics.append(fit_res.metrics)
        
        # Aggregate sparse updates
        aggregated_sparse = self._aggregate_sparse_updates(client_updates)
        aggregated_dense = self._decompress_update(aggregated_sparse)
        self._update_global_model(aggregated_dense)
        
        # Compute metrics
        avg_epsilon = np.mean([m.get('epsilon', 0) for m in client_metrics])
        avg_noise = np.mean([m.get('noise_multiplier', 0) for m in client_metrics])
        avg_sparsity = np.mean([m.get('sparsity', 0) for m in client_metrics])
        
        # Bandwidth statistics (THESIS REQUIREMENT)
        round_dense_bytes = np.sum([m.get('dense_bytes', 0) for m in client_metrics])
        round_sparse_bytes = np.sum([m.get('sparse_bytes', 0) for m in client_metrics])
        round_reduction = np.mean([m.get('communication_reduction', 0) for m in client_metrics])
        
        self.total_dense_bytes += round_dense_bytes
        self.total_sparse_bytes += round_sparse_bytes
        
        cumulative_reduction = (1 - self.total_sparse_bytes / self.total_dense_bytes) * 100
        
        self.round_bandwidth_stats.append({
            'round': server_round,
            'dense_bytes': round_dense_bytes,
            'sparse_bytes': round_sparse_bytes,
            'reduction': round_reduction,
            'cumulative_reduction': cumulative_reduction
        })
        
        # Update privacy
        self.privacy_accountant.add_round(avg_noise, self.fraction_fit)
        server_epsilon = self.privacy_accountant.get_epsilon()
        
        # Print detailed statistics
        print(f"[Server] Aggregation complete")
        print(f"  Privacy: epsilon={server_epsilon:.3f} (client avg: {avg_epsilon:.3f})")
        print(f"  Sparsity: {avg_sparsity:.2%}")
        print(f"  **BANDWIDTH (This Round):**")
        print(f"    Dense: {round_dense_bytes:,} bytes")
        print(f"    Sparse: {round_sparse_bytes:,} bytes")
        print(f"    Reduction: {round_reduction:.1f}%")
        print(f"  **BANDWIDTH (Cumulative):**")
        print(f"    Total Dense: {self.total_dense_bytes:,} bytes")
        print(f"    Total Sparse: {self.total_sparse_bytes:,} bytes")
        print(f"    **CUMULATIVE REDUCTION: {cumulative_reduction:.1f}%**")
        
        # Return updated parameters
        updated_params = [val.cpu().numpy() for val in self.model.state_dict().values()]
        parameters = ndarrays_to_parameters(updated_params)
        
        metrics = {
            'epsilon': server_epsilon,
            'avg_client_epsilon': avg_epsilon,
            'avg_sparsity': avg_sparsity,
            'round_reduction': round_reduction,
            'cumulative_reduction': cumulative_reduction,
            'num_clients': len(results)
        }
        
        return parameters, metrics
    
    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: fl.server.client_manager.ClientManager
    ) -> List[Tuple[ClientProxy, fl.common.EvaluateIns]]:
        """Configure clients for evaluation."""
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
        
        config = {'round_num': server_round}
        eval_ins = fl.common.EvaluateIns(parameters, config)
        
        return [(client, eval_ins) for client in clients]
    
    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, fl.common.EvaluateRes], BaseException]]
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results."""
        if not results:
            return None, {}
        
        total_samples = sum([r.num_examples for _, r in results])
        weighted_loss = sum([r.loss * r.num_examples for _, r in results]) / total_samples
        
        accuracies = [r.metrics.get('accuracy', 0) for _, r in results]
        avg_accuracy = np.mean(accuracies)
        
        # F1 score for NSL-KDD
        if self.model_type == 'mlp':
            f1_scores = [r.metrics.get('f1_score', 0) for _, r in results]
            avg_f1 = np.mean(f1_scores)
            
            print(f"\n[Server] Round {server_round} Evaluation:")
            print(f"  Loss: {weighted_loss:.4f}")
            print(f"  Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
            print(f"  **F1 Score: {avg_f1:.4f} ({avg_f1*100:.2f}%)** [TARGET: 91.9%]")
            
            metrics = {
                'accuracy': avg_accuracy,
                'f1_score': avg_f1,
                'num_clients_evaluated': len(results)
            }
        else:
            print(f"\n[Server] Round {server_round} Evaluation:")
            print(f"  Loss: {weighted_loss:.4f}")
            print(f"  Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
            
            metrics = {
                'accuracy': avg_accuracy,
                'num_clients_evaluated': len(results)
            }
        
        return weighted_loss, metrics
    
    def evaluate(
        self,
        server_round: int,
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Server-side evaluation (optional)."""
        return None
    
    def _aggregate_sparse_updates(self, client_updates: List[Dict]) -> Dict:
        """Aggregate sparse updates (masks cancel out)."""
        num_clients = len(client_updates)
        aggregated = {}
        
        layer_names = [k for k in client_updates[0].keys()]
        
        for layer_name in layer_names:
            all_indices = []
            all_values = []
            shape = None
            
            for update in client_updates:
                if layer_name in update:
                    indices = update[layer_name]['indices']
                    values = update[layer_name]['values']
                    shape = update[layer_name]['shape']
                    
                    all_indices.append(torch.from_numpy(indices))
                    all_values.append(torch.from_numpy(values))
            
            if not all_indices:
                continue
            
            combined_indices = torch.cat(all_indices)
            combined_values = torch.cat(all_values)
            
            # Aggregate by index
            unique_indices = torch.unique(combined_indices)
            aggregated_values = torch.zeros_like(unique_indices, dtype=torch.float32)
            
            for idx, unique_idx in enumerate(unique_indices):
                mask = combined_indices == unique_idx
                aggregated_values[idx] = combined_values[mask].sum()
            
            # Average across clients
            aggregated_values /= num_clients
            
            aggregated[layer_name] = {
                'indices': unique_indices.numpy(),
                'values': aggregated_values.numpy(),
                'shape': shape
            }
        
        return aggregated
    
    def _decompress_update(self, sparse_update: Dict) -> Dict[str, torch.Tensor]:
        """Decompress sparse update to dense."""
        dense_update = {}
        
        for layer_name, sparse_data in sparse_update.items():
            indices = torch.from_numpy(sparse_data['indices'])
            values = torch.from_numpy(sparse_data['values'])
            shape = sparse_data['shape']
            
            total_size = int(np.prod(shape))
            dense_flat = torch.zeros(total_size, dtype=torch.float32)
            dense_flat[indices] = values
            
            dense_update[layer_name] = dense_flat.reshape(shape)
        
        return dense_update
    
    def _update_global_model(self, aggregated_gradients: Dict[str, torch.Tensor]):
        """Update global model: θ ← θ - η·∇L"""
        with torch.no_grad():
            for name, param in self.model.named_parameters():
                if name in aggregated_gradients:
                    gradient = aggregated_gradients[name].to(self.device)
                    param.data -= self.learning_rate * gradient
    
    def print_final_summary(self):
        """Print final summary of training."""
        print("\n" + "=" * 70)
        print("THESIS DEFENSE SUMMARY")
        print("=" * 70)
        
        final_epsilon = self.privacy_accountant.get_epsilon()
        cumulative_reduction = (1 - self.total_sparse_bytes / self.total_dense_bytes) * 100
        
        print(f"\n[1] PRIVACY GUARANTEE:")
        print(f"    Final Privacy Budget: epsilon = {final_epsilon:.3f}")
        print(f"    Target: epsilon ≤ 1.0")
        print(f"    Status: {'✓ PASS' if final_epsilon <= 1.0 else '✗ FAIL'}")
        
        print(f"\n[2] COMMUNICATION EFFICIENCY:")
        print(f"    Total Dense Bytes: {self.total_dense_bytes:,}")
        print(f"    Total Sparse Bytes: {self.total_sparse_bytes:,}")
        print(f"    **COMMUNICATION REDUCTION: {cumulative_reduction:.1f}%**")
        print(f"    Target: ≥ 73%")
        print(f"    Status: {'✓ PASS' if cumulative_reduction >= 73 else '✗ FAIL'}")
        
        print(f"\n[3] MODEL UTILITY:")
        print(f"    See evaluation metrics above")
        print(f"    Target (NSL-KDD): F1 ≥ 91.9%")
        print(f"    Target (CIFAR-10): Accuracy ≥ 85%")
        
        print("\n" + "=" * 70)


# ============================================================================
# MAIN SERVER STARTUP
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='EnhPPFL Flower Server')
    parser.add_argument('--server-address', type=str, default='0.0.0.0:8080')
    parser.add_argument('--model-type', type=str, default='resnet18',
                       choices=['resnet18', 'mlp'])
    parser.add_argument('--total-rounds', type=int, default=100)
    parser.add_argument('--min-clients', type=int, default=5)
    parser.add_argument('--min-available-clients', type=int, default=5)
    parser.add_argument('--fraction-fit', type=float, default=0.1)
    parser.add_argument('--fraction-evaluate', type=float, default=0.1)
    parser.add_argument('--privacy-epsilon', type=float, default=1.0)
    parser.add_argument('--privacy-delta', type=float, default=1e-5)
    parser.add_argument('--learning-rate', type=float, default=0.01)
    parser.add_argument('--device', type=str, default='cpu')
    
    args = parser.parse_args()
    
    print("=" * 70)
    print("EnhPPFL Federated Learning Server - THESIS DEFENSE")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: {args.model_type.upper()}")
    print(f"  Server: {args.server_address}")
    print(f"  Rounds: {args.total_rounds}")
    print(f"  Privacy: epsilon={args.privacy_epsilon}, delta={args.privacy_delta}")

    
    # Create model
    print(f"\n[Server] Creating {args.model_type.upper()} model...")
    if args.model_type == 'resnet18':
        model = create_model('resnet18', num_classes=10)
    else:  # mlp
        model = create_model('mlp', input_dim=41, num_classes=2)
    
    # Create strategy
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
    
    # Start server
    print(f"\n[Server] Starting on {args.server_address}")
    print("[Server] Waiting for clients...\n")
    
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.total_rounds),
        strategy=strategy
    )
    
    # Print final summary
    strategy.print_final_summary()
    
    # Save model
    print(f"\n[Server] Saving model...")
    torch.save(model.state_dict(), f'enhppfl_{args.model_type}_final.pt')
    print(f"[Server] Model saved to: enhppfl_{args.model_type}_final.pt")


if __name__ == '__main__':
    main()
