"""
EnhPPFL Results Analysis Tool
==============================
Analyze logs from federated learning experiments and generate reports.

Usage:
    python analyze_results.py --log-dir ./logs/enhppfl_20251130_120000
"""

import argparse
import re
import os
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import json


class ExperimentAnalyzer:
    """Analyze EnhPPFL experiment logs."""
    
    def __init__(self, log_dir: str):
        self.log_dir = Path(log_dir)
        self.server_log = self.log_dir / 'server.log'
        self.client_logs = list(self.log_dir.glob('client_*.log'))
        
        if not self.server_log.exists():
            raise FileNotFoundError(f"Server log not found: {self.server_log}")
    
    def parse_server_log(self) -> Dict:
        """Parse server log file."""
        print(f"[Analyzer] Parsing server log: {self.server_log}")
        
        with open(self.server_log, 'r') as f:
            content = f.read()
        
        # Extract round statistics
        rounds = []
        round_pattern = r'\[Server\] Round (\d+)/(\d+)'
        epsilon_pattern = r'Server ε: ([\d.]+)'
        sparsity_pattern = r'Avg sparsity: ([\d.]+)%'
        loss_pattern = r'Loss: ([\d.]+)'
        accuracy_pattern = r'Accuracy: ([\d.]+)'
        
        for match in re.finditer(round_pattern, content):
            round_num = int(match.group(1))
            total_rounds = int(match.group(2))
            
            # Find statistics for this round
            round_start = match.start()
            next_round = content.find(f'Round {round_num + 1}/', round_start)
            if next_round == -1:
                next_round = len(content)
            
            round_content = content[round_start:next_round]
            
            # Extract metrics
            epsilon_match = re.search(epsilon_pattern, round_content)
            sparsity_match = re.search(sparsity_pattern, round_content)
            loss_match = re.search(loss_pattern, round_content)
            accuracy_match = re.search(accuracy_pattern, round_content)
            
            round_data = {
                'round': round_num,
                'epsilon': float(epsilon_match.group(1)) if epsilon_match else None,
                'sparsity': float(sparsity_match.group(1)) if sparsity_match else None,
                'loss': float(loss_match.group(1)) if loss_match else None,
                'accuracy': float(accuracy_match.group(1)) if accuracy_match else None
            }
            
            rounds.append(round_data)
        
        # Extract final privacy budget
        final_privacy_pattern = r'Final Privacy Budget: ε = ([\d.]+)'
        final_privacy_match = re.search(final_privacy_pattern, content)
        final_epsilon = float(final_privacy_match.group(1)) if final_privacy_match else None
        
        return {
            'total_rounds_configured': total_rounds if rounds else 0,
            'rounds_completed': len(rounds),
            'final_epsilon': final_epsilon,
            'rounds': rounds
        }
    
    def parse_client_logs(self) -> List[Dict]:
        """Parse all client log files."""
        print(f"[Analyzer] Parsing {len(self.client_logs)} client logs...")
        
        client_data = []
        
        for client_log in self.client_logs:
            # Extract client ID from filename
            client_id_match = re.search(r'client_(\d+)\.log', client_log.name)
            if not client_id_match:
                continue
            
            client_id = int(client_id_match.group(1))
            
            with open(client_log, 'r') as f:
                content = f.read()
            
            # Count training rounds
            fit_pattern = r'\[Client \d+\] Starting fit for round (\d+)'
            rounds_participated = len(re.findall(fit_pattern, content))
            
            # Extract privacy metrics
            epsilon_pattern = r'Privacy: ε=([\d.]+)'
            epsilon_matches = re.findall(epsilon_pattern, content)
            
            # Extract compression metrics
            compression_pattern = r'Compressing: keeping (\d+)/(\d+) \(([\d.]+)%\)'
            compression_matches = re.findall(compression_pattern, content)
            
            client_data.append({
                'client_id': client_id,
                'rounds_participated': rounds_participated,
                'epsilon_values': [float(e) for e in epsilon_matches],
                'compression_ratios': [float(c[2]) for c in compression_matches]
            })
        
        return client_data
    
    def generate_summary(self) -> Dict:
        """Generate experiment summary."""
        server_data = self.parse_server_log()
        client_data = self.parse_client_logs()
        
        # Compute statistics
        rounds = server_data['rounds']
        
        # Privacy statistics
        epsilon_values = [r['epsilon'] for r in rounds if r['epsilon'] is not None]
        
        # Performance statistics
        loss_values = [r['loss'] for r in rounds if r['loss'] is not None]
        accuracy_values = [r['accuracy'] for r in rounds if r['accuracy'] is not None]
        
        # Compression statistics
        sparsity_values = [r['sparsity'] for r in rounds if r['sparsity'] is not None]
        
        # Client statistics
        rounds_per_client = [c['rounds_participated'] for c in client_data]
        
        summary = {
            'experiment': {
                'log_dir': str(self.log_dir),
                'total_rounds_configured': server_data['total_rounds_configured'],
                'rounds_completed': server_data['rounds_completed'],
                'num_clients': len(client_data)
            },
            'privacy': {
                'final_epsilon': server_data['final_epsilon'],
                'max_epsilon': max(epsilon_values) if epsilon_values else None,
                'min_epsilon': min(epsilon_values) if epsilon_values else None,
                'avg_epsilon': np.mean(epsilon_values) if epsilon_values else None
            },
            'performance': {
                'final_loss': loss_values[-1] if loss_values else None,
                'final_accuracy': accuracy_values[-1] if accuracy_values else None,
                'best_accuracy': max(accuracy_values) if accuracy_values else None,
                'worst_accuracy': min(accuracy_values) if accuracy_values else None,
                'avg_accuracy': np.mean(accuracy_values) if accuracy_values else None
            },
            'compression': {
                'avg_sparsity': np.mean(sparsity_values) if sparsity_values else None,
                'avg_compression_ratio': 100 - np.mean(sparsity_values) if sparsity_values else None
            },
            'clients': {
                'total_clients': len(client_data),
                'avg_rounds_per_client': np.mean(rounds_per_client) if rounds_per_client else None,
                'min_rounds_per_client': min(rounds_per_client) if rounds_per_client else None,
                'max_rounds_per_client': max(rounds_per_client) if rounds_per_client else None
            }
        }
        
        return summary
    
    def print_report(self):
        """Print formatted analysis report."""
        summary = self.generate_summary()
        
        print("\n" + "=" * 70)
        print("EnhPPFL Experiment Analysis Report")
        print("=" * 70)
        
        # Experiment info
        print("\n[Experiment Information]")
        print(f"  Log Directory: {summary['experiment']['log_dir']}")
        print(f"  Total Clients: {summary['experiment']['num_clients']}")
        print(f"  Rounds Completed: {summary['experiment']['rounds_completed']}/{summary['experiment']['total_rounds_configured']}")
        
        # Privacy
        print("\n[Privacy Budget]")
        if summary['privacy']['final_epsilon']:
            print(f"  Final ε: {summary['privacy']['final_epsilon']:.4f}")
        if summary['privacy']['avg_epsilon']:
            print(f"  Average ε: {summary['privacy']['avg_epsilon']:.4f}")
            print(f"  Min ε: {summary['privacy']['min_epsilon']:.4f}")
            print(f"  Max ε: {summary['privacy']['max_epsilon']:.4f}")
        
        # Performance
        print("\n[Model Performance]")
        if summary['performance']['final_accuracy']:
            print(f"  Final Accuracy: {summary['performance']['final_accuracy']:.4f} ({summary['performance']['final_accuracy']*100:.2f}%)")
        if summary['performance']['best_accuracy']:
            print(f"  Best Accuracy: {summary['performance']['best_accuracy']:.4f} ({summary['performance']['best_accuracy']*100:.2f}%)")
            print(f"  Average Accuracy: {summary['performance']['avg_accuracy']:.4f} ({summary['performance']['avg_accuracy']*100:.2f}%)")
        if summary['performance']['final_loss']:
            print(f"  Final Loss: {summary['performance']['final_loss']:.4f}")
        
        # Compression
        print("\n[Communication Efficiency]")
        if summary['compression']['avg_sparsity']:
            print(f"  Average Sparsity: {summary['compression']['avg_sparsity']:.2f}%")
            print(f"  Average Compression: {summary['compression']['avg_compression_ratio']:.2f}% reduction")
        
        # Client participation
        print("\n[Client Participation]")
        print(f"  Average Rounds per Client: {summary['clients']['avg_rounds_per_client']:.1f}")
        print(f"  Min/Max Participation: {summary['clients']['min_rounds_per_client']}/{summary['clients']['max_rounds_per_client']}")
        
        print("\n" + "=" * 70)
    
    def save_json_report(self, output_file: str = None):
        """Save detailed report as JSON."""
        if output_file is None:
            output_file = self.log_dir / 'analysis_report.json'
        
        summary = self.generate_summary()
        server_data = self.parse_server_log()
        client_data = self.parse_client_logs()
        
        full_report = {
            'summary': summary,
            'server_rounds': server_data['rounds'],
            'clients': client_data
        }
        
        with open(output_file, 'w') as f:
            json.dump(full_report, f, indent=2)
        
        print(f"\n[Analyzer] Detailed report saved to: {output_file}")
    
    def plot_training_curves(self, output_file: str = None):
        """Plot training curves (requires matplotlib)."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Analyzer] matplotlib not installed. Skipping plots.")
            return
        
        if output_file is None:
            output_file = self.log_dir / 'training_curves.png'
        
        server_data = self.parse_server_log()
        rounds = server_data['rounds']
        
        if not rounds:
            print("[Analyzer] No round data available for plotting.")
            return
        
        # Extract data
        round_nums = [r['round'] for r in rounds]
        epsilon_values = [r['epsilon'] for r in rounds]
        accuracy_values = [r['accuracy'] for r in rounds if r['accuracy'] is not None]
        accuracy_rounds = [r['round'] for r in rounds if r['accuracy'] is not None]
        sparsity_values = [r['sparsity'] for r in rounds if r['sparsity'] is not None]
        
        # Create subplots
        fig, axes = plt.subplots(3, 1, figsize=(10, 12))
        
        # Privacy budget over time
        axes[0].plot(round_nums, epsilon_values, 'b-', linewidth=2)
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Privacy Budget (ε)')
        axes[0].set_title('Privacy Budget Accumulation')
        axes[0].grid(True, alpha=0.3)
        
        # Accuracy over time
        if accuracy_values:
            axes[1].plot(accuracy_rounds, accuracy_values, 'g-', linewidth=2)
            axes[1].set_xlabel('Round')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_title('Model Accuracy')
            axes[1].grid(True, alpha=0.3)
        
        # Sparsity over time
        if sparsity_values:
            axes[2].plot(round_nums, sparsity_values, 'r-', linewidth=2)
            axes[2].set_xlabel('Round')
            axes[2].set_ylabel('Sparsity (%)')
            axes[2].set_title('Gradient Sparsity')
            axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"[Analyzer] Training curves saved to: {output_file}")
        plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze EnhPPFL experiment results'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        required=True,
        help='Path to experiment log directory'
    )
    parser.add_argument(
        '--json-report',
        type=str,
        default=None,
        help='Path to save JSON report (default: <log-dir>/analysis_report.json)'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate training curve plots (requires matplotlib)'
    )
    parser.add_argument(
        '--plot-file',
        type=str,
        default=None,
        help='Path to save plot (default: <log-dir>/training_curves.png)'
    )
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = ExperimentAnalyzer(args.log_dir)
    
    # Print report to console
    analyzer.print_report()
    
    # Save JSON report
    analyzer.save_json_report(args.json_report)
    
    # Generate plots
    if args.plot:
        analyzer.plot_training_curves(args.plot_file)
    
    print("\n[Analyzer] Analysis complete!")


if __name__ == '__main__':
    main()
