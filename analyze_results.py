"""
EnhPPFL Results Analyser
=========================
Parse experiment logs and produce a summary report.

Usage:
    python analyze_results.py --log-dir ./logs/enhppfl_20251130_120000
    python analyze_results.py --log-dir ./logs/... --plot --json-report report.json

Authors: Navneet Mishra, Prachet Bhuyan
Affiliation: School of Computer Engineering, KIIT Deemed to be University
"""

import argparse
import re
import os
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import json


class ExperimentAnalyser:
    """Parse EnhPPFL server and client logs into summary statistics."""

    def __init__(self, log_dir: str):
        self.log_dir    = Path(log_dir)
        self.server_log = self.log_dir / 'server.log'
        self.client_logs = list(self.log_dir.glob('client_*.log'))

        if not self.server_log.exists():
            raise FileNotFoundError(f"Server log not found: {self.server_log}")

    # ------------------------------------------------------------------

    def parse_server_log(self) -> Dict:
        with open(self.server_log, 'r') as f:
            content = f.read()

        rounds = []
        round_pattern    = r'\[Server\] Round (\d+)/(\d+)'
        epsilon_pattern  = r'Privacy \(server\):\s+ε=([\d.]+)'
        sparsity_pattern = r'Sparsity:\s+([\d.]+)'
        loss_pattern     = r'Loss:\s+([\d.]+)'
        accuracy_pattern = r'Accuracy:\s+([\d.]+)'

        total_rounds = 0
        for match in re.finditer(round_pattern, content):
            round_num    = int(match.group(1))
            total_rounds = int(match.group(2))

            start     = match.start()
            nxt       = content.find(f'Round {round_num + 1}/', start)
            end       = nxt if nxt != -1 else len(content)
            chunk     = content[start:end]

            def _first(pattern, text):
                m = re.search(pattern, text)
                return float(m.group(1)) if m else None

            rounds.append({
                'round':    round_num,
                'epsilon':  _first(epsilon_pattern,  chunk),
                'sparsity': _first(sparsity_pattern, chunk),
                'loss':     _first(loss_pattern,     chunk),
                'accuracy': _first(accuracy_pattern, chunk),
            })

        fp_m = re.search(r'Final ε = ([\d.]+)', content)
        return {
            'total_rounds_configured': total_rounds,
            'rounds_completed':        len(rounds),
            'final_epsilon':           float(fp_m.group(1)) if fp_m else None,
            'rounds':                  rounds
        }

    def parse_client_logs(self) -> List[Dict]:
        client_data = []
        for client_log in self.client_logs:
            m = re.search(r'client_(\d+)\.log', client_log.name)
            if not m:
                continue
            client_id = int(m.group(1))
            with open(client_log, 'r') as f:
                content = f.read()
            rounds_participated = len(re.findall(r'\[Client \d+\] Round \d+', content))
            epsilon_values      = [float(x) for x in re.findall(r'Privacy: ε=([\d.]+)', content)]
            compression_ratios  = [float(x) for x in re.findall(r'reduction=([\d.]+)%', content)]
            client_data.append({
                'client_id':           client_id,
                'rounds_participated': rounds_participated,
                'epsilon_values':      epsilon_values,
                'compression_ratios':  compression_ratios,
            })
        return client_data

    def generate_summary(self) -> Dict:
        server_data = self.parse_server_log()
        client_data = self.parse_client_logs()
        rounds      = server_data['rounds']

        def _vals(key):
            return [r[key] for r in rounds if r.get(key) is not None]

        rounds_per_client = [c['rounds_participated'] for c in client_data]

        return {
            'experiment': {
                'log_dir':                  str(self.log_dir),
                'total_rounds_configured':  server_data['total_rounds_configured'],
                'rounds_completed':         server_data['rounds_completed'],
                'num_clients':              len(client_data)
            },
            'privacy': {
                'final_epsilon': server_data['final_epsilon'],
                'max_epsilon':   max(_vals('epsilon'))  if _vals('epsilon')  else None,
                'min_epsilon':   min(_vals('epsilon'))  if _vals('epsilon')  else None,
                'avg_epsilon':   float(np.mean(_vals('epsilon'))) if _vals('epsilon') else None,
            },
            'performance': {
                'final_loss':     _vals('loss')[-1]     if _vals('loss')     else None,
                'final_accuracy': _vals('accuracy')[-1] if _vals('accuracy') else None,
                'best_accuracy':  max(_vals('accuracy')) if _vals('accuracy') else None,
                'avg_accuracy':   float(np.mean(_vals('accuracy'))) if _vals('accuracy') else None,
            },
            'compression': {
                'avg_sparsity':         float(np.mean(_vals('sparsity'))) if _vals('sparsity') else None,
                'avg_compression_ratio': 100.0 - float(np.mean(_vals('sparsity'))) if _vals('sparsity') else None,
            },
            'clients': {
                'total_clients':        len(client_data),
                'avg_rounds_per_client': float(np.mean(rounds_per_client)) if rounds_per_client else None,
                'min_rounds_per_client': min(rounds_per_client) if rounds_per_client else None,
                'max_rounds_per_client': max(rounds_per_client) if rounds_per_client else None,
            }
        }

    # ------------------------------------------------------------------

    def print_report(self):
        s = self.generate_summary()

        print("\n" + "=" * 70)
        print("EnhPPFL Experiment Analysis")
        print("=" * 70)

        e = s['experiment']
        print(f"\n[Experiment]")
        print(f"  Log directory:  {e['log_dir']}")
        print(f"  Clients:        {e['num_clients']}")
        print(f"  Rounds:         {e['rounds_completed']} / {e['total_rounds_configured']}")

        p = s['privacy']
        print(f"\n[Privacy Budget]")
        if p['final_epsilon'] is not None:
            print(f"  Final ε:  {p['final_epsilon']:.4f}")
        if p['avg_epsilon'] is not None:
            print(f"  Avg ε:    {p['avg_epsilon']:.4f}")
            print(f"  Min / Max ε: {p['min_epsilon']:.4f} / {p['max_epsilon']:.4f}")

        perf = s['performance']
        print(f"\n[Model Performance]")
        if perf['final_accuracy'] is not None:
            print(f"  Final accuracy:  {perf['final_accuracy']:.4f}")
        if perf['best_accuracy'] is not None:
            print(f"  Best accuracy:   {perf['best_accuracy']:.4f}")
        if perf['final_loss'] is not None:
            print(f"  Final loss:      {perf['final_loss']:.4f}")

        comp = s['compression']
        print(f"\n[Communication Efficiency]")
        if comp['avg_sparsity'] is not None:
            print(f"  Avg sparsity:     {comp['avg_sparsity']:.2f}%")
            print(f"  Avg reduction:    {comp['avg_compression_ratio']:.2f}%")

        cl = s['clients']
        print(f"\n[Client Participation]")
        if cl['avg_rounds_per_client'] is not None:
            print(f"  Avg rounds/client: {cl['avg_rounds_per_client']:.1f}")
            print(f"  Min / Max:         {cl['min_rounds_per_client']} / {cl['max_rounds_per_client']}")

        print("\n" + "=" * 70)

    def save_json_report(self, output_file: Optional[str] = None):
        if output_file is None:
            output_file = str(self.log_dir / 'analysis_report.json')
        server_data = self.parse_server_log()
        client_data = self.parse_client_logs()
        report      = {
            'summary':       self.generate_summary(),
            'server_rounds': server_data['rounds'],
            'clients':       client_data
        }
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        print(f"\nJSON report saved to: {output_file}")

    def plot_training_curves(self, output_file: Optional[str] = None):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not installed — skipping plots.")
            return

        if output_file is None:
            output_file = str(self.log_dir / 'training_curves.png')

        server_data = self.parse_server_log()
        rounds      = server_data['rounds']
        if not rounds:
            print("No round data available for plotting.")
            return

        round_nums      = [r['round']    for r in rounds]
        epsilon_vals    = [r['epsilon']  for r in rounds if r['epsilon']  is not None]
        accuracy_rounds = [r['round']    for r in rounds if r['accuracy'] is not None]
        accuracy_vals   = [r['accuracy'] for r in rounds if r['accuracy'] is not None]
        sparsity_vals   = [r['sparsity'] for r in rounds if r['sparsity'] is not None]

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        if epsilon_vals:
            axes[0].plot(round_nums[:len(epsilon_vals)], epsilon_vals, 'b-', linewidth=2)
        axes[0].set_xlabel('Round')
        axes[0].set_ylabel('Privacy budget (ε)')
        axes[0].set_title('Privacy budget accumulation')
        axes[0].grid(True, alpha=0.3)

        if accuracy_vals:
            axes[1].plot(accuracy_rounds, accuracy_vals, 'g-', linewidth=2)
        axes[1].set_xlabel('Round')
        axes[1].set_ylabel('Accuracy')
        axes[1].set_title('Model accuracy')
        axes[1].grid(True, alpha=0.3)

        if sparsity_vals:
            axes[2].plot(round_nums[:len(sparsity_vals)], sparsity_vals, 'r-', linewidth=2)
        axes[2].set_xlabel('Round')
        axes[2].set_ylabel('Sparsity (%)')
        axes[2].set_title('Gradient sparsity')
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(output_file, dpi=150)
        print(f"Training curves saved to: {output_file}")
        plt.close()


# ============================================================================
# ENTRY POINT
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Analyse EnhPPFL experiment logs')
    parser.add_argument('--log-dir',     type=str, required=True,
                        help='Path to experiment log directory')
    parser.add_argument('--json-report', type=str, default=None,
                        help='Path for JSON report (default: <log-dir>/analysis_report.json)')
    parser.add_argument('--plot',        action='store_true',
                        help='Generate training curve plots (requires matplotlib)')
    parser.add_argument('--plot-file',   type=str, default=None,
                        help='Path for plot file (default: <log-dir>/training_curves.png)')
    args = parser.parse_args()

    analyser = ExperimentAnalyser(args.log_dir)
    analyser.print_report()
    analyser.save_json_report(args.json_report)
    if args.plot:
        analyser.plot_training_curves(args.plot_file)
    print("\nAnalysis complete.")


if __name__ == '__main__':
    main()
