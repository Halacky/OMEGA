"""
Utility script for analyzing cross-subject experiment results
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Dict, List
import pandas as pd


class CrossSubjectAnalyzer:
    """Analyzer for cross-subject experiment results"""
    
    def __init__(self, results_dir: Path):
        self.results_dir = Path(results_dir)
        self.results = None
        self.loso_summary = None
    
    def load_single_experiment(self, results_path: Path = None):
        """Load results from a single cross-subject experiment"""
        if results_path is None:
            results_path = self.results_dir / "cross_subject_results.json"
        
        with open(results_path, 'r') as f:
            self.results = json.load(f)
        
        print(f"✓ Loaded results from: {results_path}")
        return self.results
    
    def load_loso_summary(self, summary_path: Path = None):
        """Load LOSO summary results"""
        if summary_path is None:
            summary_path = self.results_dir / "loso_summary.json"
        
        with open(summary_path, 'r') as f:
            self.loso_summary = json.load(f)
        
        print(f"✓ Loaded LOSO summary from: {summary_path}")
        return self.loso_summary
    
    def print_single_experiment_summary(self):
        """Print summary of a single experiment"""
        if self.results is None:
            print("❌ No results loaded. Call load_single_experiment() first.")
            return
        
        config = self.results['config']
        test_results = self.results['cross_subject_test']
        
        print("\n" + "=" * 70)
        print("CROSS-SUBJECT EXPERIMENT SUMMARY")
        print("=" * 70)
        print(f"Train Subjects: {', '.join(config['train_subjects'])}")
        print(f"Test Subject:   {config['test_subject']}")
        print(f"Exercise:       {config['exercise']}")
        print(f"Common Gestures: {len(config['common_gestures'])} gestures")
        print("-" * 70)
        print(f"Test Accuracy:  {test_results['accuracy']:.4f}")
        print(f"Test F1-Macro:  {test_results['f1_macro']:.4f}")
        print("=" * 70)
        
        # Per-subject breakdown
        if 'per_subject_analysis' in self.results:
            print("\nPer-Subject Analysis:")
            print("-" * 70)
            for subject, metrics in sorted(self.results['per_subject_analysis'].items()):
                role = "TRAIN" if metrics['is_train'] else "TEST"
                print(f"{subject:10s} [{role:5s}]: Acc={metrics['accuracy']:.4f}, F1={metrics['f1_macro']:.4f}")
    
    def print_loso_summary(self):
        """Print LOSO evaluation summary"""
        if self.loso_summary is None:
            print("❌ No LOSO summary loaded. Call load_loso_summary() first.")
            return
        
        print("\n" + "=" * 70)
        print("LOSO EVALUATION SUMMARY")
        print("=" * 70)
        print(f"Subjects:    {', '.join(self.loso_summary['subjects'])}")
        print(f"Exercise:    {self.loso_summary['exercise']}")
        print(f"Total Experiments: {self.loso_summary['total_experiments']}")
        print(f"Successful:        {self.loso_summary['successful_experiments']}")
        print("-" * 70)
        
        if 'mean_accuracy' in self.loso_summary:
            print(f"Mean Accuracy: {self.loso_summary['mean_accuracy']:.4f} ± {self.loso_summary['std_accuracy']:.4f}")
            print(f"Mean F1-Macro: {self.loso_summary['mean_f1_macro']:.4f} ± {self.loso_summary['std_f1_macro']:.4f}")
            print("=" * 70)
            
            # Individual results
            print("\nIndividual Results:")
            print("-" * 70)
            for result in self.loso_summary['individual_results']:
                if result['accuracy'] is not None:
                    print(f"Test: {result['test_subject']:10s} | Acc={result['accuracy']:.4f} | F1={result['f1_macro']:.4f}")
                else:
                    print(f"Test: {result['test_subject']:10s} | ❌ FAILED")
    
    def create_comparison_table(self, save_path: Path = None):
        """Create comparison table for LOSO results"""
        if self.loso_summary is None:
            print("❌ No LOSO summary loaded.")
            return None
        
        results = self.loso_summary['individual_results']
        successful = [r for r in results if r['accuracy'] is not None]
        
        if not successful:
            print("❌ No successful results to create table.")
            return None
        
        df = pd.DataFrame({
            'Test Subject': [r['test_subject'] for r in successful],
            'Accuracy': [r['accuracy'] for r in successful],
            'F1-Macro': [r['f1_macro'] for r in successful],
            'Train Subjects': [', '.join(r['train_subjects']) for r in successful]
        })
        
        # Add statistics row
        stats_row = pd.DataFrame({
            'Test Subject': ['MEAN ± STD'],
            'Accuracy': [f"{self.loso_summary['mean_accuracy']:.4f} ± {self.loso_summary['std_accuracy']:.4f}"],
            'F1-Macro': [f"{self.loso_summary['mean_f1_macro']:.4f} ± {self.loso_summary['std_f1_macro']:.4f}"],
            'Train Subjects': ['']
        })
        
        df = pd.concat([df, stats_row], ignore_index=True)
        
        if save_path:
            df.to_csv(save_path, index=False)
            print(f"✓ Table saved to: {save_path}")
        
        print("\n" + df.to_string(index=False))
        return df
    
    def plot_loso_distribution(self, save_path: Path = None):
        """Plot distribution of LOSO results"""
        if self.loso_summary is None:
            print("❌ No LOSO summary loaded.")
            return
        
        results = self.loso_summary['individual_results']
        successful = [r for r in results if r['accuracy'] is not None]
        
        if not successful:
            print("❌ No successful results to plot.")
            return
        
        subjects = [r['test_subject'] for r in successful]
        accuracies = [r['accuracy'] for r in successful]
        f1_scores = [r['f1_macro'] for r in successful]
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Accuracy distribution
        ax1 = axes[0]
        bars1 = ax1.bar(range(len(subjects)), accuracies, color='#4CAF50', alpha=0.7, edgecolor='black')
        ax1.axhline(y=self.loso_summary['mean_accuracy'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax1.axhline(y=self.loso_summary['mean_accuracy'] + self.loso_summary['std_accuracy'], 
                   color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax1.axhline(y=self.loso_summary['mean_accuracy'] - self.loso_summary['std_accuracy'], 
                   color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax1.set_xticks(range(len(subjects)))
        ax1.set_xticklabels(subjects, rotation=45, ha='right')
        ax1.set_ylabel('Accuracy')
        ax1.set_title('LOSO: Accuracy Distribution')
        ax1.set_ylim(0, 1.0)
        ax1.grid(True, alpha=0.3, axis='y')
        ax1.legend()
        
        # Add value labels
        for bar, acc in zip(bars1, accuracies):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{acc:.3f}', ha='center', va='bottom', fontsize=9)
        
        # F1-Macro distribution
        ax2 = axes[1]
        bars2 = ax2.bar(range(len(subjects)), f1_scores, color='#2196F3', alpha=0.7, edgecolor='black')
        ax2.axhline(y=self.loso_summary['mean_f1_macro'], color='red', linestyle='--', linewidth=2, label='Mean')
        ax2.axhline(y=self.loso_summary['mean_f1_macro'] + self.loso_summary['std_f1_macro'], 
                   color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax2.axhline(y=self.loso_summary['mean_f1_macro'] - self.loso_summary['std_f1_macro'], 
                   color='red', linestyle=':', linewidth=1, alpha=0.5)
        ax2.set_xticks(range(len(subjects)))
        ax2.set_xticklabels(subjects, rotation=45, ha='right')
        ax2.set_ylabel('F1-Macro')
        ax2.set_title('LOSO: F1-Macro Distribution')
        ax2.set_ylim(0, 1.0)
        ax2.grid(True, alpha=0.3, axis='y')
        ax2.legend()
        
        # Add value labels
        for bar, f1 in zip(bars2, f1_scores):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{f1:.3f}', ha='center', va='bottom', fontsize=9)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Plot saved to: {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def analyze_generalization_gap(self):
        """Analyze the generalization gap between train and test performance"""
        if self.loso_summary is None:
            print("❌ No LOSO summary loaded.")
            return
        
        print("\n" + "=" * 70)
        print("GENERALIZATION GAP ANALYSIS")
        print("=" * 70)
        
        # Load individual experiment results to get training performance
        gaps = []
        
        for result in self.loso_summary['individual_results']:
            if result['accuracy'] is None:
                continue
            
            test_subject = result['test_subject']
            exp_dir = self.results_dir / f"test_{test_subject}"
            results_file = exp_dir / "results.json"
            
            if results_file.exists():
                with open(results_file, 'r') as f:
                    exp_results = json.load(f)
                
                # Get validation performance (proxy for train performance)
                if 'training' in exp_results and 'val' in exp_results['training']:
                    val_acc = exp_results['training']['val']['accuracy']
                    test_acc = result['accuracy']
                    gap = val_acc - test_acc
                    gaps.append({
                        'test_subject': test_subject,
                        'val_accuracy': val_acc,
                        'test_accuracy': test_acc,
                        'gap': gap,
                        'gap_percent': (gap / val_acc * 100) if val_acc > 0 else 0
                    })
        
        if gaps:
            print(f"\nAnalyzed {len(gaps)} experiments:")
            print("-" * 70)
            for g in gaps:
                print(f"{g['test_subject']:10s}: Val={g['val_accuracy']:.4f}, Test={g['test_accuracy']:.4f}, "
                      f"Gap={g['gap']:.4f} ({g['gap_percent']:.1f}%)")
            
            mean_gap = np.mean([g['gap'] for g in gaps])
            mean_gap_pct = np.mean([g['gap_percent'] for g in gaps])
            
            print("-" * 70)
            print(f"Mean Gap: {mean_gap:.4f} ({mean_gap_pct:.1f}%)")
            print("=" * 70)
            
            return gaps
        else:
            print("❌ Could not compute generalization gaps.")
            return None


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze cross-subject experiment results')
    parser.add_argument('--results_dir', type=str, required=True, help='Path to results directory')
    parser.add_argument('--mode', type=str, choices=['single', 'loso'], default='loso',
                       help='Analysis mode: single experiment or LOSO')
    parser.add_argument('--save_plot', type=str, default=None, help='Path to save plot')
    parser.add_argument('--save_table', type=str, default=None, help='Path to save comparison table (CSV)')
    
    args = parser.parse_args()
    
    analyzer = CrossSubjectAnalyzer(Path(args.results_dir))
    
    if args.mode == 'single':
        analyzer.load_single_experiment()
        analyzer.print_single_experiment_summary()
    
    elif args.mode == 'loso':
        analyzer.load_loso_summary()
        analyzer.print_loso_summary()
        
        if args.save_table:
            analyzer.create_comparison_table(Path(args.save_table))
        else:
            analyzer.create_comparison_table()
        
        if args.save_plot:
            analyzer.plot_loso_distribution(Path(args.save_plot))
        else:
            analyzer.plot_loso_distribution()
        
        analyzer.analyze_generalization_gap()


if __name__ == "__main__":
    main()