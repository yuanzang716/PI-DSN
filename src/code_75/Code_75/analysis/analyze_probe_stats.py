#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
analyze_probe_stats.py

Statistical analysis script: extract statistical metrics from probe experiment results

Features:
1. Extract metrics from all probe experiments (EMA error, predicted mean, bias, etc.)
2. Calculate success rate, failure rate
3. Compute statistics (mean, std, median, min, max, quartiles)
4. Calculate maximum bias and variance distribution
5. Generate statistical summary tables (CSV and JSON)

Outputs:
- probe_statistics/summary_stats.csv: statistical summary table
- probe_statistics/detailed_results.json: detailed results
- probe_statistics/metrics_per_seed.csv: detailed metrics for each seed
"""

import os
import sys
import json
import csv
import glob
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from pathlib import Path


# ==========================
# Configuration
# ==========================
CONFIG = {
    "OUTPUT_ROOT": os.path.join(os.path.dirname(os.path.dirname(__file__)), "results"),  # Save results to new_strategy/results (consistent with run_all_experiments.py)
    "GT_DIAMETER": 100.2,  # Ground truth diameter (um)
    "SUCCESS_THRESHOLD": 0.01,  # Success threshold: err < 1%
    "OUTPUT_DIR": None,  # Auto-set to OUTPUT_ROOT/probe_statistics
}


def to_float(x: Any, default: float = float('nan')) -> float:
    """Safely convert to float"""
    try:
        return float(x)
    except (ValueError, TypeError):
        return default


def load_metrics_csv(csv_path: str) -> Optional[Dict[str, Any]]:
    """Extract best EMA model metrics from metrics.csv"""
    if not os.path.exists(csv_path):
        return None

    try:
        df = pd.read_csv(csv_path)
        if df.empty:
            return None

        # Find the row with the minimum val_ema_score (best EMA model)
        if 'val_ema_score' not in df.columns:
            return None

        df['val_ema_score'] = pd.to_numeric(df['val_ema_score'], errors='coerce')
        best_row = df.loc[df['val_ema_score'].idxmin()]

        # Extract key metrics
        result = {
            'epoch': int(best_row.get('epoch', -1)),
            'val_ema_err_obs': to_float(best_row.get('val_ema_err_obs')),
            'val_ema_score': to_float(best_row.get('val_ema_score')),
            'val_ema_pred_mean': to_float(best_row.get('val_pred_mean')),  # EMA predicted mean
            'val_err_obs': to_float(best_row.get('val_err_obs')),
            'val_score': to_float(best_row.get('val_score')),
            'val_pred_mean': to_float(best_row.get('val_pred_mean')),
            'val_pred_std': to_float(best_row.get('val_pred_std')),
            'train_loss': to_float(best_row.get('train_loss')),
        }

        # Compute bias
        if not np.isnan(result['val_ema_pred_mean']):
            result['bias_um'] = abs(result['val_ema_pred_mean'] - CONFIG['GT_DIAMETER'])
        else:
            result['bias_um'] = float('nan')

        # Determine success
        result['is_success'] = (
            not np.isnan(result['val_ema_err_obs']) and
            result['val_ema_err_obs'] < CONFIG['SUCCESS_THRESHOLD']
        )

        return result
    except Exception as e:
        print(f"Warning: failed to read {csv_path}: {e}")
        return None


def collect_all_probe_results() -> List[Dict[str, Any]]:
    """Collect results from all probe experiments"""
    output_root = CONFIG["OUTPUT_ROOT"]
    results = []

    # Find all seed_XX/probe directories
    seed_dirs = glob.glob(os.path.join(output_root, "seed_*"))

    for seed_dir in seed_dirs:
        seed_name = os.path.basename(seed_dir)
        try:
            seed = int(seed_name.split('_')[1])
        except (ValueError, IndexError):
            continue

        probe_dir = os.path.join(seed_dir, "probe")
        metrics_csv = os.path.join(probe_dir, "metrics.csv")

        if not os.path.exists(metrics_csv):
            continue

        metrics = load_metrics_csv(metrics_csv)
        if metrics is None:
            continue

        metrics['seed'] = seed
        metrics['probe_dir'] = probe_dir
        results.append(metrics)

    return sorted(results, key=lambda x: x['seed'])


def calculate_statistics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compute statistics"""
    if not results:
        return {}

    # Extract key metrics
    ema_errors = [r['val_ema_err_obs'] for r in results if not np.isnan(r['val_ema_err_obs'])]
    biases = [r['bias_um'] for r in results if not np.isnan(r['bias_um'])]
    ema_scores = [r['val_ema_score'] for r in results if not np.isnan(r['val_ema_score'])]
    successes = [r['is_success'] for r in results]

    stats: Dict[str, Any] = {}

    # EMA error statistics
    if ema_errors:
        stats['ema_error'] = {
            'mean': float(np.mean(ema_errors)),
            'std': float(np.std(ema_errors)),
            'median': float(np.median(ema_errors)),
            'min': float(np.min(ema_errors)),
            'max': float(np.max(ema_errors)),
            'q25': float(np.percentile(ema_errors, 25)),
            'q75': float(np.percentile(ema_errors, 75)),
            'n': len(ema_errors),
        }

    # Bias statistics
    if biases:
        stats['bias'] = {
            'mean': float(np.mean(biases)),
            'std': float(np.std(biases)),
            'median': float(np.median(biases)),
            'min': float(np.min(biases)),
            'max': float(np.max(biases)),
            'q25': float(np.percentile(biases, 25)),
            'q75': float(np.percentile(biases, 75)),
            'n': len(biases),
        }

    # EMA score statistics
    if ema_scores:
        stats['ema_score'] = {
            'mean': float(np.mean(ema_scores)),
            'std': float(np.std(ema_scores)),
            'median': float(np.median(ema_scores)),
            'min': float(np.min(ema_scores)),
            'max': float(np.max(ema_scores)),
            'n': len(ema_scores),
        }

    # Success rate statistics
    if successes:
        n_success = sum(successes)
        n_total = len(successes)
        stats['success_rate'] = {
            'n_success': n_success,
            'n_total': n_total,
            'rate': float(n_success / n_total) if n_total > 0 else 0.0,
            'failure_rate': float((n_total - n_success) / n_total) if n_total > 0 else 0.0,
        }

    return stats


def save_results(results: List[Dict[str, Any]], stats: Dict[str, Any], output_dir: str):
    """Save results"""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Save detailed results (JSON)
    detailed_path = os.path.join(output_dir, "detailed_results.json")
    with open(detailed_path, 'w', encoding='utf-8') as f:
        json.dump({
            'config': CONFIG,
            'statistics': stats,
            'results': results,
        }, f, indent=2, ensure_ascii=False)

    # 2. Save per-seed metrics (CSV)
    metrics_path = os.path.join(output_dir, "metrics_per_seed.csv")
    if results:
        df = pd.DataFrame(results)
        df.to_csv(metrics_path, index=False)

    # 3. Save summary stats (CSV)
    summary_path = os.path.join(output_dir, "summary_stats.csv")
    summary_rows = []

    # EMA error statistics
    if 'ema_error' in stats:
        summary_rows.append({
            'metric': 'EMA Relative Error',
            'mean': stats['ema_error']['mean'],
            'std': stats['ema_error']['std'],
            'median': stats['ema_error']['median'],
            'min': stats['ema_error']['min'],
            'max': stats['ema_error']['max'],
            'q25': stats['ema_error']['q25'],
            'q75': stats['ema_error']['q75'],
            'n': stats['ema_error']['n'],
        })

    # Bias statistics
    if 'bias' in stats:
        summary_rows.append({
            'metric': 'Absolute Bias (um)',
            'mean': stats['bias']['mean'],
            'std': stats['bias']['std'],
            'median': stats['bias']['median'],
            'min': stats['bias']['min'],
            'max': stats['bias']['max'],
            'q25': stats['bias']['q25'],
            'q75': stats['bias']['q75'],
            'n': stats['bias']['n'],
        })

    # EMA score statistics
    if 'ema_score' in stats:
        summary_rows.append({
            'metric': 'EMA Score',
            'mean': stats['ema_score']['mean'],
            'std': stats['ema_score']['std'],
            'median': stats['ema_score']['median'],
            'min': stats['ema_score']['min'],
            'max': stats['ema_score']['max'],
            'q25': None,
            'q75': None,
            'n': stats['ema_score']['n'],
        })

    # Success rate statistics
    if 'success_rate' in stats:
        summary_rows.append({
            'metric': 'Success Rate',
            'mean': stats['success_rate']['rate'],
            'std': None,
            'median': None,
            'min': None,
            'max': None,
            'q25': None,
            'q75': None,
            'n': stats['success_rate']['n_total'],
        })

    if summary_rows:
        df_summary = pd.DataFrame(summary_rows)
        df_summary.to_csv(summary_path, index=False)

    print(f"\n Results saved to:")
    print(f"   - {detailed_path}")
    print(f"   - {metrics_path}")
    print(f"   - {summary_path}")


def print_summary(stats: Dict[str, Any], n_results: int):
    """Print statistical summary"""
    print("\n" + "=" * 80)
    print("Probe statistical analysis summary")
    print("=" * 80)
    print(f"\nTotal experiments: {n_results}")

    if 'success_rate' in stats:
        sr = stats['success_rate']
        print(f"\nSuccess rate statistics:")
        print(f"  Successes: {sr['n_success']}/{sr['n_total']}")
        print(f"  Success rate: {sr['rate']*100:.2f}%")
        print(f"  Failure rate: {sr['failure_rate']*100:.2f}%")

    if 'ema_error' in stats:
        ee = stats['ema_error']
        print(f"\nEMA relative error statistics:")
        print(f"  Mean: {ee['mean']*100:.4f}%")
        print(f"  Std: {ee['std']*100:.4f}%")
        print(f"  Median: {ee['median']*100:.4f}%")
        print(f"  Range: [{ee['min']*100:.4f}%, {ee['max']*100:.4f}%]")
        print(f"  Quartiles: Q25={ee['q25']*100:.4f}%, Q75={ee['q75']*100:.4f}%")

    if 'bias' in stats:
        bias = stats['bias']
        print(f"\nAbsolute bias statistics (um):")
        print(f"  Mean: {bias['mean']:.4f} um")
        print(f"  Std: {bias['std']:.4f} um")
        print(f"  Median: {bias['median']:.4f} um")
        print(f"  Range: [{bias['min']:.4f}, {bias['max']:.4f}] um")
        print(f"  Max bias: {bias['max']:.4f} um")

    print("\n" + "=" * 80)


def main():
    """Main function"""
    # Set output directory
    if CONFIG["OUTPUT_DIR"] is None:
        CONFIG["OUTPUT_DIR"] = os.path.join(CONFIG["OUTPUT_ROOT"], "probe_statistics")

    print("=" * 80)
    print("Probe statistical analysis")
    print("=" * 80)
    print(f"Output root: {CONFIG['OUTPUT_ROOT']}")
    print(f"GT diameter: {CONFIG['GT_DIAMETER']} um")
    print(f"Success threshold: err < {CONFIG['SUCCESS_THRESHOLD']*100}%")

    # Collect all probe results
    print("\nCollecting probe experiment results...")
    results = collect_all_probe_results()

    if not results:
        print("No probe experiment results found!")
        print("Please make sure you have run the probe stage experiments.")
        return

    print(f"Found {len(results)} probe experiment results")

    # Compute statistics
    print("\nComputing statistics...")
    stats = calculate_statistics(results)

    # Print summary
    print_summary(stats, len(results))

    # Save results
    print("\nSaving results...")
    save_results(results, stats, CONFIG["OUTPUT_DIR"])

    print("\n Statistical analysis completed!")
    print(f"\nNext step: run the visualization script to generate plots")
    print(f"  python3 visualize_probe_stats.py")


if __name__ == "__main__":
    main()
