#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""unified_mismatch_analysis.py

One-click script for unified, high-quality mismatch analysis (75mm vs 120mm).
Replicates the style and content of the reference Fig2_mismatch_analysis.pdf.

Key requirements:
- Use val paired sim + same-source real (is_paired=true) ONLY.
- Panels B/D show representative sim (median corr) to preserve fringe details.
- Panels E/F/G/H show per-real distributions across all paired sims (violin + box).
- Panel I: log10(PSD_ratio) heatmap + small 3D surface inset (publication-friendly).
- Export source data CSVs for per-pair mismatch and per-pair PSD.
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

import seaborn as sns  # kept for potential future use

warnings.filterwarnings('ignore', category=RuntimeWarning)

import importlib.util

# Optional dependencies
try:
    _ssim_spec = importlib.util.find_spec('skimage.metrics')
except ModuleNotFoundError:
    _ssim_spec = None

if _ssim_spec is not None:
    try:
        from skimage.metrics import structural_similarity as ssim
    except Exception:
        ssim = None
else:
    ssim = None

try:
    from scipy.stats import pearsonr
except ImportError:
    pearsonr = None

try:
    from scipy.stats import skew as _scipy_skew
    from scipy.stats import kurtosis as _scipy_kurtosis
    from scipy.stats import gaussian_kde as _gaussian_kde
    from scipy.stats import mannwhitneyu as _mannwhitneyu
except Exception:
    _scipy_skew = None
    _scipy_kurtosis = None
    _gaussian_kde = None
    _mannwhitneyu = None

try:
    from scipy.signal import find_peaks as _find_peaks
except Exception:
    _find_peaks = None

try:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
except Exception:
    Axes3D = None


# --- Style Configuration ---

def set_pub_style():
    plt.rcParams.update({
        'font.size': 8,
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial'],
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 9,
        'axes.titlesize': 10,
        'axes.titlepad': 5,
        'xtick.labelsize': 8,
        'ytick.labelsize': 8,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.5,
        'ytick.major.size': 3.5,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'xtick.color': '#333333',
        'ytick.color': '#333333',
        'legend.fontsize': 7,
        'legend.frameon': False,
        'figure.titlesize': 12,
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': False,
        'lines.linewidth': 1.0,
        'patch.linewidth': 1.0,
        'grid.linewidth': 0.6,
    })
    plt.rcParams['figure.facecolor'] = 'white'


PALETTE = {
    'blue': '#0072B2',
    'orange': '#D55E00',
    'gray': '#4D4D4D',
    'light': '#EAEAF2',
}
RNG = np.random.default_rng(2026)


# --- I/O ---

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_pdf_png(fig, path_no_ext: str):
    fig.savefig(f"{path_no_ext}.pdf")
    fig.savefig(f"{path_no_ext}.png")


# --- Metric helpers ---

def _safe_corr(a, b):
    a, b = a.ravel().astype(np.float64), b.ravel().astype(np.float64)
    if np.std(a) < 1e-9 or np.std(b) < 1e-9:
        return np.nan
    return pearsonr(a, b)[0] if pearsonr else np.corrcoef(a, b)[0, 1]


def _radial_psd(img):
    f = np.fft.fftshift(np.fft.fft2(img))
    psd2d = np.abs(f) ** 2
    h, w = img.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.hypot(x - cx, y - cy).astype(int)
    radial_mean = np.bincount(r.ravel(), psd2d.ravel()) / np.maximum(1, np.bincount(r.ravel()))
    return np.arange(len(radial_mean)), radial_mean


def _safe_ssim(a, b):
    if ssim is None:
        return np.nan
    dr = float(np.max([a.max(), b.max()]) - np.min([a.min(), b.min()]))
    return ssim(a.astype(np.float64), b.astype(np.float64), data_range=max(dr, 1e-6))


def _grid_stats(map2d, grid=8):
    h, w = map2d.shape
    gh, gw = h // grid, w // grid
    out = np.zeros((grid, grid))
    for i in range(grid):
        for j in range(grid):
            out[i, j] = np.mean(map2d[i * gh:(i + 1) * gh, j * gw:(j + 1) * gw])
    return out


@dataclass
class MismatchData:
    metrics: pd.DataFrame
    pair_metrics: pd.DataFrame
    psd_curves: pd.DataFrame
    image_tiles: Dict[str, Dict[str, np.ndarray]]


def process_focal_length(focal_mm: int, code_dir: str) -> MismatchData:
    print(f'Processing {focal_mm}mm...')

    sys.path.insert(0, os.path.abspath(code_dir))
    module_name = 'main_75' if focal_mm == 75 else 'main_120'
    m = __import__(module_name)
    cfg = m.TrainConfig
    
    ds = m.DiffSimRealDataset(
        dataset_root=cfg.DATASET_ROOT,
        real_root=cfg.VAL_REAL_FOLDER,
        labels_csv=cfg.VAL_LABELS_CSV,
        init_diameter_txt=cfg.INIT_DIAMETER_TXT,
        real_rp_csv=cfg.VAL_REAL_RP_CSV,
        is_train=False,
        config=cfg,
    )

    df_paired = ds.df[ds.df['is_paired'] == True].copy()
    
    grouped_metrics = []
    all_pair_metrics = []
    all_psd_curves = []
    image_tiles = {}

    for real_fname, group in df_paired.groupby('tgt_fname'):
        print(f'  Analyzing real image: {real_fname} ({len(group)} sims)')

        real_tensor = None
        per_pair_rows = []
        per_pair_psd_rows = []
        
        for idx in group.index:
            sample = ds[idx]
            full_input = sample[1]

            if real_tensor is None:
                real_tensor = full_input[0].numpy()

            sim_img = full_input[2].numpy()
            if sim_img.shape != real_tensor.shape:
                continue

            residual = real_tensor - sim_img
            abs_residual = np.abs(residual)

            corr = _safe_corr(real_tensor, sim_img)
            nmae = np.mean(abs_residual) / (np.mean(real_tensor) + 1e-12)
            nrmse = np.sqrt(np.mean(residual ** 2)) / (np.sqrt(np.mean(real_tensor ** 2)) + 1e-12)
            ssim_val = _safe_ssim(real_tensor, sim_img)

            grid_map = _grid_stats(abs_residual)
            grid_mean, grid_max = np.mean(grid_map), np.max(grid_map)
            grid_ratio = grid_max / (grid_mean + 1e-12)

            psd_real_freq, psd_real_val = _radial_psd(real_tensor)
            psd_sim_freq, psd_sim_val = _radial_psd(sim_img)
            if len(psd_sim_val) != len(psd_real_val):
                L = min(len(psd_sim_val), len(psd_real_val))
                psd_real_freq = psd_real_freq[:L]
                psd_real_val = psd_real_val[:L]
                psd_sim_val = psd_sim_val[:L]

            psd_ratio = psd_sim_val / (psd_real_val + 1e-12)

            per_pair_rows.append({
                'real_fname': real_fname,
                'idx': int(idx),
                'corr': corr,
                'nmae': nmae,
                'nrmse': nrmse,
                'ssim': ssim_val,
                'grid_mean': grid_mean,
                'grid_max': grid_max,
                'grid_max_over_mean': grid_ratio,
                'grid_max_over_mean_log10': float(np.log10(grid_ratio + 1e-12)),
            })

            per_pair_psd_rows.append(pd.DataFrame({
                'real_fname': real_fname,
                'idx': int(idx),
                'freq': psd_real_freq,
                'psd_real': psd_real_val,
                'psd_sim': psd_sim_val,
                'psd_ratio': psd_ratio,
            }))

        if real_tensor is None or len(per_pair_rows) == 0:
            continue

        df_pair = pd.DataFrame(per_pair_rows)
        all_pair_metrics.append(df_pair)

        grouped_metrics.append({
            'real_fname': real_fname,
            'n_pairs': int(len(df_pair)),
            'corr_mean': float(df_pair['corr'].mean()),
            'corr_std': float(df_pair['corr'].std(ddof=1)) if len(df_pair) > 1 else 0.0,
            'nrmse_mean': float(df_pair['nrmse'].mean()),
            'nrmse_std': float(df_pair['nrmse'].std(ddof=1)) if len(df_pair) > 1 else 0.0,
            'nmae_mean': float(df_pair['nmae'].mean()),
            'nmae_std': float(df_pair['nmae'].std(ddof=1)) if len(df_pair) > 1 else 0.0,
            'ssim_mean': float(df_pair['ssim'].mean()),
            'ssim_std': float(df_pair['ssim'].std(ddof=1)) if len(df_pair) > 1 else 0.0,
            'grid_max_over_mean_mean': float(df_pair['grid_max_over_mean'].mean()),
            'grid_max_over_mean_std': float(df_pair['grid_max_over_mean'].std(ddof=1)) if len(df_pair) > 1 else 0.0,
        })

        # Representative sim for visualization (median corr)
        example_sim_img = None
        if not df_pair.empty:
            median_corr = float(df_pair['corr'].median())
            row = df_pair.iloc[(df_pair['corr'] - median_corr).abs().argsort()[:1]]
            if not row.empty:
                representative_idx = int(row['idx'].iloc[0])
                rep_sample = ds[representative_idx]
                example_sim_img = rep_sample[1][2].numpy()

        # Retrieve real again for tiles (guarantee aligned)
        if example_sim_img is not None:
            # Use ds[...] to get the aligned real corresponding to that index
            rep_sample = ds[int(df_pair['idx'].iloc[0])]
            real_tensor = rep_sample[1][0].numpy()

        image_tiles[real_fname] = {
            'real': real_tensor,
            'example_sim': example_sim_img,
        }

        if per_pair_psd_rows:
            all_psd_curves.append(pd.concat(per_pair_psd_rows, ignore_index=True))

    sys.path.pop(0)

    return MismatchData(
        metrics=pd.DataFrame(grouped_metrics),
        pair_metrics=pd.concat(all_pair_metrics, ignore_index=True) if all_pair_metrics else pd.DataFrame(),
                        psd_curves=pd.concat(all_psd_curves, ignore_index=True) if all_psd_curves else pd.DataFrame(),
        image_tiles=image_tiles,
    )


def _summarize_violin_data(df: pd.DataFrame, focal_length: int) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame()

    metrics_to_summarize = ['corr', 'nrmse', 'nmae', 'ssim', 'grid_max_over_mean_log10']
    metrics_to_summarize = [m for m in metrics_to_summarize if m in df.columns]

    summary = df.groupby('real_fname')[metrics_to_summarize].agg(
        ['mean', 'std', 'median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)]
    ).reset_index()

    new_cols = []
    for col in summary.columns.values:
        if col[1] == '<lambda_0>':
            new_cols.append(f"{col[0]}_q25")
        elif col[1] == '<lambda_1>':
            new_cols.append(f"{col[0]}_q75")
        elif col[1] == '':
            new_cols.append(col[0])
        else:
            new_cols.append(f"{col[0]}_{col[1]}")
    summary.columns = new_cols

    summary['focal_length'] = focal_length
    return summary


def _compute_violin_morphology(df_pairs: pd.DataFrame, focal_length: int,
                              metrics: Optional[List[str]] = None,
                              kde_grid_n: int = 512,
                              peak_prom_frac: float = 0.05) -> pd.DataFrame:
    """Quantify violin-plot morphology per (real_fname, metric).

    Outputs:
      - modality: unimodal/bimodal/multimodal (via KDE peak counting)
      - n_modes
      - skewness
      - kurtosis_excess (Fisher)
      - coverage_ratio_3sigma: fraction of samples within median ± 3*robust_sigma
      - coverage_width_3sigma: width of (median ± 3*robust_sigma)
      - tail_mass_outside_3sigma: 1 - coverage_ratio_3sigma
      - kde_area (should be ~1; numerical sanity check)
      - min/q25/median/q75/max/std

    Notes:
      - "3σ-like" uses a robust sigma estimate: sigma_robust = IQR / 1.349.
        This makes the metric meaningful even for non-Gaussian distributions.
      - For multi-modal / skewed distributions, coverage_ratio_3sigma still serves
        as a compact measure of tail heaviness; interpret together with skewness
        and kurtosis_excess.
      - If SciPy is unavailable, returns empty DataFrame.
      - Peak counting uses KDE density and a prominence threshold to avoid noise peaks.
    """

    if df_pairs is None or df_pairs.empty:
        return pd.DataFrame()

    if _scipy_skew is None or _scipy_kurtosis is None or _gaussian_kde is None or _find_peaks is None:
        return pd.DataFrame()

    if metrics is None:
        metrics = ['corr', 'nrmse', 'ssim', 'grid_max_over_mean_log10']

    metrics = [m for m in metrics if m in df_pairs.columns]
    if not metrics:
        return pd.DataFrame()

    rows = []
    for (real_fname), g in df_pairs.groupby('real_fname'):
        for metric in metrics:
            x = g[metric].dropna().astype(float).to_numpy()
            if x.size < 10:
                continue

            # KDE grid and density
            lo = float(np.min(x))
            hi = float(np.max(x))
            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                continue

            kde = _gaussian_kde(x)
            grid = np.linspace(lo, hi, int(kde_grid_n))
            dens = kde(grid)

            prom = float(np.max(dens) * float(peak_prom_frac))
            peaks, _ = _find_peaks(dens, prominence=prom)
            n_modes = int(len(peaks))
            if n_modes <= 1:
                modality = 'unimodal'
            elif n_modes == 2:
                modality = 'bimodal'
            else:
                modality = 'multimodal'

            q25 = float(np.quantile(x, 0.25))
            q75 = float(np.quantile(x, 0.75))
            med = float(np.median(x))
            iqr = float(q75 - q25)
            robust_sigma = float(iqr / 1.349) if iqr > 0 else float(np.std(x, ddof=1))
            lo3 = med - 3.0 * robust_sigma
            hi3 = med + 3.0 * robust_sigma
            in3 = np.logical_and(x >= lo3, x <= hi3)
            coverage_ratio_3sigma = float(np.mean(in3))

            rows.append({
                'focal_length': int(focal_length),
                'real_fname': real_fname,
                'metric': metric,
                'n': int(x.size),
                'modality': modality,
                'n_modes': n_modes,
                'skewness': float(_scipy_skew(x, bias=False)),
                'kurtosis_excess': float(_scipy_kurtosis(x, fisher=True, bias=False)),
                'coverage_ratio_3sigma': coverage_ratio_3sigma,
                'coverage_width_3sigma': float(hi3 - lo3),
                'tail_mass_outside_3sigma': float(1.0 - coverage_ratio_3sigma),
                'kde_area': float(np.trapezoid(dens, grid)),
                'min': float(lo),
                'q25': q25,
                'median': med,
                'q75': q75,
                'max': float(hi),
                'std': float(np.std(x, ddof=1)),
            })

    return pd.DataFrame(rows)


def _cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's Delta effect size. Positive => x tends to be larger than y."""
    x = np.asarray(x)
    y = np.asarray(y)
    if x.size == 0 or y.size == 0:
        return float('nan')

    # Use a vectorized sign comparison (may be memory-heavy for huge arrays, but here it's manageable).
    cmp = np.sign(x[:, None] - y[None, :])
    return float(np.sum(cmp) / (x.size * y.size))


def _interpret_cliffs_delta(d: float) -> str:
    if not np.isfinite(d):
        return 'nan'
    ad = abs(d)
    if ad < 0.147:
        return 'negligible'
    if ad < 0.33:
        return 'small'
    if ad < 0.474:
        return 'medium'
    return 'large'


def _stat_compare_groups(df75: pd.DataFrame, df120: pd.DataFrame,
                         metrics: Optional[List[str]] = None) -> pd.DataFrame:
    """Compare 75mm vs 120mm mismatch distributions for each metric.

    Returns a table with:
      - median, IQR for each group
      - Mann-Whitney U p-value (two-sided)
      - Cliff's Delta (d120_vs_75)
      - delta_interpretation
      - skewness/kurtosis_excess for each group
    """
    if metrics is None:
        metrics = ['corr', 'nrmse', 'ssim', 'grid_max_over_mean_log10']

    rows = []
    for m in metrics:
        if m not in df75.columns or m not in df120.columns:
            continue

        a = df75[m].dropna().astype(float).to_numpy()
        b = df120[m].dropna().astype(float).to_numpy()
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if a.size < 2 or b.size < 2:
            continue

        med75 = float(np.median(a))
        med120 = float(np.median(b))
        iqr75 = float(np.quantile(a, 0.75) - np.quantile(a, 0.25))
        iqr120 = float(np.quantile(b, 0.75) - np.quantile(b, 0.25))

        if _mannwhitneyu is not None:
            try:
                _, p = _mannwhitneyu(a, b, alternative='two-sided')
                p = float(p)
            except Exception:
                p = float('nan')
        else:
            p = float('nan')

        d = _cliffs_delta(b, a)  # positive => 120 tends to be larger

        if _scipy_skew is not None and _scipy_kurtosis is not None:
            sk75 = float(_scipy_skew(a, bias=False))
            sk120 = float(_scipy_skew(b, bias=False))
            ku75 = float(_scipy_kurtosis(a, fisher=True, bias=False))
            ku120 = float(_scipy_kurtosis(b, fisher=True, bias=False))
        else:
            sk75 = sk120 = ku75 = ku120 = float('nan')

        rows.append({
            'metric': m,
            'n_75': int(a.size),
            'n_120': int(b.size),
            'median_75': med75,
            'iqr_75': iqr75,
            'median_120': med120,
            'iqr_120': iqr120,
            'p_mannwhitneyu': p,
            'cliffs_delta_120_vs_75': d,
            'cliffs_delta_interpretation': _interpret_cliffs_delta(d),
            'skewness_75': sk75,
            'kurtosis_excess_75': ku75,
            'skewness_120': sk120,
            'kurtosis_excess_120': ku120,
        })

    return pd.DataFrame(rows)


def plot_mismatch_analysis(data75: MismatchData, data120: MismatchData, out_dir: str):
    fig = plt.figure(figsize=(7.2, 8.0))
    gs = fig.add_gridspec(5, 2, hspace=0.85, wspace=0.35, left=0.10, right=0.98, top=0.95, bottom=0.08)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])
    axF = fig.add_subplot(gs[2, 1])
    axG = fig.add_subplot(gs[3, 0])
    axH = fig.add_subplot(gs[3, 1])
    axI = fig.add_subplot(gs[4, :])

    def _panel(ax, label: str, color: str = 'white'):
        ax.text(0.02, 0.98, label, transform=ax.transAxes, fontsize=12, fontweight='bold',
                va='top', ha='left', color=color)

    _violin_by_real(axA, data75.pair_metrics, 'corr', '75mm: corr(real, sim)', 'Pearson corr', PALETTE['blue'])
    _violin_by_real(axB, data120.pair_metrics, 'corr', '120mm: corr(real, sim)', 'Pearson corr', PALETTE['orange'])
    _panel(axA, 'A', color='#111111')
    _panel(axB, 'B', color='#111111')

    _violin_by_real(axC, data75.pair_metrics, 'nrmse', '75mm: NRMSE(real, sim)', 'NRMSE', PALETTE['blue'])
    _violin_by_real(axD, data120.pair_metrics, 'nrmse', '120mm: NRMSE(real, sim)', 'NRMSE', PALETTE['orange'])
    _panel(axC, 'C', color='#111111')
    _panel(axD, 'D', color='#111111')

    _violin_by_real(axE, data75.pair_metrics, 'ssim', '75mm: SSIM(real, sim)', 'SSIM', PALETTE['blue'])
    _violin_by_real(axF, data120.pair_metrics, 'ssim', '120mm: SSIM(real, sim)', 'SSIM', PALETTE['orange'])
    _panel(axE, 'E', color='#111111')
    _panel(axF, 'F', color='#111111')

    _violin_by_real(axG, data75.pair_metrics, 'grid_max_over_mean_log10', '75mm: log10(Grid Max/Mean)', 'log10(Grid Max/Mean)', PALETTE['blue'])
    _violin_by_real(axH, data120.pair_metrics, 'grid_max_over_mean_log10', '120mm: log10(Grid Max/Mean)', 'log10(Grid Max/Mean)', PALETTE['orange'])
    _panel(axG, 'G', color='#111111')
    _panel(axH, 'H', color='#111111')

    def _plot_psd_ratio_curve(ax, psd_df: pd.DataFrame, label_prefix: str, color: str,
                             q_low: float = 0.10, q_high: float = 0.90):
        if psd_df is None or psd_df.empty:
            return

        g = psd_df.groupby('freq')['psd_ratio'].agg(
            mean='mean',
            q_low=lambda x: np.nanquantile(x, q_low),
            q_high=lambda x: np.nanquantile(x, q_high),
        ).reset_index()

        x = g['freq'].to_numpy(dtype=float)
        y = g['mean'].to_numpy(dtype=float)
        ylo = g['q_low'].to_numpy(dtype=float)
        yhi = g['q_high'].to_numpy(dtype=float)

        ax.plot(x, y, color=color, linewidth=1.0, label=f"{label_prefix} mean")
        ax.fill_between(x, ylo, yhi, color=color, alpha=0.20, linewidth=0,
                        label=f"{label_prefix} {int(q_low*100)}–{int(q_high*100)}%")

    _plot_psd_ratio_curve(axI, data75.psd_curves, '75mm', PALETTE['blue'])
    _plot_psd_ratio_curve(axI, data120.psd_curves, '120mm', PALETTE['orange'])
    axI.axhline(1.0, color='black', linestyle='--', linewidth=1.0, alpha=0.8)
    axI.set_title('PSD ratio (sim / real)', fontweight='bold')
    axI.set_xlabel('Radial frequency (px)')
    axI.set_ylabel('PSD ratio')
    axI.legend(loc='upper right')
    _style_ax(axI, grid_axis='both')
    _panel(axI, 'I', color='#111111')

    save_pdf_png(fig, os.path.join(out_dir, 'Fig2_mismatch_analysis'))
    print(f'Saved mismatch analysis figure to {out_dir}')


def main():
    set_pub_style()
    
    data75 = process_focal_length(75, 'Code_75/core')
    data120 = process_focal_length(120, 'Code_120/core')

    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(script_dir, '..'))

    table_out_dir = os.path.join(project_root, 'tables')
    ensure_dir(table_out_dir)

    data75.metrics.to_csv(os.path.join(table_out_dir, 'mismatch_summary_75mm.csv'), index=False)
    data75.pair_metrics.to_csv(os.path.join(table_out_dir, 'mismatch_per_pair_75mm.csv'), index=False)
    data75.psd_curves.to_csv(os.path.join(table_out_dir, 'psd_curves_per_pair_75mm.csv'), index=False)

    data120.metrics.to_csv(os.path.join(table_out_dir, 'mismatch_summary_120mm.csv'), index=False)
    data120.pair_metrics.to_csv(os.path.join(table_out_dir, 'mismatch_per_pair_120mm.csv'), index=False)
    data120.psd_curves.to_csv(os.path.join(table_out_dir, 'psd_curves_per_pair_120mm.csv'), index=False)

    df_summary75 = data75.metrics.copy()
    df_summary120 = data120.metrics.copy()
    df_summary75['focal_length'] = 75
    df_summary120['focal_length'] = 120
    df_summary_unified = pd.concat([df_summary75, df_summary120], ignore_index=True)
    df_summary_unified.to_csv(os.path.join(table_out_dir, 'table_mismatch_summary_unified.csv'), index=False)

    df_violin_summary75 = _summarize_violin_data(data75.pair_metrics, 75)
    df_violin_summary120 = _summarize_violin_data(data120.pair_metrics, 120)
    df_violin_summary_unified = pd.concat([df_violin_summary75, df_violin_summary120], ignore_index=True)
    df_violin_summary_unified.to_csv(os.path.join(table_out_dir, 'table_violin_plot_summary_unified.csv'), index=False)

    # Violin morphology (shape) analysis
    df_morph75 = _compute_violin_morphology(data75.pair_metrics, 75)
    df_morph120 = _compute_violin_morphology(data120.pair_metrics, 120)
    df_morph_unified = pd.concat([df_morph75, df_morph120], ignore_index=True)
    df_morph_unified.to_csv(os.path.join(table_out_dir, 'table_violin_morphology_by_focal_real_metric.csv'), index=False)

    # Cross-focal statistical comparison (deeper_analysis.py logic, merged)
    df_stat = _stat_compare_groups(data75.pair_metrics, data120.pair_metrics)
    df_stat.to_csv(os.path.join(table_out_dir, 'table_mismatch_stats_75_vs_120.csv'), index=False)

    print(f'Exported source data tables to {table_out_dir}')
    
    out_dir = os.path.join(project_root, 'figures')
    ensure_dir(out_dir)

    plot_example_tiles(data75, data120, out_dir)
    plot_mismatch_analysis(data75, data120, out_dir)

    print(f'Unified analysis complete. Outputs in {project_root}')


def plot_example_tiles(data75: MismatchData, data120: MismatchData, out_dir: str):
    """Creates a separate figure for the example real/sim tiles."""
    fig = plt.figure(figsize=(3.6, 3.8))
    gs = fig.add_gridspec(2, 2, hspace=0.25, wspace=0.15, left=0.02, right=0.98, top=0.90, bottom=0.02)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])

    def _get_first_tile(data: MismatchData) -> Optional[Dict[str, np.ndarray]]:
        if not data.image_tiles:
            return None
        k = sorted(list(data.image_tiles.keys()))[0]
        return data.image_tiles[k]

    tile75 = _get_first_tile(data75)
    tile120 = _get_first_tile(data120)

    def _imshow_or_text(ax, img: Optional[np.ndarray], title: str):
        ax.set_title(title, fontweight='bold', fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        if img is None:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center')
            return
        ax.imshow(img, cmap='gray')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('#BBBBBB')
            spine.set_linewidth(1.0)

    def _panel(ax, label: str):
        ax.text(0.05, 0.95, label, transform=ax.transAxes, fontsize=12, fontweight='bold', va='top', ha='left', color='white')

    _imshow_or_text(axA, tile75['real'] if tile75 else None, '75mm: Real (aligned)')
    _imshow_or_text(axB, tile75['example_sim'] if tile75 else None, '75mm: Representative sim')
    _imshow_or_text(axC, tile120['real'] if tile120 else None, '120mm: Real (aligned)')
    _imshow_or_text(axD, tile120['example_sim'] if tile120 else None, '120mm: Representative sim')
    _panel(axA, 'A')
    _panel(axB, 'B')
    _panel(axC, 'C')
    _panel(axD, 'D')

    save_pdf_png(fig, os.path.join(out_dir, 'Fig1_example_tiles'))
    print(f'Saved example tiles figure to {out_dir}')



def _style_ax(ax, grid_axis='y'):
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, alpha=0.18, linewidth=0.6)
    ax.tick_params(length=3.2, width=1.0)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#333333')



def _violin_by_real(ax, df_pairs: pd.DataFrame, metric: str, title: str, ylabel: str, base_color: str):
    if df_pairs is None or df_pairs.empty or metric not in df_pairs.columns:
        ax.text(0.5, 0.5, f'No {metric} data', ha='center', va='center')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    order = sorted(df_pairs['real_fname'].dropna().unique().tolist())
    data_by_real_raw = [df_pairs.loc[df_pairs['real_fname'] == name, metric].dropna().to_numpy(dtype=float) for name in order]
    order_kept = [name for name, arr in zip(order, data_by_real_raw) if arr.size > 0]
    data_by_real = [arr for arr in data_by_real_raw if arr.size > 0]

    if len(data_by_real) == 0:
        ax.text(0.5, 0.5, f'No valid {metric} values', ha='center', va='center')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        return

    positions = np.arange(len(order_kept))

    parts = ax.violinplot(
        data_by_real,
        positions=positions,
        widths=0.80,
        showmeans=False,
        showmedians=False,
        showextrema=False,
    )
    for body in parts['bodies']:
        body.set_facecolor(base_color)
        body.set_edgecolor('#111111')
        body.set_alpha(0.48)
        body.set_linewidth(1.0)

    ax.boxplot(
        data_by_real,
        positions=positions,
        widths=0.22,
        patch_artist=True,
        showfliers=False,
        medianprops={'color': '#111111', 'linewidth': 1.3},
        boxprops={'facecolor': 'white', 'edgecolor': '#444444', 'linewidth': 0.9},
        whiskerprops={'color': '#444444', 'linewidth': 0.9},
        capprops={'color': '#444444', 'linewidth': 0.9},
    )

    for i, vals in enumerate(data_by_real):
        if vals.size == 0:
            continue
        k = int(min(60, vals.size))
        idxs = RNG.choice(vals.size, size=k, replace=False) if vals.size > k else np.arange(vals.size)
        yy = vals[idxs]
        xx = positions[i] + RNG.uniform(-0.12, 0.12, size=yy.size)
        ax.scatter(
            xx,
            yy,
            s=6,
            color='#6B7280',
            alpha=0.50,
            edgecolors='white',
            linewidths=0.2,
            zorder=3,
        )

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('')
    ax.set_ylabel(ylabel)
    ax.set_xticks(positions)
    ax.set_xticklabels(order_kept, rotation=20, ha='right')
    _style_ax(ax)


if __name__ == '__main__':
    main()
