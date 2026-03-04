"""
synthetic_multi_basin_analysis.py

This script provides a comprehensive numerical simulation framework to characterize 
the ill-posed nature of the diffraction-based inverse problem and the strong 
coupling between target parameters and nuisance factors.

As described in the accompanying manuscript, recovered physical parameters from 
indirect sensor measurements often suffer from biased convergence due to structural 
sim-to-real mismatch and parameter entanglement. This script replicates these 
challenges in a controlled synthetic environment by:
  1. Generating high-density 1D/2D error landscapes (loss surfaces) across 
     target (diameter 'a') and nuisance (offset 'r', nonlinearity 'gamma') dimensions.
  2. Characterizing "equivalent solution bands" where distinct parameter 
     combinations yield nearly identical diffraction waveforms under noise.
  3. Visualizing the multi-basin structure and non-convexity of the optimization 
     objective, motivating the need for the proposed dual-stem learning framework 
     over traditional fitting methods.

This analysis serves as the numerical foundation for the "Structured sim-to-real 
mismatch" and "nuisance coupling" arguments presented in the main text.
"""

import os
import math
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from concurrent.futures import ProcessPoolExecutor
from functools import partial

def set_pub_style():
    plt.rcParams.update({
        'font.size': 7,
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Liberation Sans', 'Arial'],
        'axes.linewidth': 1.0,
        'axes.edgecolor': '#333333',
        'axes.labelcolor': '#333333',
        'axes.spines.top': False,
        'axes.spines.right': False,
        'axes.labelsize': 8,
        'axes.titlesize': 8,
        'axes.titlepad': 5,
        'xtick.labelsize': 6,
        'ytick.labelsize': 6,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 3.0,
        'ytick.major.size': 3.0,
        'xtick.major.width': 1.0,
        'ytick.major.width': 1.0,
        'legend.fontsize': 6,
        'legend.frameon': False,
        'figure.titlesize': 10,
        'axes.unicode_minus': False,
        'pdf.fonttype': 42,
        'ps.fonttype': 42,
        'figure.dpi': 600,
        'savefig.dpi': 600,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'axes.grid': False,
        'lines.linewidth': 1.2,
        'patch.linewidth': 1.0,
        'grid.linewidth': 0.6,
    })
    plt.rcParams['figure.facecolor'] = 'white'


PALETTE = {
    'blue': '#0072B2',
    'orange': '#D55E00',
    'gray': '#4D4D4D',
    'light': '#EAEAF2',
    'green': '#009E73',
    'purple': '#CC79A7',
}


# ============================
# Fraunhofer model (1D slice)
# ============================

def fraunhofer_intensity_1d(a_um: float, theta: np.ndarray, wavelength_um: float) -> np.ndarray:
    x = (a_um * np.sin(theta)) / wavelength_um
    return np.sinc(x) ** 2


def normalize_max(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    m = float(np.max(np.abs(v)))
    return v / (m + eps)


def add_noise(y_norm: np.ndarray, *, kind: str, sigma: float, rng: np.random.Generator, alpha: float = 0.0, beta: float = 0.0) -> np.ndarray:
    """Add noise on a *max-normalized* signal.

    gaussian_hetero (requested):
      noisy_i ~ N(mean=y_i, std=alpha*y_i + beta)
      output_i = max(y_i, noisy_i)
    """
    y_norm = np.asarray(y_norm, dtype=np.float64)

    if kind == "gaussian_hetero":
        std = alpha * y_norm + beta
        noisy = rng.normal(loc=y_norm, scale=std, size=y_norm.shape)
        return np.maximum(y_norm, noisy)

    if kind == "stripe":
        n = y_norm.shape[0]
        t = np.linspace(0, 1, n)
        stripe = np.sin(2 * np.pi * (5.0 * t + 0.15))
        stripe = stripe / (np.max(np.abs(stripe)) + 1e-12)
        return y_norm + sigma * 0.6 * stripe

    return y_norm


# ============================
# Loss (normalized L2 only)
# ============================

def loss_norm_l2(y_meas_norm: np.ndarray, y_pred: np.ndarray) -> float:
    y_pred_norm = normalize_max(y_pred)
    d = y_meas_norm - y_pred_norm
    return float(np.mean(d * d))


# ============================
# Nuisance model and workers
# ============================

def _theta_shift(theta_obs: np.ndarray, *, r_um: float, focal_length_um: float) -> np.ndarray:
    return np.arctan(np.tan(theta_obs) + (r_um / focal_length_um))


def worker_best_over_nuisance_at_a(
    a_um: float,
    y_meas_norm: np.ndarray,
    theta_obs: np.ndarray,
    wavelength_um: float,
    focal_length_um: float,
    r_grid_um: np.ndarray,
    gamma_grid: np.ndarray,
) -> tuple[float, float, float]:
    best_l = math.inf
    best_r = float("nan")
    best_gm = float("nan")

    for r_um in r_grid_um:
        th = _theta_shift(theta_obs, r_um=float(r_um), focal_length_um=focal_length_um)
        I0 = fraunhofer_intensity_1d(float(a_um), th, wavelength_um)
        for gm in gamma_grid:
            I = np.power(np.clip(I0, 0.0, None), float(gm))
            L = loss_norm_l2(y_meas_norm, I)
            if L < best_l:
                best_l = L
                best_r = float(r_um)
                best_gm = float(gm)

    return float(best_l), best_r, best_gm


def worker_landscape_row_raw(
    r_um: float,
    a_sub: np.ndarray,
    y_meas_norm: np.ndarray,
    theta_obs: np.ndarray,
    wavelength_um: float,
    focal_length_um: float,
    gamma_fixed: float,
) -> np.ndarray:
    th = _theta_shift(theta_obs, r_um=float(r_um), focal_length_um=focal_length_um)
    row = np.zeros(len(a_sub), dtype=np.float64)
    for j, a in enumerate(a_sub):
        I0 = fraunhofer_intensity_1d(float(a), th, wavelength_um)
        I = np.power(np.clip(I0, 0.0, None), float(gamma_fixed))
        row[j] = loss_norm_l2(y_meas_norm, I)
    return row


def worker_landscape_row_min(
    r_um: float,
    a_sub: np.ndarray,
    y_meas_norm: np.ndarray,
    theta_obs: np.ndarray,
    wavelength_um: float,
    focal_length_um: float,
    gamma_grid: np.ndarray,
) -> np.ndarray:
    th = _theta_shift(theta_obs, r_um=float(r_um), focal_length_um=focal_length_um)
    row = np.zeros(len(a_sub), dtype=np.float64)
    for j, a in enumerate(a_sub):
        I0 = fraunhofer_intensity_1d(float(a), th, wavelength_um)
        best = math.inf
        for gm in gamma_grid:
            I = np.power(np.clip(I0, 0.0, None), float(gm))
            L = loss_norm_l2(y_meas_norm, I)
            if L < best:
                best = L
        row[j] = float(best)
    return row


def worker_landscape_row_gamma_min_over_r(
    gm: float,
    a_grid: np.ndarray,
    y_meas_norm: np.ndarray,
    theta_obs: np.ndarray,
    wavelength_um: float,
    focal_length_um: float,
    r_grid_um: np.ndarray,
) -> np.ndarray:
    """For a fixed gamma, compute loss(a, gamma) minimized over r."""
    row = np.zeros(len(a_grid), dtype=np.float64)
    gm = float(gm)
    for j, a in enumerate(a_grid):
        best = math.inf
        for r_um in r_grid_um:
            th = _theta_shift(theta_obs, r_um=float(r_um), focal_length_um=focal_length_um)
            I0 = fraunhofer_intensity_1d(float(a), th, wavelength_um)
            I = np.power(np.clip(I0, 0.0, None), gm)
            L = loss_norm_l2(y_meas_norm, I)
            if L < best:
                best = L
        row[j] = float(best)
    return row


# ============================
# Plot helpers
# ============================

def _panel_label(ax, label: str):
    """Draw panel label outside the plot area in lowercase."""
    label = label.lower()
    # Positioning label at top-left outside the axis
    if hasattr(ax, "text2D"):
        ax.text2D(-0.15, 1.10, label, transform=ax.transAxes, fontsize=8, fontweight='bold',
                  va='top', ha='left', color='#111111')
    else:
        ax.text(-0.15, 1.10, label, transform=ax.transAxes, fontsize=8, fontweight='bold',
                va='top', ha='left', color='#111111')


def _style_ax(ax, grid_axis: str = 'y'):
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, alpha=0.18, linewidth=0.6)
    ax.tick_params(length=3.0, width=1.0, labelsize=6)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#333333')


def save_multifigure(
    *,
    out_path_no_ext: str,
    theta_obs: np.ndarray,
    focal_length_um: float,
    y_meas_norm: np.ndarray,
    y_true_plot: np.ndarray,
    waveform_curves: list[dict],
    a_grid: np.ndarray,
    best_losses: np.ndarray,
    best_r: np.ndarray,
    best_gamma: np.ndarray,
    a_true: float,
    equiv_mask: np.ndarray,
    L_min: np.ndarray,
    a_sub: np.ndarray,
    r_sub: np.ndarray,
    r_true: float,
):
    """High-density multi-panel figure for coupling/multi-solution."""

    fig = plt.figure(figsize=(7.2, 6.8))
    gs = fig.add_gridspec(3, 2, hspace=0.65, wspace=0.45, left=0.12, right=0.82, top=0.92, bottom=0.1)

    axA = fig.add_subplot(gs[0, 0])
    axB = fig.add_subplot(gs[0, 1])
    axC = fig.add_subplot(gs[1, 0])
    axD = fig.add_subplot(gs[1, 1])
    axE = fig.add_subplot(gs[2, 0])
    axF = fig.add_subplot(gs[2, 1])

    # A: measured vs predicted waveforms
    y_axis = np.tan(theta_obs) * focal_length_um
    axA.plot(y_axis, y_meas_norm, color=PALETTE['gray'], lw=1.2, label='meas')
    axA.plot(y_axis, y_true_plot, color=PALETTE['orange'], lw=1.2, alpha=0.8, label='true')
    colors = [PALETTE['blue'], PALETTE['green'], PALETTE['purple']]
    for i, c in enumerate(waveform_curves):
        axA.plot(y_axis, c['pred_norm'], color=colors[i % len(colors)], lw=1.2, label=c['label'])
    axA.set_xlabel('y (µm)', fontsize=7)
    axA.set_ylabel('intensity', fontsize=7)
    axA.set_title('Waveforms', fontsize=8)
    axA.legend(loc='upper left', bbox_to_anchor=(1.05, 1.05), fontsize=5.5, frameon=False, handlelength=1.2)
    _style_ax(axA, grid_axis='both')
    _panel_label(axA, 'a')

    # B: residuals
    for i, c in enumerate(waveform_curves):
        axB.plot(y_axis, y_meas_norm - c['pred_norm'], color=colors[i % len(colors)], lw=1.0)
    axB.axhline(0.0, color='#111111', lw=1.0, ls='--', alpha=0.5)
    axB.set_xlabel('y (µm)', fontsize=7)
    axB.set_ylabel('residual', fontsize=7)
    axB.set_title('Residuals', fontsize=8)
    _style_ax(axB, grid_axis='both')
    _panel_label(axB, 'b')

    # C: min loss vs a (min over r,gamma)
    axC.plot(a_grid, best_losses, color=PALETTE['blue'], lw=1.2)
    axC.fill_between(a_grid, best_losses, float(np.min(best_losses)), where=equiv_mask, color=PALETTE['blue'], alpha=0.2)
    axC.axvline(a_true, color=PALETTE['orange'], ls='--', lw=1.2)
    axC.set_xlabel('a (µm)', fontsize=7)
    axC.set_ylabel('min loss', fontsize=7)
    axC.set_title('Loss vs Diameter', fontsize=8)
    _style_ax(axC, grid_axis='both')

    # C right axis: best gamma at each a (aligned to panel-c x)
    axC2 = axC.twinx()
    axC2.plot(a_grid, best_gamma, color=PALETTE['purple'], lw=1.0, ls='--', alpha=0.9)
    axC2.set_ylabel('best γ', fontsize=7, color=PALETTE['purple'])
    axC2.tick_params(axis='y', labelsize=6, colors=PALETTE['purple'])
    axC2.spines['right'].set_visible(True)
    axC2.spines['right'].set_linewidth(1.0)
    axC2.spines['right'].set_color('#333333')
    _panel_label(axC, 'c')

    # D: Zoomed view of panel C near true a (±1%)
    zoom_range = 0.01
    zoom_mask = (a_grid >= a_true * (1 - zoom_range)) & (a_grid <= a_true * (1 + zoom_range))
    axD.plot(a_grid[zoom_mask], best_losses[zoom_mask], color=PALETTE['blue'], lw=1.5)
    axD.fill_between(
        a_grid[zoom_mask],
        best_losses[zoom_mask],
        float(np.min(best_losses)),
        where=equiv_mask[zoom_mask],
        color=PALETTE['blue'],
        alpha=0.3,
    )
    axD.axvline(a_true, color=PALETTE['orange'], ls='--', lw=1.5)
    axD.set_xlabel('a (µm)', fontsize=7)
    axD.set_ylabel('min loss', fontsize=7)
    axD.set_title('Zoomed Loss (±1%)', fontsize=8)
    _style_ax(axD, grid_axis='both')
    _panel_label(axD, 'd')

    from matplotlib.patches import ConnectionPatch
    y_min_c, y_max_c = axC.get_ylim()
    x_start, x_end = a_true * (1 - zoom_range), a_true * (1 + zoom_range)
    rect = plt.Rectangle(
        (x_start, y_min_c),
        x_end - x_start,
        y_max_c - y_min_c,
        fill=False,
        edgecolor=PALETTE['gray'],
        ls='--',
        lw=1.0,
        alpha=0.6,
    )
    axC.add_patch(rect)

    con_tl = ConnectionPatch(xyA=(x_start, y_max_c), xyB=(0, 1), coordsA='data', coordsB='axes fraction', axesA=axC, axesB=axD, color=PALETTE['gray'], lw=1.0, ls='--', alpha=0.4)
    con_tr = ConnectionPatch(xyA=(x_end, y_max_c), xyB=(1, 1), coordsA='data', coordsB='axes fraction', axesA=axC, axesB=axD, color=PALETTE['gray'], lw=1.0, ls='--', alpha=0.4)
    con_bl = ConnectionPatch(xyA=(x_start, y_min_c), xyB=(0, 0), coordsA='data', coordsB='axes fraction', axesA=axC, axesB=axD, color=PALETTE['gray'], lw=1.0, ls='--', alpha=0.4)
    con_br = ConnectionPatch(xyA=(x_end, y_min_c), xyB=(1, 0), coordsA='data', coordsB='axes fraction', axesA=axC, axesB=axD, color=PALETTE['gray'], lw=1.0, ls='--', alpha=0.4)
    for con in [con_tl, con_tr, con_bl, con_br]:
        fig.add_artist(con)

    # E: iso-loss contour map on (a, r), min over gamma
    AA, RR = np.meshgrid(a_sub, r_sub)
    cf = axE.contourf(AA, RR, L_min, levels=18, cmap='viridis')
    c_lines = axE.contour(AA, RR, L_min, levels=8, colors='white', linewidths=0.6, alpha=0.7)
    axE.clabel(c_lines, inline=True, fontsize=5, fmt='%.1e')
    axE.axvline(a_true, color=PALETTE['orange'], ls='--', lw=1.2, label='true a')
    axE.axhline(r_true, color=PALETTE['green'], ls='--', lw=1.1, label='true r')
    axE.set_title('Iso-loss contour (min over γ)', fontsize=8, fontweight='semibold')
    axE.set_xlabel('a (µm)', fontsize=7)
    axE.set_ylabel('r (µm)', fontsize=7)
    cbarE = fig.colorbar(cf, ax=axE, fraction=0.046, pad=0.04)
    cbarE.set_label('loss', fontsize=6)
    cbarE.ax.tick_params(labelsize=6)
    axE.legend(loc='upper right', fontsize=5.6, frameon=False, handlelength=1.5)
    _style_ax(axE, grid_axis='both')
    _panel_label(axE, 'e')

    # F: equivalent-solution cloud (legend uses category proxies; color meaning in colorbar)
    axF.scatter(
        a_grid,
        best_gamma,
        color='#C9C9C9',
        s=10,
        alpha=0.35,
        edgecolors='none',
    )

    n_equiv = int(np.sum(equiv_mask))
    if n_equiv >= 2:
        sc = axF.scatter(
            a_grid[equiv_mask],
            best_gamma[equiv_mask],
            c=best_r[equiv_mask],
            cmap='coolwarm',
            s=24,
            alpha=0.92,
            edgecolors='none',
        )
        cbarF = fig.colorbar(sc, ax=axF, fraction=0.046, pad=0.04)
        cbarF.set_label('best r (µm)', fontsize=6)
        cbarF.ax.tick_params(labelsize=6)
    else:
        axF.scatter(
            a_grid[equiv_mask],
            best_gamma[equiv_mask],
            color=PALETTE['purple'],
            s=30,
            alpha=0.95,
            edgecolors='none',
        )

    axF.axvline(a_true, color=PALETTE['orange'], ls='--', lw=1.2)
    axF.set_title('Equivalent-solution cloud', fontsize=8, fontweight='semibold')
    axF.set_xlabel('a (µm)', fontsize=7)
    axF.set_ylabel('best γ', fontsize=7)

    from matplotlib.lines import Line2D
    from matplotlib.legend_handler import HandlerBase
    from matplotlib.patches import FancyBboxPatch, Rectangle

    class ColormapCapsule:
        pass

    class HandlerColormapCapsule(HandlerBase):
        def __init__(self, cmap, n_strips=20):
            super().__init__()
            self.cmap = cmap
            self.n_strips = n_strips

        def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
            pad_w = width * 0.04
            pad_h = height * 0.18
            x0 = xdescent + pad_w
            y0 = ydescent + pad_h
            w = width - 2 * pad_w
            h = height - 2 * pad_h
            rounding = h / 2.0

            capsule = FancyBboxPatch(
                (x0, y0),
                w,
                h,
                boxstyle=f"round,pad=0,rounding_size={rounding}",
                transform=trans,
                facecolor='none',
                edgecolor='#444444',
                linewidth=0.35,
            )

            artists = []
            for i in range(self.n_strips):
                t = (i + 0.5) / self.n_strips
                strip = Rectangle(
                    (x0 + i * w / self.n_strips, y0),
                    w / self.n_strips + 1e-6,
                    h,
                    transform=trans,
                    facecolor=self.cmap(0.08 + 0.84 * t),
                    edgecolor='none',
                    alpha=0.96,
                )
                strip.set_clip_path(capsule)
                artists.append(strip)

            artists.append(capsule)
            return artists

    equiv_handle = ColormapCapsule()
    legend_handles = [
        Line2D([0], [0], marker='o', color='none', markerfacecolor='#C9C9C9', markeredgecolor='none', markersize=4.8, alpha=0.7),
        equiv_handle,
        Line2D([0], [0], color=PALETTE['orange'], lw=1.2, ls='--'),
    ]
    legend_labels = ['all candidates', 'equivalent solutions', 'true a']

    axF.legend(
        handles=legend_handles,
        labels=legend_labels,
        loc='upper right',
        fontsize=5.6,
        frameon=False,
        handlelength=1.6,
        handler_map={ColormapCapsule: HandlerColormapCapsule(plt.get_cmap('coolwarm'))},
    )
    _style_ax(axF, grid_axis='both')
    _panel_label(axF, 'f')

    fig.savefig(f"{out_path_no_ext}.pdf", dpi=600)
    fig.savefig(f"{out_path_no_ext}.png", dpi=300)
    plt.close(fig)


# ============================
# Main
# ============================

def save_candidate_best_nuisance_tracks(*, out_path_no_ext: str, a_grid: np.ndarray, best_losses: np.ndarray, best_r: np.ndarray, best_gamma: np.ndarray, a_true: float, equiv_mask: np.ndarray):
    fig, ax1 = plt.subplots(figsize=(4.2, 3.0))
    ax1.plot(a_grid, best_losses, color=PALETTE['blue'], lw=1.4, label='min loss')
    ax1.fill_between(a_grid, best_losses, float(np.min(best_losses)), where=equiv_mask, color=PALETTE['blue'], alpha=0.20)
    ax1.axvline(a_true, color=PALETTE['orange'], ls='--', lw=1.2)
    ax1.set_xlabel('a (µm)', fontsize=8)
    ax1.set_ylabel('min loss', fontsize=8, color=PALETTE['blue'])
    ax1.tick_params(axis='y', labelcolor=PALETTE['blue'])
    _style_ax(ax1, grid_axis='both')

    ax2 = ax1.twinx()
    ax2.plot(a_grid, best_gamma, color=PALETTE['purple'], ls='--', lw=1.1, label='best γ')
    ax2.plot(a_grid, best_r, color=PALETTE['green'], ls='-.', lw=1.1, label='best r')
    ax2.set_ylabel('best γ / best r (µm)', fontsize=8)
    ax2.tick_params(labelsize=6)

    lines = ax1.get_lines() + ax2.get_lines()
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', fontsize=6)
    ax1.set_title('Coupling tracks along a', fontsize=8)

    fig.savefig(f"{out_path_no_ext}.pdf", dpi=600)
    fig.savefig(f"{out_path_no_ext}.png", dpi=300)
    plt.close(fig)


def save_candidate_iso_contour(*, out_path_no_ext: str, L_min: np.ndarray, a_sub: np.ndarray, r_sub: np.ndarray, a_true: float, r_true: float):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    AA, RR = np.meshgrid(a_sub, r_sub)
    csf = ax.contourf(AA, RR, L_min, levels=18, cmap='viridis')
    cs = ax.contour(AA, RR, L_min, levels=8, colors='white', linewidths=0.5, alpha=0.7)
    ax.clabel(cs, inline=True, fontsize=5, fmt='%.2e')
    cbar = fig.colorbar(csf, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=6)
    ax.axvline(a_true, color=PALETTE['orange'], ls='--', lw=1.2)
    ax.axhline(r_true, color=PALETTE['green'], ls='--', lw=1.0)
    ax.set_xlabel('a (µm)', fontsize=8)
    ax.set_ylabel('r (µm)', fontsize=8)
    ax.set_title('Iso-loss contours on (a, r)', fontsize=8)
    _style_ax(ax, grid_axis='both')
    fig.savefig(f"{out_path_no_ext}.pdf", dpi=600)
    fig.savefig(f"{out_path_no_ext}.png", dpi=300)
    plt.close(fig)


def save_candidate_equiv_scatter(*, out_path_no_ext: str, a_grid: np.ndarray, best_r: np.ndarray, best_gamma: np.ndarray, equiv_mask: np.ndarray, a_true: float):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))

    ax.scatter(
        a_grid,
        best_gamma,
        color='#C9C9C9',
        s=10,
        alpha=0.35,
        edgecolors='none',
        label='all candidates',
    )

    n_equiv = int(np.sum(equiv_mask))
    if n_equiv >= 2:
        sc = ax.scatter(
            a_grid[equiv_mask],
            best_gamma[equiv_mask],
            c=best_r[equiv_mask],
            cmap='coolwarm',
            s=24,
            alpha=0.92,
            edgecolors='none',
            label='equivalent solutions',
        )
        cbar = fig.colorbar(sc, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('best r (µm)', fontsize=7)
        cbar.ax.tick_params(labelsize=6)
    else:
        ax.scatter(
            a_grid[equiv_mask],
            best_gamma[equiv_mask],
            color=PALETTE['purple'],
            s=30,
            alpha=0.95,
            edgecolors='none',
            label='equivalent solution',
        )

    ax.axvline(a_true, color=PALETTE['orange'], ls='--', lw=1.2, label='true a')
    ax.set_xlabel('a (µm)', fontsize=8)
    ax.set_ylabel('best γ', fontsize=8)
    ax.set_title('Equivalent-solution cloud', fontsize=8, fontweight='semibold')
    ax.legend(loc='upper right', fontsize=6, frameon=False, handlelength=1.5)
    _style_ax(ax, grid_axis='both')
    fig.savefig(f"{out_path_no_ext}.pdf", dpi=600)
    fig.savefig(f"{out_path_no_ext}.png", dpi=300)
    plt.close(fig)


def save_candidate_loss_slices(*, out_path_no_ext: str, a_sub: np.ndarray, r_sub: np.ndarray, L_min: np.ndarray, a_true: float):
    fig, ax = plt.subplots(figsize=(4.2, 3.0))
    idxs = np.linspace(0, len(r_sub) - 1, 6).astype(int)
    for idx in idxs:
        ax.plot(a_sub, L_min[idx, :], lw=1.0, label=f'r={r_sub[idx]:.0f} µm')
    ax.axvline(a_true, color=PALETTE['orange'], ls='--', lw=1.2)
    ax.set_xlabel('a (µm)', fontsize=8)
    ax.set_ylabel('min loss over γ', fontsize=8)
    ax.set_title('Loss slices at selected r', fontsize=8)
    ax.legend(fontsize=5.5, ncol=2, loc='upper right')
    _style_ax(ax, grid_axis='both')
    fig.savefig(f"{out_path_no_ext}.pdf", dpi=600)
    fig.savefig(f"{out_path_no_ext}.png", dpi=300)
    plt.close(fig)


def run():
    import pandas as pd

    set_pub_style()

    root_out_dir = os.path.join(os.path.dirname(__file__), 'synthetic_outputs')
    os.makedirs(root_out_dir, exist_ok=True)

    wavelength_um = 0.78024
    focal_length_um = 75000.0

    # Observation window (k ~ 5..12 region)
    a_true = 100.2
    k_start = 4.3
    k_end = 12.7
    n_samples = 950

    sin_t0 = (k_start * wavelength_um) / a_true
    sin_t1 = (k_end * wavelength_um) / a_true
    sin_t0 = np.clip(sin_t0, -0.999999, 0.999999)
    sin_t1 = np.clip(sin_t1, -0.999999, 0.999999)

    y0 = focal_length_um * np.tan(np.arcsin(sin_t0))
    y1 = focal_length_um * np.tan(np.arcsin(sin_t1))
    y_obs = np.linspace(y0, y1, int(n_samples))
    theta_obs = np.arctan(y_obs / focal_length_um)

    # True nuisance
    r_true_um = 60.0
    gamma_true = 0.55
    gain_true = 1.7

    theta_true = _theta_shift(theta_obs, r_um=r_true_um, focal_length_um=focal_length_um)
    I0_true = fraunhofer_intensity_1d(a_true, theta_true, wavelength_um)
    I_true = gain_true * np.power(np.clip(I0_true, 0.0, None), gamma_true)

    # Normalize true then add noise (as specified)
    I_true_norm = normalize_max(I_true)

    # Search grids
    r_grid = np.linspace(-550.0, 550.0, 100)
    gamma_grid = np.linspace(0.1, 1.10, 100)
    a_grid = np.linspace(a_true * 0.90, a_true * 1.1, 100)

    # landscapes
    a_sub = np.linspace(a_true * 0.90, a_true * 1.1, 100)
    r_sub = np.linspace(-550.0, 550.0, 100)

    # Cases: all gaussian replaced by gaussian_hetero (alpha/beta differ)
    noise_cases = [
        ('gaussian_hetero', 0.0, {'alpha': 0.00, 'beta': 0.00}),
        ('gaussian_hetero', 0.0, {'alpha': 0.05, 'beta': 0.005}),
        ('gaussian_hetero', 0.0, {'alpha': 0.10, 'beta': 0.010}),
        ('gaussian_hetero', 0.0, {'alpha': 0.10, 'beta': 0.020}),
        ('gaussian_hetero', 0.0, {'alpha': 0.20, 'beta': 0.010}),
        ('gaussian_hetero', 0.0, {'alpha': 0.25, 'beta': 0.010}),
        ('stripe', 0.02, {}),
    ]

    rng = np.random.default_rng(42)
    num_cpus = os.cpu_count() or 4

    summary_rows = []
    tol = 0.2

    for kind, sigma, kwargs in noise_cases:
        if kind == 'gaussian_hetero':
            case_id = f"{kind}_a{kwargs.get('alpha', 0):.3f}_b{kwargs.get('beta', 0):.3f}"
        else:
            case_id = f"{kind}_s{float(sigma):.3f}"

        case_dir = os.path.join(root_out_dir, f"case_{case_id}")
        os.makedirs(case_dir, exist_ok=True)

        print(f"\n[CASE] {case_id}")

        # Add noise on normalized true
        y_meas_noisy = add_noise(I_true_norm.copy(), kind=kind, sigma=float(sigma), rng=rng, **kwargs)
        
        # Calculate the scale factor to maintain visual alignment in plots
        noise_max = np.max(y_meas_noisy)
        scale_factor = 1.0 / (noise_max if noise_max > 0 else 1.0)
        
        # Then normalize again for estimation input
        y_meas_norm = y_meas_noisy * scale_factor
        y_meas_norm = np.clip(y_meas_norm, 0.0, None)
        
        # Adjust the true reference for the plot to align with the base of the spikes
        y_true_plot = I_true_norm * scale_factor

        # 1) best loss vs a (min over r,gamma)
        with ProcessPoolExecutor(max_workers=num_cpus) as ex:
            func = partial(
                worker_best_over_nuisance_at_a,
                y_meas_norm=y_meas_norm,
                theta_obs=theta_obs,
                wavelength_um=wavelength_um,
                focal_length_um=focal_length_um,
                r_grid_um=r_grid,
                gamma_grid=gamma_grid,
            )
            res = list(ex.map(func, a_grid))

        best_losses = np.array([x[0] for x in res], dtype=np.float64)
        best_r = np.array([x[1] for x in res], dtype=np.float64)
        best_gamma = np.array([x[2] for x in res], dtype=np.float64)

        min_loss = float(np.min(best_losses))
        equiv_mask = best_losses <= min_loss * (1.0 + tol)

        # summary
        a_equiv = a_grid[equiv_mask]
        equiv_a_lo = float(np.min(a_equiv)) if a_equiv.size else float('nan')
        equiv_a_hi = float(np.max(a_equiv)) if a_equiv.size else float('nan')
        equiv_width_um = (equiv_a_hi - equiv_a_lo) if np.isfinite(equiv_a_lo) and np.isfinite(equiv_a_hi) else float('nan')
        equiv_width_pct = (equiv_width_um / a_true * 100.0) if np.isfinite(equiv_width_um) else float('nan')
        mask_1pct = equiv_mask & (a_grid >= a_true * 0.99) & (a_grid <= a_true * 1.01)
        count_equiv_in_1pct = int(np.sum(mask_1pct))

        # 2) pick waveform candidates: TRUE + Extreme Error candidates
        # We want to show cases where 'a' has a large error but waveforms are nearly identical.
        idx_true = int(np.argmin(np.abs(a_grid - a_true)))
        candidate_indices = [idx_true]

        # Find the global indices of all equivalent solutions
        idxs_equiv = np.where(equiv_mask)[0]
        
        if idxs_equiv.size > 0:
            # Sort equivalent indices by their absolute error from true 'a'
            abs_errors = np.abs(a_grid[idxs_equiv] - a_true)
            sorted_equiv_by_err = idxs_equiv[np.argsort(abs_errors)[::-1]] # Descending error
            
            # Pick top 2 most distant solutions that are still "equivalent"
            for i in range(min(2, len(sorted_equiv_by_err))):
                candidate_indices.append(int(sorted_equiv_by_err[i]))

        # build waveform curves
        waveform_curves = []
        for idx in candidate_indices[:3]: # Keep it clean with 3 curves total
            a_c = float(a_grid[idx])
            r_c = float(best_r[idx])
            gm_c = float(best_gamma[idx])
            err_pct = (a_c - a_true) / a_true * 100.0
            
            th_c = _theta_shift(theta_obs, r_um=r_c, focal_length_um=focal_length_um)
            I0 = fraunhofer_intensity_1d(a_c, th_c, wavelength_um)
            I = np.power(np.clip(I0, 0.0, None), gm_c)
            
            label = f"True" if idx == idx_true else f"Err={err_pct:+.1f}%"
            waveform_curves.append({
                'label': f"{label} (a={a_c:.2f}, r={r_c:.0f}, γ={gm_c:.2f})",
                'pred_norm': normalize_max(I),
            })

        # 2) landscape envelope: L_min(a, r) = min_gamma loss(a, r, gamma)
        with ProcessPoolExecutor(max_workers=num_cpus) as ex:
            func_min = partial(
                worker_landscape_row_min,
                a_sub=a_sub,
                y_meas_norm=y_meas_norm,
                theta_obs=theta_obs,
                wavelength_um=wavelength_um,
                focal_length_um=focal_length_um,
                gamma_grid=gamma_grid,
            )
            rows_min = list(ex.map(func_min, r_sub))
        L_min = np.vstack(rows_min)

        # multi-panel figure (panel e: iso-contour; panel f: equivalent cloud)
        out_path_no_ext = os.path.join(case_dir, f"Fig_coupling_multipanel__{case_id}")
        save_multifigure(
            out_path_no_ext=out_path_no_ext,
            theta_obs=theta_obs,
            focal_length_um=focal_length_um,
            y_meas_norm=y_meas_norm,
            y_true_plot=y_true_plot,
            waveform_curves=waveform_curves,
            a_grid=a_grid,
            best_losses=best_losses,
            best_r=best_r,
            best_gamma=best_gamma,
            a_true=a_true,
            equiv_mask=equiv_mask,
            L_min=L_min,
            a_sub=a_sub,
            r_sub=r_sub,
            r_true=r_true_um,
        )

        # candidate figure 1: best nuisance tracks along a
        save_candidate_best_nuisance_tracks(
            out_path_no_ext=os.path.join(case_dir, f"Fig_candidate_best_tracks__{case_id}"),
            a_grid=a_grid,
            best_losses=best_losses,
            best_r=best_r,
            best_gamma=best_gamma,
            a_true=a_true,
            equiv_mask=equiv_mask,
        )

        # candidate figure 2: iso-loss contour on (a, r)
        save_candidate_iso_contour(
            out_path_no_ext=os.path.join(case_dir, f"Fig_candidate_iso_contour__{case_id}"),
            L_min=L_min,
            a_sub=a_sub,
            r_sub=r_sub,
            a_true=a_true,
            r_true=r_true_um,
        )

        # candidate figure 3: equivalent-solution cloud
        save_candidate_equiv_scatter(
            out_path_no_ext=os.path.join(case_dir, f"Fig_candidate_equiv_cloud__{case_id}"),
            a_grid=a_grid,
            best_r=best_r,
            best_gamma=best_gamma,
            equiv_mask=equiv_mask,
            a_true=a_true,
        )

        # candidate figure 4: loss slices at selected r
        save_candidate_loss_slices(
            out_path_no_ext=os.path.join(case_dir, f"Fig_candidate_loss_slices__{case_id}"),
            a_sub=a_sub,
            r_sub=r_sub,
            L_min=L_min,
            a_true=a_true,
        )

        summary_rows.append({
            'case_id': case_id,
            'noise': kind,
            'sigma': float(sigma),
            'alpha': float(kwargs.get('alpha', 0.0)),
            'beta': float(kwargs.get('beta', 0.0)),
            'min_loss': float(min_loss),
            'equiv_a_lo': float(equiv_a_lo),
            'equiv_a_hi': float(equiv_a_hi),
            'equiv_width_um': float(equiv_width_um),
            'equiv_width_pct': float(equiv_width_pct),
            'count_equiv_in_1pct': int(count_equiv_in_1pct),
        })

    df = pd.DataFrame(summary_rows)
    df.to_csv(os.path.join(root_out_dir, 'summary_equiv_bands.csv'), index=False)
    print(f"\n[DONE] outputs -> {root_out_dir}")
    print(df)


if __name__ == '__main__':
    run()
