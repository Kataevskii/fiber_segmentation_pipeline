"""
visualize_results.py
====================
Save diagnostic slices, histograms and metric summaries to disk.
"""

import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.colors import ListedColormap


def _random_label_cmap(n: int, seed: int = 42) -> ListedColormap:
    """Generate a random colormap for instance labels (label 0 = black)."""
    rng = np.random.default_rng(seed)
    colors = rng.uniform(0.2, 1.0, size=(n + 1, 3))
    colors[0] = [0, 0, 0]   # background = black
    return ListedColormap(colors)


def save_diagnostic_slices(
    volume: np.ndarray,
    intensity: np.ndarray,
    pred_inst: np.ndarray,
    gt_inst: np.ndarray | None,
    out_dir: str,
    n_slices: int = 6,
    axis: int = 0,
):
    """
    Save multi-panel diagnostic slices comparing raw / intensity / prediction / GT.

    Parameters
    ----------
    volume      : (D,H,W) raw binary or grayscale volume
    intensity   : (D,H,W) predicted intensity field
    pred_inst   : (D,H,W) int -- predicted instance labels
    gt_inst     : (D,H,W) int or None -- GT instance labels (centerlines)
    out_dir     : str -- directory to save PNG files
    n_slices    : int -- number of evenly spaced slices along `axis`
    axis        : int -- 0=Z, 1=Y, 2=X
    """
    os.makedirs(out_dir, exist_ok=True)

    D = volume.shape[axis]
    slice_indices = np.linspace(D * 0.05, D * 0.95, n_slices, dtype=int)

    n_max_label = int(pred_inst.max())
    cmap_pred = _random_label_cmap(n_max_label)
    cmap_gt   = _random_label_cmap(int(gt_inst.max()) if gt_inst is not None else 1)

    for sl_idx in slice_indices:
        fig, axes = plt.subplots(1, 4 if gt_inst is not None else 3,
                                 figsize=(6 * (4 if gt_inst is not None else 3), 6))

        def get_slice(arr):
            if axis == 0: return arr[sl_idx]
            elif axis == 1: return arr[:, sl_idx, :]
            else: return arr[:, :, sl_idx]

        vol_sl  = get_slice(volume).astype(np.float32)
        int_sl  = get_slice(intensity)
        pred_sl = get_slice(pred_inst)

        axes[0].imshow(vol_sl, cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Raw Volume (axis={axis}, sl={sl_idx})', fontsize=10)

        axes[1].imshow(int_sl, cmap='hot', vmin=-0.2, vmax=1.0)
        axes[1].set_title('Predicted Intensity', fontsize=10)

        axes[2].imshow(pred_sl, cmap=cmap_pred,
                       vmin=0, vmax=max(n_max_label, 1), interpolation='nearest')
        axes[2].set_title(f'Predicted Instances ({int(np.max(pred_sl))} here)', fontsize=10)

        if gt_inst is not None:
            gt_sl = get_slice(gt_inst)
            axes[3].imshow(gt_sl, cmap=cmap_gt,
                           vmin=0, vmax=max(int(gt_inst.max()), 1), interpolation='nearest')
            axes[3].set_title('GT Centerlines', fontsize=10)

        for ax in axes:
            ax.axis('off')
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f'slice_ax{axis}_{sl_idx:04d}.png'),
                    dpi=120, bbox_inches='tight')
        plt.close(fig)

    print(f"  [viz] Saved {n_slices} diagnostic slices to {out_dir}", flush=True)


def save_fiber_length_histogram(
    pred_lengths: list[int],
    gt_lengths: list[int] | None,
    out_path: str,
):
    """Bar histogram of fiber lengths (predicted vs GT if available)."""
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.logspace(np.log10(max(1, min(pred_lengths))),
                       np.log10(max(pred_lengths) + 1), 40)

    ax.hist(pred_lengths, bins=bins, color='steelblue', alpha=0.75, label='Predicted')
    if gt_lengths:
        ax.hist(gt_lengths, bins=bins, color='tomato', alpha=0.60, label='Ground Truth')

    ax.set_xscale('log')
    ax.set_xlabel('Fiber length (voxels)', fontsize=12)
    ax.set_ylabel('Count', fontsize=12)
    ax.set_title(f'Fiber Length Distribution  (N_pred={len(pred_lengths)}'
                 + (f', N_gt={len(gt_lengths)}' if gt_lengths else '') + ')')
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    print(f"  [viz] Saved fiber length histogram to {out_path}", flush=True)


def save_metrics_text(metrics: dict, out_path: str):
    """Write metrics dict to a plain-text report file."""
    with open(out_path, 'w') as f:
        f.write("FIBER TOPOLOGY OPTIMIZER -- EVALUATION REPORT\n")
        f.write("=" * 50 + "\n")
        for k, v in metrics.items():
            if isinstance(v, float):
                f.write(f"  {k:<20s}: {v:.4f}\n")
            else:
                f.write(f"  {k:<20s}: {v}\n")
    print(f"  [viz] Saved metrics report to {out_path}", flush=True)
