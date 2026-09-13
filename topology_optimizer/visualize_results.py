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
    n_colors = max(2, min(n + 1, 4096))
    colors = rng.uniform(0.2, 1.0, size=(n_colors, 3))
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
    n_max_label: int | None = None,
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
    n_max_label : int or None -- maximum label ID for colormap sizing
    """
    os.makedirs(out_dir, exist_ok=True)

    D = volume.shape[axis]
    slice_indices = np.linspace(D * 0.05, D * 0.95, n_slices, dtype=int)

    if n_max_label is None:
        n_max_label = 1
        for sl_idx in slice_indices:
            sl = pred_inst[sl_idx] if axis == 0 else (pred_inst[:, sl_idx, :] if axis == 1 else pred_inst[:, :, sl_idx])
            n_max_label = max(n_max_label, int(np.max(sl)))

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


def export_uint16_tiff(arr: np.ndarray, out_path: str, verbose: bool = True):
    """
    Save 3D array as uint16 TIFF with compression (supports memory-mapped and large volumes).
    Uses chunked / slice-by-slice writing for volumes > 1GB to ensure low peak RAM usage.
    """
    import tifffile
    import time
    t0 = time.time()

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    D = arr.shape[0]
    total_bytes = arr.size * 2
    is_big = total_bytes > (2 * 1024 * 1024 * 1024)

    axes = 'ZYX' if arr.ndim == 3 else ('YX' if arr.ndim == 2 else None)
    meta = {'axes': axes} if axes else None

    if total_bytes < 1024 * 1024 * 1024 and not isinstance(arr, np.memmap):
        # Fast in-memory write for arrays < 1 GB (and not memory-mapped)
        u16_arr = np.asarray(arr, dtype=np.uint16)
        tifffile.imwrite(
            out_path,
            u16_arr,
            compression='zlib',
            metadata=meta,
            bigtiff=is_big,
        )
    elif arr.ndim >= 3:
        # Stream slice-by-slice for huge or memory-mapped volumes to keep resident RAM near zero
        def _slice_gen():
            for z in range(D):
                yield np.asarray(arr[z], dtype=np.uint16)

        tifffile.imwrite(
            out_path,
            _slice_gen(),
            shape=arr.shape,
            dtype=np.uint16,
            compression='zlib',
            metadata=meta,
            bigtiff=True,
        )
    else:
        u16_arr = np.asarray(arr, dtype=np.uint16)
        tifffile.imwrite(
            out_path,
            u16_arr,
            compression='zlib',
            metadata=meta,
            bigtiff=True,
        )

    if verbose:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  [export] Saved uint16 TIFF: {out_path} ({size_mb:.1f} MB in {time.time()-t0:.1f}s)", flush=True)
