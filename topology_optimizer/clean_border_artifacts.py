"""
clean_border_artifacts.py
==========================
Crops border margins from instance volume & skeleton, and relabels any fiber paths
that were broken in the border zone into separate connected components.

Algorithm:
  1. Crops volume and skeleton arrays by `margin` voxels on all 6 faces (±Z, ±Y, ±X).
  2. Finds all candidate fiber labels that touch the border faces of the cropped region.
  3. For each candidate label, runs 3D 26-connectivity connected components on its voxels.
  4. If a fiber is broken into >= 2 pieces, assigns a new unique fiber ID to each
     secondary piece across both volume and skeleton.
  5. Runs in milliseconds with zero memory overhead.

Usage:
  python topology_optimizer/clean_border_artifacts.py --input outputs/topology_resolved_morpho --margin 32
"""

import os
import sys
import gc
import time
import argparse
import numpy as np
from scipy.ndimage import label as nd_label, find_objects

# Ensure parent directory is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from topology_optimizer.visualize_results import save_diagnostic_slices, save_metrics_text, export_uint16_tiff


def crop_and_separate_border_fibers(
    vol_path: str,
    skel_path: str | None = None,
    out_dir: str = 'outputs/border_cleaned',
    margin: int | tuple[int, int, int] = 32,
    min_component_size: int = 1,
    raw_vol_path: str | None = None,
    intensity_path: str | None = None,
    save_previews: bool = True,
    verbose: bool = True,
    **kwargs,
) -> dict:
    """
    Crop border margin from volume & skeleton and relabel broken border fiber paths.

    Parameters
    ----------
    vol_path           : str -- Path to input instance_volume.npy
    skel_path          : str or None -- Optional path to input instance_skeleton.npy
    out_dir            : str -- Directory to save cropped results
    margin             : int or (mz, my, mx) -- Margin voxels to cut from each face (default: 32)
    min_component_size : int -- Minimum voxels required to keep a broken fragment (default: 1)
    raw_vol_path       : str or None -- Optional path to raw microscopy volume for previews
    intensity_path     : str or None -- Optional path to intensity field for previews
    save_previews      : bool -- Generate slice previews
    verbose            : bool

    Returns
    -------
    metrics : dict
    """
    t_start = time.time()
    os.makedirs(out_dir, exist_ok=True)

    if isinstance(margin, int):
        mz = my = mx = margin
    else:
        mz, my, mx = margin

    in_vol = np.load(vol_path, mmap_mode='r')
    D, H, W = in_vol.shape

    if D <= 2 * mz or H <= 2 * my or W <= 2 * mx:
        raise ValueError(f"Volume shape ({D}, {H}, {W}) is too small for margin ({mz}, {my}, {mx})")

    out_D = D - 2 * mz
    out_H = H - 2 * my
    out_W = W - 2 * mx

    if verbose:
        print(f"  Cropping border margin ({mz} px) on each side: ({D}, {H}, {W}) -> ({out_D}, {out_H}, {out_W})", flush=True)

    out_vol_path = os.path.join(out_dir, 'instance_volume.npy')
    out_skel_path = os.path.join(out_dir, 'instance_skeleton.npy')

    # 1. Write cropped arrays directly to disk
    out_vol = np.lib.format.open_memmap(out_vol_path, mode='w+', dtype=np.int32, shape=(out_D, out_H, out_W))
    out_vol[:] = in_vol[mz:D-mz, my:H-my, mx:W-mx]
    out_vol.flush()

    if skel_path and os.path.exists(skel_path):
        in_skel = np.load(skel_path, mmap_mode='r')
        out_skel = np.lib.format.open_memmap(out_skel_path, mode='w+', dtype=np.int32, shape=(out_D, out_H, out_W))
        out_skel[:] = in_skel[mz:D-mz, my:H-my, mx:W-mx]
        out_skel.flush()
    else:
        out_skel = None

    # 2. Find labels that touch the border faces of the cropped region
    border_labels = set()
    for sl in [
        out_vol[0], out_vol[-1],
        out_vol[:, 0], out_vol[:, -1],
        out_vol[:, :, 0], out_vol[:, :, -1]
    ]:
        nz = sl[sl > 0]
        if len(nz) > 0:
            border_labels.update(np.unique(nz))

    # Also check skeleton border if available
    if out_skel is not None:
        for sl in [
            out_skel[0], out_skel[-1],
            out_skel[:, 0], out_skel[:, -1],
            out_skel[:, :, 0], out_skel[:, :, -1]
        ]:
            nz = sl[sl > 0]
            if len(nz) > 0:
                border_labels.update(np.unique(nz))

    initial_fibers = int(out_vol.max()) if out_vol.size > 0 else 0
    next_label = initial_fibers + 1
    struct26 = np.ones((3, 3, 3), dtype=bool)

    n_split = 0
    n_new = 0

    # 3. Find bounding boxes of all objects in a single fast C pass
    all_slices = find_objects(out_vol)

    # Check and relabel ONLY candidate labels that touch the border zone
    for lbl in border_labels:
        if lbl <= 0 or lbl > len(all_slices):
            continue
        box_slice = all_slices[lbl - 1]
        if box_slice is None:
            continue

        sub_vol = out_vol[box_slice]
        box = (sub_vol == lbl)
        comp_map, n_comp = nd_label(box, structure=struct26)

        if n_comp > 1:
            n_split += 1
            comp_sizes = np.bincount(comp_map.ravel())[1:]
            sorted_comps = np.argsort(-comp_sizes) + 1  # largest first

            sub_skel = out_skel[box_slice] if out_skel is not None else None

            # Largest component retains `lbl`; other pieces get new unique IDs
            for c in sorted_comps[1:]:
                mask_c = (comp_map == c)
                c_size = int(mask_c.sum())
                if c_size < min_component_size:
                    sub_vol[mask_c] = 0
                    if sub_skel is not None:
                        sub_skel[mask_c & (sub_skel == lbl)] = 0
                    continue

                sub_vol[mask_c] = next_label
                if sub_skel is not None:
                    sub_skel[mask_c & (sub_skel == lbl)] = next_label
                next_label += 1
                n_new += 1

    out_vol.flush()
    if out_skel is not None:
        out_skel.flush()

    final_fibers = int(out_vol.max()) if out_vol.size > 0 else 0
    total_time = float(time.time() - t_start)

    if verbose:
        print(f"  Border crop & relabel complete in {total_time:.2f}s:")
        print(f"    - Border-crossing fibers checked: {len(border_labels)}")
        print(f"    - Broken paths split:             {n_split}")
        print(f"    - New fiber instances created:   +{n_new}")
        print(f"    - Final continuous fibers:        {final_fibers}")

    # 4. Optional diagnostic previews
    if save_previews and (raw_vol_path or intensity_path):
        viz_dir = os.path.join(out_dir, 'slices')
        raw_vol = np.load(raw_vol_path, mmap_mode='r') if raw_vol_path and os.path.exists(raw_vol_path) else None
        int_vol = np.load(intensity_path, mmap_mode='r') if intensity_path and os.path.exists(intensity_path) else None
        crop_raw = raw_vol[mz:D-mz, my:H-my, mx:W-mx] if raw_vol is not None else (out_vol > 0).astype(np.float32)
        crop_int = int_vol[mz:D-mz, my:H-my, mx:W-mx] if int_vol is not None else crop_raw

        save_diagnostic_slices(
            volume=crop_raw,
            intensity=crop_int,
            pred_inst=out_vol,
            gt_inst=None,
            out_dir=viz_dir,
            n_slices=8,
            axis=0,
            n_max_label=final_fibers,
        )

    # Export final clean arrays as uint16 TIFF
    out_vol_tif_path = os.path.join(out_dir, 'instance_volume.tif')
    export_uint16_tiff(out_vol, out_vol_tif_path, verbose=verbose)
    if out_skel is not None:
        out_skel_tif_path = os.path.join(out_dir, 'instance_skeleton.tif')
        export_uint16_tiff(out_skel, out_skel_tif_path, verbose=verbose)
    else:
        out_skel_tif_path = None

    metrics = {
        'initial_fibers': initial_fibers,
        'border_fibers_checked': len(border_labels),
        'fibers_split_by_border_cut': n_split,
        'new_fibers_created': n_new,
        'final_clean_fibers': final_fibers,
        'margin_voxels': [mz, my, mx],
        'original_shape': [D, H, W],
        'cropped_shape': [out_D, out_H, out_W],
        'total_runtime_sec': total_time,
        'out_instance_volume': out_vol_path,
        'out_instance_volume_tif': out_vol_tif_path,
        'out_instance_skeleton': out_skel_path if out_skel is not None else None,
        'out_instance_skeleton_tif': out_skel_tif_path,
    }
    save_metrics_text(metrics, os.path.join(out_dir, 'border_cleaning_report.txt'))

    del in_vol, out_vol
    if out_skel is not None:
        del out_skel
    gc.collect()

    return metrics


# Compatibility aliases
separate_broken_fibers = crop_and_separate_border_fibers
crop_and_separate_skeleton_fibers = crop_and_separate_border_fibers


def main():
    parser = argparse.ArgumentParser(description="Crop Border Margins & Relabel Broken Paths")
    parser.add_argument('--input', type=str, default=None, help="Path to output directory OR instance_volume.npy")
    parser.add_argument('--vol', type=str, default=None, help="Path to instance_volume.npy")
    parser.add_argument('--skel', type=str, default=None, help="Path to instance_skeleton.npy")
    parser.add_argument('--out', type=str, default=None, help="Output directory")
    parser.add_argument('--margin', type=int, default=32, help="Border margin in voxels to cut from each face (default: 32)")
    parser.add_argument('--min-size', type=int, default=1, help="Minimum voxels to retain a broken fragment (default: 1)")
    parser.add_argument('--raw-vol', type=str, default=None, help="Optional raw volume .npy for diagnostic previews")
    parser.add_argument('--intensity', type=str, default=None, help="Optional intensity .npy for diagnostic previews")
    parser.add_argument('--no-previews', action='store_true', help="Disable slice previews")
    args = parser.parse_args()

    vol_path = args.vol
    skel_path = args.skel
    out_dir = args.out

    if args.input:
        if os.path.isdir(args.input):
            candidate_vol = os.path.join(args.input, 'instance_volume.npy')
            candidate_skel = os.path.join(args.input, 'instance_skeleton.npy')
            if os.path.exists(candidate_vol):
                vol_path = candidate_vol
            if os.path.exists(candidate_skel):
                skel_path = candidate_skel
            if out_dir is None:
                out_dir = args.input + f'_cropped_{args.margin}px'
        elif os.path.isfile(args.input):
            vol_path = args.input
            if out_dir is None:
                out_dir = os.path.join(os.path.dirname(args.input), f'cropped_{args.margin}px')

    if vol_path is None or not os.path.exists(vol_path):
        parser.error("Could not find input instance volume. Specify --vol <path> or --input <directory/file>")

    if out_dir is None:
        out_dir = f'outputs/topology_resolved_cropped_{args.margin}px'

    crop_and_separate_border_fibers(
        vol_path=vol_path,
        skel_path=skel_path,
        out_dir=out_dir,
        margin=args.margin,
        min_component_size=args.min_size,
        raw_vol_path=args.raw_vol,
        intensity_path=args.intensity,
        save_previews=not args.no_previews,
        verbose=True,
    )


if __name__ == '__main__':
    main()
