"""
run_topology_optimization.py
============================
Executes the Global Fiber Topology Optimization Pipeline directly on the full volume:
  1. Memory-Mapped Input Volume Loading (0 GB resident RAM for 25 GB orientation field)
  2. 3D Potential Field Thinning & Spur Pruning
  3. Orientation-Guided Transverse H-Severing
  4. Vectorized Fragment Graph Construction & Durable Endpoint Averaging
  5. Post-H-Sever Direction-Durable Multi-Probe Gap Candidate Search
  6. Global Min-Cost Linear Assignment Matching with Degree-1 & No-Cycle Constraints (Zero False Merges)
  7. Connected-Component Bounded Label Diffusion (Zero bleed across empty voxels)
  8. Short Fiber Filtering (< 5 vx) before Label Diffusion

Modes:
  - 'direct' (default): Runs global topology optimization directly across the entire volume with strict degree-1 matching.
  - 'chunked': Runs overlap consensus stitching across 3D sub-blocks.
"""

import os
import sys
import time
import argparse
import gc
import shutil
from collections import defaultdict
import numpy as np
from scipy.ndimage import gaussian_filter, convolve
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
import tifffile
from tqdm import tqdm

# Ensure package root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from topology_optimizer.step1_sever_h_junctions import sever_h_junctions
from topology_optimizer.step2_bridge_gaps import find_bridge_candidates
from topology_optimizer.step3_build_fragment_graph import build_fragment_graph, FiberFragment
from topology_optimizer.step4_optimize_topology import optimize_topology
from topology_optimizer.step5_diffuse_labels import build_labeled_skeleton, diffuse_labels_voronoi, prune_short_fibers_and_repropagate
from topology_optimizer.clean_border_artifacts import crop_and_separate_border_fibers, separate_broken_fibers
from topology_optimizer.evaluate_against_gt import load_gt_centerlines_from_gad, compute_metrics
from topology_optimizer.visualize_results import save_fiber_length_histogram, export_uint16_tiff


def get_default_volume(target_dir='process_data'):
    """Finds and returns the first .tif, .tiff, or .npy volume file in the specified directory."""
    if os.path.isdir(target_dir):
        valid_exts = ('.tif', '.tiff', '.npy')
        files = [
            os.path.join(target_dir, f).replace('\\', '/')
            for f in sorted(os.listdir(target_dir))
            if f.lower().endswith(valid_exts)
            and not f.lower().endswith(('_intensity.npy', '_orientation.npy', '_skel.npy', '_skeleton.npy', '_instance.npy'))
            and os.path.isfile(os.path.join(target_dir, f))
        ]
        if files:
            return files[0]
    return 'process_data/COLLAGENCROP_003_0000.tif'


def get_default_intensity(target_dir='process_data', volume_path=None):
    """Finds default intensity file from process_data folder first, falling back to outputs."""
    stem = os.path.splitext(os.path.basename(volume_path))[0] if volume_path else ''

    # 1. Search in process_data folder
    if os.path.isdir(target_dir):
        if stem:
            for ext in ('.npy', '.tif', '.tiff'):
                p = os.path.join(target_dir, f"{stem}_intensity{ext}").replace('\\', '/')
                if os.path.exists(p):
                    return p
        for name in ['fiber_intensity.npy', 'intensity.npy']:
            p = os.path.join(target_dir, name).replace('\\', '/')
            if os.path.exists(p):
                return p
        for f in sorted(os.listdir(target_dir)):
            if 'intensity' in f.lower() and f.lower().endswith(('.npy', '.tif', '.tiff')):
                return os.path.join(target_dir, f).replace('\\', '/')

    # 2. Search in outputs folder
    if os.path.isdir('outputs'):
        if stem:
            for ext in ('.npy', '.tif', '.tiff'):
                p = os.path.join('outputs', f"{stem}_intensity{ext}").replace('\\', '/')
                if os.path.exists(p):
                    return p
        for name in ['fiber_intensity.npy', 'dual_collagen_64_intensity.npy', 'dual_collagen_intensity.npy', 'dual_collagen_morpho_intensity.npy']:
            p = os.path.join('outputs', name).replace('\\', '/')
            if os.path.exists(p):
                return p
        for f in sorted(os.listdir('outputs')):
            if 'intensity' in f.lower() and f.lower().endswith(('.npy', '.tif', '.tiff')):
                return os.path.join('outputs', f).replace('\\', '/')

    return os.path.join(target_dir, 'fiber_intensity.npy').replace('\\', '/')


def get_default_orientation(target_dir='process_data', volume_path=None):
    """Finds default orientation file from process_data folder first, falling back to outputs."""
    stem = os.path.splitext(os.path.basename(volume_path))[0] if volume_path else ''

    # 1. Search in process_data folder
    if os.path.isdir(target_dir):
        if stem:
            for ext in ('.npy', '.tif', '.tiff'):
                p = os.path.join(target_dir, f"{stem}_orientation{ext}").replace('\\', '/')
                if os.path.exists(p):
                    return p
        for name in ['fiber_orientation.npy', 'orientation.npy']:
            p = os.path.join(target_dir, name).replace('\\', '/')
            if os.path.exists(p):
                return p
        for f in sorted(os.listdir(target_dir)):
            if 'orientation' in f.lower() and f.lower().endswith(('.npy', '.tif', '.tiff')):
                return os.path.join(target_dir, f).replace('\\', '/')

    # 2. Search in outputs folder
    if os.path.isdir('outputs'):
        if stem:
            for ext in ('.npy', '.tif', '.tiff'):
                p = os.path.join('outputs', f"{stem}_orientation{ext}").replace('\\', '/')
                if os.path.exists(p):
                    return p
        for name in ['fiber_orientation.npy', 'dual_collagen_64_orientation.npy', 'dual_collagen_orientation.npy', 'dual_collagen_morpho_orientation.npy']:
            p = os.path.join('outputs', name).replace('\\', '/')
            if os.path.exists(p):
                return p
        for f in sorted(os.listdir('outputs')):
            if 'orientation' in f.lower() and f.lower().endswith(('.npy', '.tif', '.tiff')):
                return os.path.join('outputs', f).replace('\\', '/')

    return os.path.join(target_dir, 'fiber_orientation.npy').replace('\\', '/')


def load_volume_array(path: str, mmap_mode: str = 'r'):
    """Loads a 3D volume from .npy, .tif, or .tiff format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Volume file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path, mmap_mode=mmap_mode)
    elif ext in ('.tif', '.tiff'):
        vol = tifffile.imread(path)
        if vol.dtype == bool:
            return vol.astype(np.float32)
        elif vol.max() > 1.0:
            return (vol > 127).astype(np.float32)
        else:
            return (vol > 0.5).astype(np.float32)
    else:
        raise ValueError(f"Unsupported volume file format '{ext}' for '{path}'. Supported formats: .npy, .tif, .tiff")


def load_neural_field_array(path: str, mmap_mode: str = 'r'):
    """Loads a neural field array (intensity or orientation) from .npy, .tif, or .tiff format."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Neural field file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext == '.npy':
        return np.load(path, mmap_mode=mmap_mode)
    elif ext in ('.tif', '.tiff'):
        arr = tifffile.imread(path)
        if arr.dtype == np.uint16:
            return arr.astype(np.float32) / 65535.0
        elif arr.dtype == np.uint8:
            return arr.astype(np.float32) / 255.0
        return arr.astype(np.float32)
    else:
        return np.load(path, mmap_mode=mmap_mode)


class UnionFind:
    """Disjoint Set Union (Union-Find) with path compression for fiber identity stitching."""
    def __init__(self):
        self.parent = {}

    def find(self, item):
        if item not in self.parent:
            self.parent[item] = item
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, item1, item2):
        root1 = self.find(item1)
        root2 = self.find(item2)
        if root1 != root2:
            self.parent[root1] = root2


DEFAULT_PARAMS = dict(
    # Skeleton extraction
    core_threshold    = 0.28,
    sigma_prefilter   = 0.60,
    spur_pruning_steps= 0,

    # Step 1: H-junction severing
    max_rung_length      = 14,
    min_neighbor_length  = 30,
    perp_thresh          = 0.50,

    # Step 2: Gap bridging
    max_gap_distance  = 24.0,
    mutual_thresh     = 0.50,
    forward_thresh    = 0.35,
    durable_thresh    = 0.40,
    durable_min_frac  = 0.40,
    n_gap_samples     = 7,

    # Step 3: Fragment graph
    min_fragment_length = 1,

    # Step 5: Diffusion & Post-processing
    min_chain_length  = 5,
    min_fiber_length  = 5,
    fg_threshold      = 0.10,

    # Border cleaning
    cut_border        = 32,
    border_mode       = 'crop',
)


def run_direct_topology_optimization(
    intensity_path: str,
    orientation_path: str,
    volume_path: str,
    out_dir: str,
    gad_path: str = None,
    params: dict = None,
    verbose: bool = True,
):
    """Direct whole-volume topology optimization with global degree-1 linear assignment matching."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    os.makedirs(out_dir, exist_ok=True)
    t_total = time.time()

    print("\n" + "=" * 85, flush=True)
    print(" DIRECT GLOBAL FIBER TOPOLOGY OPTIMIZER (Memory-Mapped, Degree <= 1)", flush=True)
    print("=" * 85, flush=True)

    # 1. Load memory-mapped input volumes (0 GB resident RAM for large .npy)
    intensity = load_neural_field_array(intensity_path, mmap_mode='r')
    orientation = load_neural_field_array(orientation_path, mmap_mode='r')
    volume = load_volume_array(volume_path, mmap_mode='r')

    D, H, W = intensity.shape
    print(f"  Volume dimensions: {D} x {H} x {W}", flush=True)
    print(f"  Intensity range: [{float(intensity.min()):.3f}, {float(intensity.max()):.3f}]", flush=True)

    if orientation.shape == (D, H, W, 3):
        ori_z = orientation[:, :, :, 0]
        ori_y = orientation[:, :, :, 1]
        ori_x = orientation[:, :, :, 2]
    elif orientation.shape == (3, D, H, W):
        ori_z = orientation[0]
        ori_y = orientation[1]
        ori_x = orientation[2]
    else:
        raise ValueError(f"Unexpected orientation shape: {orientation.shape}")

    # -------------------------------------------------------------------------
    # STAGE 1: Medial Axis Thinning & Spur Pruning
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 1: Global Potential Field Thinning & Spur Pruning", flush=True)
    print("=" * 85, flush=True)
    t1 = time.time()

    if params.get('sigma_prefilter', 0.0) > 0:
        print("  Applying Gaussian pre-filter...", flush=True)
        smooth_int = gaussian_filter(intensity.astype(np.float32), sigma=params['sigma_prefilter'])
        binary_core = smooth_int >= params['core_threshold']
        del smooth_int
        gc.collect()
    else:
        binary_core = intensity >= params['core_threshold']

    print(f"  Skeletonizing binary core ({int(binary_core.sum())} voxels)...", flush=True)
    skel = skeletonize(binary_core)
    del binary_core
    gc.collect()

    if params.get('spur_pruning_steps', 0) > 0:
        struct26 = np.ones((3, 3, 3), dtype=np.uint8)
        print(f"  Pruning spurs ({params['spur_pruning_steps']} iterations)...", flush=True)
        for _ in range(params['spur_pruning_steps']):
            skel_u8 = skel.astype(np.uint8)
            convolve(skel_u8, struct26, output=skel_u8, mode='constant', cval=0)
            endpoints = skel & (skel_u8 == 2)
            del skel_u8
            if not endpoints.any():
                del endpoints
                break
            skel[endpoints] = False
            del endpoints
            gc.collect()

    n_skel_vox = int(skel.sum())
    print(f"  Skeleton extracted: {n_skel_vox} voxels in {time.time()-t1:.1f}s", flush=True)

    if n_skel_vox == 0:
        print("  [Warning] Skeleton is empty. Check threshold parameters.", flush=True)
        return {'N_final_fibers': 0, 'Total_pipeline_runtime_sec': float(time.time() - t_total)}

    # Border margin cut (Option 1): Cuts border margin from full skeleton & volumes before
    # graph construction to cleanly sever edge boundary merger loops and prevent hairpins.
    cut_b = params.get('cut_border', 0)
    if cut_b > 0:
        mz = my = mx = cut_b
        out_D = D - 2 * mz
        out_H = H - 2 * my
        out_W = W - 2 * mx
        print(f"\n  Cutting border margin ({cut_b} px) from skeleton & volumes before optimization: ({D}, {H}, {W}) -> ({out_D}, {out_H}, {out_W})", flush=True)
        skel = skel[mz:D-mz, my:H-my, mx:W-mx]
        intensity = intensity[mz:D-mz, my:H-my, mx:W-mx]
        volume = volume[mz:D-mz, my:H-my, mx:W-mx]
        ori_z = ori_z[mz:D-mz, my:H-my, mx:W-mx]
        ori_y = ori_y[mz:D-mz, my:H-my, mx:W-mx]
        ori_x = ori_x[mz:D-mz, my:H-my, mx:W-mx]
        D, H, W = out_D, out_H, out_W
        n_skel_vox = int(skel.sum())
        print(f"  Cropped skeleton: {n_skel_vox} voxels retained in clean inner domain", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 2: Orientation-Guided Transverse H-Severing
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 2: Orientation-Guided Transverse H-Severing", flush=True)
    print("=" * 85, flush=True)
    t2 = time.time()

    clean_skel = sever_h_junctions(
        skeleton=skel,
        ori_z=ori_z, ori_y=ori_y, ori_x=ori_x,
        max_rung_length=params['max_rung_length'],
        min_neighbor_length=params.get('min_neighbor_length', 30),
        perp_thresh=params.get('perp_thresh', 0.50),
        verbose=verbose,
    )
    del skel
    gc.collect()
    print(f"  H-severing complete in {time.time()-t2:.1f}s", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 3: Fragment Graph Construction & Durable Endpoints
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 3: Fragment Graph Construction & Durable Endpoint Averaging", flush=True)
    print("=" * 85, flush=True)
    t3 = time.time()

    fragments, _ = build_fragment_graph(
        skeleton=clean_skel,
        ori_z=ori_z, ori_y=ori_y, ori_x=ori_x,
        min_fragment_length=params['min_fragment_length'],
        verbose=verbose,
        return_seg_labels=False,
    )
    del clean_skel
    gc.collect()
    print(f"  Fragment graph built in {time.time()-t3:.1f}s -- {len(fragments)} fragments", flush=True)

    if not fragments:
        print("  [Warning] No valid fragments found.", flush=True)
        return {'N_final_fibers': 0, 'Total_pipeline_runtime_sec': float(time.time() - t_total)}

    # -------------------------------------------------------------------------
    # STAGE 4: Post-H-Sever Direction-Durable Multi-Probe Gap Candidate Search
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 4: Post-H-Sever Direction-Durable Multi-Probe Gap Candidate Search", flush=True)
    print("=" * 85, flush=True)
    t4 = time.time()

    candidates = find_bridge_candidates(
        fragments=fragments,
        ori_z=ori_z, ori_y=ori_y, ori_x=ori_x,
        max_gap_distance=params['max_gap_distance'],
        mutual_thresh=params['mutual_thresh'],
        forward_thresh=params['forward_thresh'],
        durable_thresh=params['durable_thresh'],
        durable_min_fraction=params['durable_min_frac'],
        n_gap_samples=params['n_gap_samples'],
        verbose=verbose,
    )
    print(f"  Gap candidate search complete in {time.time()-t4:.1f}s -- {len(candidates)} candidate bridges", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 5: Global Min-Cost Topology Optimization of post-H-sever fragments
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 5: Global Min-Cost Topology Optimization (Degree <= 1)", flush=True)
    print("=" * 85, flush=True)
    t5 = time.time()

    chains = optimize_topology(
        fragments=fragments,
        candidates=candidates,
        verbose=verbose,
    )
    print(f"  Topology optimization complete in {time.time()-t5:.1f}s -- {len(chains)} continuous fibers resolved", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 6: Multi-Label Voronoi Label Diffusion (Disk-Backed)
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 6: Multi-Label Voronoi Label Diffusion", flush=True)
    print("=" * 85, flush=True)
    t6 = time.time()

    inst_skel_path = os.path.join(out_dir, 'instance_skeleton.npy')
    inst_vol_path = os.path.join(out_dir, 'instance_volume.npy')

    inst_skel = np.lib.format.open_memmap(inst_skel_path, mode='w+', dtype=np.int32, shape=(D, H, W))
    min_flen = params.get('min_fiber_length', params.get('min_chain_length', 5))
    inst_skel, n_kept_chains = build_labeled_skeleton(
        fragments=fragments,
        chains=chains,
        seg_labels=None,
        volume_shape=(D, H, W),
        min_chain_length=min_flen,
        out_skel_mmap=inst_skel,
        verbose=verbose
    )
    inst_skel.flush()

    inst_vol = np.lib.format.open_memmap(inst_vol_path, mode='w+', dtype=np.int32, shape=(D, H, W))
    diffuse_labels_voronoi(
        inst_skel=inst_skel,
        volume=volume,
        intensity=intensity,
        fg_threshold=params['fg_threshold'],
        out_vol_mmap=inst_vol,
        chunk_size=1000000,
        verbose=verbose
    )
    inst_vol.flush()
    print(f"  Diffusion complete in {time.time()-t6:.1f}s -- {n_kept_chains} continuous fibers resolved in final volume", flush=True)

    # -------------------------------------------------------------------------
    # STAGE 7: Exporting Volumes & Centerlines
    # -------------------------------------------------------------------------
    print("\n" + "=" * 85, flush=True)
    print(" STAGE 7: Exporting uint16 TIFF & Centerlines", flush=True)
    print("=" * 85, flush=True)

    # Export final outputs as uint16 TIFF
    inst_vol_tif_path = os.path.join(out_dir, 'instance_volume.tif')
    inst_skel_tif_path = os.path.join(out_dir, 'instance_skeleton.tif')
    export_uint16_tiff(inst_vol, inst_vol_tif_path, verbose=verbose)
    export_uint16_tiff(inst_skel, inst_skel_tif_path, verbose=verbose)

    unique_ids, counts = np.unique(inst_skel[inst_skel > 0], return_counts=True)
    chain_lengths = [int(c) for c in counts]
    save_fiber_length_histogram(chain_lengths, None, os.path.join(out_dir, 'fiber_length_histogram.png'))

    total_runtime = float(time.time() - t_total)
    metrics = {
        'N_final_fibers': len(chain_lengths),
        'Mean_fiber_length_voxels': float(np.mean(chain_lengths)) if chain_lengths else 0,
        'Max_fiber_length_voxels': float(max(chain_lengths)) if chain_lengths else 0,
        'Total_pipeline_runtime_sec': total_runtime,
        'instance_volume_npy': inst_vol_path,
        'instance_volume_tif': inst_vol_tif_path,
        'instance_skeleton_npy': inst_skel_path,
        'instance_skeleton_tif': inst_skel_tif_path,
    }

    print("\n" + "=" * 85, flush=True)
    print(f" TOPOLOGY OPTIMIZATION COMPLETE in {total_runtime:.1f}s")
    print(f"  - Final Continuous Fibers Resolved:  {len(chain_lengths)}")
    print(f"  - Mean Fiber Length:                 {metrics['Mean_fiber_length_voxels']:.1f} voxels")
    print(f"  - Max Fiber Length:                  {metrics['Max_fiber_length_voxels']:.1f} voxels")
    print(f"  - Instance Volume Array (npy):       {inst_vol_path}")
    print(f"  - Instance Volume TIFF (uint16):     {inst_vol_tif_path}")
    print(f"  - Labeled Skeleton Array (npy):      {inst_skel_path}")
    print(f"  - Labeled Skeleton TIFF (uint16):    {inst_skel_tif_path}")
    print("=" * 85 + "\n", flush=True)

    return metrics


def run_chunked_topology_optimization(
    intensity_path: str,
    orientation_path: str,
    volume_path: str,
    out_dir: str,
    gad_path: str = None,
    params: dict = None,
    chunk_size: int = 512,
    overlap: int = 256,
    min_overlap_ratio: float = 0.80,
    verbose: bool = True,
):
    """Chunked topology optimization with 50% overlap consensus stitching."""
    if params is None:
        params = DEFAULT_PARAMS.copy()

    os.makedirs(out_dir, exist_ok=True)
    tmp_chunk_dir = os.path.join(out_dir, '_chunk_consensus_tmp')
    os.makedirs(tmp_chunk_dir, exist_ok=True)

    t_total = time.time()

    print("\n" + "=" * 85, flush=True)
    print(" CHUNKED FIBER TOPOLOGY OPTIMIZER -- OVERLAP CONSENSUS STITCHING (Memory-Mapped)", flush=True)
    print("=" * 85, flush=True)

    # 1. Load memory-mapped input volumes (0 GB resident RAM for large .npy)
    intensity = load_neural_field_array(intensity_path, mmap_mode='r')
    orientation = load_neural_field_array(orientation_path, mmap_mode='r')
    volume = load_volume_array(volume_path, mmap_mode='r')

    D, H, W = intensity.shape
    print(f"  Volume dimensions: {D} x {H} x {W}", flush=True)
    print(f"  Intensity range: [{float(intensity.min()):.3f}, {float(intensity.max()):.3f}]", flush=True)

    if orientation.shape == (D, H, W, 3):
        ori_is_last = True
    elif orientation.shape == (3, D, H, W):
        ori_is_last = False
    else:
        raise ValueError(f"Unexpected orientation shape: {orientation.shape}")

    # Border margin cut: Slices inputs before chunking to eliminate edge artifacts
    cut_b = params.get('cut_border', 0)
    if cut_b > 0:
        mz = my = mx = cut_b
        out_D = D - 2 * mz
        out_H = H - 2 * my
        out_W = W - 2 * mx
        print(f"  Cutting border margin ({cut_b} px) from inputs before chunking: ({D}, {H}, {W}) -> ({out_D}, {out_H}, {out_W})", flush=True)
        intensity = intensity[mz:D-mz, my:H-my, mx:W-mx]
        volume = volume[mz:D-mz, my:H-my, mx:W-mx]
        if ori_is_last:
            orientation = orientation[mz:D-mz, my:H-my, mx:W-mx, :]
        else:
            orientation = orientation[:, mz:D-mz, my:H-my, mx:W-mx]
        D, H, W = out_D, out_H, out_W

    # Build overlapping chunk grid
    step_z = max(1, chunk_size - overlap)
    step_y = max(1, chunk_size - overlap)
    step_x = max(1, chunk_size - overlap)

    z_starts = list(range(0, max(1, D - overlap), step_z))
    y_starts = list(range(0, max(1, H - overlap), step_y))
    x_starts = list(range(0, max(1, W - overlap), step_x))

    if z_starts[-1] + chunk_size < D:
        z_starts.append(D - chunk_size if D >= chunk_size else 0)
    if y_starts[-1] + chunk_size < H:
        y_starts.append(H - chunk_size if H >= chunk_size else 0)
    if x_starts[-1] + chunk_size < W:
        x_starts.append(W - chunk_size if W >= chunk_size else 0)

    z_starts = sorted(list(set(z_starts)))
    y_starts = sorted(list(set(y_starts)))
    x_starts = sorted(list(set(x_starts)))

    chunk_boxes = []
    for z0 in z_starts:
        z1 = min(D, z0 + chunk_size)
        for y0 in y_starts:
            y1 = min(H, y0 + chunk_size)
            for x0 in x_starts:
                x1 = min(W, x0 + chunk_size)
                chunk_boxes.append((z0, z1, y0, y1, x0, x1))

    n_total_chunks = len(chunk_boxes)
    print(f"  Chunk Grid: {len(z_starts)}x{len(y_starts)}x{len(x_starts)} = {n_total_chunks} chunks ({chunk_size}x{chunk_size}x{chunk_size}) with {overlap}-vx overlap", flush=True)

    # Stage 1: Local optimization
    t1 = time.time()
    struct26 = np.ones((3, 3, 3), dtype=np.uint8)
    active_chunk_indices = []

    pbar = tqdm(enumerate(chunk_boxes), total=n_total_chunks, desc="  Optimizing Chunks", unit="chunk", ncols=95)
    for c_idx, (z0, z1, y0, y1, x0, x1) in pbar:
        sub_int = intensity[z0:z1, y0:y1, x0:x1]
        if np.max(sub_int) < params['core_threshold']:
            continue

        if ori_is_last:
            sub_oz = orientation[z0:z1, y0:y1, x0:x1, 0]
            sub_oy = orientation[z0:z1, y0:y1, x0:x1, 1]
            sub_ox = orientation[z0:z1, y0:y1, x0:x1, 2]
        else:
            sub_oz = orientation[0, z0:z1, y0:y1, x0:x1]
            sub_oy = orientation[1, z0:z1, y0:y1, x0:x1]
            sub_ox = orientation[2, z0:z1, y0:y1, x0:x1]

        sub_vol = volume[z0:z1, y0:y1, x0:x1]

        if params.get('sigma_prefilter', 0.0) > 0:
            smooth_int = gaussian_filter(sub_int.astype(np.float32), sigma=params['sigma_prefilter'])
            binary_core = smooth_int >= params['core_threshold']
            del smooth_int
        else:
            binary_core = sub_int >= params['core_threshold']

        if not np.any(binary_core):
            continue

        sub_skel = skeletonize(binary_core)

        for _ in range(params['spur_pruning_steps']):
            skel_u8 = sub_skel.astype(np.uint8)
            convolve(skel_u8, struct26, output=skel_u8, mode='constant', cval=0)
            endpoints = sub_skel & (skel_u8 == 2)
            del skel_u8
            if not endpoints.any():
                del endpoints
                break
            sub_skel[endpoints] = False
            del endpoints

        sub_clean_skel = sever_h_junctions(
            skeleton=sub_skel,
            ori_z=sub_oz, ori_y=sub_oy, ori_x=sub_ox,
            max_rung_length=params['max_rung_length'],
            min_neighbor_length=params.get('min_neighbor_length', 30),
            perp_thresh=params.get('perp_thresh', 0.50),
            verbose=False
        )

        sub_frags, _ = build_fragment_graph(
            skeleton=sub_clean_skel,
            ori_z=sub_oz, ori_y=sub_oy, ori_x=sub_ox,
            min_fragment_length=params['min_fragment_length'],
            verbose=False
        )

        if not sub_frags:
            continue

        sub_candidates = find_bridge_candidates(
            fragments=sub_frags,
            ori_z=sub_oz, ori_y=sub_oy, ori_x=sub_ox,
            max_gap_distance=params['max_gap_distance'],
            mutual_thresh=params['mutual_thresh'],
            forward_thresh=params['forward_thresh'],
            durable_thresh=params['durable_thresh'],
            durable_min_fraction=params['durable_min_frac'],
            n_gap_samples=params['n_gap_samples'],
            verbose=False
        )

        sub_chains = optimize_topology(
            fragments=sub_frags,
            candidates=sub_candidates,
            verbose=False
        )

        sub_inst_skel, _ = build_labeled_skeleton(
            fragments=sub_frags,
            chains=sub_chains,
            seg_labels=None,
            volume_shape=sub_int.shape,
            min_chain_length=params.get('min_fiber_length', params.get('min_chain_length', 5)),
            verbose=False
        )

        sub_fg = (sub_vol > 0.05) | (sub_int >= params['fg_threshold'])
        sub_inst_vol = diffuse_labels_voronoi(
            inst_skel=sub_inst_skel,
            fg_mask=sub_fg,
            verbose=False
        )

        fg_idx = np.where(sub_inst_vol > 0)
        lbls = sub_inst_vol[fg_idx]
        skel_idx = np.where(sub_inst_skel > 0)
        skel_lbls = sub_inst_skel[skel_idx]

        chunk_file = os.path.join(tmp_chunk_dir, f'chunk_{c_idx}.npz')
        np.savez_compressed(
            chunk_file,
            z=fg_idx[0].astype(np.int16),
            y=fg_idx[1].astype(np.int16),
            x=fg_idx[2].astype(np.int16),
            label=lbls.astype(np.int32),
            skel_z=skel_idx[0].astype(np.int16),
            skel_y=skel_idx[1].astype(np.int16),
            skel_x=skel_idx[2].astype(np.int16),
            skel_label=skel_lbls.astype(np.int32),
            box=np.array([z0, z1, y0, y1, x0, x1], dtype=np.int32)
        )
        active_chunk_indices.append(c_idx)
        pbar.set_postfix({'Active': len(active_chunk_indices), 'Fibers': len(sub_chains)})

    print(f"  Local chunk optimization complete in {time.time()-t1:.1f}s -- {len(active_chunk_indices)} active chunks processed.", flush=True)

    if not active_chunk_indices:
        print("  [Warning] No active fiber chunks resolved.", flush=True)
        shutil.rmtree(tmp_chunk_dir, ignore_errors=True)
        return {'N_final_fibers': 0, 'Total_pipeline_runtime_sec': float(time.time() - t_total)}

    # Stage 2: Consensus matching
    t2 = time.time()
    chunk_meta = {}
    for c_idx in active_chunk_indices:
        chunk_file = os.path.join(tmp_chunk_dir, f'chunk_{c_idx}.npz')
        with np.load(chunk_file) as d:
            chunk_meta[c_idx] = {
                'box': tuple(d['box']),
                'z': d['z'], 'y': d['y'], 'x': d['x'],
                'label': d['label'],
                'skel_z': d['skel_z'], 'skel_y': d['skel_y'], 'skel_x': d['skel_x'],
                'skel_label': d['skel_label']
            }

    chunk_pairs = []
    for i in range(len(active_chunk_indices)):
        cA = active_chunk_indices[i]
        boxA = chunk_meta[cA]['box']
        for j in range(i + 1, len(active_chunk_indices)):
            cB = active_chunk_indices[j]
            boxB = chunk_meta[cB]['box']

            z0_ov = max(boxA[0], boxB[0])
            z1_ov = min(boxA[1], boxB[1])
            y0_ov = max(boxA[2], boxB[2])
            y1_ov = min(boxA[3], boxB[3])
            x0_ov = max(boxA[4], boxB[4])
            x1_ov = min(boxA[5], boxB[5])

            if z0_ov < z1_ov and y0_ov < y1_ov and x0_ov < x1_ov:
                chunk_pairs.append((cA, cB, (z0_ov, z1_ov, y0_ov, y1_ov, x0_ov, x1_ov)))

    uf = UnionFind()
    matches_found = 0

    pbar_ov = tqdm(chunk_pairs, desc="  Matching Overlapping Chunks", unit="pair", ncols=95)
    for (cA, cB, ov_box) in pbar_ov:
        z0_ov, z1_ov, y0_ov, y1_ov, x0_ov, x1_ov = ov_box
        boxA = chunk_meta[cA]['box']
        boxB = chunk_meta[cB]['box']

        dA = chunk_meta[cA]
        gzA = dA['z'] + boxA[0]
        gyA = dA['y'] + boxA[2]
        gxA = dA['x'] + boxA[4]
        in_ovA = (gzA >= z0_ov) & (gzA < z1_ov) & (gyA >= y0_ov) & (gyA < y1_ov) & (gxA >= x0_ov) & (gxA < x1_ov)

        dB = chunk_meta[cB]
        gzB = dB['z'] + boxB[0]
        gyB = dB['y'] + boxB[2]
        gxB = dB['x'] + boxB[4]
        in_ovB = (gzB >= z0_ov) & (gzB < z1_ov) & (gyB >= y0_ov) & (gyB < y1_ov) & (gxB >= x0_ov) & (gxB < x1_ov)

        if not np.any(in_ovA) or not np.any(in_ovB):
            continue

        ov_shape = (z1_ov - z0_ov, y1_ov - y0_ov, x1_ov - x0_ov)
        gridA = np.zeros(ov_shape, dtype=np.int32)
        gridB = np.zeros(ov_shape, dtype=np.int32)

        gridA[gzA[in_ovA] - z0_ov, gyA[in_ovA] - y0_ov, gxA[in_ovA] - x0_ov] = dA['label'][in_ovA]
        gridB[gzB[in_ovB] - z0_ov, gyB[in_ovB] - y0_ov, gxB[in_ovB] - x0_ov] = dB['label'][in_ovB]

        both_fg = (gridA > 0) & (gridB > 0)
        if not np.any(both_fg):
            continue

        pairs, counts = np.unique(np.column_stack([gridA[both_fg], gridB[both_fg]]), return_counts=True, axis=0)

        volsA = np.bincount(gridA.ravel())
        volsB = np.bincount(gridB.ravel())

        for (idA, idB), count in zip(pairs, counts):
            volA = int(volsA[idA])
            volB = int(volsB[idB])
            min_vol = min(volA, volB)
            if min_vol == 0:
                continue

            overlap_ratio = count / min_vol
            if overlap_ratio >= min_overlap_ratio:
                uf.union((cA, int(idA)), (cB, int(idB)))
                matches_found += 1

        pbar_ov.set_postfix({'Matches': matches_found})

    print(f"  Consensus matching complete in {time.time()-t2:.1f}s -- Found {matches_found} matching fiber overlaps.", flush=True)

    # Stage 3: Assembly
    t3 = time.time()
    global_id_map = {}
    next_global_id = 1

    for c_idx in active_chunk_indices:
        for lbl in np.unique(chunk_meta[c_idx]['label']):
            if lbl == 0:
                continue
            root = uf.find((c_idx, int(lbl)))
            if root not in global_id_map:
                global_id_map[root] = next_global_id
                next_global_id += 1

    n_global_fibers = next_global_id - 1
    print(f"  Unified into {n_global_fibers} global continuous fiber instances.", flush=True)

    inst_vol_path = os.path.join(out_dir, 'instance_volume.npy')
    inst_skel_path = os.path.join(out_dir, 'instance_skeleton.npy')

    inst_vol = np.lib.format.open_memmap(inst_vol_path, mode='w+', dtype=np.int32, shape=(D, H, W))
    inst_vol[:] = 0

    inst_skel = np.lib.format.open_memmap(inst_skel_path, mode='w+', dtype=np.int32, shape=(D, H, W))
    inst_skel[:] = 0

    for c_idx in tqdm(active_chunk_indices, desc="  Assembling Final Volume", unit="chunk", ncols=95):
        data = chunk_meta[c_idx]
        box = data['box']

        gz = data['z'] + box[0]
        gy = data['y'] + box[2]
        gx = data['x'] + box[4]
        g_labels = np.array([global_id_map[uf.find((c_idx, int(l)))] for l in data['label']], dtype=np.int32)
        inst_vol[gz, gy, gx] = g_labels

        sgz = data['skel_z'] + box[0]
        sgy = data['skel_y'] + box[2]
        sgx = data['skel_x'] + box[4]
        sg_labels = np.array([global_id_map[uf.find((c_idx, int(l)))] for l in data['skel_label']], dtype=np.int32)
        inst_skel[sgz, sgy, sgx] = sg_labels

    inst_vol.flush()
    inst_skel.flush()

    del chunk_meta
    gc.collect()

    shutil.rmtree(tmp_chunk_dir, ignore_errors=True)


    # Export final outputs as uint16 TIFF
    inst_vol_tif_path = os.path.join(out_dir, 'instance_volume.tif')
    inst_skel_tif_path = os.path.join(out_dir, 'instance_skeleton.tif')
    export_uint16_tiff(inst_vol, inst_vol_tif_path, verbose=verbose)
    export_uint16_tiff(inst_skel, inst_skel_tif_path, verbose=verbose)

    unique_ids, counts = np.unique(inst_skel[inst_skel > 0], return_counts=True)
    chain_lengths = [int(c) for c in counts]

    save_fiber_length_histogram(chain_lengths, None, os.path.join(out_dir, 'fiber_length_histogram.png'))

    total_runtime = float(time.time() - t_total)
    metrics = {
        'N_final_fibers': len(chain_lengths),
        'Mean_fiber_length_voxels': float(np.mean(chain_lengths)) if chain_lengths else 0,
        'Max_fiber_length_voxels': float(max(chain_lengths)) if chain_lengths else 0,
        'Total_pipeline_runtime_sec': total_runtime,
        'instance_volume_npy': inst_vol_path,
        'instance_volume_tif': inst_vol_tif_path,
        'instance_skeleton_npy': inst_skel_path,
        'instance_skeleton_tif': inst_skel_tif_path,
    }

    print("\n" + "=" * 85, flush=True)
    print(f" CHUNKED TOPOLOGY OPTIMIZATION COMPLETE in {total_runtime:.1f}s")
    print(f"  - Final Continuous Fibers Resolved:  {len(chain_lengths)}")
    print(f"  - Mean Fiber Length:                 {metrics['Mean_fiber_length_voxels']:.1f} voxels")
    print(f"  - Max Fiber Length:                  {metrics['Max_fiber_length_voxels']:.1f} voxels")
    print(f"  - Instance Volume Array (npy):       {inst_vol_path}")
    print(f"  - Instance Volume TIFF (uint16):     {inst_vol_tif_path}")
    print(f"  - Labeled Skeleton Array (npy):      {inst_skel_path}")
    print(f"  - Labeled Skeleton TIFF (uint16):    {inst_skel_tif_path}")
    print("=" * 85 + "\n", flush=True)

    return metrics


def run_topology_optimization(
    intensity_path: str = None,
    orientation_path: str = None,
    volume_path: str = None,
    out_dir: str = 'outputs/topology_resolved',
    gad_path: str = None,
    params: dict = None,
    mode: str = 'direct',
    chunk_size: int = 512,
    overlap: int = 256,
    min_overlap_ratio: float = 0.80,
    verbose: bool = True,
):
    if volume_path is None or not os.path.exists(volume_path):
        volume_path = get_default_volume('process_data')

    if intensity_path is None or not os.path.exists(intensity_path):
        intensity_path = get_default_intensity('process_data', volume_path=volume_path)

    if orientation_path is None or not os.path.exists(orientation_path):
        orientation_path = get_default_orientation('process_data', volume_path=volume_path)

    if mode.lower() in ('chunked', 'chunks', 'tile', 'tiled'):
        return run_chunked_topology_optimization(
            intensity_path=intensity_path,
            orientation_path=orientation_path,
            volume_path=volume_path,
            out_dir=out_dir,
            gad_path=gad_path,
            params=params,
            chunk_size=chunk_size,
            overlap=overlap,
            min_overlap_ratio=min_overlap_ratio,
            verbose=verbose
        )
    else:
        return run_direct_topology_optimization(
            intensity_path=intensity_path,
            orientation_path=orientation_path,
            volume_path=volume_path,
            out_dir=out_dir,
            gad_path=gad_path,
            params=params,
            verbose=verbose
        )


if __name__ == '__main__':
    default_vol = get_default_volume('process_data')
    default_int = get_default_intensity('process_data', volume_path=default_vol)
    default_ori = get_default_orientation('process_data', volume_path=default_vol)

    parser = argparse.ArgumentParser(description="Global Fiber Topology Optimization Runner")
    parser.add_argument('--volume', type=str, default=default_vol,
                        help=f"Path to raw/binary volume (.tif, .tiff, .npy) (default: {default_vol})")
    parser.add_argument('--intensity', type=str, default=default_int,
                        help=f"Path to predicted intensity .npy/.tif (default: {default_int})")
    parser.add_argument('--orientation', type=str, default=default_ori,
                        help=f"Path to predicted orientation .npy/.tif (default: {default_ori})")
    parser.add_argument('--out', type=str, default='outputs/topology_resolved',
                        help="Output directory (default: outputs/topology_resolved)")
    parser.add_argument('--mode', type=str, default='direct', choices=['direct', 'chunked'],
                        help="Optimization mode: 'direct' (default, whole-volume degree-1 matching) or 'chunked'")
    parser.add_argument('--chunk-size', type=int, default=512, help="3D cube chunk size for chunked mode (default: 512)")
    parser.add_argument('--overlap', type=int, default=256, help="Overlap margin in voxels for chunked mode (default: 256)")
    parser.add_argument('--sigma-prefilter', type=float, default=0.60, help="Gaussian smoothing sigma before skeletonization (default: 0.60)")
    parser.add_argument('--core-threshold', type=float, default=0.28, help="Potential field threshold for centerline core extraction (default: 0.28)")
    parser.add_argument('--gad', default=None, help="Optional path to GAD ground truth file")
    parser.add_argument('--cut-border', type=int, default=32, help="Margin in voxels to cut from boundary to eliminate loop artifacts (default: 32)")
    parser.add_argument('--border-mode', type=str, default='crop', choices=['crop', 'zero'],
                        help="Border cut mode: 'crop' (default, trims shape) or 'zero' (preserves shape, zeroes boundary)")
    parser.add_argument('--min-fiber-length', type=int, default=5,
                        help="Minimum fiber centerline length in voxels (default: 5). Shorter fibers are filtered before diffusion.")
    args = parser.parse_args()

    params = DEFAULT_PARAMS.copy()
    params['sigma_prefilter'] = args.sigma_prefilter
    params['core_threshold'] = args.core_threshold
    params['cut_border'] = args.cut_border
    params['border_mode'] = args.border_mode
    params['min_fiber_length'] = args.min_fiber_length
    params['min_chain_length'] = args.min_fiber_length

    run_topology_optimization(
        intensity_path=args.intensity,
        orientation_path=args.orientation,
        volume_path=args.volume,
        out_dir=args.out,
        gad_path=args.gad,
        params=params,
        mode=args.mode,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        # min_overlap_ratio=args.min_overlap_ratio
    )
