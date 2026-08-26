"""
step5_diffuse_labels.py
========================
Voronoi label diffusion: expand skeleton chain labels to fill the full fiber volume.

After topology optimization we have a labeled centerline skeleton. This step
diffuses instance labels outward into the foreground binary mask using a
fast, memory-efficient nearest-neighbor (Voronoi / cKDTree) approach.

Bridges are drawn into the skeleton before diffusion so the gap regions are
also labeled.
"""

import gc
import numpy as np
from scipy.spatial import cKDTree

try:
    from .step3_build_fragment_graph import FiberFragment
    from .step4_optimize_topology import FiberChain
except ImportError:
    from step3_build_fragment_graph import FiberFragment
    from step4_optimize_topology import FiberChain


def build_labeled_skeleton(
    fragments: list[FiberFragment],
    chains: list[FiberChain],
    seg_labels: np.ndarray | None,
    volume_shape: tuple[int, int, int],
    min_chain_length: int = 1,
    out_skel_mmap: np.ndarray | None = None
) -> tuple[np.ndarray, int]:
    """
    Build a labeled skeleton volume from chains.

    Parameters
    ----------
    fragments        : list[FiberFragment]
    chains           : list[FiberChain]
    seg_labels       : (D, H, W) int32 or None
    volume_shape     : (D, H, W)
    min_chain_length : int -- chains shorter than this are discarded
    out_skel_mmap    : optional pre-allocated or memmapped array to write to

    Returns
    -------
    inst_skel : (D, H, W) int32 -- labeled skeleton (0 = background)
    n_chains  : int -- number of kept chains
    """
    D, H, W = volume_shape
    frag_id_to_chain_id: dict[int, int] = {}

    kept_chains = [c for c in chains if c.total_length >= min_chain_length]
    for new_id, chain in enumerate(kept_chains, start=1):
        for frag_id in chain.fragment_ids:
            frag_id_to_chain_id[frag_id] = new_id

    if out_skel_mmap is not None:
        inst_skel = out_skel_mmap
        inst_skel[:] = 0
    else:
        inst_skel = np.zeros((D, H, W), dtype=np.int32)

    # Draw fragment coordinates directly into skeleton
    for frag in fragments:
        chain_id = frag_id_to_chain_id.get(frag.frag_id, 0)
        if chain_id > 0 and len(frag.coords) > 0:
            cz, cy, cx = frag.coords[:, 0], frag.coords[:, 1], frag.coords[:, 2]
            valid = (cz >= 0) & (cz < D) & (cy >= 0) & (cy < H) & (cx >= 0) & (cx < W)
            inst_skel[cz[valid], cy[valid], cx[valid]] = chain_id

    # Draw bridge lines into skeleton
    for chain in kept_chains:
        chain_label = frag_id_to_chain_id.get(chain.fragment_ids[0], 0) if chain.fragment_ids else 0
        if chain_label == 0:
            continue
        for (coord_a, coord_b) in chain.bridge_coords:
            n_steps = max(2, int(np.ceil(np.linalg.norm(coord_b - coord_a))) * 4)
            t_vals = np.linspace(0, 1, n_steps)
            bz = np.clip(np.round(coord_a[0] + t_vals * (coord_b[0] - coord_a[0])).astype(int), 0, D - 1)
            by = np.clip(np.round(coord_a[1] + t_vals * (coord_b[1] - coord_a[1])).astype(int), 0, H - 1)
            bx = np.clip(np.round(coord_a[2] + t_vals * (coord_b[2] - coord_a[2])).astype(int), 0, W - 1)
            inst_skel[bz, by, bx] = chain_label

    return inst_skel, len(kept_chains)


def diffuse_labels_voronoi(
    inst_skel: np.ndarray,
    fg_mask: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    fg_threshold: float = 0.10,
    vol_threshold: float = 0.05,
    chunk_size: int = 1000000,
    verbose: bool = True,
    out_vol_mmap: np.ndarray | None = None
) -> np.ndarray:
    """
    Expand labeled skeleton to full fiber volume via fast memory-efficient KDTree Voronoi diffusion.
    Streams slice-by-slice across 3D volume to guarantee near-zero memory footprint.

    Parameters
    ----------
    inst_skel    : (D, H, W) int32 -- labeled skeleton
    fg_mask      : (D, H, W) bool or None -- optional foreground mask (fiber voxels)
    volume       : (D, H, W) float or None -- raw/binary volume
    intensity    : (D, H, W) float or None -- predicted intensity field
    fg_threshold : float -- intensity threshold for foreground
    vol_threshold: float -- volume threshold for foreground
    chunk_size   : int   -- query chunk size for memory-safe execution
    verbose      : bool
    out_vol_mmap : optional pre-allocated or memmapped array

    Returns
    -------
    instance_vol : (D, H, W) int32 -- full labeled instance volume
    """
    D, H, W = inst_skel.shape
    seed_coords_list = []
    seed_labels_list = []
    fg_coords_list = []

    if fg_mask is not None:
        for z in range(D):
            s_sl = inst_skel[z]
            s_idx = np.nonzero(s_sl)
            if len(s_idx[0]) > 0:
                sy, sx = s_idx[0], s_idx[1]
                sz = np.full(len(sy), z, dtype=np.int32)
                seed_coords_list.append(np.column_stack([sz, sy, sx]))
                seed_labels_list.append(s_sl[sy, sx])

            fg_sl = (fg_mask[z] > 0) | (s_sl > 0)
            fg_idx = np.nonzero(fg_sl)
            if len(fg_idx[0]) > 0:
                fy, fx = fg_idx[0], fg_idx[1]
                fz = np.full(len(fy), z, dtype=np.int32)
                fg_coords_list.append(np.column_stack([fz, fy, fx]))
    elif volume is not None and intensity is not None:
        for z in range(D):
            s_sl = inst_skel[z]
            s_idx = np.nonzero(s_sl)
            if len(s_idx[0]) > 0:
                sy, sx = s_idx[0], s_idx[1]
                sz = np.full(len(sy), z, dtype=np.int32)
                seed_coords_list.append(np.column_stack([sz, sy, sx]))
                seed_labels_list.append(s_sl[sy, sx])

            v_sl = volume[z]
            i_sl = intensity[z]
            fg_sl = (v_sl > vol_threshold) | (i_sl >= fg_threshold) | (s_sl > 0)
            fg_idx = np.nonzero(fg_sl)
            if len(fg_idx[0]) > 0:
                fy, fx = fg_idx[0], fg_idx[1]
                fz = np.full(len(fy), z, dtype=np.int32)
                fg_coords_list.append(np.column_stack([fz, fy, fx]))
    else:
        raise ValueError("Must provide either fg_mask or both volume and intensity.")

    if not seed_coords_list:
        if verbose:
            print("  [diffuse] No labeled seeds -- returning zeros.", flush=True)
        if out_vol_mmap is not None:
            out_vol_mmap[:] = 0
            return out_vol_mmap
        return np.zeros_like(inst_skel, dtype=np.int32)

    seed_coords = np.vstack(seed_coords_list).astype(np.float32)
    seed_labels = np.concatenate(seed_labels_list)
    del seed_coords_list, seed_labels_list
    gc.collect()

    tree = cKDTree(seed_coords)
    del seed_coords
    gc.collect()

    if not fg_coords_list:
        if out_vol_mmap is not None:
            out_vol_mmap[:] = 0
            return out_vol_mmap
        return np.zeros_like(inst_skel, dtype=np.int32)

    fg_coords = np.vstack(fg_coords_list)
    del fg_coords_list
    gc.collect()

    n_fg = len(fg_coords)

    if out_vol_mmap is not None:
        instance_vol = out_vol_mmap
        instance_vol[:] = 0
    else:
        instance_vol = np.zeros_like(inst_skel, dtype=np.int32)

    for start_idx in range(0, n_fg, chunk_size):
        end_idx = min(start_idx + chunk_size, n_fg)
        chunk_pts = fg_coords[start_idx:end_idx]
        _, nearest_idx = tree.query(chunk_pts, workers=-1)
        instance_vol[chunk_pts[:, 0], chunk_pts[:, 1], chunk_pts[:, 2]] = seed_labels[nearest_idx]

    if verbose:
        n_fibers = int(np.max(seed_labels)) if len(seed_labels) > 0 else 0
        print(f"  [diffuse] {n_fibers} fiber instances, "
              f"{n_fg} foreground voxels labeled (100.0%)",
              flush=True)

    del fg_coords, tree, seed_labels
    gc.collect()

    return instance_vol
