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
    frag_id_to_frag: dict[int, FiberFragment] = {f.frag_id: f for f in fragments}

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
    fg_mask: np.ndarray,
    chunk_size: int = 500000,
    verbose: bool = True,
    out_vol_mmap: np.ndarray | None = None
) -> np.ndarray:
    """
    Expand labeled skeleton to full fiber volume via fast memory-efficient KDTree Voronoi diffusion.

    Parameters
    ----------
    inst_skel    : (D, H, W) int32 -- labeled skeleton
    fg_mask      : (D, H, W) bool  -- foreground mask (fiber voxels)
    chunk_size   : int             -- query chunk size for memory-safe execution
    out_vol_mmap : optional pre-allocated or memmapped array

    Returns
    -------
    instance_vol : (D, H, W) int32 -- full labeled instance volume
    """
    seed_mask = inst_skel > 0
    fg_mask = fg_mask | seed_mask   # ensure skeleton voxels are always in foreground

    if not np.any(seed_mask):
        if verbose:
            print("  [diffuse] No labeled seeds -- returning zeros.", flush=True)
        return np.zeros_like(inst_skel, dtype=np.int32)

    seed_coords = np.argwhere(seed_mask)
    seed_labels = inst_skel[seed_coords[:, 0], seed_coords[:, 1], seed_coords[:, 2]]
    tree = cKDTree(seed_coords)

    fg_coords = np.argwhere(fg_mask)
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
        n_fibers = int(np.max(instance_vol))
        n_labeled = int((instance_vol > 0).sum())
        print(f"  [diffuse] {n_fibers} fiber instances, "
              f"{n_labeled}/{n_fg} foreground voxels labeled ({100*n_labeled/max(n_fg,1):.1f}%)",
              flush=True)

    return instance_vol
