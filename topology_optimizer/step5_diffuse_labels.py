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
from collections import defaultdict
import numpy as np
from scipy.ndimage import label as nd_label, find_objects
from scipy.spatial import cKDTree
from skimage.draw import line_nd
from skimage.segmentation import watershed

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
    min_chain_length: int = 5,
    out_skel_mmap: np.ndarray | None = None,
    verbose: bool = True
) -> tuple[np.ndarray, int]:
    """
    Build a labeled skeleton volume from chains, removing short fibers (< min_chain_length).

    Parameters
    ----------
    fragments        : list[FiberFragment]
    chains           : list[FiberChain]
    seg_labels       : (D, H, W) int32 or None
    volume_shape     : (D, H, W)
    min_chain_length : int -- chains shorter than this are discarded (default: 5)
    out_skel_mmap    : optional pre-allocated or memmapped array to write to
    verbose          : bool

    Returns
    -------
    inst_skel : (D, H, W) int32 -- labeled skeleton (0 = background)
    n_chains  : int -- number of kept chains
    """
    D, H, W = volume_shape
    frag_id_to_chain_id: dict[int, int] = {}

    kept_chains = [c for c in chains if c.total_length >= min_chain_length]
    n_removed = len(chains) - len(kept_chains)
    if verbose:
        if n_removed > 0:
            print(f"  [skeleton] Filtered {n_removed} short fiber chains (< {min_chain_length} vx) -> {len(kept_chains)} continuous fibers retained for diffusion.", flush=True)
        else:
            print(f"  [skeleton] All {len(kept_chains)} fiber chains kept (len >= {min_chain_length} vx).", flush=True)

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
            pt_a = np.clip(np.round(np.asarray(coord_a)).astype(int), 0, [D - 1, H - 1, W - 1])
            pt_b = np.clip(np.round(np.asarray(coord_b)).astype(int), 0, [D - 1, H - 1, W - 1])
            line_pts = line_nd(pt_a, pt_b, endpoint=True)
            inst_skel[line_pts] = chain_label

    return inst_skel, len(kept_chains)


def diffuse_labels_voronoi(
    inst_skel: np.ndarray,
    fg_mask: np.ndarray | None = None,
    volume: np.ndarray | None = None,
    intensity: np.ndarray | None = None,
    fg_threshold: float = 0.10,
    vol_threshold: float = 0.05,
    chunk_size: int = 1000000,
    max_watershed_voxels: int = 2000000,
    verbose: bool = True,
    out_vol_mmap: np.ndarray | None = None
) -> np.ndarray:
    """
    Expand labeled skeleton to full fiber volume strictly through connected components.
    Guarantees color is NEVER diffused across empty background voxels or into unseeded components.
    Uses memory-efficient slice streaming and connected-component LUT mapping to prevent OOM
    on massive 3D volumes.

    Parameters
    ----------
    inst_skel    : (D, H, W) int32 -- labeled skeleton
    fg_mask      : (D, H, W) bool or None -- optional foreground mask (fiber voxels)
    volume       : (D, H, W) float or None -- raw/binary volume
    intensity    : (D, H, W) float or None -- predicted intensity field
    fg_threshold : float -- intensity threshold for foreground
    vol_threshold: float -- volume threshold for foreground
    chunk_size   : int   -- query chunk size
    verbose      : bool
    out_vol_mmap : optional pre-allocated or memmapped array

    Returns
    -------
    instance_vol : (D, H, W) int32 -- full labeled instance volume
    """
    D, H, W = inst_skel.shape

    # 1. Build foreground binary mask strictly from fiber voxels (slice-by-slice to prevent multi-GB RAM spikes)
    full_fg = np.zeros((D, H, W), dtype=bool)
    if fg_mask is not None:
        for z in range(D):
            full_fg[z] = (fg_mask[z] > 0) | (inst_skel[z] > 0)
    elif volume is not None and intensity is not None:
        for z in range(D):
            full_fg[z] = (volume[z] > vol_threshold) | (intensity[z] >= fg_threshold) | (inst_skel[z] > 0)
    elif volume is not None:
        for z in range(D):
            full_fg[z] = (volume[z] > vol_threshold) | (inst_skel[z] > 0)
    elif intensity is not None:
        for z in range(D):
            full_fg[z] = (intensity[z] >= fg_threshold) | (inst_skel[z] > 0)
    else:
        for z in range(D):
            full_fg[z] = (inst_skel[z] > 0)

    # Collect skeleton seeds slice-by-slice to avoid huge boolean index allocations
    seed_coords_list = []
    seed_labels_list = []
    for z in range(D):
        s_sl = inst_skel[z]
        s_idx = np.nonzero(s_sl)
        if len(s_idx[0]) > 0:
            sy, sx = s_idx[0], s_idx[1]
            sz = np.full(len(sy), z, dtype=np.int32)
            seed_coords_list.append(np.column_stack([sz, sy, sx]))
            seed_labels_list.append(s_sl[sy, sx])

    if not seed_coords_list:
        if verbose:
            print("  [diffuse] No labeled seeds -- returning zeros.", flush=True)
        if out_vol_mmap is not None:
            out_vol_mmap[:] = 0
            return out_vol_mmap
        return np.zeros_like(inst_skel, dtype=np.int32)

    skel_coords = np.vstack(seed_coords_list)
    skel_labels = np.concatenate(seed_labels_list)
    del seed_coords_list, seed_labels_list
    gc.collect()

    skel_z = skel_coords[:, 0]
    skel_y = skel_coords[:, 1]
    skel_x = skel_coords[:, 2]

    # 2. Extract 3D Connected Components on foreground
    if verbose:
        print("  [diffuse] Computing 3D connected components on foreground mask...", flush=True)
    cc_labels, num_cc = nd_label(full_fg, structure=np.ones((3, 3, 3), dtype=bool))
    if num_cc == 0:
        if out_vol_mmap is not None:
            out_vol_mmap[:] = 0
            return out_vol_mmap
        return np.zeros_like(inst_skel, dtype=np.int32)

    # full_fg is no longer needed since cc_labels > 0 defines foreground voxels
    del full_fg
    gc.collect()

    # 3. Associate skeleton seeds with their containing connected components
    skel_ccs = cc_labels[skel_z, skel_y, skel_x]

    comp_seed_indices = defaultdict(list)
    for i, c_id in enumerate(skel_ccs):
        if c_id > 0:
            comp_seed_indices[c_id].append(i)

    # 4. Classify connected components: single-fiber vs multi-fiber
    single_fiber_comps = {}
    multi_fiber_comps = {}

    for c_id, s_idxs in comp_seed_indices.items():
        s_lbls = skel_labels[s_idxs]
        unique_lbls = np.unique(s_lbls)
        if len(unique_lbls) == 1:
            single_fiber_comps[c_id] = unique_lbls[0]
        else:
            multi_fiber_comps[c_id] = (s_idxs, unique_lbls)

    if verbose:
        n_unseeded = num_cc - len(comp_seed_indices)
        print(f"  [diffuse] {num_cc} connected components detected: "
              f"{len(single_fiber_comps)} single-fiber, {len(multi_fiber_comps)} multi-fiber, "
              f"{n_unseeded} unseeded (ignored).", flush=True)

    # 5. Build lookup table for single-fiber components (0 for unseeded/multi-fiber)
    lut = np.zeros(num_cc + 1, dtype=np.int32)
    for c_id, lbl in single_fiber_comps.items():
        lut[c_id] = lbl

    if out_vol_mmap is not None:
        instance_vol = out_vol_mmap
        instance_vol[:] = 0
    else:
        instance_vol = np.zeros_like(inst_skel, dtype=np.int32)

    # 6. Multi-fiber components handling
    objs = find_objects(cc_labels) if multi_fiber_comps else None
    kdtree_comps = {}
    any_watershed_done = False

    for c_id, (s_idxs, unique_lbls) in multi_fiber_comps.items():
        sl = objs[c_id - 1] if objs is not None and c_id <= len(objs) else None
        z_slice = sl[0] if sl is not None else slice(0, D)
        y_slice = sl[1] if sl is not None else slice(0, H)
        x_slice = sl[2] if sl is not None else slice(0, W)

        box_size = (z_slice.stop - z_slice.start) * (y_slice.stop - y_slice.start) * (x_slice.stop - x_slice.start)

        labeled = False
        if max_watershed_voxels > 0 and box_size <= max_watershed_voxels and sl is not None:
            try:
                sub_mask = (cc_labels[sl] == c_id)
                s_z = skel_z[s_idxs] - z_slice.start
                s_y = skel_y[s_idxs] - y_slice.start
                s_x = skel_x[s_idxs] - x_slice.start
                valid_seeds = (s_z >= 0) & (s_z < sub_mask.shape[0]) & \
                              (s_y >= 0) & (s_y < sub_mask.shape[1]) & \
                              (s_x >= 0) & (s_x < sub_mask.shape[2])

                if np.any(valid_seeds):
                    vs_z = s_z[valid_seeds]
                    vs_y = s_y[valid_seeds]
                    vs_x = s_x[valid_seeds]
                    in_sub = sub_mask[vs_z, vs_y, vs_x]
                    if np.any(in_sub):
                        sub_markers = np.zeros(sub_mask.shape, dtype=np.int32)
                        sub_markers[vs_z[in_sub], vs_y[in_sub], vs_x[in_sub]] = skel_labels[s_idxs][valid_seeds][in_sub]
                        sub_labeled = watershed(np.zeros(sub_mask.shape, dtype=np.uint8), markers=sub_markers, mask=sub_mask, connectivity=3)

                        target_chunk = instance_vol[sl]
                        target_chunk[sub_mask] = sub_labeled[sub_mask]
                        instance_vol[sl] = target_chunk
                        labeled = True
                        any_watershed_done = True
            except (MemoryError, Exception):
                labeled = False

        if not labeled:
            comp_s_coords = np.column_stack([skel_z[s_idxs], skel_y[s_idxs], skel_x[s_idxs]]).astype(np.float32)
            tree = cKDTree(comp_s_coords)
            kdtree_comps[c_id] = (tree, skel_labels[s_idxs], z_slice.start, z_slice.stop)

    del objs
    gc.collect()

    active_comps_per_z = defaultdict(list)
    for c_id, (tree, seed_lbls, z_start, z_stop) in kdtree_comps.items():
        for z in range(z_start, z_stop):
            active_comps_per_z[z].append(c_id)

    # 7. Stream slice-by-slice across 3D volume (near-zero resident RAM footprint)
    n_labeled_voxels = 0
    total_fg = 0

    for z in range(D):
        cc_sl = cc_labels[z]

        # Single-fiber components via LUT
        v_sl = lut[cc_sl]

        # Preserve any watershed results in this slice
        if any_watershed_done:
            exist_sl = instance_vol[z]
            exist_mask = (exist_sl > 0)
            if exist_mask.any():
                v_sl[exist_mask] = exist_sl[exist_mask]

        # Multi-fiber components via KDTree
        active_c_ids = active_comps_per_z.get(z, [])
        for c_id in active_c_ids:
            mask = (cc_sl == c_id)
            if not mask.any():
                continue
            tree, seed_lbls, _, _ = kdtree_comps[c_id]
            y_pts, x_pts = np.nonzero(mask)
            n_pts = len(y_pts)
            z_pts = np.full(n_pts, z, dtype=np.float32)
            pts = np.column_stack([z_pts, y_pts.astype(np.float32), x_pts.astype(np.float32)])

            for start_i in range(0, n_pts, chunk_size):
                end_i = min(start_i + chunk_size, n_pts)
                sub_pts = pts[start_i:end_i]
                _, nearest = tree.query(sub_pts, workers=-1)
                v_sl[y_pts[start_i:end_i], x_pts[start_i:end_i]] = seed_lbls[nearest]

        # Hard-lock skeleton seeds on foreground in this slice
        s_sl = inst_skel[z]
        s_mask = (s_sl > 0)
        if s_mask.any():
            fg_s_mask = s_mask & (cc_sl > 0)
            if fg_s_mask.any():
                v_sl[fg_s_mask] = s_sl[fg_s_mask]

        instance_vol[z] = v_sl

        n_labeled_voxels += int(np.count_nonzero(v_sl))
        total_fg += int(np.count_nonzero(cc_sl > 0))

    if hasattr(instance_vol, 'flush'):
        instance_vol.flush()

    if verbose:
        n_fibers = int(np.max(skel_labels)) if len(skel_labels) > 0 else 0
        pct = (n_labeled_voxels / total_fg * 100.0) if total_fg > 0 else 0.0
        print(f"  [diffuse] {n_fibers} fiber instances, {n_labeled_voxels}/{total_fg} foreground voxels labeled ({pct:.1f}%) across {len(comp_seed_indices)} seeded connected components", flush=True)

    del cc_labels, kdtree_comps, lut, skel_coords, skel_labels
    gc.collect()

    return instance_vol


def prune_short_fibers_and_repropagate(
    inst_skel: np.ndarray,
    inst_vol: np.ndarray,
    min_length: int = 5,
    chunk_size: int = 1000000,
    verbose: bool = True,
) -> tuple[np.ndarray, np.ndarray, int]:
    """
    At the very end of topology optimization:
    1. Identifies small fiber centerlines (shorter than min_length voxels).
    2. Removes these small fiber centerlines from inst_skel (set to 0).
    3. Re-propagates the color / instance label of the nearest legitimate fiber (length >= min_length)
       onto all the volume voxels that belonged to the pruned small fibers.
    4. Compactly renumbers surviving fiber instances 1 .. N.

    Parameters
    ----------
    inst_skel  : (D, H, W) int32 labeled skeleton (in-place or memmap)
    inst_vol   : (D, H, W) int32 labeled instance volume (in-place or memmap)
    min_length : int (default: 5) minimum centerline length in voxels
    chunk_size : int query chunk size for memory safety
    verbose    : bool

    Returns
    -------
    inst_skel   : (D, H, W) int32 cleaned skeleton
    inst_vol    : (D, H, W) int32 cleaned instance volume with color propagation
    n_surviving : int count of surviving fibers
    """
    D, H, W = inst_skel.shape
    unique_ids, skel_counts = np.unique(inst_skel[inst_skel > 0], return_counts=True)
    if len(unique_ids) == 0:
        return inst_skel, inst_vol, 0

    id_to_count = dict(zip(unique_ids, skel_counts))
    short_ids = {int(fid) for fid, cnt in id_to_count.items() if cnt < min_length}
    surviving_ids = [int(fid) for fid, cnt in id_to_count.items() if cnt >= min_length]

    if not short_ids:
        if verbose:
            print(f"  [prune_short_fibers] All {len(surviving_ids)} fibers have centerline >= {min_length} vx. No short fibers to prune.", flush=True)
        return inst_skel, inst_vol, len(surviving_ids)

    if not surviving_ids:
        if verbose:
            print(f"  [prune_short_fibers] Warning: No fibers >= {min_length} vx found. Zeroing outputs.", flush=True)
        inst_skel[:] = 0
        inst_vol[:] = 0
        return inst_skel, inst_vol, 0

    if verbose:
        print(f"  [prune_short_fibers] Pruning {len(short_ids)} short fibers (< {min_length} vx) and propagating color from {len(surviving_ids)} surviving fibers...", flush=True)

    # 1. Compact map for surviving fiber IDs: old_id -> 1 .. N_surviving
    compact_map = {old_id: new_id for new_id, old_id in enumerate(surviving_ids, start=1)}

    # 2. Build cKDTree of surviving skeleton seeds
    surviving_coords_list = []
    surviving_labels_list = []

    for z in range(D):
        s_sl = inst_skel[z]
        mask = np.isin(s_sl, surviving_ids)
        if mask.any():
            sy, sx = np.nonzero(mask)
            sz = np.full(len(sy), z, dtype=np.int32)
            surviving_coords_list.append(np.column_stack([sz, sy, sx]))
            raw_lbls = s_sl[sy, sx]
            new_lbls = np.array([compact_map[int(l)] for l in raw_lbls], dtype=np.int32)
            surviving_labels_list.append(new_lbls)

    if not surviving_coords_list:
        inst_skel[:] = 0
        inst_vol[:] = 0
        return inst_skel, inst_vol, 0

    seed_coords = np.vstack(surviving_coords_list).astype(np.float32)
    seed_labels = np.concatenate(surviving_labels_list)
    del surviving_coords_list, surviving_labels_list
    gc.collect()

    tree = cKDTree(seed_coords)
    del seed_coords
    gc.collect()

    # 3. Vectorized lookup table for surviving labels
    max_id = max(int(np.max(unique_ids)), int(np.max(inst_vol))) if len(unique_ids) > 0 else 0
    lookup = np.zeros(max_id + 1, dtype=np.int32)
    for old_id, new_id in compact_map.items():
        if old_id <= max_id:
            lookup[old_id] = new_id

    short_ids_list = list(short_ids)
    n_repropagated_voxels = 0

    # 4. Stream slice by slice to guarantee near-zero resident RAM on large memmaps
    for z in range(D):
        s_sl = inst_skel[z]
        v_sl = inst_vol[z]

        # Clean skeleton: keep surviving with new IDs, wipe short
        s_surv_mask = np.isin(s_sl, surviving_ids)
        new_s_sl = np.zeros_like(s_sl)
        new_s_sl[s_surv_mask] = lookup[s_sl[s_surv_mask]]
        inst_skel[z] = new_s_sl

        # Clean volume: remap surviving, re-propagate short
        v_surv_mask = np.isin(v_sl, surviving_ids)
        v_short_mask = np.isin(v_sl, short_ids_list)

        new_v_sl = np.zeros_like(v_sl)
        new_v_sl[v_surv_mask] = lookup[v_sl[v_surv_mask]]

        if v_short_mask.any():
            sy, sx = np.nonzero(v_short_mask)
            sz = np.full(len(sy), z, dtype=np.int32)
            short_pts = np.column_stack([sz, sy, sx])

            for start_i in range(0, len(short_pts), chunk_size):
                end_i = min(start_i + chunk_size, len(short_pts))
                sub_pts = short_pts[start_i:end_i]
                _, nearest_idx = tree.query(sub_pts, workers=-1)
                new_v_sl[sub_pts[:, 1], sub_pts[:, 2]] = seed_labels[nearest_idx]

            n_repropagated_voxels += len(short_pts)

        # Hard-lock surviving skeleton seeds into volume slice
        new_v_sl[s_surv_mask] = new_s_sl[s_surv_mask]
        inst_vol[z] = new_v_sl

    if hasattr(inst_skel, 'flush'):
        inst_skel.flush()
    if hasattr(inst_vol, 'flush'):
        inst_vol.flush()

    if verbose:
        print(f"  [prune_short_fibers] Successfully removed {len(short_ids)} short fiber centerlines (< {min_length} vx), re-propagated color to {n_repropagated_voxels} voxels, retained {len(surviving_ids)} clean continuous fibers.", flush=True)

    del tree, seed_labels, lookup
    gc.collect()

    return inst_skel, inst_vol, len(surviving_ids)
