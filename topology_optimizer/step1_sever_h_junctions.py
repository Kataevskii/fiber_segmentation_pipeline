"""
step1_sever_h_junctions.py
===========================
Detect and sever transverse H-junction rungs from the fiber skeleton.

Core insight
------------
An H-rung is:
  1. SHORT  -- small number of voxels (< max_rung_length)
  2. Adjacent to LONG neighbors -- the trunks it bridges are much longer
  3. PERPENDICULAR to those neighbors -- its geometric direction disagrees
     with the orientation of the long fibers it connects

This is a purely geometric/topological criterion. It does NOT rely on the
orientation field at the junction itself (which can be noisy or averaged).
Instead it uses:
  - the neighbor BRANCH orientations  (reliable: long straight fibers)
  - the rung's own geometric direction (start-to-end vector)

Strategy
--------
For each skeleton branch B:
  1. If len(B) > max_rung_length  -> skip (too long to be a rung)
  2. Spatially dilate B and find all adjacent branches
  3. Among adjacent branches, keep only the LONG ones (>= min_neighbor_length)
     These are the "trunk candidates"
  4. If no long neighbors -> skip (this is a real fiber stub, not a rung)
  5. Compute mean orientation of the long neighbor branches
  6. Compute geometric direction of rung: rung_dir = normalize(end - start)
  7. perp = 1 - |dot(rung_dir, neighbor_mean_ori)|
     perp ~ 1  -> rung is perpendicular to trunks -> SEVER
     perp ~ 0  -> rung is aligned with trunks (real fiber stub) -> KEEP
  8. Sever if perp > perp_thresh

This cleanly handles all cases:
  - H-rung between two | fibers: rung points left-right, neighbor ori is up-down -> SEVER
  - X-crossing stub: stub is aligned with neighbor fiber -> KEEP
  - Real fiber endpoint: no long neighbors -> KEEP
"""

import numpy as np
from scipy.ndimage import convolve, label as nd_label, binary_dilation


def sever_h_junctions(
    skeleton: np.ndarray,
    ori_z: np.ndarray,
    ori_y: np.ndarray,
    ori_x: np.ndarray,
    max_rung_length: int = 14,
    min_neighbor_length: int = 30,
    perp_thresh: float = 0.50,
    verbose: bool = True,
) -> np.ndarray:
    """
    Remove H-junction rungs from a binary skeleton.

    Parameters
    ----------
    skeleton             : (D, H, W) bool
    ori_z/y/x            : (D, H, W) float  -- unit orientation field
    max_rung_length      : int   -- branches longer than this are never rungs
    min_neighbor_length  : int   -- a neighbor branch must be at least this long
                           for the rung test to apply (avoids short-short pairs)
    perp_thresh          : float -- sever if perp = 1 - |dot(rung_dir, trunk_ori)|
                           > this value. 0.50 means rung makes > 60 deg with trunk.
    verbose              : bool

    Returns
    -------
    clean_skeleton : (D, H, W) bool
    """
    struct26      = np.ones((3, 3, 3), dtype=np.uint8)
    struct26_bool = np.ones((3, 3, 3), dtype=bool)

    # ------------------------------------------------------------------ #
    # Step 1: label all branch segments (non-junction components)         #
    # ------------------------------------------------------------------ #
    skel_u8  = skeleton.astype(np.uint8)
    n_count  = convolve(skel_u8, struct26, mode='constant', cval=0) - skel_u8
    junc_mask = skeleton & (n_count >= 3)
    non_junc  = skeleton & ~junc_mask

    labeled_branches, n_branches = nd_label(non_junc, structure=struct26_bool)

    if verbose:
        print(f"  [H-sever] {int(junc_mask.sum())} junction voxels, "
              f"{n_branches} branch segments.", flush=True)

    if n_branches == 0:
        return skeleton.copy()

    # ------------------------------------------------------------------ #
    # Step 2: compute per-branch stats & collect coords (vectorized)      #
    # ------------------------------------------------------------------ #
    bz, by, bx = np.where(labeled_branches > 0)
    b_ids = labeled_branches[bz, by, bx]

    # Branch sizes
    branch_sizes = np.bincount(b_ids, minlength=n_branches + 1)

    # Mean orientation per branch (vectorized using ONLY branch skeleton voxels)
    oz_vals = ori_z[bz, by, bx]
    oy_vals = ori_y[bz, by, bx]
    ox_vals = ori_x[bz, by, bx]

    sum_oz = np.bincount(b_ids, weights=oz_vals, minlength=n_branches + 1)
    sum_oy = np.bincount(b_ids, weights=oy_vals, minlength=n_branches + 1)
    sum_ox = np.bincount(b_ids, weights=ox_vals, minlength=n_branches + 1)

    counts = branch_sizes.astype(np.float32)
    counts[0] = 1.0
    branch_mean_ori = np.column_stack([
        sum_oz / counts, sum_oy / counts, sum_ox / counts
    ])  # (n_branches+1, 3)
    norms = np.linalg.norm(branch_mean_ori, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    branch_mean_ori = branch_mean_ori / norms

    # ------------------------------------------------------------------ #
    # Step 3: collect start/end coords per branch for rung_dir           #
    # ------------------------------------------------------------------ #
    sort_order = np.argsort(b_ids, kind='stable')
    bz, by, bx, b_ids = bz[sort_order], by[sort_order], bx[sort_order], b_ids[sort_order]
    unique_bids, split_idx = np.unique(b_ids, return_index=True)
    coord_z_splits = np.split(bz, split_idx[1:])
    coord_y_splits = np.split(by, split_idx[1:])
    coord_x_splits = np.split(bx, split_idx[1:])

    # ------------------------------------------------------------------ #
    # Step 4: build branch adjacency THROUGH junction clusters            #
    # Branches are separated by junction voxels; branches that touch the  #
    # same junction cluster are adjacent neighbors.                       #
    # ------------------------------------------------------------------ #
    from scipy.ndimage import maximum_filter
    from collections import defaultdict

    labeled_juncs, n_juncs = nd_label(junc_mask, structure=struct26_bool)
    dilated_juncs = maximum_filter(labeled_juncs, size=3, mode='constant', cval=0)

    touch_mask = (dilated_juncs > 0) & non_junc
    t_junc = dilated_juncs[touch_mask]
    t_branch = labeled_branches[touch_mask]
    touch_pairs = np.unique(np.column_stack([t_junc, t_branch]), axis=0)

    branch_to_juncs = defaultdict(set)
    junc_to_branches = defaultdict(set)
    for jid, bid in touch_pairs:
        branch_to_juncs[bid].add(jid)
        junc_to_branches[jid].add(bid)

    adjacency: list[set] = [set() for _ in range(n_branches + 1)]
    for jid, br_set in junc_to_branches.items():
        br_list = list(br_set)
        for i in range(len(br_list)):
            for j in range(len(br_list)):
                if i != j:
                    adjacency[br_list[i]].add(br_list[j])

    # ------------------------------------------------------------------ #
    # Step 5: classify each short branch                                  #
    # ------------------------------------------------------------------ #
    sever_mask    = np.zeros(n_branches + 1, dtype=bool)
    severed_count = 0

    for idx, bid in enumerate(unique_bids):
        bid = int(bid)
        b_len = int(branch_sizes[bid])

        if b_len > max_rung_length:
            continue

        cz_s = coord_z_splits[idx]
        cy_s = coord_y_splits[idx]
        cx_s = coord_x_splits[idx]
        p_start = np.array([cz_s[0],  cy_s[0],  cx_s[0]],  dtype=np.float32)
        p_end   = np.array([cz_s[-1], cy_s[-1], cx_s[-1]], dtype=np.float32)
        rung_vec  = p_end - p_start
        rung_dist = float(np.linalg.norm(rung_vec))

        juncs_touched = branch_to_juncs.get(bid, set())

        # Degenerate tiny branch touching junctions -> sever
        if rung_dist < 1.0:
            if len(juncs_touched) >= 1:
                sever_mask[bid] = True
                severed_count += 1
            continue

        rung_dir = rung_vec / rung_dist

        # 1. Orientation along the rung itself from predicted orientation field
        oz_m = float(np.mean(ori_z[cz_s, cy_s, cx_s]))
        oy_m = float(np.mean(ori_y[cz_s, cy_s, cx_s]))
        ox_m = float(np.mean(ori_x[cz_s, cy_s, cx_s]))
        o_mean = np.array([oz_m, oy_m, ox_m], dtype=np.float32)
        o_norm = float(np.linalg.norm(o_mean))
        if o_norm > 1e-8:
            o_mean /= o_norm
        else:
            o_mean = np.array([1.0, 0.0, 0.0], dtype=np.float32)

        rung_internal_align = float(abs(np.dot(rung_dir, o_mean)))

        # Rule A: Connects >= 2 distinct junction clusters and is misaligned
        if len(juncs_touched) >= 2 and rung_internal_align < 0.60:
            sever_mask[bid] = True
            severed_count += 1
            continue

        # Rule B: Adjacent to long trunks and perpendicular
        neighbors = adjacency[bid]
        long_neighbors = [nb for nb in neighbors if int(branch_sizes[nb]) >= min_neighbor_length]
        if long_neighbors:
            trunk_ori_vecs = branch_mean_ori[long_neighbors]
            ref = trunk_ori_vecs[0]
            for k in range(1, len(trunk_ori_vecs)):
                if np.dot(trunk_ori_vecs[k], ref) < 0:
                    trunk_ori_vecs[k] = -trunk_ori_vecs[k]
            trunk_mean_ori = trunk_ori_vecs.mean(axis=0)
            tm_norm = np.linalg.norm(trunk_mean_ori)
            if tm_norm > 1e-8:
                trunk_mean_ori /= tm_norm
                alignment_trunk = float(abs(np.dot(rung_dir, trunk_mean_ori)))
                if (1.0 - alignment_trunk) > perp_thresh:
                    sever_mask[bid] = True
                    severed_count += 1
                    continue

    # ------------------------------------------------------------------ #
    # Step 6: remove severed branches                                     #
    # ------------------------------------------------------------------ #
    clean_skel = skeleton.copy()
    rung_voxels = sever_mask[labeled_branches] & (labeled_branches > 0)
    clean_skel[rung_voxels] = False

    if verbose:
        print(f"  [H-sever] Severed {severed_count} H-junction rungs "
              f"(short<={max_rung_length}, neighbor>={min_neighbor_length}, "
              f"perp>{perp_thresh}).", flush=True)
        removed = int(skeleton.sum()) - int(clean_skel.sum())
        print(f"  [H-sever] Skeleton: {int(skeleton.sum())} -> "
              f"{int(clean_skel.sum())} ({removed} voxels removed).", flush=True)

    return clean_skel
