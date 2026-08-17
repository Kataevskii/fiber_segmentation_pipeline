"""
step3_build_fragment_graph.py
==============================
Build a compact fragment graph from the cleaned skeleton.

After H-severing, the skeleton consists of clean segments separated by junctions
or endpoints. This step:

  1. Re-labels connected components as "fragments" (= candidate fiber pieces)
  2. For each fragment, computes:
       - length (voxels)
       - two endpoint coordinates (from the degree-1 voxels)
       - endpoint orientations (averaged over the nearest k voxels -- durable)
  3. Returns a list of FiberFragment objects and a dense label map.

Design note: path ordering is skipped entirely. The optimizer only needs
endpoint positions and orientations, not the full ordered path. This makes
the whole stage vectorized and ~100x faster than BFS-per-fragment.
"""

import numpy as np
from scipy.ndimage import label as nd_label, convolve, maximum_filter
from scipy.spatial import cKDTree
from dataclasses import dataclass, field


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class FiberFragment:
    """One continuous skeleton segment (no junctions inside)."""
    frag_id: int
    # All voxel coordinates (unordered) -- (N, 3) int ZYX
    coords: np.ndarray
    # Unit orientations at those voxels -- (N, 3) float
    oris: np.ndarray
    length: int = 0

    # Endpoint A (arbitrary which is head/tail -- the optimizer handles polarity)
    head_coord: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tail_coord: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    head_ori:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tail_ori:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __post_init__(self):
        self.length = len(self.coords)


# ---------------------------------------------------------------------------
# Main builder -- fully vectorized, no per-fragment Python loops
# ---------------------------------------------------------------------------

def build_fragment_graph(
    skeleton: np.ndarray,
    ori_z: np.ndarray,
    ori_y: np.ndarray,
    ori_x: np.ndarray,
    min_fragment_length: int = 8,
    endpoint_avg_k: int = 5,
    verbose: bool = True,
) -> tuple[list[FiberFragment], np.ndarray]:
    """
    Decompose skeleton into fragments and return a label map.

    Parameters
    ----------
    skeleton            : (D, H, W) bool -- clean skeleton (post H-severing)
    ori_z/y/x           : (D, H, W) float -- unit orientation field
    min_fragment_length : int -- discard fragments shorter than this
    endpoint_avg_k      : int -- number of nearest voxels to average for endpoint
                          orientation (makes it "durable", not just last pixel)

    Returns
    -------
    fragments  : list[FiberFragment]
    seg_labels : (D, H, W) int32 -- label map (0 = bg, new_id = fragment)
    """
    import time
    t0 = time.time()

    D, H, W = skeleton.shape
    struct26      = np.ones((3, 3, 3), dtype=np.uint8)
    struct26_bool = np.ones((3, 3, 3), dtype=bool)

    # ------------------------------------------------------------------ #
    # Step 1: remove junction voxels, label connected components          #
    # ------------------------------------------------------------------ #
    skel_u8 = skeleton.astype(np.uint8)
    n_count  = convolve(skel_u8, struct26, mode='constant', cval=0) - skel_u8
    junc_mask = skeleton & (n_count >= 3)
    non_junc  = skeleton & ~junc_mask

    seg_labels_raw, n_segs = nd_label(non_junc, structure=struct26_bool)

    if verbose:
        print(f"  [graph] {n_segs} raw segments before length filtering.", flush=True)

    # ------------------------------------------------------------------ #
    # Step 2: collect non-zero voxel coords & filter by length           #
    # ------------------------------------------------------------------ #
    vox_z, vox_y, vox_x = np.where(seg_labels_raw > 0)
    vox_ids = seg_labels_raw[vox_z, vox_y, vox_x]

    seg_sizes = np.bincount(vox_ids, minlength=n_segs + 1)
    valid_ids = np.where(seg_sizes >= min_fragment_length)[0]
    valid_ids = valid_ids[valid_ids > 0]

    # Only keep voxels belonging to valid (long enough) segments
    keep_mask = np.isin(vox_ids, valid_ids)
    vox_z  = vox_z[keep_mask]
    vox_y  = vox_y[keep_mask]
    vox_x  = vox_x[keep_mask]
    vox_ids = vox_ids[keep_mask]

    # Orientation at every kept voxel
    oz = ori_z[vox_z, vox_y, vox_x]
    oy = ori_y[vox_z, vox_y, vox_x]
    ox = ori_x[vox_z, vox_y, vox_x]

    # Sort by segment id so we can split cheaply
    sort_idx = np.argsort(vox_ids, kind='stable')
    vox_z   = vox_z[sort_idx]
    vox_y   = vox_y[sort_idx]
    vox_x   = vox_x[sort_idx]
    oz      = oz[sort_idx]
    oy      = oy[sort_idx]
    ox      = ox[sort_idx]
    vox_ids = vox_ids[sort_idx]

    unique_ids, split_idx = np.unique(vox_ids, return_index=True)
    coord_z_splits = np.split(vox_z, split_idx[1:])
    coord_y_splits = np.split(vox_y, split_idx[1:])
    coord_x_splits = np.split(vox_x, split_idx[1:])
    oz_splits = np.split(oz, split_idx[1:])
    oy_splits = np.split(oy, split_idx[1:])
    ox_splits = np.split(ox, split_idx[1:])

    # ------------------------------------------------------------------ #
    # Step 4: find endpoints per segment -- vectorized degree-1 mask      #
    # ------------------------------------------------------------------ #
    ep_n_count = convolve(non_junc.astype(np.uint8), struct26,
                          mode='constant', cval=0) - non_junc.astype(np.uint8)
    ep_mask = non_junc & (ep_n_count == 1)

    ep_z, ep_y, ep_x = np.where(ep_mask)
    ep_seg = seg_labels_raw[ep_z, ep_y, ep_x]

    # Group endpoints by segment id
    ep_sort = np.argsort(ep_seg, kind='stable')
    ep_z    = ep_z[ep_sort]
    ep_y    = ep_y[ep_sort]
    ep_x    = ep_x[ep_sort]
    ep_seg  = ep_seg[ep_sort]

    ep_unique, ep_split = np.unique(ep_seg, return_index=True)
    ep_z_splits = np.split(ep_z, ep_split[1:])
    ep_y_splits = np.split(ep_y, ep_split[1:])
    ep_x_splits = np.split(ep_x, ep_split[1:])
    ep_seg_to_idx = {int(sid): i for i, sid in enumerate(ep_unique)}

    # ------------------------------------------------------------------ #
    # Step 5: build FiberFragment objects -- one small loop per fragment  #
    # ------------------------------------------------------------------ #
    seg_labels = np.zeros_like(seg_labels_raw, dtype=np.int32)
    fragments: list[FiberFragment] = []
    new_id = 1

    for i, old_id in enumerate(unique_ids):
        old_id = int(old_id)
        cz = coord_z_splits[i]
        cy = coord_y_splits[i]
        cx = coord_x_splits[i]
        n  = len(cz)

        coords = np.column_stack([cz, cy, cx]).astype(np.int32)  # (N,3)
        oris   = np.column_stack([oz_splits[i], oy_splits[i], ox_splits[i]]).astype(np.float32)
        nrms   = np.linalg.norm(oris, axis=1, keepdims=True)
        nrms[nrms < 1e-8] = 1.0
        oris /= nrms

        # --- find endpoints for this segment ---------------------------
        ep_i = ep_seg_to_idx.get(old_id)
        if ep_i is not None:
            ezs = ep_z_splits[ep_i]
            eys = ep_y_splits[ep_i]
            exs = ep_x_splits[ep_i]
        else:
            ezs, eys, exs = np.array([cz[0]]), np.array([cy[0]]), np.array([cx[0]])

        if len(ezs) >= 2:
            head_coord_raw = np.array([ezs[0], eys[0], exs[0]], dtype=np.float32)
            tail_coord_raw = np.array([ezs[1], eys[1], exs[1]], dtype=np.float32)
        elif len(ezs) == 1:
            head_coord_raw = np.array([ezs[0], eys[0], exs[0]], dtype=np.float32)
            tail_coord_raw = np.array([cz[-1], cy[-1], cx[-1]], dtype=np.float32)
        else:
            head_coord_raw = np.array([cz[0],  cy[0],  cx[0]],  dtype=np.float32)
            tail_coord_raw = np.array([cz[-1], cy[-1], cx[-1]], dtype=np.float32)

        # Durable endpoint orientations: average k nearest voxels to each endpoint
        all_pts = np.column_stack([cz, cy, cx]).astype(np.float32)

        def avg_ori_near(ep_coord, k=endpoint_avg_k):
            dists = np.sum((all_pts - ep_coord) ** 2, axis=1)
            nearest = np.argpartition(dists, min(k, n) - 1)[:min(k, n)]
            avg = oris[nearest].mean(axis=0)
            nrm = np.linalg.norm(avg)
            return avg / nrm if nrm > 1e-8 else oris[0]

        head_ori_v = avg_ori_near(head_coord_raw)
        tail_ori_v = avg_ori_near(tail_coord_raw)

        frag = FiberFragment(
            frag_id=new_id,
            coords=coords,
            oris=oris,
        )
        frag.head_coord = head_coord_raw
        frag.tail_coord = tail_coord_raw
        frag.head_ori   = head_ori_v
        frag.tail_ori   = tail_ori_v

        fragments.append(frag)
        seg_labels[cz, cy, cx] = new_id
        new_id += 1

    if verbose:
        lengths = [f.length for f in fragments]
        print(f"  [graph] {len(fragments)} fragments kept (len >= {min_fragment_length}). "
              f"Length: min={min(lengths) if lengths else 0}, "
              f"mean={int(np.mean(lengths)) if lengths else 0}, "
              f"max={max(lengths) if lengths else 0}  "
              f"({time.time()-t0:.1f}s)", flush=True)

    return fragments, seg_labels
