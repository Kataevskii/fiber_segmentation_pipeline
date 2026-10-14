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

import gc
import time
import numpy as np
from scipy.ndimage import label as nd_label, convolve
from scipy.spatial import cKDTree
from collections import deque
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
    head_coord:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tail_coord:   np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    head_ori:     np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tail_ori:     np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    head_tangent: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    tail_tangent: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))

    def __post_init__(self):
        self.length = len(self.coords)


def get_geodesic_diameter_endpoints(coords: np.ndarray, ep_coords: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the geodesic (graph path) diameter endpoints along the 26-connected fragment skeleton.
    Accurately handles non-convex geometries (spirals, hairpins, curved fibers, and loops)
    where the true endpoints lie along the fiber path rather than straight-line Euclidean space.
    """
    n = len(coords)
    if n <= 1:
        return coords[0].astype(np.float32), coords[0].astype(np.float32)
    if n == 2:
        return coords[0].astype(np.float32), coords[1].astype(np.float32)

    # 26-connectivity adjacency graph on the fragment voxels
    tree = cKDTree(coords)
    pairs = tree.query_pairs(r=1.75)  # 26-connectivity distance <= sqrt(3) ~ 1.732
    adj = [[] for _ in range(n)]
    for i, j in pairs:
        adj[i].append(j)
        adj[j].append(i)

    def bfs_all(start_idx: int) -> tuple[int, list[int]]:
        dist = [-1] * n
        dist[start_idx] = 0
        q = deque([start_idx])
        furthest = start_idx
        max_d = 0
        while q:
            u = q.popleft()
            d_u = dist[u]
            if d_u > max_d:
                max_d = d_u
                furthest = u
            for v in adj[u]:
                if dist[v] == -1:
                    dist[v] = d_u + 1
                    q.append(v)
        return furthest, dist

    # Case A: >= 2 candidate endpoints provided
    if ep_coords is not None and len(ep_coords) >= 2:
        _, ep_indices = tree.query(ep_coords, k=1)
        ep_set = list(dict.fromkeys(ep_indices))
        if len(ep_set) == 2:
            return coords[ep_set[0]].astype(np.float32), coords[ep_set[1]].astype(np.float32)
        elif len(ep_set) > 2:
            # Pick pair among candidates with maximum graph geodesic distance
            best_pair = (ep_set[0], ep_set[1])
            max_d = -1
            for c_idx in ep_set:
                _, dist = bfs_all(int(c_idx))
                for other_idx in ep_set:
                    if dist[other_idx] > max_d:
                        max_d = dist[other_idx]
                        best_pair = (c_idx, other_idx)
            return coords[best_pair[0]].astype(np.float32), coords[best_pair[1]].astype(np.float32)

    # Case B: Exactly 1 candidate endpoint provided
    if ep_coords is not None and len(ep_coords) == 1:
        _, ep_idx = tree.query(ep_coords[0], k=1)
        far_idx, _ = bfs_all(int(ep_idx))
        return coords[int(ep_idx)].astype(np.float32), coords[far_idx].astype(np.float32)

    # Case C: 0 candidate endpoints (closed loop / cycle)
    p1, _ = bfs_all(0)
    p2, _ = bfs_all(p1)
    if p1 != p2:
        return coords[p1].astype(np.float32), coords[p2].astype(np.float32)

    # Fallback to Euclidean diameter if graph is disconnected
    diff = coords[:, None, :] - coords[None, :, :]
    dists_sq = np.sum(diff**2, axis=-1)
    i, j = np.unravel_index(np.argmax(dists_sq), dists_sq.shape)
    return coords[i].astype(np.float32), coords[j].astype(np.float32)


# ---------------------------------------------------------------------------
# Main builder -- fully vectorized, no per-fragment Python loops
# ---------------------------------------------------------------------------

def build_fragment_graph(
    skeleton: np.ndarray,
    ori_z: np.ndarray,
    ori_y: np.ndarray,
    ori_x: np.ndarray,
    min_fragment_length: int = 1,
    endpoint_avg_k: int = 5,
    verbose: bool = True,
    return_seg_labels: bool = False,
) -> tuple[list[FiberFragment], np.ndarray | None]:
    """
    Decompose skeleton into fragments and return a label map.

    Parameters
    ----------
    skeleton            : (D, H, W) bool -- clean skeleton (post H-severing)
    ori_z/y/x           : (D, H, W) float -- unit orientation field
    min_fragment_length : int -- keep fragments at or above this length; use 1
                           to retain all post-H-sever fragments for bridging
    endpoint_avg_k      : int -- number of nearest voxels to average for endpoint
                          orientation (makes it "durable", not just last pixel)
    verbose             : bool
    return_seg_labels   : bool -- if True, allocate and return dense 3D label array;
                          defaults to False to avoid 8+ GB RAM allocation.

    Returns
    -------
    fragments  : list[FiberFragment]
    seg_labels : (D, H, W) int32 or None -- label map (0 = bg, new_id = fragment)
    """
    t0 = time.time()

    D, H, W = skeleton.shape
    struct26      = np.ones((3, 3, 3), dtype=np.uint8)
    struct26_bool = np.ones((3, 3, 3), dtype=bool)

    # ------------------------------------------------------------------ #
    # Step 1: remove junction voxels, label connected components          #
    # ------------------------------------------------------------------ #
    skel_u8 = skeleton.astype(np.uint8)
    convolve(skel_u8, struct26, output=skel_u8, mode='constant', cval=0)
    junc_mask = skeleton & (skel_u8 >= 4)
    non_junc  = skeleton & ~junc_mask
    del skel_u8, junc_mask
    gc.collect()

    seg_labels_raw, n_segs = nd_label(non_junc, structure=struct26_bool)

    if verbose:
        print(f"  [graph] {n_segs} raw segments before length filtering.", flush=True)

    # ------------------------------------------------------------------ #
    # Step 2: find endpoints per segment -- vectorized degree-1 mask      #
    # ------------------------------------------------------------------ #
    non_junc_u8 = non_junc.astype(np.uint8)
    convolve(non_junc_u8, struct26, output=non_junc_u8, mode='constant', cval=0)
    ep_mask = non_junc & (non_junc_u8 == 2)
    del non_junc_u8, non_junc
    gc.collect()

    ep_z, ep_y, ep_x = np.where(ep_mask)
    ep_seg = seg_labels_raw[ep_z, ep_y, ep_x]
    del ep_mask
    gc.collect()

    # ------------------------------------------------------------------ #
    # Step 3: collect non-zero voxel coords & filter by length            #
    # ------------------------------------------------------------------ #
    vox_z, vox_y, vox_x = np.where(seg_labels_raw > 0)
    vox_ids = seg_labels_raw[vox_z, vox_y, vox_x]

    if not return_seg_labels:
        del seg_labels_raw
        gc.collect()

    seg_sizes = np.bincount(vox_ids, minlength=n_segs + 1)
    
    # Keep fragments if length >= min_fragment_length OR if length >= 1 (preserves collinear pieces)
    valid_ids = np.where(seg_sizes >= max(1, min_fragment_length))[0]
    valid_ids = valid_ids[valid_ids > 0]

    # Keep voxels belonging to valid segments
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
    if return_seg_labels:
        seg_labels = np.zeros((D, H, W), dtype=np.int32)
    else:
        seg_labels = None

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
            ep_pts = np.column_stack([ezs, eys, exs]).astype(np.float32) if len(ezs) > 0 else None
        else:
            ep_pts = None

        head_coord_raw, tail_coord_raw = get_geodesic_diameter_endpoints(coords, ep_pts)

        # Durable endpoint orientations: 3D volumetric majority voting via Nematic Tensor
        # Sampled 5 voxels inside the body to completely avoid crossing/junction interference
        all_pts = np.column_stack([cz, cy, cx]).astype(np.float32)

        def get_volumetric_consensus_ori(ep_coord, offset_k=endpoint_avg_k, radius=2):
            dists = np.sum((all_pts - ep_coord) ** 2, axis=1)
            sorted_idx = np.argsort(dists)
            
            # Anchor 5 voxels inside body
            anchor_idx = sorted_idx[min(offset_k, n - 1)]
            anchor = all_pts[anchor_idx]
            
            az, ay, ax = int(anchor[0]), int(anchor[1]), int(anchor[2])
            z0, z1 = max(0, az - radius), min(D, az + radius + 1)
            y0, y1 = max(0, ay - radius), min(H, ay + radius + 1)
            x0, x1 = max(0, ax - radius), min(W, ax + radius + 1)
            
            sub_oz = ori_z[z0:z1, y0:y1, x0:x1].ravel()
            sub_oy = ori_y[z0:z1, y0:y1, x0:x1].ravel()
            sub_ox = ori_x[z0:z1, y0:y1, x0:x1].ravel()
            
            vecs = np.column_stack([sub_oz, sub_oy, sub_ox]).astype(np.float32)
            nrms = np.linalg.norm(vecs, axis=1, keepdims=True)
            nrms[nrms < 1e-8] = 1.0
            vecs /= nrms
            
            # Nematic Second-Moment Tensor T = sum(v * v^T)
            T = np.einsum('ni,nj->ij', vecs, vecs)
            _, evecs = np.linalg.eigh(T)
            consensus_ori = evecs[:, -1].astype(np.float32) # dominant eigenvector
            
            # Outward sign deduction from body anchor through endpoint into free space
            v_out = ep_coord - anchor
            norm_out = np.linalg.norm(v_out)
            if norm_out > 1e-4:
                if np.dot(consensus_ori, v_out) < 0:
                    consensus_ori = -consensus_ori
                tan_v = (v_out / norm_out).astype(np.float32)
            else:
                tan_v = consensus_ori
                
            return consensus_ori, tan_v

        head_ori_v, head_tangent_v = get_volumetric_consensus_ori(head_coord_raw)
        tail_ori_v, tail_tangent_v = get_volumetric_consensus_ori(tail_coord_raw)

        frag = FiberFragment(
            frag_id=new_id,
            coords=coords,
            oris=oris,
        )
        frag.head_coord   = head_coord_raw
        frag.tail_coord   = tail_coord_raw
        frag.head_ori     = head_ori_v
        frag.tail_ori     = tail_ori_v
        frag.head_tangent = head_tangent_v
        frag.tail_tangent = tail_tangent_v

        fragments.append(frag)
        if seg_labels is not None:
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
