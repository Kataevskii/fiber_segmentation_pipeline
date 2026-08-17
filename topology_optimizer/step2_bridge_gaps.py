"""
step2_bridge_gaps.py
====================
Direction-durable gap bridging between fiber fragment endpoints.

Key principle:
  Uses the explicit endpoints (head and tail) of all FiberFragments in the fragment graph.
  This guarantees that ALL internal cut points (where H-rungs and crossings were severed)
  are tested for reconnection, including 1-2 voxel gaps.

Acceptance criteria for candidate bridges:
  1. Gap distance <= max_gap_distance voxels (e.g. 1.0 to 18.0 vx)
  2. Mutual alignment between endpoint body orientations >= mutual_thresh
  3. Forward alignment: endpoint vectors point into the gap >= forward_thresh
  4. Durable alignment: sampled orientation along the gap path agrees with gap direction
"""

import numpy as np
from scipy.spatial import cKDTree
from dataclasses import dataclass, field
from typing import Optional

try:
    from .step3_build_fragment_graph import FiberFragment
except ImportError:
    from step3_build_fragment_graph import FiberFragment


@dataclass
class BridgeCandidate:
    """A candidate bridge between two fragment endpoints."""
    # Fragment IDs and which end ('head' or 'tail')
    seg_id_a: int
    seg_id_b: int
    end_a: str  # 'head' or 'tail'
    end_b: str  # 'head' or 'tail'
    # 3D coordinates
    coord_a: np.ndarray     # (3,) float
    coord_b: np.ndarray     # (3,) float
    # Orientations at the endpoints (pointing outward into the gap)
    ori_a: np.ndarray       # (3,) float unit
    ori_b: np.ndarray       # (3,) float unit
    # Gap distance (Euclidean)
    gap_distance: float
    # Alignment scores
    mutual_align: float     # |dot(ori_a, ori_b)|
    forward_align_a: float  # dot(gap_dir, ori_a)
    forward_align_b: float  # dot(-gap_dir, ori_b)
    durable_fraction: float # fraction of gap sample points with good alignment
    # Orientation samples along the gap path
    path_ori_samples: Optional[np.ndarray] = field(default=None, repr=False)
    score: float = 0.0

    def __post_init__(self):
        self.score = self._compute_score()

    def _compute_score(self) -> float:
        return (
            self.mutual_align    * 0.40
          + self.forward_align_a * 0.25
          + self.forward_align_b * 0.25
          + self.durable_fraction * 0.20
          - (self.gap_distance / 25.0) * 0.10
        )


def find_bridge_candidates(
    fragments: list[FiberFragment],
    ori_z: np.ndarray,
    ori_y: np.ndarray,
    ori_x: np.ndarray,
    max_gap_distance: float = 16.0,
    mutual_thresh: float = 0.65,
    forward_thresh: float = 0.55,
    durable_thresh: float = 0.50,
    durable_min_fraction: float = 0.50,
    n_gap_samples: int = 7,
    verbose: bool = True,
) -> list[BridgeCandidate]:
    """
    Find all admissible bridge candidates between fragment endpoints.
    """
    D, H, W = ori_z.shape
    if not fragments:
        return []

    # ------------------------------------------------------------------ #
    # Build array of all fragment endpoints (2 per fragment)             #
    # ------------------------------------------------------------------ #
    ep_coords = []
    ep_oris = []
    ep_meta = []  # (frag_id, 'head'/'tail')

    for frag in fragments:
        ep_coords.append(frag.head_coord)
        ep_oris.append(frag.head_ori)
        ep_meta.append((frag.frag_id, 'head'))

        ep_coords.append(frag.tail_coord)
        ep_oris.append(frag.tail_ori)
        ep_meta.append((frag.frag_id, 'tail'))

    ep_coords = np.array(ep_coords, dtype=np.float32)
    ep_oris   = np.array(ep_oris, dtype=np.float32)
    num_eps   = len(ep_coords)

    if verbose:
        print(f"  [bridge] {num_eps} fragment endpoints from {len(fragments)} fragments. "
              f"Searching within {max_gap_distance:.1f} vx ...", flush=True)

    # ------------------------------------------------------------------ #
    # Candidate search via KD-tree                                        #
    # ------------------------------------------------------------------ #
    tree = cKDTree(ep_coords)
    candidate_pairs = tree.query_pairs(r=max_gap_distance)

    candidates: list[BridgeCandidate] = []

    for i, j in candidate_pairs:
        sid_a, end_a = ep_meta[i]
        sid_b, end_b = ep_meta[j]
        if sid_a == sid_b:
            continue  # Don't bridge the head and tail of the same short fragment

        coord_a = ep_coords[i]
        coord_b = ep_coords[j]
        ori_a   = ep_oris[i]
        ori_b   = ep_oris[j]

        gap_vec  = coord_b - coord_a
        gap_dist = float(np.linalg.norm(gap_vec))
        if gap_dist < 0.5:
            # Virtually identical endpoint (touching 1-voxel gap)
            gap_dir = ori_a.copy()
            gap_dist = 0.5
        else:
            gap_dir = gap_vec / gap_dist

        # ---- Mutual alignment (polarity-invariant) ------------------- #
        mutual_align = float(abs(np.dot(ori_a, ori_b)))
        if mutual_align < mutual_thresh:
            continue

        # ---- Forward alignment (orient vectors to point into gap) ---- #
        d_a = ori_a if float(np.dot(ori_a, gap_dir)) >= 0 else -ori_a
        d_b_toward_a = ori_b if float(np.dot(ori_b, -gap_dir)) >= 0 else -ori_b

        fwd_a = float(np.dot(gap_dir, d_a))
        fwd_b = float(np.dot(-gap_dir, d_b_toward_a))

        # For very short gaps (<= 3 vx), relax forward threshold slightly
        eff_fwd_thresh = forward_thresh if gap_dist > 3.0 else (forward_thresh - 0.15)
        if fwd_a < eff_fwd_thresh or fwd_b < eff_fwd_thresh:
            continue

        # ---- Durable alignment along gap path ------------------------ #
        if gap_dist > 2.0:
            t_samples = np.linspace(0.15, 0.85, n_gap_samples)
            sample_pts = coord_a[None, :] + t_samples[:, None] * gap_vec[None, :]

            sz = np.clip(np.round(sample_pts[:, 0]).astype(int), 0, D - 1)
            sy = np.clip(np.round(sample_pts[:, 1]).astype(int), 0, H - 1)
            sx = np.clip(np.round(sample_pts[:, 2]).astype(int), 0, W - 1)

            path_oz = ori_z[sz, sy, sx]
            path_oy = ori_y[sz, sy, sx]
            path_ox = ori_x[sz, sy, sx]
            path_oris = np.column_stack([path_oz, path_oy, path_ox])
            path_norms = np.linalg.norm(path_oris, axis=1, keepdims=True)
            path_norms[path_norms < 1e-8] = 1.0
            path_oris /= path_norms

            path_aligns = np.abs(path_oris @ gap_dir)
            durable_fraction = float(np.mean(path_aligns >= durable_thresh))
            if durable_fraction < durable_min_fraction:
                continue
        else:
            path_oris = np.stack([d_a, d_b_toward_a], axis=0)
            durable_fraction = 1.0

        cand = BridgeCandidate(
            seg_id_a=sid_a,
            seg_id_b=sid_b,
            end_a=end_a,
            end_b=end_b,
            coord_a=coord_a,
            coord_b=coord_b,
            ori_a=d_a,
            ori_b=d_b_toward_a,
            gap_distance=gap_dist,
            mutual_align=mutual_align,
            forward_align_a=fwd_a,
            forward_align_b=fwd_b,
            durable_fraction=durable_fraction,
            path_ori_samples=path_oris,
        )
        candidates.append(cand)

    candidates.sort(key=lambda c: c.score, reverse=True)

    if verbose:
        print(f"  [bridge] Found {len(candidates)} admissible bridge candidates "
              f"(mutual>={mutual_thresh:.2f}, fwd>={forward_thresh:.2f}, "
              f"durable>={durable_min_fraction:.2f}).", flush=True)

    return candidates
