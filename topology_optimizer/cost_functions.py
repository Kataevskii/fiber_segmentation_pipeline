"""
cost_functions.py
=================
Modular cost/score definitions for fiber topology optimization.

A good fiber is:
- Long (rewarded)
- Smoothly curved (low mean curvature, no sharp bends)
- Consistent in direction (gradual turns OK, sudden reversals penalized heavily)

All scores are formulated as "lower is better" costs.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Constants / tuneable weights
# ---------------------------------------------------------------------------

# How strongly we reward each extra voxel of fiber length
WEIGHT_LENGTH_BONUS   = 0.10   # subtracted from cost per voxel

# Penalty for mean skeleton curvature (radians per voxel, averaged over length)
WEIGHT_CURVATURE      = 8.0

# Extra penalty if a single junction angle exceeds this threshold (radians)
SHARP_ANGLE_THRESHOLD = np.deg2rad(55)   # 55° is a "hard bend"
SHARP_ANGLE_PENALTY   = 50.0            # flat penalty per offending junction

# Stub penalty: added per fragment to discourage leaving short isolated stubs
# Effective penalty = STUB_PENALTY / fragment_length
STUB_PENALTY          = 200.0

# Bridge gap cost: proportional to gap voxel distance
WEIGHT_GAP_DISTANCE   = 0.25

# Orientation dip at junction: penalty for the angle between the arriving and
# departing orientation vectors at a merged junction (measures direction change)
WEIGHT_JUNCTION_ORI_CHANGE = 5.0

# Minimum fragment length to even consider as a "real" fiber segment
MIN_FRAGMENT_LENGTH   = 12   # voxels -- shorter stubs treated as noise


# ---------------------------------------------------------------------------
# Elementary cost primitives
# ---------------------------------------------------------------------------

def length_bonus(n_voxels: int) -> float:
    """Negative cost (reward) for a longer fiber."""
    return -WEIGHT_LENGTH_BONUS * n_voxels


def curvature_cost(segment_coords: np.ndarray) -> float:
    """
    Mean absolute angular change along a skeleton path (radian/step).

    Parameters
    ----------
    segment_coords : (N, 3) float array  -- ordered ZYX voxel coordinates

    Returns
    -------
    float  -- mean curvature cost  (0 = perfectly straight)
    """
    if len(segment_coords) < 3:
        return 0.0

    vecs = np.diff(segment_coords.astype(np.float32), axis=0)  # (N-1, 3)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    unit_vecs = vecs / norms

    # cos(angle) between consecutive tangent vectors
    dots = np.einsum('ij,ij->i', unit_vecs[:-1], unit_vecs[1:])
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)  # radians, all >= 0

    return float(WEIGHT_CURVATURE * np.mean(angles))


def sharp_bend_cost(segment_coords: np.ndarray) -> float:
    """
    Flat penalty for every junction in a path where the angle exceeds
    SHARP_ANGLE_THRESHOLD.
    """
    if len(segment_coords) < 3:
        return 0.0

    vecs = np.diff(segment_coords.astype(np.float32), axis=0)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1e-8
    unit_vecs = vecs / norms

    dots = np.einsum('ij,ij->i', unit_vecs[:-1], unit_vecs[1:])
    dots = np.clip(dots, -1.0, 1.0)
    angles = np.arccos(dots)

    n_sharp = int(np.sum(angles > SHARP_ANGLE_THRESHOLD))
    return float(n_sharp * SHARP_ANGLE_PENALTY)


def stub_cost(n_voxels: int) -> float:
    """
    Penalty for being a short isolated stub (incentivises merging short stubs).
    Returns 0 for long segments.
    """
    if n_voxels >= MIN_FRAGMENT_LENGTH:
        return 0.0
    return STUB_PENALTY / max(n_voxels, 1)


def gap_cost(gap_distance: float) -> float:
    """
    Cost of drawing a virtual bridge of 'gap_distance' voxels through empty space.
    """
    return WEIGHT_GAP_DISTANCE * gap_distance


def junction_orientation_change_cost(ori_in: np.ndarray, ori_out: np.ndarray) -> float:
    """
    Cost for the direction change at a merged junction.

    Parameters
    ----------
    ori_in  : (3,) float -- incoming orientation vector (pointing toward junction)
    ori_out : (3,) float -- outgoing orientation vector (pointing away from junction)

    Both are unit vectors. Flip ambiguity is handled by taking |dot|.
    """
    dot = float(np.dot(ori_in, ori_out))
    # |dot| = 1 -> perfectly aligned (0 cost); |dot| = 0 -> perpendicular (max cost)
    # angle = arccos(|dot|), cost = weight * angle
    angle = np.arccos(np.clip(abs(dot), 0.0, 1.0))
    return float(WEIGHT_JUNCTION_ORI_CHANGE * angle)


# ---------------------------------------------------------------------------
# Composite costs
# ---------------------------------------------------------------------------

def segment_self_cost(coords: np.ndarray) -> float:
    """
    Total intrinsic cost of a single segment (no bridging involved).
    Lower = this segment is "good on its own".
    """
    n = len(coords)
    return (
        length_bonus(n)
      + curvature_cost(coords)
      + sharp_bend_cost(coords)
      + stub_cost(n)
    )


def bridge_cost(
    seg_a_tail_ori: np.ndarray,
    seg_b_head_ori: np.ndarray,
    gap_distance: float,
    path_ori_samples: np.ndarray | None = None,
    forward_align_a: float | None = None,
    forward_align_b: float | None = None,
) -> float:
    """
    Cost of bridging two segment endpoints across an empty gap.

    Parameters
    ----------
    seg_a_tail_ori   : (3,) float -- orientation/tangent at endpoint of segment A
    seg_b_head_ori   : (3,) float -- orientation/tangent at endpoint of segment B
    gap_distance     : float -- Euclidean distance of the gap in voxels
    path_ori_samples : (K, 3) float or None -- orientation field sampled along the gap path
    forward_align_a  : float or None -- cos(theta_A) turning angle from fragment A into gap
    forward_align_b  : float or None -- cos(theta_B) turning angle from gap into fragment B

    Returns
    -------
    float -- cost of this bridge (lower = better match)
    """
    # Polarity-invariant mutual alignment
    mutual_dot = float(np.dot(seg_a_tail_ori, seg_b_head_ori))
    mutual_align = abs(mutual_dot)

    # Orientation dip at the join
    ori_change = junction_orientation_change_cost(seg_a_tail_ori, seg_b_head_ori)

    # Turn penalties and hard sharp bend check
    turn_penalty = 0.0
    if forward_align_a is not None and forward_align_b is not None:
        theta_a = float(np.arccos(np.clip(forward_align_a, -1.0, 1.0)))
        theta_b = float(np.arccos(np.clip(forward_align_b, -1.0, 1.0)))
        turn_penalty += WEIGHT_CURVATURE * (theta_a + theta_b)
        if theta_a > SHARP_ANGLE_THRESHOLD or theta_b > SHARP_ANGLE_THRESHOLD:
            turn_penalty += SHARP_ANGLE_PENALTY

    # If we have samples along the gap path, check they all agree with direction
    durable_penalty = 0.0
    if path_ori_samples is not None and len(path_ori_samples) > 0:
        mean_gap_ori = path_ori_samples.mean(axis=0)
        norm = np.linalg.norm(mean_gap_ori)
        if norm > 1e-8:
            mean_gap_ori /= norm
            gap_align_a = abs(float(np.dot(seg_a_tail_ori, mean_gap_ori)))
            gap_align_b = abs(float(np.dot(seg_b_head_ori, mean_gap_ori)))
            durable_penalty = WEIGHT_JUNCTION_ORI_CHANGE * (
                max(0.0, 0.70 - gap_align_a)
              + max(0.0, 0.70 - gap_align_b)
            ) * 10.0

    total = gap_cost(gap_distance) + ori_change + turn_penalty + durable_penalty
    return total


def merged_fiber_cost(
    seg_a_coords: np.ndarray,
    seg_b_coords: np.ndarray,
    seg_a_tail_ori: np.ndarray,
    seg_b_head_ori: np.ndarray,
    gap_distance: float,
    path_ori_samples: np.ndarray | None = None,
) -> float:
    """
    Full cost of merging segment A -> [gap] -> segment B into a single fiber.

    This is used to decide whether to commit a bridge. The optimizer will
    choose bridges that minimise this cost summed over all fibers.
    """
    # Combined length
    n_total = len(seg_a_coords) + len(seg_b_coords)

    # Intrinsic costs of each segment (we discount the stub penalty of A since
    # it gains from being merged)
    cost_a = curvature_cost(seg_a_coords) + sharp_bend_cost(seg_a_coords)
    cost_b = curvature_cost(seg_b_coords) + sharp_bend_cost(seg_b_coords)

    cost_bridge = bridge_cost(
        seg_a_tail_ori, seg_b_head_ori,
        gap_distance, path_ori_samples
    )

    return (
        length_bonus(n_total)     # reward the combined length
      + cost_a + cost_b
      + cost_bridge
      + stub_cost(n_total)        # stub penalty on combined length (usually 0)
    )
