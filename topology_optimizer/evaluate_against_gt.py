"""
evaluate_against_gt.py
=======================
Evaluate fiber topology optimizer results against ground-truth GAD annotations.

Metrics reported:
  - N_pred  : number of predicted fibers (chains)
  - N_gt    : number of ground-truth fibers
  - Matched : number of GT fibers with a dominant predicted match (IoU > threshold)
  - Split   : GT fibers split into multiple predictions
  - Merge   : multiple GT fibers merged into one prediction
  - Missing : GT fibers with no prediction overlap
  - Mean matched IoU
  - Panoptic Quality (PQ) = SQ x RQ
"""

import json
import numpy as np
from scipy.spatial import cKDTree
from collections import defaultdict


def load_gt_centerlines_from_gad(
    gad_path: str,
    volume_shape: tuple[int, int, int],
) -> np.ndarray:
    """
    Load ground-truth centerline instance labels from a GAD file.

    Returns
    -------
    gt_centerline_vol : (D, H, W) int32 -- each fiber has a unique integer label
    n_gt_fibers       : int
    """
    with open(gad_path, 'r', encoding='utf-8') as f:
        gad = json.load(f)

    D, H, W = volume_shape
    voxel_length = gad['Domain']['VoxelLength'][0]
    domain_length = np.array([
        gad['Domain']['LengthZ'][0] / voxel_length,
        gad['Domain']['LengthY'][0] / voxel_length,
        gad['Domain']['LengthX'][0] / voxel_length,
    ], dtype=np.float32)

    num_objects = gad['NumberOfObjects']
    gt_vol = np.zeros((D, H, W), dtype=np.int32)

    for obj_idx in range(1, num_objects + 1):
        obj_key = f'Object{obj_idx}'
        obj = gad.get(obj_key, {})
        p_keys = sorted([k for k in obj.keys() if k.startswith('Point')],
                        key=lambda x: int(x[5:]))
        pts_list = []
        for pk in p_keys:
            p_val = obj[pk]
            if isinstance(p_val, dict) and 'Coord' in p_val:
                pts_list.append(p_val['Coord'][0])
            elif isinstance(p_val, (list, tuple)):
                pts_list.append(p_val[0])

        if len(pts_list) < 2:
            continue

        pts = np.array(pts_list, dtype=np.float32) / voxel_length
        pts_zyx = pts[:, [2, 1, 0]] % domain_length

        # Dense interpolation
        n_pts = len(pts_zyx)
        t_dense = np.linspace(0, 1, n_pts * 12)
        t_sparse = np.linspace(0, 1, n_pts)
        dense_z = np.interp(t_dense, t_sparse, pts_zyx[:, 0])
        dense_y = np.interp(t_dense, t_sparse, pts_zyx[:, 1])
        dense_x = np.interp(t_dense, t_sparse, pts_zyx[:, 2])

        iz = np.clip(np.round(dense_z).astype(int), 0, D - 1)
        iy = np.clip(np.round(dense_y).astype(int), 0, H - 1)
        ix = np.clip(np.round(dense_x).astype(int), 0, W - 1)
        gt_vol[iz, iy, ix] = obj_idx

    return gt_vol, num_objects


def compute_metrics(
    pred_vol: np.ndarray,
    gt_vol: np.ndarray,
    iou_threshold: float = 0.50,
    verbose: bool = True,
    note: str = "",
) -> dict:
    """
    Compute fiber-level detection / segmentation metrics.

    Both volumes should be int32 with 0 = background.
    """
    gt_ids  = np.unique(gt_vol[gt_vol > 0])
    pred_ids = np.unique(pred_vol[pred_vol > 0])
    N_gt   = len(gt_ids)
    N_pred = len(pred_ids)

    if N_gt == 0:
        return {'N_gt': 0, 'N_pred': N_pred, 'error': 'No GT fibers'}

    # ------------------------------------------------------------------ #
    # Build overlap matrix                                                #
    # (only on voxels where both are nonzero)                            #
    # ------------------------------------------------------------------ #
    overlap_mask = (gt_vol > 0) & (pred_vol > 0)
    gt_at_overlap   = gt_vol[overlap_mask].astype(np.int64)
    pred_at_overlap = pred_vol[overlap_mask].astype(np.int64)

    # Count overlaps per (gt_id, pred_id) pair
    pair_ids = gt_at_overlap * (int(N_pred) + 1) + pred_at_overlap
    pair_counts = defaultdict(int)
    for pid in pair_ids:
        pair_counts[pid] += 1

    # GT voxel counts
    gt_counts = {int(g): int(np.sum(gt_vol == g)) for g in gt_ids}
    pred_counts = {int(p): int(np.sum(pred_vol == p)) for p in pred_ids}

    # For each GT fiber, find best matching prediction
    matched_iou: list[float] = []
    matched_gt = set()
    matched_pred = set()
    split_count  = 0
    merge_count  = 0
    missing_count = 0

    for g in gt_ids:
        g = int(g)
        # Find all predictions overlapping this GT fiber
        preds_for_g = {}
        for p in pred_ids:
            p = int(p)
            key = g * (int(N_pred) + 1) + p
            overlap = pair_counts.get(key, 0)
            if overlap > 0:
                preds_for_g[p] = overlap

        if not preds_for_g:
            missing_count += 1
            continue

        # Best matching prediction by IoU
        best_p, best_overlap = max(preds_for_g.items(), key=lambda x: x[1])
        union = gt_counts[g] + pred_counts[best_p] - best_overlap
        iou = best_overlap / max(union, 1)

        if iou >= iou_threshold:
            matched_iou.append(iou)
            matched_gt.add(g)
            matched_pred.add(best_p)
        elif len(preds_for_g) > 1:
            split_count += 1
        else:
            missing_count += 1

    # Merge: predictions that cover multiple GT fibers dominantly
    for p in pred_ids:
        p = int(p)
        if p in matched_pred:
            continue
        gts_for_p = {}
        for g in gt_ids:
            g = int(g)
            key = g * (int(N_pred) + 1) + p
            overlap = pair_counts.get(key, 0)
            if overlap > 0:
                gts_for_p[g] = overlap
        if len(gts_for_p) > 1:
            merge_count += 1

    n_matched = len(matched_gt)
    SQ = float(np.mean(matched_iou)) if matched_iou else 0.0
    RQ = n_matched / max(N_gt, 1)
    PQ = SQ * RQ

    results = {
        'N_gt':    N_gt,
        'N_pred':  N_pred,
        'Matched': n_matched,
        'Split':   split_count,
        'Merge':   merge_count,
        'Missing': missing_count,
        'Mean_IoU': float(np.mean(matched_iou)) if matched_iou else 0.0,
        'SQ':  SQ,
        'RQ':  RQ,
        'PQ':  PQ,
    }

    if verbose:
        print("\n" + "=" * 60)
        print(" EVALUATION RESULTS")
        print("=" * 60)
        print(f"  GT fibers      : {N_gt}")
        print(f"  Pred fibers    : {N_pred}")
        print(f"  Matched (IoU>{iou_threshold}): {n_matched}")
        print(f"  Split          : {split_count}")
        print(f"  Merge          : {merge_count}")
        print(f"  Missing        : {missing_count}")
        print(f"  Mean matched IoU: {results['Mean_IoU']:.3f}")
        print(f"  SQ (seg qual) : {SQ:.3f}")
        print(f"  RQ (rec qual) : {RQ:.3f}")
        print(f"  PQ (panoptic) : {PQ:.3f}")
        print("=" * 60 + "\n")

    return results
