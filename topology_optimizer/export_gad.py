"""
export_gad.py
=============
Exports segmented 3D fiber centerlines into GeoDict .gad format (JSON-based).
Fully compatible with GeoDict, the Altendorf-Jeulin Model, and evaluate_against_gt.py.
"""

import os
import json
import time
import platform
from datetime import datetime
import numpy as np
from scipy.spatial import cKDTree


def order_centerline_points(pts: np.ndarray) -> np.ndarray:
    """
    Orders 3D skeleton points along the continuous curve from one tip to the other.
    
    Parameters
    ----------
    pts : (N, 3) float or int [z, y, x]

    Returns
    -------
    ordered_pts : (N, 3) float64 [z, y, x]
    """
    n = len(pts)
    if n <= 2:
        return pts.astype(np.float64)

    pts = pts.astype(np.float64)
    tree = cKDTree(pts)
    centroid = pts.mean(axis=0)
    start_idx = int(np.argmax(np.linalg.norm(pts - centroid, axis=1)))

    unvisited = set(range(n))
    unvisited.remove(start_idx)
    ordered_indices = [start_idx]
    curr = start_idx

    while unvisited:
        k_query = min(16, len(pts))
        dists, neighbors = tree.query(pts[curr], k=k_query)
        found = False
        for d, nbr in zip(dists, neighbors):
            if nbr in unvisited:
                if d > 5.0:  # voxel gap tolerance in skeleton
                    break
                curr = nbr
                unvisited.remove(curr)
                ordered_indices.append(curr)
                found = True
                break
        if not found:
            sub = list(unvisited)
            dists_sub = np.linalg.norm(pts[sub] - pts[curr], axis=1)
            nearest_sub = np.argmin(dists_sub)
            if dists_sub[nearest_sub] > 8.0:
                break
            curr = sub[nearest_sub]
            unvisited.remove(curr)
            ordered_indices.append(curr)

    return pts[ordered_indices]


def export_fibers_as_gad(
    inst_skel: np.ndarray,
    inst_vol: np.ndarray | None = None,
    out_path: str = 'outputs/instance_fibers.gad',
    voxel_size: float = 1e-06,
    is_periodic: bool = False,
    min_points: int = 2,
    default_diameter_vx: float = 7.0,
    verbose: bool = True
) -> str:
    """
    Export resolved 3D fibers from labeled skeleton volume into GeoDict .gad format.

    Parameters
    ----------
    inst_skel           : (D, H, W) int32 labeled skeleton volume (0 = background)
    inst_vol            : optional (D, H, W) int32 instance volume (used to estimate fiber diameter)
    out_path            : path to output .gad file
    voxel_size          : physical voxel size in meters (default: 1e-06 m = 1 um)
    is_periodic         : whether domain is periodic (default: False)
    min_points          : minimum points along centerline to export as a fiber (default: 2)
    default_diameter_vx : fallback fiber diameter in voxels (default: 7.0)
    verbose             : bool

    Returns
    -------
    out_path : str
    """
    t0 = time.time()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    D, H, W = inst_skel.shape

    unique_fids = np.unique(inst_skel[inst_skel > 0])
    now = datetime.now()

    system_dict = {
        "Header": {
            "Release": now.strftime("%Y"),
            "Revision": 0,
            "BuildDate": now.strftime("%d %b %Y"),
            "CreationDate": now.strftime("%d %b %Y"),
            "CreationTime": now.strftime("%H:%M:%S"),
            "Creator": "FiberResolutionPipeline",
            "Platform": platform.system()
        },
        "Description": "Segmented fibers resolved by Dual Neural Field & Topology Optimization Pipeline",
        "Domain": {
            "PeriodicX": is_periodic,
            "PeriodicY": is_periodic,
            "PeriodicZ": is_periodic,
            "DomainMode": "Length",
            "OriginX": [0.0, "m"],
            "OriginY": [0.0, "m"],
            "OriginZ": [0.0, "m"],
            "VoxelLength": [voxel_size, "m"],
            "LengthX": [float(W * voxel_size), "m"],
            "LengthY": [float(H * voxel_size), "m"],
            "LengthZ": [float(D * voxel_size), "m"],
            "OverlapMode": "NewMaterial",
            "HollowMaterialID": 0,
            "PostProcessing": {
                "ResolveOverlap": True,
                "MarkContactVoxels": False,
                "ContactMaterialID": 3
            }
        }
    }

    object_count = 0
    for fid in unique_fids:
        pts_zyx = np.argwhere(inst_skel == fid)
        if len(pts_zyx) < min_points:
            continue

        ordered_zyx = order_centerline_points(pts_zyx)
        if len(ordered_zyx) < min_points:
            continue

        # Estimate diameter in meters
        if inst_vol is not None:
            vol_count = int(np.sum(inst_vol == fid))
            cross_sec_area = vol_count / max(len(ordered_zyx), 1)
            radius_vx = np.sqrt(max(cross_sec_area, 0.5) / np.pi)
            diameter_m = float(2.0 * radius_vx * voxel_size)
        else:
            diameter_m = float(default_diameter_vx * voxel_size)

        object_count += 1
        obj_key = f"Object{object_count}"

        # GeoDict coordinates are in meters (X, Y, Z)
        fiber_dict = {
            "MaterialIDModel": "Constant",
            "MaterialID": 1,
            "Type": "CurvedCircularFiber",
            "NumberOfSegments": len(ordered_zyx) - 1,
            "FiberEndType1": "Rounded",
            "FiberEndType2": "Rounded",
            "CuttingPlane1": [0.0, 0.0, 0.0],
            "CuttingPlane2": [0.0, 0.0, 0.0],
            "Diameter": diameter_m
        }

        for j, pt_zyx in enumerate(ordered_zyx, start=1):
            z_vx, y_vx, x_vx = pt_zyx
            x_m = float(x_vx * voxel_size)
            y_m = float(y_vx * voxel_size)
            z_m = float(z_vx * voxel_size)
            fiber_dict[f"Point{j}"] = [[x_m, y_m, z_m], "m"]

        system_dict[obj_key] = fiber_dict

    system_dict["NumberOfObjects"] = object_count

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(system_dict, f, indent=2)

    if verbose:
        size_mb = os.path.getsize(out_path) / (1024 * 1024)
        print(f"  [export] Saved {object_count} fibers to GAD file: {out_path} ({size_mb:.2f} MB in {time.time()-t0:.2f}s)", flush=True)

    return out_path

