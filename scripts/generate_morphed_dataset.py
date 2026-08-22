import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import argparse
import numpy as np
import tifffile
from scipy.spatial import cKDTree
from core.fiber_morpher import (
    RealFiberLibrary,
    deform_real_fiber_along_spline,
    compute_parallel_transport_frames
)

def augment_and_morph_gad_model(
    gad_path,
    library,
    out_vol_path,
    out_intensity_path,
    out_ori_path,
    max_shift=2.0,
    sigma=1.0,
    sigma_cross=1.5
):
    """
    Renders an entire 500x500x500 synthetic GAD model by:
    1. Applying bounded per-fiber 3D spatial translations (<= 2.0 voxels) guaranteeing >= 4.5 vx clearance.
    2. Morphing real biological fiber sleeves from RealFiberLibrary along every centerline.
    3. Synthesizing exact analytical ground-truth continuous intensity I(x) and unit tangents O(x).
    """
    t0 = time.time()
    print(f"Loading GAD geometry: '{gad_path}'...", flush=True)
    with open(gad_path, 'r', encoding='utf-8') as f:
        gad = json.load(f)

    voxel_length = gad['Domain']['VoxelLength'][0]
    domain_length = np.array([
        gad['Domain']['LengthZ'][0] / voxel_length,
        gad['Domain']['LengthY'][0] / voxel_length,
        gad['Domain']['LengthX'][0] / voxel_length
    ], dtype=np.float32)

    D, H, W = int(domain_length[0]), int(domain_length[1]), int(domain_length[2])
    num_objects = gad['NumberOfObjects']
    print(f"Domain: {D}x{H}x{W} voxels | {num_objects} fiber objects.", flush=True)

    aug_curves = []
    aug_tangents = []

    print(f"Augmenting synthetic spline centerlines (bounded 3D shift <= {max_shift:.1f} vx, no wobble)...", flush=True)
    for obj_idx in range(1, num_objects + 1):
        obj_key = f'Object{obj_idx}'
        obj = gad[obj_key]
        p_keys = sorted([k for k in obj.keys() if k.startswith('Point')], key=lambda x: int(x[5:]))

        pts_list = []
        for pk in p_keys:
            p_val = obj[pk]
            if isinstance(p_val, dict) and 'Coord' in p_val:
                pts_list.append(p_val['Coord'][0])
            elif isinstance(p_val, (list, tuple)):
                pts_list.append(p_val[0])
            elif isinstance(p_val, dict) and 'Coord' not in p_val:
                pts_list.append(list(p_val.values())[0])

        pts = np.array(pts_list, dtype=np.float32)
        pts_zyx = (pts / voxel_length)[:, [2, 1, 0]]
        if len(pts_zyx) < 2:
            continue

        n_pts = len(pts_zyx)
        # Bounded 3D spatial shift (norm <= max_shift, guarantees >= 4.57 vx clearance between fibers)
        shift_dir = np.random.normal(0, 1, size=(1, 3))
        shift_dir /= (np.linalg.norm(shift_dir) + 1e-8)
        shift_mag = np.random.uniform(0.0, max_shift)
        shift = shift_dir * shift_mag

        pts_displaced = pts_zyx + shift
        pts_displaced[:, 0] %= D
        pts_displaced[:, 1] %= H
        pts_displaced[:, 2] %= W

        # Dense spline interpolation
        t_dense = np.linspace(0, 1, n_pts * 8)
        t_sparse = np.linspace(0, 1, n_pts)
        dense_z = np.interp(t_dense, t_sparse, pts_displaced[:, 0])
        dense_y = np.interp(t_dense, t_sparse, pts_displaced[:, 1])
        dense_x = np.interp(t_dense, t_sparse, pts_displaced[:, 2])
        dense_pts = np.column_stack([dense_z, dense_y, dense_x]).astype(np.float32)

        # Tangents
        tangs = np.zeros_like(dense_pts, dtype=np.float32)
        tangs[0] = dense_pts[1] - dense_pts[0]
        tangs[-1] = dense_pts[-1] - dense_pts[-2]
        if len(dense_pts) > 2:
            tangs[1:-1] = (dense_pts[2:] - dense_pts[:-2]) / 2.0
        t_norms = np.linalg.norm(tangs, axis=1, keepdims=True)
        t_norms[t_norms == 0] = 1.0
        tangs /= t_norms

        aug_curves.append(dense_pts)
        aug_tangents.append(tangs)

    print(f"Morphing {len(aug_curves)} real biological fiber sleeves into {D}x{H}x{W} volume...", flush=True)
    morphed_vol = np.zeros((D, H, W), dtype=bool)

    t_morph = time.time()
    for i, curve in enumerate(aug_curves):
        real_fiber = library.sample_fiber(boundary_only=True)
        fiber_mask, _ = deform_real_fiber_along_spline(
            real_fiber, curve, dest_shape=(D, H, W), radius_margin=6.0
        )
        morphed_vol |= fiber_mask
        if (i + 1) % 25 == 0 or (i + 1) == len(aug_curves):
            print(f"  Morphed {i + 1}/{len(aug_curves)} fibers ({time.time() - t_morph:.1f}s)...", flush=True)

    print(f"Total foreground fiber voxels: {morphed_vol.sum()} ({np.mean(morphed_vol):.4%})", flush=True)

    # Ground Truth Target Synthesis
    print("Synthesizing exact analytical Ground Truth intensity & unit orientation fields...", flush=True)
    all_spine_pts = np.vstack(aug_curves)
    all_spine_tangs = np.vstack(aug_tangents)
    all_spine_objs = np.concatenate([np.full(len(c), i + 1, dtype=np.uint16) for i, c in enumerate(aug_curves)])

    spine_tree = cKDTree(all_spine_pts)
    fiber_z, fiber_y, fiber_x = np.where(morphed_vol)
    fiber_coords = np.column_stack([fiber_z, fiber_y, fiber_x])

    print(f"Querying nearest spine points for {len(fiber_coords)} foreground voxels...", flush=True)
    dists, indices = spine_tree.query(fiber_coords, k=8)

    primary_dists = dists[:, 0]
    primary_indices = indices[:, 0]
    primary_objs = all_spine_objs[primary_indices]

    # Find secondary distance to different fiber object for intersection dip
    k_objs = all_spine_objs[indices]
    diff_mask = (k_objs != primary_objs[:, None])
    has_diff = np.any(diff_mask, axis=1)
    diff_col_idx = np.argmax(diff_mask, axis=1)
    sec_dists = np.where(has_diff, dists[np.arange(len(dists)), diff_col_idx], 999.0)

    # 1. Centerline Gaussian G1(x) in [0, 1]
    g1 = np.exp(-(primary_dists ** 2) / (2.0 * (sigma ** 2)))
    g1[g1 < 1e-4] = 0.0

    # 2. Continuous Intersection Gaussian Field G_cross(x)
    cross_mask = (primary_dists <= 3.0) & (sec_dists <= 3.0)
    g_cross = np.zeros_like(primary_dists)
    g_cross[cross_mask] = np.exp(-(primary_dists[cross_mask]**2 + sec_dists[cross_mask]**2) / (2.0 * (sigma_cross ** 2)))

    # 3. Signed Continuous Intensity Target I(x) in [-1, 1]
    signed_intensity = np.clip(g1 - 1.5 * g_cross, -1.0, 1.0)
    intensity_vol = np.zeros((D, H, W), dtype=np.float32)
    intensity_vol[fiber_z, fiber_y, fiber_x] = signed_intensity

    # 4. Unit Orientation Vector Field O(x)
    ori_vol = np.zeros((3, D, H, W), dtype=np.float32)
    chosen_tangs = all_spine_tangs[primary_indices]
    for c in range(3):
        ori_vol[c, fiber_z, fiber_y, fiber_x] = chosen_tangs[:, c]

    # Save to disk
    os.makedirs(os.path.dirname(out_vol_path), exist_ok=True)
    print(f"Saving output files to '{os.path.dirname(out_vol_path)}'...", flush=True)

    np.save(out_vol_path.replace('.tif', '.npy'), morphed_vol.astype(bool))
    tifffile.imwrite(out_vol_path, (morphed_vol.astype(np.uint8) * 255), compression='zlib')

    np.save(out_intensity_path, intensity_vol.astype(np.float32))
    np.save(out_ori_path, ori_vol.astype(np.float32))

    print(f"Successfully generated morphed model in {time.time() - t0:.1f}s!", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Generate Full 3D Synthetic Dataset Morphed with Real Biological Fibers")
    parser.add_argument('--models', type=int, nargs='+', default=[1], help="Model indices to generate (e.g. 1 2 3 4)")
    parser.add_argument('--raw-dir', type=str, default='raw_data', help="Directory containing AJ_model_*.gad files")
    parser.add_argument('--curated-dir', type=str, default='real_train_data/curated_patches', help="Curated real patches dir")
    parser.add_argument('--out-dir', type=str, default='augmented_data', help="Output directory for generated morphed models")
    args = parser.parse_args()

    library = RealFiberLibrary(curated_dir=args.curated_dir, cache_path='real_train_data/fiber_library.pkl')

    for m_idx in args.models:
        gad_file = os.path.join(args.raw_dir, f"AJ_model_{m_idx}.gad")
        if not os.path.exists(gad_file):
            print(f"GAD file not found: {gad_file}, skipping...")
            continue

        out_vol = os.path.join(args.out_dir, f"model_{m_idx}_morphed_vol.tif")
        out_int = os.path.join(args.out_dir, f"model_{m_idx}_morphed_intensity.npy")
        out_ori = os.path.join(args.out_dir, f"model_{m_idx}_morphed_ori.npy")

        print("\n" + "=" * 80)
        print(f" PROCESSING SYNTHETIC MODEL #{m_idx} -> MORPHED REAL BIOLOGICAL TEXTURE")
        print("=" * 80)
        augment_and_morph_gad_model(
            gad_path=gad_file,
            library=library,
            out_vol_path=out_vol,
            out_intensity_path=out_int,
            out_ori_path=out_ori
        )

if __name__ == '__main__':
    main()
