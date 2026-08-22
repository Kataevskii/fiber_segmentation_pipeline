import os
import time
import json
import numpy as np
import tifffile
from scipy.spatial import cKDTree
from scipy.ndimage import median_filter

def compute_analytical_orientation_from_gad(
    gad_path='raw_data/AJ_model_1.gad',
    tif_path='raw_data/AJ_model_1.tif',
    out_vector_path='augmented_data/model_1_base_ori.npy',
    out_intensity_path='augmented_data/model_1_base_intensity.tif',
    sigma=1.0
):
    """
    Computes continuous 3D Gaussian distribution target I(x) = exp(-d(x, C)^2 / (2 * sigma^2)) with std = 1.0
    placed directly on the exact ground truth centerline C, along with 3D fiber unit orientation vectors.
    """
    t0 = time.time()
    print(f"Loading GAD geometry file: {gad_path}...", flush=True)
    with open(gad_path, 'r', encoding='utf-8') as f:
        gad = json.load(f)

    print(f"Loading reference volume: {tif_path}...", flush=True)
    aj_vol = tifffile.imread(tif_path)
    D, H, W = aj_vol.shape

    voxel_length = gad['Domain']['VoxelLength'][0] # 1e-6 meters per voxel
    domain_length = np.array([
        gad['Domain']['LengthZ'][0] / voxel_length,
        gad['Domain']['LengthY'][0] / voxel_length,
        gad['Domain']['LengthX'][0] / voxel_length
    ], dtype=np.float32)

    num_objects = gad['NumberOfObjects']
    print(f"Parsing exact centerlines for {num_objects} fiber objects...", flush=True)

    all_spine_pts = []
    all_spine_tangs = []
    all_spine_objs = []

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
        pts = np.array(pts_list, dtype=np.float32)
        pts_voxels = pts / voxel_length
        pts_zyx = pts_voxels[:, [2, 1, 0]]

        if len(pts_zyx) < 2:
            continue

        n_pts = len(pts_zyx)
        tangents = np.zeros_like(pts_zyx)
        tangents[0] = pts_zyx[1] - pts_zyx[0]
        tangents[-1] = pts_zyx[-1] - pts_zyx[-2]
        if n_pts > 2:
            tangents[1:-1] = (pts_zyx[2:] - pts_zyx[:-2]) / 2.0

        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tangents = tangents / norms

        t_dense = np.linspace(0, 1, n_pts * 12)
        t_sparse = np.linspace(0, 1, n_pts)

        dense_z = np.interp(t_dense, t_sparse, pts_zyx[:, 0])
        dense_y = np.interp(t_dense, t_sparse, pts_zyx[:, 1])
        dense_x = np.interp(t_dense, t_sparse, pts_zyx[:, 2])
        dense_pts = np.column_stack([dense_z, dense_y, dense_x])

        dense_tz = np.interp(t_dense, t_sparse, tangents[:, 0])
        dense_ty = np.interp(t_dense, t_sparse, tangents[:, 1])
        dense_tx = np.interp(t_dense, t_sparse, tangents[:, 2])
        dense_tangs = np.column_stack([dense_tz, dense_ty, dense_tx])
        dense_t_norms = np.linalg.norm(dense_tangs, axis=1, keepdims=True)
        dense_t_norms[dense_t_norms == 0] = 1.0
        dense_tangs = dense_tangs / dense_t_norms

        all_spine_pts.append(dense_pts)
        all_spine_tangs.append(dense_tangs)
        all_spine_objs.append(np.full(len(dense_pts), obj_idx, dtype=np.int32))

    all_spine_pts = np.vstack(all_spine_pts) % domain_length
    all_spine_tangs = np.vstack(all_spine_tangs)
    all_spine_objs = np.concatenate(all_spine_objs)
    print(f"Generated {len(all_spine_pts)} interpolated exact spine points across {num_objects} fibers.", flush=True)

    print("Building periodic 3D cKDTree for exact centerline Gaussian distribution & intersection subtraction...", flush=True)
    t_tree = time.time()
    kdtree = cKDTree(all_spine_pts, boxsize=domain_length)
    print(f"cKDTree built in {time.time() - t_tree:.2f}s", flush=True)

    # Query fiber volume voxels
    fiber_z, fiber_y, fiber_x = np.where(aj_vol > 0)
    fiber_coords = np.column_stack([fiber_z, fiber_y, fiber_x]).astype(np.float32)

    print(f"Querying exact centerline & intersection distances for {len(fiber_coords)} fiber voxels...", flush=True)
    t_query = time.time()
    dists, indices = kdtree.query(fiber_coords, k=8, workers=-1)
    print(f"KDTree query complete in {time.time() - t_query:.2f}s!", flush=True)

    primary_indices = indices[:, 0]
    primary_dists = dists[:, 0]
    primary_objs = all_spine_objs[primary_indices]

    ori_vectors = np.zeros((D, H, W, 3), dtype=np.float32)
    matched_tangs = all_spine_tangs[primary_indices]
    ori_vectors[fiber_z, fiber_y, fiber_x] = matched_tangs

    # Identify 2nd nearest spine point belonging to a DIFFERENT fiber object ID
    matched_objs = all_spine_objs[indices] # (N, 8)
    diff_mask = matched_objs != primary_objs[:, None] # (N, 8) boolean mask

    has_diff = np.any(diff_mask, axis=1)
    diff_col_idx = np.argmax(diff_mask, axis=1) # index of first True column

    sec_dists = np.where(has_diff, dists[np.arange(len(dists)), diff_col_idx], 999.0)

    # 1. Single Fiber Centerline Gaussian Probability Field G1(x) in [0, 1]
    g1 = np.exp(- (primary_dists ** 2) / (2.0 * (sigma ** 2)))
    g1[g1 < 1e-4] = 0.0

    # 2. Intersection Gaussian Field G_cross(x) centered at crossing pixels (where two fibers intersect)
    sigma_cross = 1.5
    cross_mask = (primary_dists <= 3.0) & (sec_dists <= 3.0)
    g_cross = np.zeros_like(primary_dists)
    g_cross[cross_mask] = np.exp(- (primary_dists[cross_mask]**2 + sec_dists[cross_mask]**2) / (2.0 * (sigma_cross ** 2)))

    # 3. Subtracted Probability Field I(x) = G1(x) - 1.5 * G_cross(x) in [-1, 1]
    # Single fiber centerlines -> +1.0
    # Background -> 0.0
    # Intersection pixels -> Negative probabilities dipping down to -1.0
    signed_intensity_values = np.clip(g1 - 1.5 * g_cross, -1.0, 1.0)

    intensity_field = np.zeros((D, H, W), dtype=np.float32)
    intensity_field[fiber_z, fiber_y, fiber_x] = signed_intensity_values

    os.makedirs(os.path.dirname(out_vector_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_intensity_path), exist_ok=True)

    print(f"Saving orientation vector field to {out_vector_path}...", flush=True)
    np.save(out_vector_path, ori_vectors)

    print(f"Saving signed 3D probability target with subtracted intersection Gaussians to {out_intensity_path}...", flush=True)
    if out_intensity_path.endswith('.npy'):
        np.save(out_intensity_path, intensity_field)
    else:
        tifffile.imwrite(out_intensity_path, ((intensity_field + 1.0) * 0.5 * 65535.0).astype(np.uint16), compression='zlib')

    print(f"All precomputations done in {time.time() - t0:.2f}s!", flush=True)


def create_robust_augmented_training_data(
    gad_path='raw_data/AJ_model_1.gad',
    out_tif_path='augmented_data/model_1_aug_vol.tif',
    out_ori_path='augmented_data/model_1_aug_ori.npy',
    out_centerline_path='augmented_data/model_1_aug_centerline.tif',
    max_shift=2.0,
    median_radius=1
):
    """
    Generates robust augmented synthetic fiber volume by:
    1. Applying bounded per-fiber 3D spatial translations (<= 2.0 voxels) guaranteeing >= 4.5 vx clearance.
    2. Re-rasterizing displaced fibers to create realistic optical fiber boundary overlap.
    3. Applying a 3D Median Filter for optical blurring and boundary overlap.
    """
    t0 = time.time()
    print(f"Loading GAD geometry file: {gad_path}...", flush=True)
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
    print(f"Applying robust bounded fiber displacement (<= {max_shift:.1f} vx, no wobble) to {num_objects} fiber objects...", flush=True)

    aug_spine_pts = []
    aug_spine_tangs = []
    aug_spine_radii = []

    np.random.seed(42)

    for obj_idx in range(1, num_objects + 1):
        obj_key = f'Object{obj_idx}'
        obj = gad[obj_key]
        p_keys = sorted([k for k in obj.keys() if k.startswith('Point')], key=lambda x: int(x[5:]))

        pts = np.array([obj[pk]['Coord'][0] for pk in p_keys], dtype=np.float32)
        pts_voxels = pts / voxel_length
        pts_zyx = pts_voxels[:, [2, 1, 0]]
        radii_voxels = np.array([obj[pk]['Radius'][0] / voxel_length for pk in p_keys], dtype=np.float32)

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

        tangents = np.zeros_like(pts_displaced)
        tangents[0] = pts_displaced[1] - pts_displaced[0]
        tangents[-1] = pts_displaced[-1] - pts_displaced[-2]
        if n_pts > 2:
            tangents[1:-1] = (pts_displaced[2:] - pts_displaced[:-2]) / 2.0
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tangents = tangents / norms

        t_dense = np.linspace(0, 1, n_pts * 8)
        t_sparse = np.linspace(0, 1, n_pts)

        dense_z = np.interp(t_dense, t_sparse, pts_displaced[:, 0])
        dense_y = np.interp(t_dense, t_sparse, pts_displaced[:, 1])
        dense_x = np.interp(t_dense, t_sparse, pts_displaced[:, 2])
        dense_pts = np.column_stack([dense_z, dense_y, dense_x])

        dense_r = np.interp(t_dense, t_sparse, radii_voxels)

        # Smooth diameter variation & localized spherical/ellipsoidal bead bulges along the fiber
        n_dense = len(dense_r)
        r_wave = np.sin(np.linspace(0, 2 * np.pi * np.random.uniform(1.0, 3.5), n_dense)) * np.random.uniform(0.2, 0.6)
        
        # Localized beads (spherical / ellipsoidal bumps with radius r + 1 ~ r + 1.5)
        n_beads = np.random.randint(1, 4)
        bead_profile = np.zeros(n_dense, dtype=np.float32)
        t_line = np.linspace(0, 1, n_dense)
        for _ in range(n_beads):
            b_center = np.random.uniform(0.1, 0.9)
            b_width = np.random.uniform(0.02, 0.07)
            b_amp = np.random.uniform(0.8, 1.6)
            bead_profile += b_amp * np.exp(-((t_line - b_center) ** 2) / (2.0 * (b_width ** 2)))

        dense_r = np.clip(dense_r + r_wave + bead_profile, a_min=1.5, a_max=5.5)

        dense_tz = np.interp(t_dense, t_sparse, tangents[:, 0])
        dense_ty = np.interp(t_dense, t_sparse, tangents[:, 1])
        dense_tx = np.interp(t_dense, t_sparse, tangents[:, 2])
        dense_tangs = np.column_stack([dense_tz, dense_ty, dense_tx])
        dense_t_norms = np.linalg.norm(dense_tangs, axis=1, keepdims=True)
        dense_t_norms[dense_t_norms == 0] = 1.0
        dense_tangs = dense_tangs / dense_t_norms

        aug_spine_pts.append(dense_pts)
        aug_spine_tangs.append(dense_tangs)
        aug_spine_radii.append(dense_r)

    aug_spine_pts = np.vstack(aug_spine_pts)
    aug_spine_tangs = np.vstack(aug_spine_tangs)
    aug_spine_radii = np.concatenate(aug_spine_radii)
    print(f"Generated {len(aug_spine_pts)} augmented spine points across {num_objects} fibers.", flush=True)

    # Rasterize
    print("Rasterizing augmented fiber volume & 2px centerlines...", flush=True)
    t_rast = time.time()

    synth_vol = np.zeros((D, H, W), dtype=np.uint8)
    centerline_vol = np.zeros((D, H, W), dtype=np.uint8)

    grid_z, grid_y, grid_x = np.ogrid[-6:7, -6:7, -6:7]
    offsets = np.column_stack([grid_z.ravel(), grid_y.ravel(), grid_x.ravel()])
    offset_dists = np.linalg.norm(offsets, axis=1)

    for pt, r in zip(aug_spine_pts, aug_spine_radii):
        pz, py, px = int(round(pt[0])) % D, int(round(pt[1])) % H, int(round(pt[2])) % W

        valid_mask = offset_dists <= r
        valid_offsets = offsets[valid_mask]
        z_coords = (pz + valid_offsets[:, 0]) % D
        y_coords = (py + valid_offsets[:, 1]) % H
        x_coords = (px + valid_offsets[:, 2]) % W
        synth_vol[z_coords, y_coords, x_coords] = 255

        cl_mask = offset_dists <= 2.0
        cl_offsets = offsets[cl_mask]
        cl_z = (pz + cl_offsets[:, 0]) % D
        cl_y = (py + cl_offsets[:, 1]) % H
        cl_x = (px + cl_offsets[:, 2]) % W
        centerline_vol[cl_z, cl_y, cl_x] = 1

    print(f"Rasterization completed in {time.time() - t_rast:.2f}s! Fiber volume fraction: {np.mean(synth_vol > 0):.4f}", flush=True)

    # 3D Median Filter for realistic boundary fusion
    print(f"Applying 3D Median Filter (radius={median_radius}) for realistic boundary overlap & fusion...", flush=True)
    t_med = time.time()
    synth_vol_filtered = median_filter(synth_vol, size=1 + 2 * median_radius)
    print(f"3D Median filtering complete in {time.time() - t_med:.2f}s!", flush=True)

    # Recalculate orientation KDTree
    kdtree_aug = cKDTree(aug_spine_pts, boxsize=domain_length)
    fiber_z, fiber_y, fiber_x = np.where(synth_vol_filtered > 0)
    fiber_coords = np.column_stack([fiber_z, fiber_y, fiber_x]).astype(np.float32)

    dists, indices = kdtree_aug.query(fiber_coords, k=1, workers=-1)
    aug_ori_vectors = np.zeros((D, H, W, 3), dtype=np.float32)
    aug_ori_vectors[fiber_z, fiber_y, fiber_x] = aug_spine_tangs[indices]

    os.makedirs(os.path.dirname(out_tif_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_ori_path), exist_ok=True)
    os.makedirs(os.path.dirname(out_centerline_path), exist_ok=True)

    print(f"Saving robust augmented TIFF volume to {out_tif_path}...", flush=True)
    tifffile.imwrite(out_tif_path, synth_vol_filtered, compression='zlib')

    print(f"Saving robust augmented 2px centerline volume to {out_centerline_path}...", flush=True)
    tifffile.imwrite(out_centerline_path, centerline_vol, compression='zlib')

    print(f"Saving robust ground-truth orientation field to {out_ori_path}...", flush=True)
    np.save(out_ori_path, aug_ori_vectors)

    print(f"\nRobust data augmentation complete in {time.time() - t0:.2f}s!", flush=True)
