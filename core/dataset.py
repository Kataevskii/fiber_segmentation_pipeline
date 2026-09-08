import os
import json
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from scipy.spatial import cKDTree

def ensure_individual_fibers_extracted(real_data_dir='data/curated', padding=2, verbose=True):
    """
    Scans real curated training patch blocks (e.g. in data/curated/patches/)
    and extracts all individual single-fiber sub-volume stamps into
    data/curated/individual_fibers/ if missing or incomplete.
    """
    if not real_data_dir or not os.path.exists(real_data_dir):
        return 0

    if os.path.basename(real_data_dir) == 'patches':
        curated_dir = real_data_dir
        indiv_dir = os.path.join(os.path.dirname(real_data_dir), 'individual_fibers')
    else:
        curated_dir = os.path.join(real_data_dir, 'patches')
        indiv_dir = os.path.join(real_data_dir, 'individual_fibers')

    if not os.path.exists(curated_dir):
        return 0

    os.makedirs(indiv_dir, exist_ok=True)
    patch_vols = sorted([
        os.path.join(curated_dir, f)
        for f in os.listdir(curated_dir)
        if f.endswith('_vol.npy')
    ])

    newly_extracted = 0
    total_stamps = 0

    for p_vol_path in patch_vols:
        p_name = os.path.basename(p_vol_path).replace('_vol.npy', '')
        p_inst_path = os.path.join(curated_dir, f"{p_name}_instance.npy")
        p_int_path = os.path.join(curated_dir, f"{p_name}_intensity.npy")
        p_ori_path = os.path.join(curated_dir, f"{p_name}_ori.npy")

        if not (os.path.exists(p_inst_path) and os.path.exists(p_int_path) and os.path.exists(p_ori_path)):
            continue

        inst = np.load(p_inst_path)
        unique_ids = np.unique(inst[inst > 0])
        total_stamps += len(unique_ids)

        missing_fibers = []
        for fib_id in unique_ids:
            v_out = os.path.join(indiv_dir, f"{p_name}_fiber_{fib_id:02d}_vol.npy")
            i_out = os.path.join(indiv_dir, f"{p_name}_fiber_{fib_id:02d}_intensity.npy")
            o_out = os.path.join(indiv_dir, f"{p_name}_fiber_{fib_id:02d}_ori.npy")
            if not (os.path.exists(v_out) and os.path.exists(i_out) and os.path.exists(o_out)):
                missing_fibers.append(fib_id)

        if not missing_fibers:
            continue

        vol = np.load(p_vol_path)
        intensity = np.load(p_int_path)
        ori = np.load(p_ori_path)

        for fib_id in missing_fibers:
            mask = (inst == fib_id)
            coords = np.argwhere(mask)
            if len(coords) < 3:
                continue

            zmin, ymin, xmin = coords.min(axis=0)
            zmax, ymax, xmax = coords.max(axis=0) + 1

            z0, z1 = max(0, zmin - padding), min(vol.shape[0], zmax + padding)
            y0, y1 = max(0, ymin - padding), min(vol.shape[1], ymax + padding)
            x0, x1 = max(0, xmin - padding), min(vol.shape[2], xmax + padding)

            sub_mask = (inst[z0:z1, y0:y1, x0:x1] == fib_id)
            sub_vol = (vol[z0:z1, y0:y1, x0:x1] * sub_mask).astype(bool)
            sub_int = (intensity[z0:z1, y0:y1, x0:x1] * sub_mask).astype(np.float32)
            sub_ori = (ori[:, z0:z1, y0:y1, x0:x1] * sub_mask).astype(np.float32)

            out_prefix = os.path.join(indiv_dir, f"{p_name}_fiber_{fib_id:02d}")
            np.save(f"{out_prefix}_vol.npy", sub_vol)
            np.save(f"{out_prefix}_intensity.npy", sub_int)
            np.save(f"{out_prefix}_ori.npy", sub_ori)
            newly_extracted += 1

    if newly_extracted > 0 and verbose:
        print(f"Auto-extracted {newly_extracted} new individual fiber stamps from curated patch blocks into '{indiv_dir}'.", flush=True)

    return total_stamps


def _augment_stamp(stamp_vol, stamp_int, stamp_ori):
    """Randomly flips and rotates an individual fiber stamp of arbitrary 3D shape."""
    s_vol = stamp_vol.copy()
    s_int = stamp_int.copy()
    s_ori = stamp_ori.copy()

    # Ensure channel-first format for ori: (3, d, h, w)
    if s_ori.ndim == 4 and s_ori.shape[-1] == 3 and s_ori.shape[0] != 3:
        s_ori = np.transpose(s_ori, (3, 0, 1, 2)).copy()

    # 1. Random 3D Flips
    if random.random() > 0.5:
        s_vol = np.flip(s_vol, axis=0).copy()
        s_int = np.flip(s_int, axis=0).copy()
        s_ori = np.flip(s_ori, axis=1).copy()
        s_ori[0] = -s_ori[0]

    if random.random() > 0.5:
        s_vol = np.flip(s_vol, axis=1).copy()
        s_int = np.flip(s_int, axis=1).copy()
        s_ori = np.flip(s_ori, axis=2).copy()
        s_ori[1] = -s_ori[1]

    if random.random() > 0.5:
        s_vol = np.flip(s_vol, axis=2).copy()
        s_int = np.flip(s_int, axis=2).copy()
        s_ori = np.flip(s_ori, axis=3).copy()
        s_ori[2] = -s_ori[2]

    # 2. Random 3D Orthogonal Rotations (using np.stack to accommodate rectangular stamps)
    k_yx = random.randint(0, 3)
    if k_yx > 0:
        s_vol = np.rot90(s_vol, k=k_yx, axes=(1, 2)).copy()
        s_int = np.rot90(s_int, k=k_yx, axes=(1, 2)).copy()
        ori_z = np.rot90(s_ori[0], k=k_yx, axes=(1, 2)).copy()
        ori_y = np.rot90(s_ori[1], k=k_yx, axes=(1, 2)).copy()
        ori_x = np.rot90(s_ori[2], k=k_yx, axes=(1, 2)).copy()
        if k_yx == 1:
            s_ori = np.stack([ori_z, -ori_x, ori_y], axis=0)
        elif k_yx == 2:
            s_ori = np.stack([ori_z, -ori_y, -ori_x], axis=0)
        elif k_yx == 3:
            s_ori = np.stack([ori_z, ori_x, -ori_y], axis=0)

    k_zx = random.randint(0, 3)
    if k_zx > 0:
        s_vol = np.rot90(s_vol, k=k_zx, axes=(0, 2)).copy()
        s_int = np.rot90(s_int, k=k_zx, axes=(0, 2)).copy()
        ori_z = np.rot90(s_ori[0], k=k_zx, axes=(0, 2)).copy()
        ori_y = np.rot90(s_ori[1], k=k_zx, axes=(0, 2)).copy()
        ori_x = np.rot90(s_ori[2], k=k_zx, axes=(0, 2)).copy()
        if k_zx == 1:
            s_ori = np.stack([-ori_x, ori_y, ori_z], axis=0)
        elif k_zx == 2:
            s_ori = np.stack([-ori_z, ori_y, -ori_x], axis=0)
        elif k_zx == 3:
            s_ori = np.stack([ori_x, ori_y, -ori_z], axis=0)

    k_zy = random.randint(0, 3)
    if k_zy > 0:
        s_vol = np.rot90(s_vol, k=k_zy, axes=(0, 1)).copy()
        s_int = np.rot90(s_int, k=k_zy, axes=(0, 1)).copy()
        ori_z = np.rot90(s_ori[0], k=k_zy, axes=(0, 1)).copy()
        ori_y = np.rot90(s_ori[1], k=k_zy, axes=(0, 1)).copy()
        ori_x = np.rot90(s_ori[2], k=k_zy, axes=(0, 1)).copy()
        if k_zy == 1:
            s_ori = np.stack([-ori_y, ori_z, ori_x], axis=0)
        elif k_zy == 2:
            s_ori = np.stack([-ori_z, -ori_y, ori_x], axis=0)
        elif k_zy == 3:
            s_ori = np.stack([ori_y, -ori_z, ori_x], axis=0)

    return s_vol, s_int, s_ori


def stamp_fiber_if_separable(
    vol_patch: np.ndarray,
    int_patch: np.ndarray,
    ori_patch: np.ndarray,
    stamp_vol: np.ndarray,
    stamp_int: np.ndarray,
    stamp_ori: np.ndarray,
    min_centerline_dist: float = 6.0,
    max_attempts: int = 15,
    intersection_dip: float = -0.5
):
    """
    Pastes an individual fiber stamp onto a 3D patch ONLY if its centerline
    remains separated by >= min_centerline_dist (default 6.0 voxels) from all
    existing fiber centerlines in the patch.

    This ensures that touching/grazing fiber bodies can intersect naturally
    with a negative probability valley while strictly preventing centerline
    merging or artificial X-crossing H-junctions.
    """
    D, H, W = vol_patch.shape
    d, h, w = stamp_vol.shape

    # Crop stamp if larger than destination patch
    if d > D or h > H or w > W:
        stamp_vol = stamp_vol[:min(d, D), :min(h, H), :min(w, W)]
        stamp_int = stamp_int[:min(d, D), :min(h, H), :min(w, W)]
        stamp_ori = stamp_ori[:, :min(d, D), :min(h, H), :min(w, W)]
        d, h, w = stamp_vol.shape

    # Extract existing centerlines (voxels with positive potential I > 0.5)
    exist_cl_idx = np.where(int_patch > 0.5)
    has_existing = len(exist_cl_idx[0]) > 0
    exist_tree = cKDTree(np.column_stack(exist_cl_idx).astype(np.float32)) if has_existing else None

    # Extract candidate stamp centerlines
    stamp_cl_idx = np.where(stamp_int > 0.5)
    if len(stamp_cl_idx[0]) == 0:
        return vol_patch, int_patch, ori_patch, False

    stamp_cl_coords = np.column_stack(stamp_cl_idx).astype(np.float32)

    for _ in range(max_attempts):
        z0 = random.randint(0, max(0, D - d))
        y0 = random.randint(0, max(0, H - h))
        x0 = random.randint(0, max(0, W - w))
        offset = np.array([z0, y0, x0], dtype=np.float32)

        # Centerline 6-voxel separability check
        if exist_tree is not None:
            stamp_cl_world = stamp_cl_coords + offset
            dists, _ = exist_tree.query(stamp_cl_world, k=1)
            if np.min(dists) < min_centerline_dist:
                continue  # Rejected: centerlines closer than 6.0 voxels

        # Accepted! Merge into patch
        z1, y1, x1 = z0 + d, y0 + h, x0 + w
        sub_vol = vol_patch[z0:z1, y0:y1, x0:x1]
        sub_int = int_patch[z0:z1, y0:y1, x0:x1]
        sub_ori = ori_patch[:, z0:z1, y0:y1, x0:x1]
        # Identify body overlap regions (where outer fiber volumes touch)
        overlap_mask = (sub_vol > 0) & (stamp_vol > 0)

        # Merge binary volume
        vol_patch[z0:z1, y0:y1, x0:x1] = (sub_vol > 0) | (stamp_vol > 0)

        # Merge intensity field with continuous Gaussian intersection subtraction (consistent with GT generator)
        merged_int = np.where(stamp_vol > 0, np.maximum(sub_int, stamp_int), sub_int)
        if np.any(overlap_mask) and exist_tree is not None:
            overlap_coords_sub = np.argwhere(overlap_mask)
            overlap_coords_world = overlap_coords_sub.astype(np.float32) + offset

            stamp_tree = cKDTree(stamp_cl_coords)
            d1, _ = exist_tree.query(overlap_coords_world, k=1)
            d2, _ = stamp_tree.query(overlap_coords_sub.astype(np.float32), k=1)

            cross_mask = (d1 <= 3.5) & (d2 <= 3.5)
            if np.any(cross_mask):
                g_cross = np.exp(-(d1[cross_mask]**2 + d2[cross_mask]**2) / (2.0 * 1.5**2))
                pts = overlap_coords_sub[cross_mask]
                merged_int[pts[:, 0], pts[:, 1], pts[:, 2]] -= 1.5 * g_cross

        int_patch[z0:z1, y0:y1, x0:x1] = np.clip(merged_int, -1.0, 1.0)

        # Merge orientation field
        for c in range(3):
            ori_patch[c, z0:z1, y0:y1, x0:x1] = np.where(
                stamp_vol > 0, stamp_ori[c], sub_ori[c]
            )

        return vol_patch, int_patch, ori_patch, True

    return vol_patch, int_patch, ori_patch, False


def collect_dataset_triplets(data_dir):
    """Collect volume/intensity/orientation triplets from one directory or a list of directories."""
    dir_list = [data_dir] if isinstance(data_dir, str) else list(data_dir)
    if isinstance(data_dir, str) and ',' in data_dir:
        dir_list = [d.strip() for d in data_dir.split(',') if d.strip()]

    volume_paths = []
    intensity_paths = []
    orientation_paths = []

    for d in dir_list:
        if not os.path.exists(d):
            continue

        files = sorted(os.listdir(d))
        v_sub = sorted([os.path.join(d, f) for f in files if f.endswith('_vol.npy')])
        if not v_sub:
            v_sub = sorted([os.path.join(d, f) for f in files if f.endswith('_vol.tif')])

        for v_path in v_sub:
            if v_path.endswith('_vol.npy'):
                i_path = v_path.replace('_vol.npy', '_intensity.npy')
                if not os.path.exists(i_path):
                    i_path = v_path.replace('_vol.npy', '_intensity.tif')
                if not os.path.exists(i_path):
                    i_path = v_path.replace('_vol.npy', '_centerline.npy')
                if not os.path.exists(i_path):
                    i_path = v_path.replace('_vol.npy', '_centerline.tif')
                o_path = v_path.replace('_vol.npy', '_ori.npy')
            else:
                i_path = v_path.replace('_vol.tif', '_intensity.tif')
                if not os.path.exists(i_path):
                    i_path = v_path.replace('_vol.tif', '_centerline.tif')
                o_path = v_path.replace('_vol.tif', '_ori.npy')

            if os.path.exists(v_path) and os.path.exists(i_path) and os.path.exists(o_path):
                volume_paths.append(v_path)
                intensity_paths.append(i_path)
                orientation_paths.append(o_path)

    return volume_paths, intensity_paths, orientation_paths


def split_real_validation_triplets(data_dir, holdout_every=5):
    """Split a real dataset directory into deterministic train and validation triplets."""
    volume_paths, intensity_paths, orientation_paths = collect_dataset_triplets(data_dir)
    if not volume_paths:
        return ([], [], []), ([], [], [])

    train_vols, train_ints, train_oris = [], [], []
    val_vols, val_ints, val_oris = [], [], []

    for index, (v_path, i_path, o_path) in enumerate(zip(volume_paths, intensity_paths, orientation_paths)):
        if holdout_every > 0 and index % holdout_every == 0:
            val_vols.append(v_path)
            val_ints.append(i_path)
            val_oris.append(o_path)
        else:
            train_vols.append(v_path)
            train_ints.append(i_path)
            train_oris.append(o_path)

    if not val_vols and train_vols:
        val_vols.append(train_vols.pop())
        val_ints.append(train_ints.pop())
        val_oris.append(train_oris.pop())

    return (train_vols, train_ints, train_oris), (val_vols, val_ints, val_oris)


class Fiber3DPatchDataset(Dataset):
    """
    Zero-Copy Memory-Mapped 3D Patch Dataset for PyTorch DataLoader (Optimized for Windows Multiprocessing).
    Stores ONLY filepaths and small integer coordinate arrays to ensure dataset pickling size is <1MB,
    preventing IPC pipe buffer serialization crashes (OSError Errno 22).
    """
    def __init__(
        self,
        data_dir='data/synthetic/precomputed/train',
        volume_paths=None,
        intensity_paths=None,
        orientation_paths=None,
        patch_size=64,
        samples_per_epoch=1200,
        augment=True,
        fg_prob=0.85,
        jitter_voxels=2,
        real_data_dir='data/curated',
        real_stamp_prob=0.25,
        verbose=False
    ):
        super().__init__()
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.fg_prob = fg_prob
        self.jitter_voxels = jitter_voxels
        self.real_stamp_prob = real_stamp_prob
        self.verbose = verbose

        # Ensure all individual fiber stamps are pre-extracted from curated blocks
        if real_data_dir and os.path.exists(real_data_dir):
            ensure_individual_fibers_extracted(real_data_dir=real_data_dir, verbose=verbose)

        # Discover curated real fiber stamps if available
        self.real_fibers = []
        if real_data_dir and os.path.exists(real_data_dir):
            real_vol_files = []
            indiv_dir = os.path.join(real_data_dir, 'individual_fibers')
            target_dir = indiv_dir if os.path.exists(indiv_dir) else real_data_dir
            for root, _, files in os.walk(target_dir):
                for f in sorted(files):
                    if f.endswith('_vol.npy'):
                        real_vol_files.append(os.path.join(root, f))
            for rv in real_vol_files:
                ri = rv.replace('_vol.npy', '_intensity.npy')
                ro = rv.replace('_vol.npy', '_ori.npy')
                if os.path.exists(ri) and os.path.exists(ro):
                    try:
                        r_vol = np.load(rv)
                        r_int = np.load(ri)
                        r_ori = np.load(ro)
                        self.real_fibers.append((r_vol, r_int, r_ori))
                    except Exception:
                        pass
            if self.real_fibers:
                print(f"Loaded {len(self.real_fibers)} curated real fiber stamps from '{target_dir}'.", flush=True)

        # Auto-discover dataset triplets from data_dir if paths not specified
        if volume_paths is None or intensity_paths is None or orientation_paths is None:
            dir_list = [data_dir] if isinstance(data_dir, str) else list(data_dir)
            if isinstance(data_dir, str) and ',' in data_dir:
                dir_list = [d.strip() for d in data_dir.split(',') if d.strip()]

            v_files = []
            for d in dir_list:
                if os.path.exists(d):
                    files = sorted(os.listdir(d))
                    v_sub = sorted([os.path.join(d, f) for f in files if f.endswith('_vol.npy')])
                    if not v_sub:
                        v_sub = sorted([os.path.join(d, f) for f in files if f.endswith('_vol.tif')])
                    v_files.extend(v_sub)

            volume_paths = v_files
            intensity_paths = []
            orientation_paths = []

            for v in v_files:
                if v.endswith('_vol.npy'):
                    i_p = v.replace('_vol.npy', '_intensity.npy')
                    if not os.path.exists(i_p):
                        i_p = v.replace('_vol.npy', '_intensity.tif')
                    if not os.path.exists(i_p):
                        i_p = v.replace('_vol.npy', '_centerline.npy')
                    if not os.path.exists(i_p):
                        i_p = v.replace('_vol.npy', '_centerline.tif')
                    ori_p = v.replace('_vol.npy', '_ori.npy')
                else:
                    i_p = v.replace('_vol.tif', '_intensity.tif')
                    if not os.path.exists(i_p):
                        i_p = v.replace('_vol.tif', '_centerline.tif')
                    ori_p = v.replace('_vol.tif', '_ori.npy')

                intensity_paths.append(i_p)
                orientation_paths.append(ori_p)
            else:
                volume_paths = volume_paths or []
                intensity_paths = intensity_paths or []
                orientation_paths = orientation_paths or []

        self.volume_paths = []
        self.intensity_paths = []
        self.orientation_paths = []
        self.fiber_coords_list = []

        # Validate paths and pre-extract tiny integer coordinate arrays
        for v_path, i_path, o_path in zip(volume_paths, intensity_paths, orientation_paths):
            if not os.path.exists(v_path) or not os.path.exists(i_path) or not os.path.exists(o_path):
                print(f"Dataset path missing: {v_path}, {i_path}, or {o_path}, skipping...", flush=True)
                continue

            if self.verbose:
                print(f"Loading fiber volume metadata: {v_path}", flush=True)
            if i_path.endswith('.npy'):
                intensity_arr = np.load(i_path, mmap_mode='r')
            else:
                intensity_arr = tifffile.imread(i_path)
                if intensity_arr.dtype == np.uint16:
                    intensity_arr = intensity_arr.astype(np.float32) / 65535.0

            # Pre-extract high potential voxel coordinates for foreground sampling (both centerlines & negative intersection pixels)
            z_idx, y_idx, x_idx = np.where(np.abs(intensity_arr) > 0.1)
            fiber_coords = np.column_stack([z_idx, y_idx, x_idx]).astype(np.int32)

            self.volume_paths.append(v_path)
            self.intensity_paths.append(i_path)
            self.orientation_paths.append(o_path)
            self.fiber_coords_list.append(fiber_coords)
            if self.verbose:
                print(f"  Found {len(fiber_coords)} valid potential locations (|I| > 0.1).", flush=True)

        if not self.volume_paths:
            raise ValueError(f"No valid dataset triplets found in '{data_dir}' or provided paths!")

    def __len__(self):
        return self.samples_per_epoch

    def _load_slice(self, file_path, z1, z2, y1, y2, x1, x2):
        if file_path.endswith('.npy'):
            mmap = np.load(file_path, mmap_mode='r')
            return np.array(mmap[z1:z2, y1:y2, x1:x2], dtype=np.float32)
        else:
            vol = tifffile.imread(file_path)
            patch = vol[z1:z2, y1:y2, x1:x2].astype(np.float32)
            if vol.dtype == np.uint16 and 'intensity' in file_path.lower():
                patch /= 65535.0
            return patch

    def __getitem__(self, idx):
        # Pick volume randomly
        vol_idx = random.randint(0, len(self.volume_paths) - 1)
        v_path = self.volume_paths[vol_idx]
        i_path = self.intensity_paths[vol_idx]
        o_path = self.orientation_paths[vol_idx]
        fiber_coords = self.fiber_coords_list[vol_idx]

        # Read shape from orientation mmap shape without loading full array into RAM
        ori_mmap = np.load(o_path, mmap_mode='r')
        if ori_mmap.shape[0] == 3:  # (3, D, H, W)
            _, D, H, W = ori_mmap.shape
            is_ch_first = True
        else:  # (D, H, W, 3)
            D, H, W, _ = ori_mmap.shape
            is_ch_first = False
        p = self.patch_size

        # Foreground sampling vs Uniform sampling
        if random.random() < self.fg_prob and len(fiber_coords) > 0:
            center_coord = fiber_coords[random.randint(0, len(fiber_coords) - 1)]
            z_c, y_c, x_c = int(center_coord[0]), int(center_coord[1]), int(center_coord[2])

            # Add a small local jitter around the sampled fiber center
            z_start = z_c - p // 2 + random.randint(-self.jitter_voxels, self.jitter_voxels)
            y_start = y_c - p // 2 + random.randint(-self.jitter_voxels, self.jitter_voxels)
            x_start = x_c - p // 2 + random.randint(-self.jitter_voxels, self.jitter_voxels)
        else:
            z_start = random.randint(0, max(0, D - p))
            y_start = random.randint(0, max(0, H - p))
            x_start = random.randint(0, max(0, W - p))

        # Clamp boundaries
        z_start = max(0, min(max(0, D - p), z_start))
        y_start = max(0, min(max(0, H - p), y_start))
        x_start = max(0, min(max(0, W - p), x_start))

        z_end, y_end, x_end = z_start + p, y_start + p, x_start + p

        # Crop patches directly from zero-copy memory maps
        vol_patch = self._load_slice(v_path, z_start, z_end, y_start, y_end, x_start, x_end)
        intensity_patch = self._load_slice(i_path, z_start, z_end, y_start, y_end, x_start, x_end)

        if is_ch_first:
            ori_patch = np.array(ori_mmap[:, z_start:z_end, y_start:y_end, x_start:x_end], dtype=np.float32).copy()
        else:
            ori_patch = np.array(ori_mmap[z_start:z_end, y_start:y_end, x_start:x_end], dtype=np.float32)
            ori_patch = np.transpose(ori_patch, (3, 0, 1, 2)).copy()

        # Smart Separable Fiber Stamping: Pastes curated fibers only if centerlines maintain >= 6.0 voxels distance
        if self.augment and self.real_fibers and random.random() < self.real_stamp_prob:
            num_stamps = random.randint(1, 2)
            for _ in range(num_stamps):
                r_vol, r_int, r_ori = random.choice(self.real_fibers)
                s_vol, s_int, s_ori = _augment_stamp(r_vol, r_int, r_ori)
                vol_patch, intensity_patch, ori_patch, _ = stamp_fiber_if_separable(
                    vol_patch, intensity_patch, ori_patch,
                    s_vol, s_int, s_ori,
                    min_centerline_dist=6.0,
                    max_attempts=15,
                    intersection_dip=-0.5
                )

        # Data Augmentations: Random 3D Flips and Orthogonal Rotations only
        if self.augment:
            if random.random() > 0.5:
                vol_patch = np.flip(vol_patch, axis=0).copy()
                intensity_patch = np.flip(intensity_patch, axis=0).copy()
                ori_patch = np.flip(ori_patch, axis=1).copy()
                ori_patch[0] = -ori_patch[0]

            if random.random() > 0.5:
                vol_patch = np.flip(vol_patch, axis=1).copy()
                intensity_patch = np.flip(intensity_patch, axis=1).copy()
                ori_patch = np.flip(ori_patch, axis=2).copy()
                ori_patch[1] = -ori_patch[1]

            if random.random() > 0.5:
                vol_patch = np.flip(vol_patch, axis=2).copy()
                intensity_patch = np.flip(intensity_patch, axis=2).copy()
                ori_patch = np.flip(ori_patch, axis=3).copy()
                ori_patch[2] = -ori_patch[2]

            k_yx = random.randint(0, 3)
            if k_yx > 0:
                vol_patch = np.rot90(vol_patch, k=k_yx, axes=(1, 2)).copy()
                intensity_patch = np.rot90(intensity_patch, k=k_yx, axes=(1, 2)).copy()
                ori_z = np.rot90(ori_patch[0], k=k_yx, axes=(1, 2)).copy()
                ori_y = np.rot90(ori_patch[1], k=k_yx, axes=(1, 2)).copy()
                ori_x = np.rot90(ori_patch[2], k=k_yx, axes=(1, 2)).copy()
                if k_yx == 1:
                    ori_patch[0], ori_patch[1], ori_patch[2] = ori_z, -ori_x, ori_y
                elif k_yx == 2:
                    ori_patch[0], ori_patch[1], ori_patch[2] = ori_z, -ori_y, -ori_x
                elif k_yx == 3:
                    ori_patch[0], ori_patch[1], ori_patch[2] = ori_z, ori_x, -ori_y

            k_zx = random.randint(0, 3)
            if k_zx > 0:
                vol_patch = np.rot90(vol_patch, k=k_zx, axes=(0, 2)).copy()
                intensity_patch = np.rot90(intensity_patch, k=k_zx, axes=(0, 2)).copy()
                ori_z = np.rot90(ori_patch[0], k=k_zx, axes=(0, 2)).copy()
                ori_y = np.rot90(ori_patch[1], k=k_zx, axes=(0, 2)).copy()
                ori_x = np.rot90(ori_patch[2], k=k_zx, axes=(0, 2)).copy()
                if k_zx == 1:
                    ori_patch[0], ori_patch[1], ori_patch[2] = -ori_x, ori_y, ori_z
                elif k_zx == 2:
                    ori_patch[0], ori_patch[1], ori_patch[2] = -ori_z, ori_y, -ori_x
                elif k_zx == 3:
                    ori_patch[0], ori_patch[1], ori_patch[2] = ori_x, ori_y, -ori_z

            k_zy = random.randint(0, 3)
            if k_zy > 0:
                vol_patch = np.rot90(vol_patch, k=k_zy, axes=(0, 1)).copy()
                intensity_patch = np.rot90(intensity_patch, k=k_zy, axes=(0, 1)).copy()
                ori_z = np.rot90(ori_patch[0], k=k_zy, axes=(0, 1)).copy()
                ori_y = np.rot90(ori_patch[1], k=k_zy, axes=(0, 1)).copy()
                ori_x = np.rot90(ori_patch[2], k=k_zy, axes=(0, 1)).copy()
                if k_zy == 1:
                    ori_patch[0], ori_patch[1], ori_patch[2] = -ori_y, ori_z, ori_x
                elif k_zy == 2:
                    ori_patch[0], ori_patch[1], ori_patch[2] = -ori_z, -ori_y, ori_x
                elif k_zy == 3:
                    ori_patch[0], ori_patch[1], ori_patch[2] = ori_y, -ori_z, ori_x

        # Convert to PyTorch Tensors (strictly binary 0.0/1.0 volume)
        vol_patch = (vol_patch > 0.5).astype(np.float32)
        vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0)
        intensity_tensor = torch.from_numpy(intensity_patch).unsqueeze(0)
        ori_tensor = torch.from_numpy(ori_patch)

        return vol_tensor, intensity_tensor, ori_tensor


class GADSplineBank:
    """Fast in-memory cache of synthetic GAD spline centerlines for dynamic patch cropping."""
    def __init__(self, raw_dir='data/synthetic/raw'):
        self.models = []
        if not os.path.exists(raw_dir):
            return

        files = sorted([f for f in os.listdir(raw_dir) if f.startswith('AJ_model_') and f.endswith('.gad')])
        for f in files:
            m_path = os.path.join(raw_dir, f)
            try:
                with open(m_path, 'r', encoding='utf-8') as fp:
                    gad = json.load(fp)
                voxel_len = gad['Domain']['VoxelLength'][0]
                curves = []
                num_objs = gad.get('NumberOfObjects', 0)
                for o_idx in range(1, num_objs + 1):
                    obj = gad.get(f'Object{o_idx}', {})
                    p_keys = sorted([k for k in obj.keys() if k.startswith('Point')], key=lambda x: int(x[5:]))
                    pts_list = []
                    for pk in p_keys:
                        p_val = obj[pk]
                        if isinstance(p_val, dict) and 'Coord' in p_val:
                            pts_list.append(p_val['Coord'][0])
                        elif isinstance(p_val, (list, tuple)):
                            pts_list.append(p_val[0])
                        elif isinstance(p_val, dict):
                            pts_list.append(list(p_val.values())[0])
                    pts = np.array(pts_list, dtype=np.float32)
                    if len(pts) >= 2:
                        pts_zyx = (pts / voxel_len)[:, [2, 1, 0]]
                        # Dense interpolation along spline
                        n_pts = len(pts_zyx)
                        t_d = np.linspace(0, 1, n_pts * 6)
                        t_s = np.linspace(0, 1, n_pts)
                        dz = np.interp(t_d, t_s, pts_zyx[:, 0])
                        dy = np.interp(t_d, t_s, pts_zyx[:, 1])
                        dx = np.interp(t_d, t_s, pts_zyx[:, 2])
                        curves.append(np.column_stack([dz, dy, dx]).astype(np.float32))
                if curves:
                    self.models.append(curves)
            except Exception as e:
                print(f"Warning: Failed to load GAD spline file '{m_path}': {e}", flush=True)

    def sample_crop_curves(self, patch_size=64, min_fibers=2):
        if not self.models:
            return []
        model = random.choice(self.models)
        for _ in range(30):
            orig = np.random.uniform(15, 500 - patch_size - 15, size=3)
            box_min = orig
            box_max = orig + patch_size

            crop_curves = []
            for c in model:
                inside = (c[:, 0] >= box_min[0]) & (c[:, 0] < box_max[0]) & \
                         (c[:, 1] >= box_min[1]) & (c[:, 1] < box_max[1]) & \
                         (c[:, 2] >= box_min[2]) & (c[:, 2] < box_max[2])
                if np.sum(inside) >= 6:
                    sub_c = c[inside] - box_min
                    crop_curves.append(sub_c.astype(np.float32))
            if len(crop_curves) >= min_fibers:
                return crop_curves
        return crop_curves


class OnTheFlyMorphedDataset(Dataset):
    """
    High-Throughput Morphed Biological Fiber Dataset with Full-Volume Sampling.

    Architecture:
    1. Maintains an active pool of 96x96x96 parent volumes (both synthetic-morphed and real-curated).
    2. Samples 64x64x64 sub-crops uniformly across the ENTIRE 3D volume of any parent in the pool:
       (z0, y0, x0) in [0, 32]^3 with random 3D flips and orthogonal rotations.
    3. Continuously evolves the pool with fresh biological blocks for infinite dataset variety.
    4. Delivers ultra-high throughput (>500 patches/sec) with zero GPU dataloader starvation.
    """
    def __init__(
        self,
        raw_dir='data/synthetic/raw',
        real_data_dir='data/curated/patches',
        patch_size=64,
        parent_block_size=96,
        pool_size=10,
        samples_per_epoch=800,
        augment=True,
        jitter_std=2.5,
        wobble_amplitude=1.2,
        real_patch_prob=0.30,
        cache_library_path='data/curated/fiber_library.pkl'
    ):
        super().__init__()
        self.patch_size = patch_size
        self.parent_block_size = max(patch_size, parent_block_size)
        self.pool_size = max(1, pool_size)
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.jitter_std = jitter_std
        self.wobble_amplitude = wobble_amplitude
        self.real_patch_prob = real_patch_prob

        # Import fiber morpher components
        from core.fiber_morpher import RealFiberLibrary, render_morphed_synthetic_patch
        self.render_fn = render_morphed_synthetic_patch
        self.library = RealFiberLibrary(curated_dir=real_data_dir, cache_path=cache_library_path)
        self.spline_bank = GADSplineBank(raw_dir=raw_dir)

        # Discover and preload unique real curated 96^3 patches (deduplicate .tif and .npy)
        self.real_pool = []
        if real_data_dir and os.path.exists(real_data_dir):
            bases = sorted(list(set(
                f.replace('_vol.tif', '').replace('_vol.npy', '')
                for f in os.listdir(real_data_dir)
                if f.endswith('_vol.tif') or f.endswith('_vol.npy')
            )))
            for base in bases:
                v_p_tif = os.path.join(real_data_dir, f"{base}_vol.tif")
                v_p_npy = os.path.join(real_data_dir, f"{base}_vol.npy")
                v_p = v_p_tif if os.path.exists(v_p_tif) else v_p_npy
                i_p = os.path.join(real_data_dir, f"{base}_intensity.npy")
                o_p = os.path.join(real_data_dir, f"{base}_ori.npy")
                if os.path.exists(v_p) and os.path.exists(i_p) and os.path.exists(o_p):
                    v = tifffile.imread(v_p) if v_p.endswith('.tif') else np.load(v_p)
                    v = (v > 0).astype(bool)
                    int_t = np.load(i_p).astype(np.float32)
                    ori_t = np.load(o_p).astype(np.float32)
                    if ori_t.ndim == 3:
                        ori_t = ori_t[None, ...]
                    self.real_pool.append((v, int_t, ori_t))

        # Initialize active morphed synthetic pool
        self.synthetic_pool = []
        print(f"Pre-rendering initial pool of {self.pool_size} diverse 96^3 morphed biological blocks...", flush=True)
        for i in range(self.pool_size):
            self.synthetic_pool.append(self._generate_parent_block())

        print(f"Initialized OnTheFlyMorphedDataset: Pool of {len(self.synthetic_pool)} morphed blocks + "
              f"{len(self.real_pool)} unique real curated blocks. Sampling uniformly across all 3D coordinates.", flush=True)

    def __len__(self):
        return self.samples_per_epoch

    def _augment_curves(self, curves, S=96):
        aug = []
        flip_z = (random.random() > 0.5)
        flip_y = (random.random() > 0.5)
        flip_x = (random.random() > 0.5)
        rot_k = random.randint(0, 3)
        shift = np.random.normal(0, self.jitter_std, size=(1, 3)).astype(np.float32)

        for c in curves:
            c_aug = c.copy() + shift

            # 3D Flips
            if flip_z: c_aug[:, 0] = (S - 1) - c_aug[:, 0]
            if flip_y: c_aug[:, 1] = (S - 1) - c_aug[:, 1]
            if flip_x: c_aug[:, 2] = (S - 1) - c_aug[:, 2]

            # 3D Orthogonal Rotations
            if rot_k > 0:
                for _ in range(rot_k):
                    y_old, x_old = c_aug[:, 1].copy(), c_aug[:, 2].copy()
                    c_aug[:, 1] = x_old
                    c_aug[:, 2] = (S - 1) - y_old

            # Biological micro-crimp / sinusoidal wobble
            n_pts = len(c_aug)
            if n_pts > 4 and self.wobble_amplitude > 0:
                t = np.linspace(0, 2 * np.pi, n_pts)
                wobble_dir = np.random.normal(0, 1, size=(1, 3))
                wobble_dir /= (np.linalg.norm(wobble_dir) + 1e-8)
                c_aug += np.sin(t)[:, None] * wobble_dir * random.uniform(0.5, self.wobble_amplitude)

            aug.append(c_aug.astype(np.float32))
        return aug

    def _generate_parent_block(self):
        """Generates a fresh 96x96x96 morphed biological parent block."""
        P = self.parent_block_size
        curves = self.spline_bank.sample_crop_curves(patch_size=P, min_fibers=3)
        if not curves:
            vol = np.zeros((P, P, P), dtype=bool)
            int_t = np.zeros((P, P, P), dtype=np.float32)
            ori_t = np.zeros((3, P, P, P), dtype=np.float32)
            return vol, int_t, ori_t

        if self.augment:
            curves = self._augment_curves(curves, S=P)

        vol_p, int_p, ori_p = self.render_fn(curves, self.library, patch_size=P)
        return vol_p, int_p, ori_p

    def refresh_epoch_pool(self, n_blocks=None):
        """Morphs a fresh batch of 96^3 biological blocks for the new epoch."""
        n = n_blocks or self.pool_size
        self.synthetic_pool = [self._generate_parent_block() for _ in range(n)]

    def refresh_pool_block(self):
        """Evolves the pool by replacing a random block with a newly rendered morphed volume."""
        self.refresh_epoch_pool()

    def __getitem__(self, idx):
        # 1. Select block from real pool or synthetic morphed pool
        if self.real_pool and random.random() < self.real_patch_prob:
            p_vol, p_int, p_ori = random.choice(self.real_pool)
        else:
            p_vol, p_int, p_ori = random.choice(self.synthetic_pool)

        P_z, P_y, P_x = p_vol.shape
        S = self.patch_size

        # 2. Sample uniformly across the WHOLE 3D volume
        z0 = random.randint(0, max(0, P_z - S))
        y0 = random.randint(0, max(0, P_y - S))
        x0 = random.randint(0, max(0, P_x - S))

        vol_patch = p_vol[z0:z0+S, y0:y0+S, x0:x0+S].copy()
        int_patch = p_int[z0:z0+S, y0:y0+S, x0:x0+S].copy()
        ori_patch = p_ori[:, z0:z0+S, y0:y0+S, x0:x0+S].copy()

        # 3. Apply full 3D spatial transformations (flips & orthogonal rotations)
        if self.augment:
            if random.random() > 0.5:
                vol_patch = np.flip(vol_patch, axis=0).copy()
                int_patch = np.flip(int_patch, axis=0).copy()
                ori_patch = np.flip(ori_patch, axis=1).copy()
                ori_patch[0] = -ori_patch[0]
            if random.random() > 0.5:
                vol_patch = np.flip(vol_patch, axis=1).copy()
                int_patch = np.flip(int_patch, axis=1).copy()
                ori_patch = np.flip(ori_patch, axis=2).copy()
                ori_patch[1] = -ori_patch[1]
            if random.random() > 0.5:
                vol_patch = np.flip(vol_patch, axis=2).copy()
                int_patch = np.flip(int_patch, axis=2).copy()
                ori_patch = np.flip(ori_patch, axis=3).copy()
                ori_patch[2] = -ori_patch[2]

            k_rot = random.randint(0, 3)
            if k_rot > 0:
                vol_patch = np.rot90(vol_patch, k=k_rot, axes=(1, 2)).copy()
                int_patch = np.rot90(int_patch, k=k_rot, axes=(1, 2)).copy()
                oz = np.rot90(ori_patch[0], k=k_rot, axes=(1, 2)).copy()
                oy = np.rot90(ori_patch[1], k=k_rot, axes=(1, 2)).copy()
                ox = np.rot90(ori_patch[2], k=k_rot, axes=(1, 2)).copy()
                if k_rot == 1:
                    ori_patch[0], ori_patch[1], ori_patch[2] = oz, -ox, oy
                elif k_rot == 2:
                    ori_patch[0], ori_patch[1], ori_patch[2] = oz, -oy, -ox
                elif k_rot == 3:
                    ori_patch[0], ori_patch[1], ori_patch[2] = oz, ox, -oy

        vol_tensor = torch.from_numpy(vol_patch.astype(np.float32)).unsqueeze(0)
        int_tensor = torch.from_numpy(int_patch).unsqueeze(0)
        ori_tensor = torch.from_numpy(ori_patch)

        return vol_tensor, int_tensor, ori_tensor

