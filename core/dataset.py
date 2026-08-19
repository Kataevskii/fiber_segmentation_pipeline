import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile

def ensure_individual_fibers_extracted(real_data_dir='real_train_data', padding=2, verbose=True):
    """
    Scans real curated training patch blocks (e.g. in real_train_data/curated_patches/)
    and extracts all individual single-fiber sub-volume stamps into
    real_train_data/individual_fibers/ if missing or incomplete.
    """
    if not real_data_dir or not os.path.exists(real_data_dir):
        return 0

    curated_dir = os.path.join(real_data_dir, 'curated_patches')
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
        data_dir='augmented_data',
        volume_paths=None,
        intensity_paths=None,
        orientation_paths=None,
        patch_size=64,
        samples_per_epoch=1200,
        augment=True,
        fg_prob=0.85,
        jitter_voxels=2,
        real_data_dir='real_train_data',
        real_stamp_prob=0.50,
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
