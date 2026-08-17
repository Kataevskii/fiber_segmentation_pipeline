import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile
from scipy.ndimage import (
    binary_closing, binary_dilation, binary_erosion, binary_opening,
    generate_binary_structure, gaussian_filter, distance_transform_edt
)

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
        real_data_dir='real_train_data',
        real_stamp_prob=0.50,
        verbose=False
    ):
        super().__init__()
        self.patch_size = patch_size
        self.samples_per_epoch = samples_per_epoch
        self.augment = augment
        self.fg_prob = fg_prob
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

            # Add slight random jitter
            z_start = z_c - p // 2 + random.randint(-4, 4)
            y_start = y_c - p // 2 + random.randint(-4, 4)
            x_start = x_c - p // 2 + random.randint(-4, 4)
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

        # Data Augmentations: Random 3D Flips, Rotations & Intensity Scaling
        if self.augment:
            # 1. Flip along Z, Y, X axes
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

            # 2. Full 3D Orthogonal Rotations across all 3 spatial planes (YX, ZX, ZY)
            # Plane 1: YX plane (axes 1, 2)
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

            # Plane 2: ZX plane (axes 0, 2)
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

            # Plane 3: ZY plane (axes 0, 1)
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

            # 3. Dense Multi-Fiber Collision & Touching Bundle Overlay (75% probability)
            if random.random() < 0.75:
                alt_idx = random.randint(0, len(self.volume_paths) - 1)
                alt_v_path = self.volume_paths[alt_idx]
                alt_o_path = self.orientation_paths[alt_idx]
                alt_i_path = self.intensity_paths[alt_idx]
                alt_coords = self.fiber_coords_list[alt_idx]

                alt_ori_mmap = np.load(alt_o_path, mmap_mode='r')
                if alt_ori_mmap.shape[0] == 3:
                    _, alt_D, alt_H, alt_W = alt_ori_mmap.shape
                    alt_ch_first = True
                else:
                    alt_D, alt_H, alt_W, _ = alt_ori_mmap.shape
                    alt_ch_first = False

                if alt_D >= p and alt_H >= p and alt_W >= p and len(alt_coords) > 0:
                    alt_center = alt_coords[random.randint(0, len(alt_coords) - 1)]
                    shift_z = random.randint(-8, 8)
                    shift_y = random.randint(-8, 8)
                    shift_x = random.randint(-8, 8)

                    alt_z = max(0, min(alt_D - p, int(alt_center[0]) - p // 2 + shift_z))
                    alt_y = max(0, min(alt_H - p, int(alt_center[1]) - p // 2 + shift_y))
                    alt_x = max(0, min(alt_W - p, int(alt_center[2]) - p // 2 + shift_x))

                    alt_vol = self._load_slice(alt_v_path, alt_z, alt_z + p, alt_y, alt_y + p, alt_x, alt_x + p)
                    alt_int = self._load_slice(alt_i_path, alt_z, alt_z + p, alt_y, alt_y + p, alt_x, alt_x + p)

                    if alt_ch_first:
                        alt_ori = np.array(alt_ori_mmap[:, alt_z:alt_z+p, alt_y:alt_y+p, alt_x:alt_x+p], dtype=np.float32).copy()
                    else:
                        alt_ori = np.array(alt_ori_mmap[alt_z:alt_z+p, alt_y:alt_y+p, alt_x:alt_x+p], dtype=np.float32)
                        alt_ori = np.transpose(alt_ori, (3, 0, 1, 2)).copy()

                    if alt_vol.shape == (p, p, p):
                        # Apply random 3D flip/rotation to colliding fiber
                        if random.random() > 0.5:
                            alt_vol = np.flip(alt_vol, axis=0).copy()
                            alt_int = np.flip(alt_int, axis=0).copy()
                            alt_ori = np.flip(alt_ori, axis=1).copy()
                            alt_ori[0] = -alt_ori[0]
                        if random.random() > 0.5:
                            alt_vol = np.rot90(alt_vol, k=random.randint(1, 3), axes=(1, 2)).copy()
                            alt_int = np.rot90(alt_int, k=random.randint(1, 3), axes=(1, 2)).copy()
                            alt_ori = np.rot90(alt_ori, k=random.randint(1, 3), axes=(2, 3)).copy()

                        # Merge fiber volumes
                        alt_bin = (alt_vol > 0.5).astype(np.float32)
                        base_bin = (vol_patch > 0.5).astype(np.float32)
                        overlap_mask = (base_bin > 0) & (alt_bin > 0)

                        vol_patch = np.maximum(vol_patch, alt_bin)

                        # Update orientation: where alt fiber is stronger, assign alt orientation
                        use_alt = (alt_int > intensity_patch) | ((base_bin == 0) & (alt_bin > 0))
                        for c in range(3):
                            ori_patch[c] = np.where(use_alt, alt_ori[c], ori_patch[c])

                        # At overlapping collision interface, create negative intersection dip
                        intensity_patch = np.where(overlap_mask, -1.0, np.maximum(intensity_patch, alt_int))

            # 4. 3D Copy-Paste Real Fiber Stamping (if curated real data exists)
            if len(self.real_fibers) > 0 and random.random() < self.real_stamp_prob:
                num_stamps = random.randint(1, min(2, len(self.real_fibers)))
                for _ in range(num_stamps):
                    r_vol, r_int, r_ori = random.choice(self.real_fibers)
                    r_vol, r_int, r_ori = r_vol.copy(), r_int.copy(), r_ori.copy()
                    if r_ori.shape[-1] == 3 and r_ori.ndim == 4:
                        r_ori = np.transpose(r_ori, (3, 0, 1, 2))

                    # Random 3D spatial flip
                    if random.random() > 0.5:
                        r_vol = np.flip(r_vol, axis=0).copy()
                        r_int = np.flip(r_int, axis=0).copy()
                        r_ori[0] = -np.flip(r_ori[0], axis=0).copy()
                        r_ori[1] = np.flip(r_ori[1], axis=0).copy()
                        r_ori[2] = np.flip(r_ori[2], axis=0).copy()
                    if random.random() > 0.5:
                        r_vol = np.flip(r_vol, axis=1).copy()
                        r_int = np.flip(r_int, axis=1).copy()
                        r_ori[0] = np.flip(r_ori[0], axis=1).copy()
                        r_ori[1] = -np.flip(r_ori[1], axis=1).copy()
                        r_ori[2] = np.flip(r_ori[2], axis=1).copy()
                    if random.random() > 0.5:
                        r_vol = np.flip(r_vol, axis=2).copy()
                        r_int = np.flip(r_int, axis=2).copy()
                        r_ori[0] = np.flip(r_ori[0], axis=2).copy()
                        r_ori[1] = np.flip(r_ori[1], axis=2).copy()
                        r_ori[2] = -np.flip(r_ori[2], axis=2).copy()

                    # Random 90 deg rotation in YX plane
                    k = random.randint(0, 3)
                    if k > 0:
                        r_vol = np.rot90(r_vol, k=k, axes=(1, 2)).copy()
                        r_int = np.rot90(r_int, k=k, axes=(1, 2)).copy()
                        oz = np.rot90(r_ori[0], k=k, axes=(1, 2)).copy()
                        oy = np.rot90(r_ori[1], k=k, axes=(1, 2)).copy()
                        ox = np.rot90(r_ori[2], k=k, axes=(1, 2)).copy()
                        if k == 1:
                            r_ori = np.stack([oz, -ox, oy], axis=0)
                        elif k == 2:
                            r_ori = np.stack([oz, -oy, -ox], axis=0)
                        elif k == 3:
                            r_ori = np.stack([oz, ox, -oy], axis=0)

                    rd, rh, rw = r_vol.shape
                    # Crop if larger than patch size p
                    if rd > p or rh > p or rw > p:
                        sz_start = random.randint(0, max(0, rd - p)) if rd > p else 0
                        sy_start = random.randint(0, max(0, rh - p)) if rh > p else 0
                        sx_start = random.randint(0, max(0, rw - p)) if rw > p else 0
                        r_vol = r_vol[sz_start:sz_start+min(rd, p), sy_start:sy_start+min(rh, p), sx_start:sx_start+min(rw, p)]
                        r_int = r_int[sz_start:sz_start+min(rd, p), sy_start:sy_start+min(rh, p), sx_start:sx_start+min(rw, p)]
                        r_ori = r_ori[:, sz_start:sz_start+min(rd, p), sy_start:sy_start+min(rh, p), sx_start:sx_start+min(rw, p)]
                        rd, rh, rw = r_vol.shape

                    # Pick random insertion origin inside patch
                    dz0 = random.randint(0, max(0, p - rd))
                    dy0 = random.randint(0, max(0, p - rh))
                    dx0 = random.randint(0, max(0, p - rw))

                    # Overlap / collision detection
                    fg_target = (r_vol > 0.5)
                    fg_dest = (vol_patch[dz0:dz0+rd, dy0:dy0+rh, dx0:dx0+rw] > 0.5)
                    collision_mask = fg_target & fg_dest

                    # Blend binary volume
                    vol_patch[dz0:dz0+rd, dy0:dy0+rh, dx0:dx0+rw] = np.maximum(
                        vol_patch[dz0:dz0+rd, dy0:dy0+rh, dx0:dx0+rw], r_vol
                    )

                    # Blend intensity potential field (with -1.0 dip at crossing collision)
                    dest_int = intensity_patch[dz0:dz0+rd, dy0:dy0+rh, dx0:dx0+rw]
                    blended_int = np.where(fg_target, np.maximum(dest_int, r_int), dest_int)
                    blended_int[collision_mask] = -1.0
                    intensity_patch[dz0:dz0+rd, dy0:dy0+rh, dx0:dx0+rw] = blended_int

                    # Blend orientation vectors where real fiber is foreground
                    dest_ori = ori_patch[:, dz0:dz0+rd, dy0:dy0+rh, dx0:dx0+rw]
                    for c in range(3):
                        dest_ori[c] = np.where(fg_target, r_ori[c], dest_ori[c])
                    ori_patch[:, dz0:dz0+rd, dy0:dy0+rh, dx0:dx0+rw] = dest_ori

            # 5. Fiber Diameter & Localized Radius Variation (Spherical Beads, Ellipsoids, and Bulges)
            if random.random() > 0.3:
                fg_indices = np.argwhere(vol_patch > 0.5)
                if len(fg_indices) > 0:
                    centerline_indices = np.argwhere(intensity_patch > 0.25)
                    sample_pool = centerline_indices if len(centerline_indices) > 0 else fg_indices
                    
                    num_bulges = random.randint(2, 6)
                    sampled_centers = sample_pool[np.random.choice(len(sample_pool), size=min(num_bulges, len(sample_pool)), replace=False)]
                    
                    D_p, H_p, W_p = vol_patch.shape
                    
                    for z_c, y_c, x_c in sampled_centers:
                        var_type = random.choice(['sphere', 'ellipsoid', 'thinning'])
                        
                        local_ori = ori_patch[:, z_c, y_c, x_c]
                        norm_ori = np.linalg.norm(local_ori)
                        if norm_ori > 1e-6:
                            local_ori = local_ori / norm_ori
                        else:
                            local_ori = np.array([1.0, 0.0, 0.0], dtype=np.float32)
                            
                        if var_type == 'sphere':
                            r_var = random.uniform(2.8, 4.5)
                            k = int(np.ceil(r_var))
                            z_min, z_max = max(0, z_c - k), min(D_p, z_c + k + 1)
                            y_min, y_max = max(0, y_c - k), min(H_p, y_c + k + 1)
                            x_min, x_max = max(0, x_c - k), min(W_p, x_c + k + 1)
                            
                            gz, gy, gx = np.ogrid[z_min-z_c:z_max-z_c, y_min-y_c:y_max-y_c, x_min-x_c:x_max-x_c]
                            mask_shape = (gz**2 + gy**2 + gx**2) <= (r_var**2)
                            
                            sub_vol = vol_patch[z_min:z_max, y_min:y_max, x_min:x_max]
                            newly_added = mask_shape & (sub_vol <= 0.5)
                            vol_patch[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(sub_vol, mask_shape.astype(np.float32))
                            
                            for c in range(3):
                                ori_sub = ori_patch[c, z_min:z_max, y_min:y_max, x_min:x_max]
                                ori_patch[c, z_min:z_max, y_min:y_max, x_min:x_max] = np.where(newly_added, local_ori[c], ori_sub)
                                
                        elif var_type == 'ellipsoid':
                            rz = random.uniform(2.2, 5.0)
                            ry = random.uniform(2.2, 5.0)
                            rx = random.uniform(2.2, 5.0)
                            
                            dom_axis = int(np.argmax(np.abs(local_ori)))
                            if dom_axis == 0: rz = random.uniform(3.5, 6.0)
                            elif dom_axis == 1: ry = random.uniform(3.5, 6.0)
                            else: rx = random.uniform(3.5, 6.0)
                            
                            kz, ky, kx = int(np.ceil(rz)), int(np.ceil(ry)), int(np.ceil(rx))
                            z_min, z_max = max(0, z_c - kz), min(D_p, z_c + kz + 1)
                            y_min, y_max = max(0, y_c - ky), min(H_p, y_c + ky + 1)
                            x_min, x_max = max(0, x_c - kx), min(W_p, x_c + kx + 1)
                            
                            gz, gy, gx = np.ogrid[z_min-z_c:z_max-z_c, y_min-y_c:y_max-y_c, x_min-x_c:x_max-x_c]
                            mask_shape = ((gz / rz)**2 + (gy / ry)**2 + (gx / rx)**2) <= 1.0
                            
                            sub_vol = vol_patch[z_min:z_max, y_min:y_max, x_min:x_max]
                            newly_added = mask_shape & (sub_vol <= 0.5)
                            vol_patch[z_min:z_max, y_min:y_max, x_min:x_max] = np.maximum(sub_vol, mask_shape.astype(np.float32))
                            
                            for c in range(3):
                                ori_sub = ori_patch[c, z_min:z_max, y_min:y_max, x_min:x_max]
                                ori_patch[c, z_min:z_max, y_min:y_max, x_min:x_max] = np.where(newly_added, local_ori[c], ori_sub)
                                
                        elif var_type == 'thinning':
                            r_thin = random.uniform(1.8, 3.0)
                            k = int(np.ceil(r_thin))
                            offset_dir = np.random.normal(0, 1, 3)
                            offset_dir -= np.dot(offset_dir, local_ori) * local_ori
                            norm_off = np.linalg.norm(offset_dir)
                            if norm_off > 1e-6:
                                offset_dir = offset_dir / norm_off * random.uniform(1.8, 2.5)
                                z_off = int(np.clip(round(z_c + offset_dir[0]), 0, D_p - 1))
                                y_off = int(np.clip(round(y_c + offset_dir[1]), 0, H_p - 1))
                                x_off = int(np.clip(round(x_c + offset_dir[2]), 0, W_p - 1))
                                
                                z_min, z_max = max(0, z_off - k), min(D_p, z_off + k + 1)
                                y_min, y_max = max(0, y_off - k), min(H_p, y_off + k + 1)
                                x_min, x_max = max(0, x_off - k), min(W_p, x_off + k + 1)
                                
                                gz, gy, gx = np.ogrid[z_min-z_off:z_max-z_off, y_min-y_off:y_max-y_off, x_min-x_off:x_max-x_off]
                                mask_bite = (gz**2 + gy**2 + gx**2) <= (r_thin**2)
                                
                                sub_intensity = intensity_patch[z_min:z_max, y_min:y_max, x_min:x_max]
                                can_remove = mask_bite & (sub_intensity < 0.4)
                                vol_patch[z_min:z_max, y_min:y_max, x_min:x_max][can_remove] = 0.0

            # 5. 3D Morphological Filters for Inter-Fiber Visual Blending & Sintering (CT Simulation)
            # In real CT images, fibers that are near or touching merge across small gaps due to PSF / partial-volume effects.
            if random.random() > 0.30:
                bin_vol = (vol_patch > 0.5)
                if np.any(bin_vol):
                    # 5a. Inter-Fiber Morphological Closing & Partial-Volume Saddle Bridging
                    conn = random.choice([1, 2, 3]) # 6, 18, 26 connectivity
                    struct = generate_binary_structure(3, conn)
                    iters = random.choice([1, 2])
                    closed = binary_closing(bin_vol, structure=struct, iterations=iters)
                    new_bridges = closed & ~bin_vol
                    
                    if np.any(new_bridges):
                        # Keep a random fraction of bridging voxels for organic irregular fusion
                        keep_fraction = random.uniform(0.5, 0.95)
                        bridge_mask = new_bridges & (np.random.random(size=vol_patch.shape) < keep_fraction)
                        
                        if np.any(bridge_mask):
                            # Distance transform to assign nearest fiber orientation to the bridge voxels
                            _, (ind_z, ind_y, ind_x) = distance_transform_edt(~bin_vol, return_indices=True)
                            
                            bin_vol = bin_vol | bridge_mask
                            vol_patch = bin_vol.astype(np.float32)
                            
                            for c in range(3):
                                ori_patch[c] = np.where(bridge_mask, ori_patch[c, ind_z, ind_y, ind_x], ori_patch[c])
                                
                            # Bridges are NOT centerlines -> mark intensity as non-centerline (<= 0.0)
                            intensity_patch = np.where(bridge_mask, np.minimum(intensity_patch, 0.0), intensity_patch)

            # 5b. Anisotropic Morphological Dilation / Erosion (Eccentricity & Flattening)
            if random.random() > 0.45:
                bin_vol = (vol_patch > 0.5)
                if np.any(bin_vol):
                    struct_aniso = np.zeros((3, 3, 3), dtype=bool)
                    struct_aniso[1, 1, 1] = True
                    num_nbrs = random.randint(2, 5)
                    coords = np.argwhere(generate_binary_structure(3, 1))
                    chosen = coords[np.random.choice(len(coords), size=num_nbrs, replace=False)]
                    for z_c, y_c, x_c in chosen:
                        struct_aniso[z_c, y_c, x_c] = True
                        
                    if random.random() > 0.4:
                        dilated = binary_dilation(bin_vol, structure=struct_aniso, iterations=1)
                        new_vox = dilated & ~bin_vol
                        if np.any(new_vox):
                            _, (ind_z, ind_y, ind_x) = distance_transform_edt(~bin_vol, return_indices=True)
                            bin_vol = dilated
                            vol_patch = bin_vol.astype(np.float32)
                            for c in range(3):
                                ori_patch[c] = np.where(new_vox, ori_patch[c, ind_z, ind_y, ind_x], ori_patch[c])
                    else:
                        eroded = binary_erosion(bin_vol, structure=struct_aniso, iterations=1)
                        core_protected = (intensity_patch > 0.35)
                        bin_vol = eroded | core_protected
                        vol_patch = bin_vol.astype(np.float32)

            # 5c. CT Point-Spread-Function (PSF) Gaussian Convolution + Soft Thresholding
            if random.random() > 0.45:
                sigma_psf = random.uniform(0.5, 1.1)
                blurred_vol = gaussian_filter(vol_patch, sigma=sigma_psf)
                noise = np.random.normal(0, 0.03, size=vol_patch.shape).astype(np.float32)
                noisy_blurred = np.clip(blurred_vol + noise, 0.0, 1.0)
                thresh = random.uniform(0.35, 0.65)
                vol_patch = (noisy_blurred >= thresh).astype(np.float32)

            # 6. Binary Fiber Border Roughness & Surface Scalloping (Flipping 0/1 on Fiber Boundaries)
            if random.random() > 0.40:
                bin_vol = (vol_patch > 0.5)
                struct_scallop = generate_binary_structure(3, 1)
                dilated = binary_dilation(bin_vol, structure=struct_scallop)
                eroded = binary_erosion(bin_vol, structure=struct_scallop)
                boundary_mask = dilated & ~eroded
                flip_mask = boundary_mask & (np.random.random(size=vol_patch.shape) < 0.12)
                bin_vol[flip_mask] = ~bin_vol[flip_mask]
                vol_patch = bin_vol.astype(np.float32)

            # 7. Binary Salt-and-Pepper Microscopy Noise
            if random.random() > 0.50:
                sp_mask = (np.random.random(size=vol_patch.shape) < 0.005)
                bin_vol = (vol_patch > 0.5)
                bin_vol[sp_mask] = ~bin_vol[sp_mask]
                vol_patch = bin_vol.astype(np.float32)

        # Convert to PyTorch Tensors (strictly binary 0.0/1.0 volume)
        vol_patch = (vol_patch > 0.5).astype(np.float32)
        vol_tensor = torch.from_numpy(vol_patch).unsqueeze(0)
        intensity_tensor = torch.from_numpy(intensity_patch).unsqueeze(0)
        ori_tensor = torch.from_numpy(ori_patch)

        return vol_tensor, intensity_tensor, ori_tensor
