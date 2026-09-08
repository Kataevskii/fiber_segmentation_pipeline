import os
import json
import time
import pickle
import random
import numpy as np
from scipy.spatial import cKDTree
from scipy.ndimage import map_coordinates, gaussian_filter1d
import tifffile

def compute_parallel_transport_frames(curve):
    """
    Computes twist-free orthonormal moving frames (T, N, B) along a 3D curve
    using the Double Reflection Rotation-Minimizing Frame (RMF) algorithm.
    """
    N = len(curve)
    if N < 2:
        T = np.array([[0, 0, 1]], dtype=np.float32)
        N_vec = np.array([[0, 1, 0]], dtype=np.float32)
        B = np.array([[1, 0, 0]], dtype=np.float32)
        return T, N_vec, B

    tangents = np.zeros_like(curve, dtype=np.float32)
    tangents[0] = curve[1] - curve[0]
    tangents[-1] = curve[-1] - curve[-2]
    if N > 2:
        tangents[1:-1] = (curve[2:] - curve[:-2]) / 2.0
    norms = np.linalg.norm(tangents, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    tangents /= norms

    t0 = tangents[0]
    init_v = np.array([1.0, 0.0, 0.0], dtype=np.float32) if abs(t0[0]) < 0.9 else np.array([0.0, 1.0, 0.0], dtype=np.float32)
    n0 = np.cross(t0, init_v)
    n0 /= (np.linalg.norm(n0) + 1e-8)
    b0 = np.cross(t0, n0)

    normals = np.zeros_like(curve, dtype=np.float32)
    binormals = np.zeros_like(curve, dtype=np.float32)
    normals[0] = n0
    binormals[0] = b0

    for i in range(N - 1):
        v1 = curve[i+1] - curve[i]
        c1 = np.dot(v1, v1)
        if c1 < 1e-8:
            normals[i+1] = normals[i]
            binormals[i+1] = binormals[i]
            continue
        r_i = normals[i] - (2.0 / c1) * np.dot(v1, normals[i]) * v1
        t_i = tangents[i] - (2.0 / c1) * np.dot(v1, tangents[i]) * v1
        v2 = tangents[i+1] - t_i
        c2 = np.dot(v2, v2)
        if c2 < 1e-8:
            normals[i+1] = r_i
        else:
            normals[i+1] = r_i - (2.0 / c2) * np.dot(v2, r_i) * v2
        normals[i+1] /= (np.linalg.norm(normals[i+1]) + 1e-8)
        binormals[i+1] = np.cross(tangents[i+1], normals[i+1])

    return tangents, normals, binormals


def center_curve_to_mask(curve, mask):
    """
    Refines skeleton curve points so that every point lies exactly at the
    cross-sectional center of mass of the fiber's binary volume.
    Eliminates 1-2 voxel skeletonization drift.
    """
    mask_coords = np.argwhere(mask > 0.5).astype(np.float32)
    if len(mask_coords) == 0 or len(curve) < 2:
        return curve

    tree = cKDTree(curve)
    _, indices = tree.query(mask_coords, k=1)

    new_curve = curve.copy()
    for k in range(len(curve)):
        assigned = (indices == k)
        if np.sum(assigned) >= 3:
            new_curve[k] = mask_coords[assigned].mean(axis=0)

    new_curve = gaussian_filter1d(new_curve, sigma=1.0, axis=0)
    return new_curve.astype(np.float32)


class RealFiberInstance:
    """Encapsulates a single extracted real biological fiber instance."""
    def __init__(self, fiber_id, patch_index, curve_pts, mask_cropped, crop_bbox, is_boundary_continuous=True):
        self.fiber_id = fiber_id
        self.patch_index = patch_index
        self.mask = mask_cropped.astype(np.float32)
        self.crop_bbox = crop_bbox
        self.is_boundary_continuous = is_boundary_continuous

        # Auto-center curve to cross-sectional center of mass
        raw_curve = np.array(curve_pts, dtype=np.float32)
        self.curve = center_curve_to_mask(raw_curve, self.mask)
        self.num_points = len(self.curve)

        self.tangents, self.normals, self.binormals = compute_parallel_transport_frames(self.curve)
        segs = np.linalg.norm(np.diff(self.curve, axis=0), axis=1) if self.num_points > 1 else np.array([1.0])
        self.cum_arc = np.insert(np.cumsum(segs), 0, 0.0)
        self.total_length = float(self.cum_arc[-1])


class RealFiberLibrary:
    """Library of extracted continuous real fibers from curated patches."""
    def __init__(self, curated_dir='data/curated/patches', cache_path='data/curated/fiber_library.pkl', min_length=75):
        self.curated_dir = curated_dir
        self.cache_path = cache_path
        self.min_length = min_length
        self.fibers = []
        self.load_or_build_library()

    def _is_intact_through_fiber(self, ordered_curve, cube_size=96, radius_margin=3.5):
        """
        Validates that a fiber is an intact through-volume fiber using orientation and boundary geometry:
        1. Both endpoints cleanly penetrate distinct boundary faces with transverse angles (|T_normal| >= 0.20).
        2. The interior trunk NEVER runs parallel along any bounding face (|T_normal| < 0.35 within radius_margin).
           This explicitly excludes fibers that were sliced longitudinally along their trunk by the cube boundary.
        3. Arc length >= self.min_length.
        """
        N = len(ordered_curve)
        if N < 15:
            return False

        segs = np.linalg.norm(np.diff(ordered_curve, axis=0), axis=1)
        total_len = float(np.sum(segs))
        if total_len < self.min_length:
            return False

        tangents = np.zeros_like(ordered_curve, dtype=np.float32)
        tangents[0] = ordered_curve[1] - ordered_curve[0]
        tangents[-1] = ordered_curve[-1] - ordered_curve[-2]
        tangents[1:-1] = (ordered_curve[2:] - ordered_curve[:-2]) / 2.0
        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        tangents /= norms

        faces = [
            ('z_min', 0, 0.0), ('z_max', 0, cube_size - 1.0),
            ('y_min', 1, 0.0), ('y_max', 1, cube_size - 1.0),
            ('x_min', 2, 0.0), ('x_max', 2, cube_size - 1.0)
        ]

        # 1. Interior trunk check: reject if parallel and touching any border
        start_m = max(3, int(0.06 * N))
        end_m = N - start_m
        interior_idx = np.arange(start_m, end_m)
        for face_name, axis, val in faces:
            dists = np.abs(ordered_curve[:, axis] - val)
            t_norm = np.abs(tangents[:, axis])
            # If trunk touches the border face AND runs parallel to it, it was sliced along the trunk
            if np.sum((dists[interior_idx] <= radius_margin) & (t_norm[interior_idx] < 0.35)) >= 3:
                return False

        # 2. Endpoint check: entrance and exit must touch different boundary faces cleanly
        p0, p1 = ordered_curve[0], ordered_curve[-1]
        t0, t1 = tangents[0], tangents[-1]

        def check_endpoint(p, t):
            best_f = None
            min_d = 999.0
            norm_c = 0.0
            for name, axis, val in faces:
                d = abs(p[axis] - val)
                if d < min_d:
                    min_d = d
                    best_f = name
                    norm_c = abs(t[axis])
            if min_d <= 2.5 and norm_c >= 0.20:
                return best_f
            return None

        face_start = check_endpoint(p0, t0)
        face_end = check_endpoint(p1, t1)

        if not face_start or not face_end or face_start == face_end:
            return False

        return True

    def load_or_build_library(self, force_rebuild=False):
        if not force_rebuild and os.path.exists(self.cache_path):
            try:
                with open(self.cache_path, 'rb') as f:
                    self.fibers = pickle.load(f)
                if self.fibers:
                    # Filter for intact through-fibers in cached objects
                    intact_count = sum(1 for f in self.fibers if f.is_boundary_continuous)
                    print(f"Loaded {len(self.fibers)} continuous real fibers from cache '{self.cache_path}' ({intact_count} intact through-volume).", flush=True)
                    return
            except Exception:
                pass

        print(f"Building real fiber library from '{self.curated_dir}'...", flush=True)
        self.fibers = []

        meta_files = sorted([f for f in os.listdir(self.curated_dir) if f.startswith('patch_') and f.endswith('_meta.json')])
        for mf in meta_files:
            p_idx = int(mf.split('_')[1].split('.')[0])
            m_path = os.path.join(self.curated_dir, mf)
            inst_path = os.path.join(self.curated_dir, f"patch_{p_idx:04d}_instance.npy")
            skel_path = os.path.join(self.curated_dir, f"patch_{p_idx:04d}_centerline.npy")

            if not os.path.exists(inst_path) or not os.path.exists(skel_path):
                continue

            with open(m_path, 'r', encoding='utf-8') as fp:
                meta = json.load(fp)

            inst_vol = np.load(inst_path)
            skel_vol = np.load(skel_path)
            cube_size = meta.get('cube_size', 96)

            unique_fids = [fid for fid in np.unique(inst_vol) if fid > 0]
            for fid in unique_fids:
                skel_coords = np.argwhere(skel_vol == fid)
                if len(skel_coords) < self.min_length:
                    continue

                ordered_curve = self._order_curve_points(skel_coords)
                if len(ordered_curve) < self.min_length:
                    continue

                # Check export metadata quality tag first, fallback to geometric evaluation
                q_meta = meta.get('fiber_quality', {}).get(str(fid))
                if q_meta is not None:
                    is_boundary = bool(q_meta.get('is_valid_donor', False))
                else:
                    is_boundary = self._is_intact_through_fiber(ordered_curve, cube_size=cube_size, radius_margin=3.5)

                fiber_mask = (inst_vol == fid)
                z_idx, y_idx, x_idx = np.where(fiber_mask)
                margin = 4
                z0, z1 = max(0, z_idx.min() - margin), min(inst_vol.shape[0], z_idx.max() + margin + 1)
                y0, y1 = max(0, y_idx.min() - margin), min(inst_vol.shape[1], y_idx.max() + margin + 1)
                x0, x1 = max(0, x_idx.min() - margin), min(inst_vol.shape[2], x_idx.max() + margin + 1)

                cropped_mask = fiber_mask[z0:z1, y0:y1, x0:x1]
                curve_rel = ordered_curve - np.array([z0, y0, x0], dtype=np.float32)

                inst = RealFiberInstance(
                    fiber_id=int(fid),
                    patch_index=p_idx,
                    curve_pts=curve_rel,
                    mask_cropped=cropped_mask,
                    crop_bbox=(z0, z1, y0, y1, x0, x1),
                    is_boundary_continuous=is_boundary
                )
                self.fibers.append(inst)

        intact_count = sum(1 for f in self.fibers if f.is_boundary_continuous)
        print(f"Built library with {len(self.fibers)} real fiber instances ({intact_count} intact through-volume).", flush=True)
        os.makedirs(os.path.dirname(self.cache_path) if os.path.dirname(self.cache_path) else '.', exist_ok=True)
        try:
            with open(self.cache_path, 'wb') as f:
                pickle.dump(self.fibers, f)
        except Exception:
            pass

    def _order_curve_points(self, coords):
        if len(coords) < 3:
            return coords.astype(np.float32)

        tree = cKDTree(coords)
        centered = coords - coords.mean(axis=0)
        u, s, vh = np.linalg.svd(centered, full_matrices=False)
        proj = centered @ vh[0]
        start_idx = np.argmin(proj)

        visited = [start_idx]
        curr = start_idx
        while len(visited) < len(coords):
            dists, idxs = tree.query(coords[curr], k=min(12, len(coords)))
            next_idx = None
            for idx in idxs:
                if idx not in visited:
                    next_idx = idx
                    break
            if next_idx is None:
                remaining = [i for i in range(len(coords)) if i not in visited]
                dists_rem = np.linalg.norm(coords[remaining] - coords[curr], axis=1)
                next_idx = remaining[np.argmin(dists_rem)]
            visited.append(next_idx)
            curr = next_idx

        return coords[visited].astype(np.float32)

    def sample_fiber(self, boundary_only=True):
        pool = [f for f in self.fibers if f.is_boundary_continuous] if boundary_only else self.fibers
        if not pool:
            pool = self.fibers
        return random.choice(pool)


def project_points_onto_curve_continuous(pts, curve, t_cum, N_vec, B_vec):
    """
    Computes continuous perpendicular projections onto piecewise-linear 3D curve segments.
    Fully vectorized in NumPy for maximum throughput.
    """
    N_curve = len(curve)
    if N_curve < 2 or len(pts) == 0:
        return pts.copy(), np.zeros(len(pts), dtype=np.float32), N_vec[:1].repeat(len(pts), axis=0), B_vec[:1].repeat(len(pts), axis=0)

    tree = cKDTree(curve)
    _, nearest_idx = tree.query(pts, k=1)

    # Check segment (k-1, k)
    k0_a = np.clip(nearest_idx - 1, 0, N_curve - 2)
    k1_a = k0_a + 1
    v_a = curve[k1_a] - curve[k0_a]
    v_a_sq = np.sum(v_a**2, axis=1, keepdims=True) + 1e-8
    t_a = np.clip(np.sum((pts - curve[k0_a]) * v_a, axis=1, keepdims=True) / v_a_sq, 0.0, 1.0)
    proj_a = curve[k0_a] + t_a * v_a
    dist_a_sq = np.sum((pts - proj_a)**2, axis=1)
    arc_a = t_cum[k0_a] + t_a[:, 0] * (t_cum[k1_a] - t_cum[k0_a])
    N_a = (1.0 - t_a) * N_vec[k0_a] + t_a * N_vec[k1_a]
    B_a = (1.0 - t_a) * B_vec[k0_a] + t_a * B_vec[k1_a]

    # Check segment (k, k+1)
    k0_b = np.clip(nearest_idx, 0, N_curve - 2)
    k1_b = k0_b + 1
    v_b = curve[k1_b] - curve[k0_b]
    v_b_sq = np.sum(v_b**2, axis=1, keepdims=True) + 1e-8
    t_b = np.clip(np.sum((pts - curve[k0_b]) * v_b, axis=1, keepdims=True) / v_b_sq, 0.0, 1.0)
    proj_b = curve[k0_b] + t_b * v_b
    dist_b_sq = np.sum((pts - proj_b)**2, axis=1)
    arc_b = t_cum[k0_b] + t_b[:, 0] * (t_cum[k1_b] - t_cum[k0_b])
    N_b = (1.0 - t_b) * N_vec[k0_b] + t_b * N_vec[k1_b]
    B_b = (1.0 - t_b) * B_vec[k0_b] + t_b * B_vec[k1_b]

    use_b = (dist_b_sq < dist_a_sq)[:, None]
    proj_pts = np.where(use_b, proj_b, proj_a)
    proj_arcs = np.where(use_b[:, 0], arc_b, arc_a)
    proj_N = np.where(use_b, N_b, N_a)
    proj_B = np.where(use_b, B_b, B_a)

    proj_N /= (np.linalg.norm(proj_N, axis=1, keepdims=True) + 1e-8)
    proj_B /= (np.linalg.norm(proj_B, axis=1, keepdims=True) + 1e-8)

    return proj_pts, proj_arcs, proj_N, proj_B


def get_candidate_voxels_fast(curve, radius_margin=6.0, dest_shape=(64, 64, 64)):
    """Fast spherical dilation along 3D curve to select candidate voxels without large bounding boxes."""
    if len(curve) == 0:
        return np.zeros((0, 3), dtype=np.float32)
    step = max(1, int(radius_margin / 2))
    pts = np.round(curve[::step]).astype(np.int32)
    R = int(np.ceil(radius_margin))
    r_grid = np.mgrid[-R:R+1, -R:R+1, -R:R+1].reshape(3, -1).T
    r_sphere = r_grid[np.sum(r_grid**2, axis=1) <= radius_margin**2]

    cand = (pts[:, None, :] + r_sphere[None, :, :]).reshape(-1, 3)
    cand = cand[(cand[:, 0] >= 0) & (cand[:, 0] < dest_shape[0]) &
                (cand[:, 1] >= 0) & (cand[:, 1] < dest_shape[1]) &
                (cand[:, 2] >= 0) & (cand[:, 2] < dest_shape[2])]

    if len(cand) == 0:
        return np.zeros((0, 3), dtype=np.float32)

    flat_idx = cand[:, 0].astype(np.int64) * (dest_shape[1] * dest_shape[2]) + cand[:, 1].astype(np.int64) * dest_shape[2] + cand[:, 2].astype(np.int64)
    unique_flat = np.unique(flat_idx)

    z = unique_flat // (dest_shape[1] * dest_shape[2])
    rem = unique_flat % (dest_shape[1] * dest_shape[2])
    y = rem // dest_shape[2]
    x = rem % dest_shape[2]

    return np.column_stack([z, y, x]).astype(np.float32)


def deform_real_fiber_along_spline(real_fiber, target_curve, dest_shape=(64, 64, 64), radius_margin=6.0):
    if len(target_curve) < 2:
        return np.zeros(dest_shape, dtype=bool), np.zeros((0, 3), dtype=np.int32)

    t_T, t_N, t_B = compute_parallel_transport_frames(target_curve)
    t_segs = np.linalg.norm(np.diff(target_curve, axis=0), axis=1)
    t_cum = np.insert(np.cumsum(t_segs), 0, 0.0)
    t_len = t_cum[-1]
    if t_len < 1e-4:
        return np.zeros(dest_shape, dtype=bool), np.zeros((0, 3), dtype=np.int32)

    pts_to_sample = get_candidate_voxels_fast(target_curve, radius_margin=radius_margin, dest_shape=dest_shape)
    if len(pts_to_sample) == 0:
        return np.zeros(dest_shape, dtype=bool), np.zeros((0, 3), dtype=np.int32)

    # Exact continuous projection onto target curve segments
    proj_pts, proj_arcs, proj_N, proj_B = project_points_onto_curve_continuous(
        pts_to_sample, target_curve, t_cum, t_N, t_B
    )

    # Perpendicular coordinates (u, v)
    diffs = pts_to_sample - proj_pts
    u = np.sum(diffs * proj_N, axis=1)
    v = np.sum(diffs * proj_B, axis=1)

    # Map arc length to real fiber: 1:1 isometric mapping when possible to prevent distortion
    if t_len <= real_fiber.total_length:
        offset = float(np.random.uniform(0.0, real_fiber.total_length - t_len))
        r_arc_target = offset + proj_arcs
    else:
        r_arc_target = (proj_arcs / t_len) * real_fiber.total_length

    r_cum = real_fiber.cum_arc
    r_curve = real_fiber.curve
    r_N = real_fiber.normals
    r_B = real_fiber.binormals

    r_idx = np.searchsorted(r_cum, r_arc_target) - 1
    r_idx = np.clip(r_idx, 0, len(r_curve) - 2)
    alpha = (r_arc_target - r_cum[r_idx]) / (r_cum[r_idx + 1] - r_cum[r_idx] + 1e-8)
    alpha = np.clip(alpha, 0.0, 1.0)[:, None]

    r_pos = (1.0 - alpha) * r_curve[r_idx] + alpha * r_curve[r_idx + 1]
    r_N_interp = (1.0 - alpha) * r_N[r_idx] + alpha * r_N[r_idx + 1]
    r_B_interp = (1.0 - alpha) * r_B[r_idx] + alpha * r_B[r_idx + 1]

    src_coords = r_pos + u[:, None] * r_N_interp + v[:, None] * r_B_interp

    sampled_vals = map_coordinates(real_fiber.mask, src_coords.T, order=1, mode='constant', cval=0.0)

    # Clean endpoint clipping: voxels strictly beyond curve start/end plane do not extrude
    if len(target_curve) >= 2:
        t_start, t_end = target_curve[0], target_curve[-1]
        tan_start, tan_end = t_T[0], t_T[-1]
        past_start = np.sum((pts_to_sample - t_start) * (-tan_start), axis=1) > 0.5
        past_end = np.sum((pts_to_sample - t_end) * tan_end, axis=1) > 0.5
        sampled_vals[past_start | past_end] = 0.0

    dest_vol = np.zeros(dest_shape, dtype=bool)
    fg_mask = (sampled_vals > 0.40)
    dest_pts = pts_to_sample[fg_mask].astype(int)
    dest_vol[dest_pts[:, 0], dest_pts[:, 1], dest_pts[:, 2]] = True

    return dest_vol, dest_pts


def render_morphed_synthetic_patch(curves_list, library, patch_size=64):
    S = patch_size
    vol_patch = np.zeros((S, S, S), dtype=bool)
    int_patch = np.zeros((S, S, S), dtype=np.float32)
    ori_patch = np.zeros((3, S, S, S), dtype=np.float32)

    if not curves_list:
        return vol_patch, int_patch, ori_patch

    all_curve_pts = []
    fiber_trees = []

    for curve in curves_list:
        if len(curve) < 2:
            continue

        real_fiber = library.sample_fiber(boundary_only=True)
        fiber_vol, _ = deform_real_fiber_along_spline(real_fiber, curve, dest_shape=(S, S, S), radius_margin=6.0)
        vol_patch |= fiber_vol

        all_curve_pts.append(curve)
        fiber_trees.append(cKDTree(curve))

    if not all_curve_pts:
        return vol_patch, int_patch, ori_patch

    all_skel_coords = np.vstack(all_curve_pts)
    skel_tree = cKDTree(all_skel_coords)

    fg_coords = np.argwhere(vol_patch)
    if len(fg_coords) > 0:
        dists, _ = skel_tree.query(fg_coords, k=1)
        valid_mask = (dists <= 6.5)
        valid_fg = fg_coords[valid_mask]
        valid_dists = dists[valid_mask]

        int_patch[valid_fg[:, 0], valid_fg[:, 1], valid_fg[:, 2]] = np.exp(-(valid_dists**2) / (2.0 * 1.0**2))

        if len(fiber_trees) >= 2:
            per_fiber_dists = np.stack([tree.query(valid_fg)[0] for tree in fiber_trees], axis=0)
            sorted_dists = np.sort(per_fiber_dists, axis=0)
            d1, d2 = sorted_dists[0], sorted_dists[1]
            cross_mask = (d1 <= 3.0) & (d2 <= 3.0)
            if np.any(cross_mask):
                g_cross = np.exp(-(d1[cross_mask]**2 + d2[cross_mask]**2) / (2.0 * 1.5**2))
                pts = valid_fg[cross_mask]
                int_patch[pts[:, 0], pts[:, 1], pts[:, 2]] -= 1.5 * g_cross

        int_patch = np.clip(int_patch, -1.0, 1.0)

        for tree, curve in zip(fiber_trees, all_curve_pts):
            if len(curve) < 2:
                continue
            tangents = np.zeros_like(curve, dtype=np.float32)
            tangents[0] = curve[1] - curve[0]
            tangents[-1] = curve[-1] - curve[-2]
            if len(curve) > 2:
                tangents[1:-1] = (curve[2:] - curve[:-2]) / 2.0
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            tangents /= norms

            c_dists, c_indices = tree.query(valid_fg, k=1)
            near_mask = (c_dists <= 5.0)
            if np.any(near_mask):
                near_pts = valid_fg[near_mask]
                near_tangs = tangents[c_indices[near_mask]]
                for c in range(3):
                    ori_patch[c, near_pts[:, 0], near_pts[:, 1], near_pts[:, 2]] = near_tangs[:, c]

    return vol_patch, int_patch, ori_patch
