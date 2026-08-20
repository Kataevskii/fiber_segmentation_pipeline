import os
import json
import time
import base64
import io
import numpy as np
import tifffile
from PIL import Image
from scipy.ndimage import (
    distance_transform_edt,
    convolve,
    label as nd_label,
    center_of_mass
)
from scipy.spatial import cKDTree
from skimage.graph import route_through_array

class RealDataCurationEngine:
    """
    Deterministic 96x96x96 Real Data Annotation & Geodesic Curvature Resolution Engine.
    Uses ONLY the thresholded 0,1 binary microscopy volume (zero model inference).
    """
    def __init__(
        self,
        raw_volume_path='process_data/COLLAGENCROP_003_0000.tif',
        curated_output_dir='real_train_data/curated_patches',
        cube_size=96
    ):
        self.raw_volume_path = raw_volume_path
        self.curated_output_dir = curated_output_dir
        self.cube_size = cube_size
        os.makedirs(self.curated_output_dir, exist_ok=True)

        print(f"Loading real microscopy volume from {self.raw_volume_path}...", flush=True)
        if not os.path.exists(self.raw_volume_path):
            raise FileNotFoundError(f"Microscopy volume not found: {self.raw_volume_path}")

        raw_vol = tifffile.imread(self.raw_volume_path)
        if raw_vol.dtype == bool:
            self.full_vol = raw_vol
        else:
            self.full_vol = raw_vol > (127 if raw_vol.max() > 1.0 else 0.5)

        self.D, self.H, self.W = self.full_vol.shape
        print(f"Loaded volume shape: ({self.D}, {self.H}, {self.W}), overall density: {np.mean(self.full_vol):.4f}", flush=True)

        # Precomputed coordinate grid for fast sub-box routing without per-step allocations
        self.grid_coords = np.stack(np.indices((self.cube_size, self.cube_size, self.cube_size)), axis=-1).astype(np.float32)

        # Current working patch state
        self.current_patch_origin = (0, 0, 0)
        self.current_patch_bin = None
        self.current_dt = None
        self.current_cost_grid = None
        self.current_seeds = []
        self.current_resolution = None
        self.patch_counter = len([f for f in os.listdir(self.curated_output_dir) if f.endswith('_vol.npy')])

    def extract_random_patch(self, min_density=0.03, max_density=0.25, max_attempts=50):
        """Extracts a random 96x96x96 subvolume with valid fiber material."""
        S = self.cube_size
        for attempt in range(max_attempts):
            z = int(np.random.randint(0, max(1, self.D - S + 1)))
            y = int(np.random.randint(0, max(1, self.H - S + 1)))
            x = int(np.random.randint(0, max(1, self.W - S + 1)))

            patch = self.full_vol[z:z+S, y:y+S, x:x+S]
            density = float(np.mean(patch))
            if min_density <= density <= max_density:
                return self.set_current_patch(z, y, x)

        # Fallback
        z = max(0, (self.D - S) // 2)
        y = max(0, (self.H - S) // 2)
        x = max(0, (self.W - S) // 2)
        return self.set_current_patch(z, y, x)

    def extract_patch_at(self, z, y, x):
        """
        Extracts a 96x96x96 subvolume at manually specified (z, y, x) origin coordinates.
        Coordinates are clamped to valid ranges [0, max_valid_dim].
        """
        self.current_loaded_index = None
        return self.set_current_patch(int(z), int(y), int(x))

    def get_volume_info(self):
        """Returns volume spatial shape and maximal origin bounds for UI/API clients."""
        S = self.cube_size
        return {
            'raw_volume_path': self.raw_volume_path,
            'shape': [int(self.D), int(self.H), int(self.W)],
            'cube_size': int(S),
            'max_origin': [int(max(0, self.D - S)), int(max(0, self.H - S)), int(max(0, self.W - S))],
            'curated_total': self.patch_counter,
            'overall_density': float(np.mean(self.full_vol))
        }

    def get_cropped_view_data(self, z_min=0, z_max=95, y_min=0, y_max=95, x_min=0, x_max=95, step=4):
        """
        Extracts visual sub-box crop features (faces, sub-MIPs, point cloud, slices)
        within the current 96x96x96 patch without modifying the underlying 96³ volume.
        All coordinate inputs are clamped to [0, cube_size - 1].
        """
        if self.current_patch_bin is None:
            raise ValueError("No active patch loaded.")

        S = self.cube_size
        z_min = max(0, min(S - 1, int(z_min)))
        z_max = max(z_min, min(S - 1, int(z_max)))
        y_min = max(0, min(S - 1, int(y_min)))
        y_max = max(y_min, min(S - 1, int(y_max)))
        x_min = max(0, min(S - 1, int(x_min)))
        x_max = max(x_min, min(S - 1, int(x_max)))

        # Subvolume binary slice
        sub_bin = self.current_patch_bin[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]

        # 6 Boundary face images for the cropped sub-box
        faces = {
            'z_min': self.current_patch_bin[z_min, y_min:y_max+1, x_min:x_max+1],      # (Y, X) -> Row=Y, Col=X
            'z_max': self.current_patch_bin[z_max, y_min:y_max+1, x_min:x_max+1],      # (Y, X) -> Row=Y, Col=X
            'y_min': self.current_patch_bin[z_min:z_max+1, y_min, x_min:x_max+1],      # (Z, X) -> Row=Z, Col=X
            'y_max': self.current_patch_bin[z_min:z_max+1, y_max, x_min:x_max+1],      # (Z, X) -> Row=Z, Col=X
            'x_min': self.current_patch_bin[z_min:z_max+1, y_min:y_max+1, x_min],      # (Z, Y) -> Row=Z, Col=Y
            'x_max': self.current_patch_bin[z_min:z_max+1, y_min:y_max+1, x_max]       # (Z, Y) -> Row=Z, Col=Y
        }

        face_imgs = {}
        face_imgs_rgba = {}
        for name, mask in faces.items():
            face_imgs[name] = self._array_to_png_base64(mask)
            h, w = mask.shape
            rgba = np.zeros((h, w, 4), dtype=np.uint8)
            rgba[mask] = [255, 255, 255, 240]
            rgba[~mask] = [15, 18, 28, 20]
            face_imgs_rgba[name] = self._array_to_png_base64(rgba)

        # Cropped Surface Point Cloud
        from scipy.ndimage import binary_erosion
        surf = self.current_patch_bin & ~binary_erosion(self.current_patch_bin)
        sub_surf = np.zeros_like(surf, dtype=bool)
        sub_surf[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1] = surf[z_min:z_max+1, y_min:y_max+1, x_min:x_max+1]
        vox_coords = np.argwhere(sub_surf)
        if len(vox_coords) > 6000:
            sub_idx = np.random.choice(len(vox_coords), size=6000, replace=False)
            vox_coords = vox_coords[sub_idx]
        point_cloud = vox_coords.tolist()

        return {
            'crop_bounds': [z_min, z_max, y_min, y_max, x_min, x_max],
            'crop_shape': [int(z_max - z_min + 1), int(y_max - y_min + 1), int(x_max - x_min + 1)],
            'density': float(np.mean(sub_bin)) if sub_bin.size > 0 else 0.0,
            'face_images': face_imgs,
            'face_images_rgba': face_imgs_rgba,
            'point_cloud': point_cloud
        }
    def resolve_connections(self, seeds, crop_bounds=None):
        """
        Executes deterministic geodesic minimal-curvature path resolution:
        1. Groups seeds by fiber_id.
        2. For pairs (e.g. 1---1, 2---2): computes optimal 3D geodesic path through distance ridge.
        3. For singletons (e.g. 3, 4): traces along DT ridge until fiber terminates.
        4. Reconstructs multi-label 3D instance segmentation and analytical orientation vectors via fast KDTree.
        """
        t0 = time.time()
        self.current_seeds = seeds
        S = self.cube_size

        fibers_dict = {}
        for s in seeds:
            fid = int(s['fiber_id'])
            if fid not in fibers_dict:
                fibers_dict[fid] = []
            fibers_dict[fid].append(s)

        resolved_curves = {}
        curves_3d_json = {}
        inst_skel = np.zeros((S, S, S), dtype=np.uint16)
        dynamic_cost = self.current_cost_grid.copy()

        for fid, seed_list in fibers_dict.items():
            if len(seed_list) >= 2:
                # Connected Path with optional intermediate Waypoints (P_start -> W_1 -> ... -> P_end)
                snapped_pts = []
                for s in seed_list:
                    p = tuple(int(c) for c in s['pos3d'])
                    p_snapped = self._snap_to_foreground(p)
                    if p_snapped is not None:
                        snapped_pts.append(p_snapped)

                if len(snapped_pts) >= 2:
                    try:
                        segments = []
                        for i in range(len(snapped_pts) - 1):
                            seg_path = self._route_fiber_lane_guided(snapped_pts[i], snapped_pts[i+1], dynamic_cost)
                            if i > 0 and len(seg_path) > 1:
                                seg_path = seg_path[1:] # avoid duplicate seam voxel
                            segments.append(seg_path)

                        path_arr = np.vstack(segments)
                        resolved_curves[fid] = path_arr
                        curves_3d_json[fid] = path_arr.tolist()
                        inst_skel[path_arr[:, 0], path_arr[:, 1], path_arr[:, 2]] = fid

                        # Apply soft 1-voxel penalty along centerline
                        for pt in path_arr:
                            dynamic_cost[pt[0], pt[1], pt[2]] += 2.0
                    except Exception as e:
                        print(f"Warning: Geodesic route failed for Fiber {fid}: {e}", flush=True)

            elif len(seed_list) == 1:
                # Terminating singleton
                s = seed_list[0]
                p_start = tuple(int(c) for c in s['pos3d'])
                p_snapped = self._snap_to_foreground(p_start)
                if p_snapped is not None:
                    path_arr = self._trace_terminating_fiber(p_snapped)
                    if len(path_arr) > 0:
                        resolved_curves[fid] = path_arr
                        curves_3d_json[fid] = path_arr.tolist()
                        inst_skel[path_arr[:, 0], path_arr[:, 1], path_arr[:, 2]] = fid

                        for pt in path_arr:
                            dynamic_cost[pt[0], pt[1], pt[2]] += 2.0

        # Accelerated Multi-Label 3D Voronoi Diffusion & Target Generation via cKDTree
        inst_vol = np.zeros((S, S, S), dtype=np.uint16)
        clean_patch_bin = np.zeros((S, S, S), dtype=bool)
        ori_vol = np.zeros((3, S, S, S), dtype=np.float32)
        intensity_target = np.full((S, S, S), -0.5, dtype=np.float32)

        if len(resolved_curves) > 0:
            # Gather all skeleton points and their fiber IDs
            all_pts = []
            all_fids = []
            for fid, curve in resolved_curves.items():
                all_pts.append(curve)
                all_fids.append(np.full(len(curve), fid, dtype=np.uint16))
            all_skel_coords = np.vstack(all_pts)
            all_skel_fids = np.concatenate(all_fids)

            # Query distance from foreground voxels to nearest skeleton point in < 1ms
            fg_coords = np.argwhere(self.current_patch_bin > 0)
            if len(fg_coords) > 0:
                skel_tree = cKDTree(all_skel_coords)
                dists, indices = skel_tree.query(fg_coords, k=1)

                # Prune points exceeding fiber radius threshold (<= 6.5 vx)
                valid_mask = (dists <= 6.5)
                valid_fg = fg_coords[valid_mask]
                valid_dists = dists[valid_mask]
                nearest_fids = all_skel_fids[indices[valid_mask]]

                inst_vol[valid_fg[:, 0], valid_fg[:, 1], valid_fg[:, 2]] = nearest_fids
                clean_patch_bin = (inst_vol > 0)

                # Centerline Gaussian probability field G1(x) with sigma = 1.0
                intensity_target[valid_fg[:, 0], valid_fg[:, 1], valid_fg[:, 2]] = np.exp(
                    -(valid_dists**2) / (2.0 * 1.0**2)
                )

                # Multi-fiber intersection dip calculation (when >= 2 fibers present)
                if len(resolved_curves) >= 2:
                    fiber_trees = [cKDTree(curve) for curve in resolved_curves.values()]
                    per_fiber_dists = np.stack([tree.query(valid_fg)[0] for tree in fiber_trees], axis=0)
                    sorted_dists = np.sort(per_fiber_dists, axis=0)
                    d1 = sorted_dists[0]
                    d2 = sorted_dists[1]
                    cross_mask = (d1 <= 3.0) & (d2 <= 3.0)
                    if np.any(cross_mask):
                        g_cross = np.exp(-(d1[cross_mask]**2 + d2[cross_mask]**2) / (2.0 * 1.5**2))
                        cross_pts = valid_fg[cross_mask]
                        intensity_target[cross_pts[:, 0], cross_pts[:, 1], cross_pts[:, 2]] -= 1.5 * g_cross

                # Fast Analytical Orientation Field
                for fid, curve in resolved_curves.items():
                    if len(curve) >= 2:
                        tangents = np.zeros_like(curve, dtype=np.float32)
                        tangents[0] = curve[1] - curve[0]
                        tangents[-1] = curve[-1] - curve[-2]
                        if len(curve) > 2:
                            tangents[1:-1] = (curve[2:] - curve[:-2]) / 2.0
                        norms = np.linalg.norm(tangents, axis=1, keepdims=True)
                        norms[norms == 0] = 1.0
                        tangents /= norms

                        mean_tang = np.mean(tangents, axis=0)
                        mean_norm = np.linalg.norm(mean_tang)
                        if mean_norm > 0:
                            mean_tang /= mean_norm

                        f_mask = (inst_vol == fid)
                        if np.any(f_mask):
                            for c in range(3):
                                ori_vol[c, f_mask] = mean_tang[c]

        resolve_time = time.time() - t0

        self.current_resolution = {
            'resolved_curves': resolved_curves,
            'inst_skel': inst_skel,
            'inst_vol': inst_vol,
            'clean_patch_bin': clean_patch_bin,
            'ori_vol': ori_vol,
            'intensity_target': intensity_target,
            'num_fibers': len(resolved_curves),
            'resolve_time_ms': round(resolve_time * 1000, 1)
        }

        return {
            'success': True,
            'num_fibers': len(resolved_curves),
            'resolve_time_ms': round(resolve_time * 1000, 1),
            'curves_3d': curves_3d_json
        }

    def _snap_to_foreground(self, pt, search_radius=4):
        S = self.cube_size
        z, y, x = pt
        if 0 <= z < S and 0 <= y < S and 0 <= x < S and self.current_patch_bin[z, y, x]:
            return (z, y, x)

        z1, z2 = max(0, z - search_radius), min(S, z + search_radius + 1)
        y1, y2 = max(0, y - search_radius), min(S, y + search_radius + 1)
        x1, x2 = max(0, x - search_radius), min(S, x + search_radius + 1)

        sub_dt = self.current_dt[z1:z2, y1:y2, x1:x2]
        if np.any(sub_dt > 0):
            max_idx = np.unravel_index(np.argmax(sub_dt), sub_dt.shape)
            return (z1 + max_idx[0], y1 + max_idx[1], x1 + max_idx[2])
        return None

    def _route_fiber_lane_guided(self, p1, p2, dynamic_cost, perpendicular_weight=0.20):
        """
        Fast bounded Dijkstra routing between p1 and p2 using distance ridge + perpendicular lane penalty.
        Executes on a tight sub-bounding box around (p1, p2) for 20x to 50x faster path computation.
        """
        S = self.cube_size
        v_dir = np.array(p2, dtype=np.float32) - np.array(p1, dtype=np.float32)
        v_len = float(np.linalg.norm(v_dir))
        if v_len < 1e-4:
            return np.array([p1], dtype=np.intp)

        v_unit = v_dir / v_len

        # Fast local sub-bounding box routing
        pad = 16
        z_min = max(0, min(p1[0], p2[0]) - pad)
        z_max = min(S, max(p1[0], p2[0]) + pad + 1)
        y_min = max(0, min(p1[1], p2[1]) - pad)
        y_max = min(S, max(p1[1], p2[1]) + pad + 1)
        x_min = max(0, min(p1[2], p2[2]) - pad)
        x_max = min(S, max(p1[2], p2[2]) + pad + 1)

        sub_coords = self.grid_coords[z_min:z_max, y_min:y_max, x_min:x_max]
        pts = sub_coords - np.array(p1, dtype=np.float32)
        proj = np.sum(pts * v_unit, axis=-1)
        proj_clamped = np.clip(proj, 0.0, v_len)
        closest_on_line = np.array(p1, dtype=np.float32) + proj_clamped[..., None] * v_unit
        perp_dist = np.linalg.norm(sub_coords - closest_on_line, axis=-1)

        sub_cost = dynamic_cost[z_min:z_max, y_min:y_max, x_min:x_max] + (perpendicular_weight * perp_dist).astype(np.float32)

        sub_p1 = (p1[0] - z_min, p1[1] - y_min, p1[2] - x_min)
        sub_p2 = (p2[0] - z_min, p2[1] - y_min, p2[2] - x_min)

        try:
            sub_path, _ = route_through_array(
                sub_cost,
                sub_p1,
                sub_p2,
                fully_connected=True
            )
            path = np.array(sub_path, dtype=np.intp)
            path[:, 0] += z_min
            path[:, 1] += y_min
            path[:, 2] += x_min
            return path
        except Exception:
            # Fallback to full volume routing if local box hit a disconnected barrier
            pts_full = self.grid_coords - np.array(p1, dtype=np.float32)
            proj_full = np.clip(np.sum(pts_full * v_unit, axis=-1), 0.0, v_len)
            closest_full = np.array(p1, dtype=np.float32) + proj_full[..., None] * v_unit
            perp_full = np.linalg.norm(self.grid_coords - closest_full, axis=-1)
            guided_cost = dynamic_cost + (perpendicular_weight * perp_full).astype(np.float32)
            path, _ = route_through_array(guided_cost, p1, p2, fully_connected=True)
            return np.array(path, dtype=np.intp)

    def _trace_terminating_fiber(self, start_pt, max_steps=120):
        """
        Traces an entering fiber from start_pt inward along the distance ridge
        of its connected component until the fiber terminates inside the volume.
        """
        S = self.cube_size
        fg = self.current_patch_bin
        if not fg[start_pt[0], start_pt[1], start_pt[2]]:
            return np.array([start_pt], dtype=np.intp)

        # Connected component containing start_pt
        lbl, num = nd_label(fg, structure=np.ones((3, 3, 3), dtype=bool))
        comp_id = lbl[start_pt[0], start_pt[1], start_pt[2]]
        comp_mask = (lbl == comp_id)

        comp_coords = np.argwhere(comp_mask)
        if len(comp_coords) == 0:
            return np.array([start_pt], dtype=np.intp)

        dists = np.linalg.norm(comp_coords - start_pt, axis=1)

        # Prefer internal termination points away from other outer faces
        internal_mask = (
            (comp_coords[:, 0] > 1) & (comp_coords[:, 0] < S - 2) &
            (comp_coords[:, 1] > 1) & (comp_coords[:, 1] < S - 2) &
            (comp_coords[:, 2] > 1) & (comp_coords[:, 2] < S - 2)
        )

        if np.any(internal_mask):
            cand_coords = comp_coords[internal_mask]
            cand_dists = dists[internal_mask]
            best_idx = np.argmax(cand_dists)
            end_pt = tuple(int(c) for c in cand_coords[best_idx])
        else:
            best_idx = np.argmax(dists)
            end_pt = tuple(int(c) for c in comp_coords[best_idx])

        try:
            path, cost = route_through_array(
                np.where(comp_mask, self.current_cost_grid, np.inf),
                start_pt,
                end_pt,
                fully_connected=True
            )
            return np.array(path, dtype=np.intp)
        except Exception:
            path = [start_pt]
            curr = start_pt
            visited = set([start_pt])
            offsets = [(dz, dy, dx) for dz in (-1, 0, 1) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if not (dz==0 and dy==0 and dx==0)]
            for _ in range(max_steps):
                best_n = None
                best_v = -1.0
                for dz, dy, dx in offsets:
                    nz, ny, nx = curr[0] + dz, curr[1] + dy, curr[2] + dx
                    if 0 <= nz < S and 0 <= ny < S and 0 <= nx < S and (nz, ny, nx) not in visited:
                        if fg[nz, ny, nx]:
                            v = self.current_dt[nz, ny, nx]
                            if v > best_v:
                                best_v = v
                                best_n = (nz, ny, nx)
                if best_n is None or best_v < 0.5:
                    break
                curr = best_n
                visited.add(curr)
                path.append(curr)
            return np.array(path, dtype=np.intp)

    def list_saved_patches(self):
        """Returns metadata list of all saved curated patches."""
        patches = []
        for f in sorted(os.listdir(self.curated_output_dir)):
            if f.endswith('_meta.json'):
                try:
                    with open(os.path.join(self.curated_output_dir, f), 'r', encoding='utf-8') as fp:
                        meta = json.load(fp)
                        patches.append({
                            'index': meta.get('patch_index', 0),
                            'name': f.replace('_meta.json', ''),
                            'num_fibers': meta.get('num_fibers', 0),
                            'density': meta.get('density', 0.0),
                            'origin': meta.get('origin_zyx', [0, 0, 0]),
                            'timestamp': meta.get('timestamp', '')
                        })
                except Exception:
                    pass
        patches.sort(key=lambda x: x['index'])
        return patches

    def load_curated_patch(self, patch_index):
        """Loads an already-saved patch with its exact origin, volume, and seeds."""
        meta_path = os.path.join(self.curated_output_dir, f"patch_{patch_index:04d}_meta.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"Patch metadata not found: {meta_path}")

        with open(meta_path, 'r', encoding='utf-8') as fp:
            meta = json.load(fp)

        origin = meta['origin_zyx']
        patch_data = self.set_current_patch(origin[0], origin[1], origin[2])
        self.current_loaded_index = patch_index

        # Re-resolve with loaded seeds
        seeds = meta.get('seeds', [])
        res = self.resolve_connections(seeds)

        patch_data['loaded_patch_index'] = patch_index
        patch_data['loaded_seeds'] = seeds
        patch_data['curves_3d'] = res['curves_3d']
        patch_data['num_fibers'] = res['num_fibers']
        return patch_data

    def save_current_curated_sample(self, overwrite=False):
        if self.current_resolution is None:
            raise ValueError("No resolved patch to save. Run resolve_connections first!")

        if overwrite and hasattr(self, 'current_loaded_index') and self.current_loaded_index is not None:
            idx = self.current_loaded_index
        else:
            existing = [int(f.split('_')[1].split('.')[0]) for f in os.listdir(self.curated_output_dir) if f.startswith('patch_') and f.endswith('_meta.json')]
            idx = (max(existing) + 1) if existing else 1
            self.current_loaded_index = idx

        prefix = os.path.join(self.curated_output_dir, f"patch_{idx:04d}")

        patch_bin = self.current_resolution.get('clean_patch_bin', self.current_patch_bin)
        inst_vol = self.current_resolution['inst_vol']
        inst_skel = self.current_resolution['inst_skel']
        ori_vol = self.current_resolution['ori_vol']
        intensity_target = self.current_resolution['intensity_target']

        tifffile.imwrite(f"{prefix}_vol.tif", (patch_bin.astype(np.uint8) * 255), compression='zlib')
        np.save(f"{prefix}_vol.npy", patch_bin.astype(bool))

        tifffile.imwrite(f"{prefix}_instance.tif", inst_vol.astype(np.uint16), compression='zlib')
        np.save(f"{prefix}_instance.npy", inst_vol.astype(np.uint16))

        tifffile.imwrite(f"{prefix}_centerline.tif", inst_skel.astype(np.uint16), compression='zlib')
        np.save(f"{prefix}_centerline.npy", inst_skel.astype(np.uint16))

        np.save(f"{prefix}_ori.npy", ori_vol.astype(np.float32))
        np.save(f"{prefix}_intensity.npy", intensity_target.astype(np.float32))

        clean_seeds = []
        for s in self.current_seeds:
            clean_seeds.append({
                'face': str(s.get('face', '')),
                'u': float(s.get('u', 0)),
                'v': float(s.get('v', 0)),
                'pos3d': [int(c) for c in s.get('pos3d', [])],
                'fiber_id': int(s.get('fiber_id', 1)),
                'is_waypoint': bool(s.get('is_waypoint', False))
            })

        meta = {
            'patch_index': int(idx),
            'origin_zyx': [int(c) for c in self.current_patch_origin],
            'cube_size': int(self.cube_size),
            'num_fibers': int(self.current_resolution['num_fibers']),
            'density': float(np.mean(patch_bin)),
            'seeds': clean_seeds,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(f"{prefix}_meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        self.patch_counter = len([f for f in os.listdir(self.curated_output_dir) if f.endswith('_meta.json')])
        print(f"Saved Curated Real Training Sample #{idx:04d} to {self.curated_output_dir}/", flush=True)

        return {
            'saved_index': idx,
            'curated_total': self.patch_counter,
            'prefix': prefix
        }

    # ==========================================================================
    # Fast PIL PNG Base64 Image Generation (Thread-Safe & Instantaneous)
    # ==========================================================================

    def _array_to_png_base64(self, arr):
        """Ultra-fast 2D/3D numpy array to PNG base64 data URL using PIL."""
        if arr.ndim == 2:
            if arr.dtype == bool:
                img_arr = (arr * 255).astype(np.uint8)
            elif arr.dtype != np.uint8:
                norm = arr.astype(np.float32)
                if norm.max() > 0: norm = norm / norm.max()
                img_arr = (norm * 255).astype(np.uint8)
            else:
                img_arr = arr
            img = Image.fromarray(img_arr, mode='L')
        elif arr.ndim == 3 and arr.shape[2] == 4:
            img = Image.fromarray(arr.astype(np.uint8), mode='RGBA')
        elif arr.ndim == 3 and arr.shape[2] == 3:
            img = Image.fromarray(arr.astype(np.uint8), mode='RGB')
        else:
            raise ValueError(f"Unsupported array shape: {arr.shape}")

        buf = io.BytesIO()
        img.save(buf, format='PNG', compress_level=1)
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode('utf-8')

    def get_face_images_base64(self, transparent_zeros=True):
        S = self.cube_size
        faces = {
            'z_min': self.current_patch_bin[0, :, :],
            'z_max': self.current_patch_bin[S-1, :, :],
            'y_min': self.current_patch_bin[:, 0, :],
            'y_max': self.current_patch_bin[:, S-1, :],
            'x_min': self.current_patch_bin[:, :, 0],
            'x_max': self.current_patch_bin[:, :, S-1]
        }

        res = {}
        res_rgba = {}
        for name, mask in faces.items():
            res[name] = self._array_to_png_base64(mask)

            # Generate RGBA where 0 is transparent and 1 is bright solid cyan/white
            rgba = np.zeros((S, S, 4), dtype=np.uint8)
            rgba[mask] = [255, 255, 255, 240]    # Solid white fibers
            rgba[~mask] = [15, 18, 28, 20]        # Ultra-faint transparent background
            res_rgba[name] = self._array_to_png_base64(rgba)

        return res, res_rgba

    def set_current_patch(self, z, y, x):
        """Sets the working 96x96x96 patch and precomputes distance transform and face features."""
        S = self.cube_size
        z = int(max(0, min(self.D - S, z)))
        y = int(max(0, min(self.H - S, y)))
        x = int(max(0, min(self.W - S, x)))

        self.current_patch_origin = (z, y, x)
        self.current_patch_bin = self.full_vol[z:z+S, y:y+S, x:x+S].copy()
        
        # Compute exact Euclidean distance transform
        self.current_dt = distance_transform_edt(self.current_patch_bin)
        
        # Precompute inverted squared distance cost for fast geodesic marching
        self.current_cost_grid = np.full((S, S, S), 1e6, dtype=np.float32)
        fg = self.current_patch_bin > 0
        self.current_cost_grid[fg] = 1.0 / (self.current_dt[fg] + 0.1)**2

        self.current_seeds = []
        self.current_resolution = None

        # Extract true 3D surface voxels of fibers for 3D volume rendering
        from scipy.ndimage import binary_erosion
        surf = self.current_patch_bin & ~binary_erosion(self.current_patch_bin)
        vox_coords = np.argwhere(surf)
        if len(vox_coords) > 7000:
            sub_idx = np.random.choice(len(vox_coords), size=7000, replace=False)
            vox_coords = vox_coords[sub_idx]
        point_cloud = vox_coords.tolist()

        face_imgs, face_imgs_rgba = self.get_face_images_base64()

        return {
            'origin': list(self.current_patch_origin),
            'cube_size': S,
            'volume_shape': [int(self.D), int(self.H), int(self.W)],
            'max_origin': [int(max(0, self.D - S)), int(max(0, self.H - S)), int(max(0, self.W - S))],
            'density': float(np.mean(self.current_patch_bin)),
            'face_images': face_imgs,
            'face_images_rgba': face_imgs_rgba,
            'point_cloud': point_cloud,
            'curated_total': self.patch_counter
        }
