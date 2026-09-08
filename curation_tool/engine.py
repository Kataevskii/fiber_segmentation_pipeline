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
    center_of_mass,
    gaussian_filter1d
)
from scipy.spatial import cKDTree
from scipy.interpolate import splprep, splev
from skimage.graph import route_through_array
from skimage.morphology import skeletonize

class RealDataCurationEngine:
    """
    Deterministic 96x96x96 Real Data Annotation & Geodesic Curvature Resolution Engine.
    Supports both raw thresholded microscopy volumes AND pre-segmented instance volumes
    with automatic skeletonization, endpoint extraction, and wiring correction.
    """
    def __init__(
        self,
        raw_volume_path='data/fibers_to_segment/COLLAGENCROP_003_0000.tif',
        curated_output_dir='data/curated/patches',
        cube_size=96
    ):
        if raw_volume_path and not os.path.exists(raw_volume_path) and os.path.isdir('data/fibers_to_segment'):
            cands = [
                os.path.join('data/fibers_to_segment', f).replace('\\', '/')
                for f in sorted(os.listdir('data/fibers_to_segment'))
                if f.lower().endswith(('.tif', '.tiff', '.npy'))
            ]
            if cands:
                raw_volume_path = cands[0]
        self.raw_volume_path = raw_volume_path.replace('\\', '/') if raw_volume_path else None
        self.instance_volume_path = None
        self.curated_output_dir = curated_output_dir
        self.cube_size = cube_size
        os.makedirs(self.curated_output_dir, exist_ok=True)

        self.full_instance_vol = None
        self.full_instance_skel = None

        print(f"Loading initial volume from {self.raw_volume_path}...", flush=True)
        if self.raw_volume_path and os.path.exists(self.raw_volume_path):
            raw_vol = tifffile.imread(self.raw_volume_path)
            if raw_vol.dtype == bool:
                self.full_vol = raw_vol
            else:
                self.full_vol = raw_vol > (127 if raw_vol.max() > 1.0 else 0.5)
            self.current_source_info = {
                'name': os.path.basename(self.raw_volume_path),
                'type': 'raw_microscopy',
                'path': self.raw_volume_path,
                'has_skeleton': False
            }
        else:
            self.full_vol = np.zeros((cube_size, cube_size, cube_size), dtype=bool)
            self.current_source_info = {
                'name': 'None',
                'type': 'empty',
                'path': '',
                'has_skeleton': False
            }

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

    def open_volume_file(self, file_path, raw_path=None, z=None, y=None, x=None):
        """
        Dynamically loads a new 3D volume (.tif, .tiff, .npy).
        Automatically detects whether file_path is:
          - A segmented instance volume (e.g. integer labels > 1): enables auto-skeletonization and fiber wiring extraction.
          - A raw microscopy volume (binary or grayscale).
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File not found: {file_path}")

        file_path_clean = file_path.replace('\\', '/')
        file_ext = os.path.splitext(file_path)[1].lower()

        # Load file
        if file_ext == '.npy':
            arr = np.load(file_path, mmap_mode='r')
        else:
            arr = tifffile.imread(file_path)

        # Detect if it's a segmented instance volume
        is_instance = False
        if np.issubdtype(arr.dtype, np.integer) and arr.max() > 1:
            is_instance = True
        elif 'instance' in file_path_clean.lower() or 'seg' in file_path_clean.lower():
            is_instance = True

        if is_instance:
            self.instance_volume_path = file_path_clean
            self.full_instance_vol = arr
            self.D, self.H, self.W = self.full_instance_vol.shape

            # Look for matching skeleton file
            skel_candidate = file_path_clean.replace('instance_volume', 'instance_skeleton').replace('_volume', '_skeleton')
            if os.path.exists(skel_candidate):
                print(f"Auto-detected matching skeleton file: {skel_candidate}", flush=True)
                if skel_candidate.endswith('.npy'):
                    self.full_instance_skel = np.load(skel_candidate, mmap_mode='r')
                else:
                    self.full_instance_skel = tifffile.imread(skel_candidate)
            else:
                self.full_instance_skel = None

            # Raw volume pairing
            if raw_path and os.path.exists(raw_path):
                raw_arr = np.load(raw_path) if raw_path.endswith('.npy') else tifffile.imread(raw_path)
                if raw_arr.shape == self.full_instance_vol.shape:
                    self.full_vol = raw_arr > (127 if raw_arr.max() > 1.0 else 0.5)
                    self.raw_volume_path = raw_path
                else:
                    self.full_vol = (self.full_instance_vol > 0)
                    self.raw_volume_path = file_path_clean
            else:
                self.full_vol = (self.full_instance_vol > 0)
                self.raw_volume_path = file_path_clean

            self.current_source_info = {
                'name': os.path.basename(file_path_clean),
                'type': 'segmented_instance',
                'path': file_path_clean,
                'has_skeleton': self.full_instance_skel is not None
            }
        else:
            self.raw_volume_path = file_path_clean
            self.instance_volume_path = None
            self.full_instance_vol = None
            self.full_instance_skel = None
            if arr.dtype == bool:
                self.full_vol = arr
            else:
                self.full_vol = arr > (127 if arr.max() > 1.0 else 0.5)
            self.D, self.H, self.W = self.full_vol.shape
            self.current_source_info = {
                'name': os.path.basename(file_path_clean),
                'type': 'raw_microscopy',
                'path': file_path_clean,
                'has_skeleton': False
            }

        print(f"Successfully opened {self.current_source_info['type']}: {file_path_clean} shape=({self.D}, {self.H}, {self.W})", flush=True)

        if z is not None and y is not None and x is not None:
            return self.extract_patch_at(z, y, x)
        else:
            return self.extract_random_patch()

    def extract_fibers_from_instance_patch(self, patch_inst, patch_skel=None, max_fibers=None, min_voxels=8):
        """
        Extracts 3D fiber centerlines, endpoints, and waypoints from a segmented 96³ patch.
        Maps them into interactive Curation Engine seeds ready for wiring inspection & fixing.
        Extracts all valid fibers in the subvolume without artificial capping.
        """
        S = self.cube_size
        unique_fids = [int(fid) for fid in np.unique(patch_inst) if fid > 0]
        if not unique_fids:
            return [], {}

        # Sort fibers by voxel volume (most prominent first)
        fid_sizes = [(fid, int((patch_inst == fid).sum())) for fid in unique_fids if (patch_inst == fid).sum() >= min_voxels]
        fid_sizes.sort(key=lambda x: x[1], reverse=True)
        if max_fibers is not None:
            fid_sizes = fid_sizes[:max_fibers]

        def get_face_coords(pt):
            z, y, x = pt
            # Check proximity to 6 boundary faces (tolerance 1 vx)
            if z <= 1: return 'z_min', float(y) / (S - 1), float(x) / (S - 1)
            if z >= S - 2: return 'z_max', float(y) / (S - 1), float(x) / (S - 1)
            if y <= 1: return 'y_min', float(z) / (S - 1), float(x) / (S - 1)
            if y >= S - 2: return 'y_max', float(z) / (S - 1), float(x) / (S - 1)
            if x <= 1: return 'x_min', float(z) / (S - 1), float(y) / (S - 1)
            if x >= S - 2: return 'x_max', float(z) / (S - 1), float(y) / (S - 1)
            return 'internal', 0.5, 0.5

        extracted_seeds = []
        curves_3d = {}

        for rank, (fid, size) in enumerate(fid_sizes, start=1):
            mask = (patch_inst == fid)
            if patch_skel is not None:
                sk = (patch_skel == fid)
                if sk.sum() == 0:
                    sk = skeletonize(mask)
            else:
                sk = skeletonize(mask)

            pts = np.argwhere(sk)
            if len(pts) < 3:
                sk = skeletonize(mask)
                pts = np.argwhere(sk)
                if len(pts) < 3:
                    continue

            # Order points from one end of the fiber to the other
            centroid = pts.mean(axis=0)
            start_idx = np.argmax(np.linalg.norm(pts - centroid, axis=1))

            visited = [start_idx]
            curr = start_idx
            unvisited = set(range(len(pts))) - {start_idx}
            ordered = [pts[start_idx]]
            while unvisited:
                sub = list(unvisited)
                dists = np.linalg.norm(pts[sub] - pts[curr], axis=1)
                nearest = np.argmin(dists)
                if dists[nearest] > 4.5:
                    break
                curr = sub[nearest]
                visited.append(curr)
                unvisited.remove(curr)
                ordered.append(pts[curr])

            if len(ordered) < 2:
                continue

            curve = np.array(ordered, dtype=np.float32)
            curves_3d[rank] = curve.tolist()

            p_start = [int(c) for c in curve[0]]
            p_end = [int(c) for c in curve[-1]]

            face_s, u_s, v_s = get_face_coords(p_start)
            face_e, u_e, v_e = get_face_coords(p_end)

            # Start seed
            extracted_seeds.append({
                'fiber_id': rank,
                'face': face_s,
                'u': u_s,
                'v': v_s,
                'pos3d': p_start,
                'is_waypoint': False
            })

            # Intermediate waypoints to preserve curved paths
            if len(curve) >= 45:
                w1 = [int(c) for c in curve[len(curve)//3]]
                w2 = [int(c) for c in curve[2*len(curve)//3]]
                extracted_seeds.append({
                    'fiber_id': rank,
                    'face': 'internal',
                    'u': 0.5,
                    'v': 0.5,
                    'pos3d': w1,
                    'is_waypoint': True
                })
                extracted_seeds.append({
                    'fiber_id': rank,
                    'face': 'internal',
                    'u': 0.5,
                    'v': 0.5,
                    'pos3d': w2,
                    'is_waypoint': True
                })
            elif len(curve) >= 18:
                w_mid = [int(c) for c in curve[len(curve)//2]]
                extracted_seeds.append({
                    'fiber_id': rank,
                    'face': 'internal',
                    'u': 0.5,
                    'v': 0.5,
                    'pos3d': w_mid,
                    'is_waypoint': True
                })

            # End seed
            extracted_seeds.append({
                'fiber_id': rank,
                'face': face_e,
                'u': u_e,
                'v': v_e,
                'pos3d': p_end,
                'is_waypoint': False
            })

        return extracted_seeds, curves_3d

    def extract_random_patch(self, min_density=0.03, max_density=0.35, max_attempts=50):
        """Extracts a random 96x96x96 subvolume with valid fiber material."""
        S = self.cube_size
        for attempt in range(max_attempts):
            z = int(np.random.randint(0, max(1, self.D - S + 1)))
            y = int(np.random.randint(0, max(1, self.H - S + 1)))
            x = int(np.random.randint(0, max(1, self.W - S + 1)))

            patch = self.full_vol[z:z+S, y:y+S, x:x+S]
            density = float(np.mean(patch))
            if min_density <= density <= max_density:
                return self.extract_patch_at(z, y, x)

        # Fallback
        z = max(0, (self.D - S) // 2)
        y = max(0, (self.H - S) // 2)
        x = max(0, (self.W - S) // 2)
        return self.extract_patch_at(z, y, x)

    def extract_patch_at(self, z, y, x):
        """
        Extracts a 96x96x96 subvolume at manually specified (z, y, x) origin coordinates.
        If a segmented instance volume is loaded, automatically pre-extracts fiber centerlines & seeds!
        """
        self.current_loaded_index = None
        patch_info = self.set_current_patch(int(z), int(y), int(x))

        if self.full_instance_vol is not None:
            S = self.cube_size
            oz, oy, ox = self.current_patch_origin
            patch_inst = self.full_instance_vol[oz:oz+S, oy:oy+S, ox:ox+S]
            patch_skel = self.full_instance_skel[oz:oz+S, oy:oy+S, ox:ox+S] if self.full_instance_skel is not None else None

            extracted_seeds, initial_curves = self.extract_fibers_from_instance_patch(patch_inst, patch_skel=patch_skel)

            if extracted_seeds:
                res = self.resolve_connections(extracted_seeds)
                patch_info['loaded_seeds'] = extracted_seeds
                patch_info['curves_3d'] = res['curves_3d']
                patch_info['num_fibers'] = res['num_fibers']
                patch_info['is_segmented_source'] = True
            else:
                patch_info['loaded_seeds'] = []
                patch_info['curves_3d'] = {}
                patch_info['num_fibers'] = 0
                patch_info['is_segmented_source'] = True
        else:
            patch_info['is_segmented_source'] = False

        patch_info['source_info'] = getattr(self, 'current_source_info', {
            'name': os.path.basename(self.raw_volume_path) if self.raw_volume_path else 'None',
            'type': 'raw_microscopy',
            'path': self.raw_volume_path or ''
        })
        return patch_info

    def list_available_files(self):
        """
        Scans workspace for available 3D TIFF and NPY volumes.
        Categorizes them into Segmented Instance volumes and Raw Microscopy volumes.
        """
        import glob
        segmented_files = []
        raw_files = []

        candidate_dirs = ['data/fibers_to_segment', 'outputs', 'data/curated/patches']
        for cdir in candidate_dirs:
            if not os.path.exists(cdir): continue
            for ext in ('*.tif', '*.tiff', '*.npy'):
                for fpath in glob.glob(os.path.join(cdir, '**', ext), recursive=True):
                    fpath_clean = fpath.replace('\\', '/')
                    lower = fpath_clean.lower()
                    if 'skeleton' in lower or 'meta' in lower or 'centerline' in lower or 'intensity' in lower or 'ori' in lower:
                        continue
                    try:
                        sz_mb = round(os.path.getsize(fpath_clean) / (1024 * 1024), 1)
                    except:
                        sz_mb = 0

                    item = {
                        'path': fpath_clean,
                        'name': os.path.basename(fpath_clean),
                        'dir': os.path.dirname(fpath_clean),
                        'size_mb': sz_mb
                    }

                    if 'instance' in lower or 'seg' in lower or 'resolved' in lower:
                        segmented_files.append(item)
                    else:
                        raw_files.append(item)

        return {
            'segmented': segmented_files,
            'raw': raw_files,
            'current': getattr(self, 'current_source_info', {
                'name': os.path.basename(self.raw_volume_path) if self.raw_volume_path else 'None',
                'type': 'raw_microscopy',
                'path': self.raw_volume_path or ''
            })
        }

    def get_volume_info(self):
        """Returns volume spatial shape and maximal origin bounds for UI/API clients."""
        S = self.cube_size
        return {
            'raw_volume_path': self.raw_volume_path,
            'instance_volume_path': self.instance_volume_path,
            'source_info': getattr(self, 'current_source_info', {
                'name': os.path.basename(self.raw_volume_path) if self.raw_volume_path else 'None',
                'type': 'raw_microscopy',
                'path': self.raw_volume_path or ''
            }),
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
        if len(vox_coords) > 24000:
            sub_idx = np.random.choice(len(vox_coords), size=24000, replace=False)
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

    def _center_curve_to_mask(self, curve, mask, smoothing_sigma=2.5, resample_step=1.0):
        """
        Refines curve points so they lie exactly at the cross-sectional center of mass of the mask,
        fits a continuous cubic B-spline along the arc length, and resamples with uniform step size
        to eliminate all discrete voxel staircasing and jaggedness.
        """
        if len(curve) < 2:
            return curve.astype(np.float32)

        mask_coords = np.argwhere(mask > 0).astype(np.float32)
        if len(mask_coords) == 0:
            return curve.astype(np.float32)

        # 1. Project mask coordinates to find cross-sectional centroids
        tree = cKDTree(curve)
        _, indices = tree.query(mask_coords, k=1)

        centered = curve.copy().astype(np.float32)
        for k in range(len(curve)):
            assigned = (indices == k)
            if np.sum(assigned) >= 3:
                centered[k] = mask_coords[assigned].mean(axis=0)

        # 2. Gaussian smoothing on centered points
        smoothed_pts = gaussian_filter1d(centered, sigma=smoothing_sigma, axis=0)

        # 3. Parametric cubic B-spline arc-length resampling
        diffs = np.linalg.norm(np.diff(smoothed_pts, axis=0), axis=1)
        keep_idx = np.insert(diffs > 1e-4, 0, True)
        valid_pts = smoothed_pts[keep_idx]

        if len(valid_pts) < 4:
            return smoothed_pts.astype(np.float32)

        try:
            tck, u = splprep([valid_pts[:, 0], valid_pts[:, 1], valid_pts[:, 2]], k=min(3, len(valid_pts)-1), s=len(valid_pts) * 0.5)
            u_fine = np.linspace(0, 1, len(valid_pts) * 4)
            z_fine, y_fine, x_fine = splev(u_fine, tck)
            fine_pts = np.column_stack([z_fine, y_fine, x_fine])

            arc_lengths = np.insert(np.cumsum(np.linalg.norm(np.diff(fine_pts, axis=0), axis=1)), 0, 0.0)
            total_len = arc_lengths[-1]
            n_resampled = max(3, int(np.round(total_len / resample_step)))
            u_uniform = np.linspace(0, 1, n_resampled)
            z_u, y_u, x_u = splev(u_uniform, tck)
            return np.column_stack([z_u, y_u, x_u]).astype(np.float32)
        except Exception:
            return smoothed_pts.astype(np.float32)

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

        resolved_curves = {}
        curves_3d_json = {}
        inst_skel = np.zeros((S, S, S), dtype=np.uint16)

        # Build dynamic cost tensor (1 / EDT^2)
        base_cost = 1.0 / (self.current_dt**2 + 1e-4)
        dynamic_cost = np.ascontiguousarray(base_cost.copy(), dtype=np.float64)

        # Group seeds by fiber ID
        fiber_groups = {}
        for s in seeds:
            fid = s['fiber_id']
            fiber_groups.setdefault(fid, []).append(s)

        for fid, seed_list in fiber_groups.items():
            if len(seed_list) >= 2:
                # Order seeds: waypoints in middle
                endpoints = [s for s in seed_list if not s.get('is_waypoint', False)]
                waypoints = [s for s in seed_list if s.get('is_waypoint', False)]
                ordered_seeds = endpoints[:1] + waypoints + endpoints[1:]

                segments = []
                for i in range(len(ordered_seeds) - 1):
                    p1 = tuple(int(c) for c in ordered_seeds[i]['pos3d'])
                    p2 = tuple(int(c) for c in ordered_seeds[i+1]['pos3d'])

                    p1_snapped = self._snap_to_foreground(p1)
                    p2_snapped = self._snap_to_foreground(p2)

                    if p1_snapped is None or p2_snapped is None:
                        continue

                    try:
                        path_indices, _ = route_through_array(
                            dynamic_cost,
                            p1_snapped,
                            p2_snapped,
                            fully_connected=True,
                            geometric=True
                        )
                        path_arr = np.array(path_indices, dtype=np.float32)
                        segments.append(path_arr if i == 0 else path_arr[1:])
                    except Exception as e:
                        print(f"Warning: Route segment failed for Fiber {fid}: {e}", flush=True)

                if segments:
                    try:
                        full_path = np.vstack(segments)
                        resolved_curves[fid] = full_path
                        curves_3d_json[fid] = full_path.tolist()

                        for pt in full_path.astype(int):
                            dynamic_cost[pt[0], pt[1], pt[2]] += 2.0
                    except Exception as e:
                        print(f"Warning: Geodesic route failed for Fiber {fid}: {e}", flush=True)

            elif len(seed_list) == 1:
                s = seed_list[0]
                p_start = tuple(int(c) for c in s['pos3d'])
                p_snapped = self._snap_to_foreground(p_start)
                if p_snapped is not None:
                    path_arr = self._trace_terminating_fiber(p_snapped)
                    if len(path_arr) > 0:
                        resolved_curves[fid] = path_arr
                        curves_3d_json[fid] = path_arr.tolist()

                        for pt in path_arr.astype(int):
                            dynamic_cost[pt[0], pt[1], pt[2]] += 2.0

        # Accelerated Multi-Label 3D Voronoi Diffusion & Target Generation via cKDTree
        inst_vol = np.zeros((S, S, S), dtype=np.uint16)
        clean_patch_bin = np.zeros((S, S, S), dtype=bool)
        ori_vol = np.zeros((3, S, S, S), dtype=np.float32)
        intensity_target = np.zeros((S, S, S), dtype=np.float32)

        if len(resolved_curves) > 0:
            # Pass 1: Initial Voronoi multi-label diffusion from coarse Dijkstra paths
            all_pts = [curve for curve in resolved_curves.values()]
            all_fids = [np.full(len(curve), fid, dtype=np.uint16) for fid, curve in resolved_curves.items()]
            all_skel_coords = np.vstack(all_pts)
            all_skel_fids = np.concatenate(all_fids)

            fg_coords = np.argwhere(self.current_patch_bin > 0)
            if len(fg_coords) > 0:
                skel_tree = cKDTree(all_skel_coords)
                dists, indices = skel_tree.query(fg_coords, k=1)
                valid_mask = (dists <= 6.5)
                valid_fg = fg_coords[valid_mask]
                nearest_fids = all_skel_fids[indices[valid_mask]]

                inst_vol[valid_fg[:, 0], valid_fg[:, 1], valid_fg[:, 2]] = nearest_fids
                clean_patch_bin = (inst_vol > 0)

                # Pass 2: Auto-refine each curve to its exact cross-sectional center of mass
                inst_skel.fill(0)
                for fid in list(resolved_curves.keys()):
                    f_mask = (inst_vol == fid)
                    if np.sum(f_mask) >= 5:
                        refined = self._center_curve_to_mask(resolved_curves[fid], f_mask)
                        resolved_curves[fid] = refined
                        curves_3d_json[fid] = refined.tolist()

                    r_int = np.clip(np.round(resolved_curves[fid]).astype(int), 0, S - 1)
                    inst_skel[r_int[:, 0], r_int[:, 1], r_int[:, 2]] = fid

                # Pass 3: Recompute exact distance field and targets using centered curves
                all_pts = [curve for curve in resolved_curves.values()]
                all_skel_coords = np.vstack(all_pts)
                skel_tree = cKDTree(all_skel_coords)
                dists, _ = skel_tree.query(fg_coords, k=1)
                valid_mask = (dists <= 6.5)
                valid_fg = fg_coords[valid_mask]
                valid_dists = dists[valid_mask]

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

                intensity_target = np.clip(intensity_target, -1.0, 1.0)

                # Fast Analytical Orientation Field (Continuous Local Tangents via cKDTree)
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

                        c_tree = cKDTree(curve)
                        f_mask = (inst_vol == fid)
                        if np.any(f_mask):
                            f_coords = np.argwhere(f_mask)
                            _, nearest_c_idx = c_tree.query(f_coords, k=1)
                            local_tangs = tangents[nearest_c_idx]
                            for c in range(3):
                                ori_vol[c, f_coords[:, 0], f_coords[:, 1], f_coords[:, 2]] = local_tangs[:, c]

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

        # Evaluate donor fiber quality using orientation and boundary geometry
        fiber_quality = {}
        intact_count = 0
        resolved_curves = self.current_resolution.get('resolved_curves', [])
        S = self.cube_size
        faces = [
            ('z_min', 0, 0.0), ('z_max', 0, S - 1.0),
            ('y_min', 1, 0.0), ('y_max', 1, S - 1.0),
            ('x_min', 2, 0.0), ('x_max', 2, S - 1.0)
        ]

        for item in resolved_curves:
            if isinstance(item, tuple) and len(item) == 2:
                fid, curve = item
            else:
                continue

            curve = np.array(curve, dtype=np.float32)
            N = len(curve)
            if N < 15:
                fiber_quality[str(fid)] = {'is_valid_donor': False, 'reason': 'too_short', 'length': float(N)}
                continue

            segs = np.linalg.norm(np.diff(curve, axis=0), axis=1)
            total_len = float(np.sum(segs))
            if total_len < 75.0:
                fiber_quality[str(fid)] = {'is_valid_donor': False, 'reason': f'short_length_{total_len:.1f}', 'length': total_len}
                continue

            tangents = np.zeros_like(curve, dtype=np.float32)
            tangents[0] = curve[1] - curve[0]
            tangents[-1] = curve[-1] - curve[-2]
            tangents[1:-1] = (curve[2:] - curve[:-2]) / 2.0
            norms = np.linalg.norm(tangents, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            tangents /= norms

            # 1. Interior trunk check: reject if parallel and touching any border
            start_m = max(3, int(0.06 * N))
            end_m = N - start_m
            interior_idx = np.arange(start_m, end_m)
            is_sliced = False
            for face_name, axis, val in faces:
                dists = np.abs(curve[:, axis] - val)
                t_norm = np.abs(tangents[:, axis])
                if np.sum((dists[interior_idx] <= 3.5) & (t_norm[interior_idx] < 0.35)) >= 3:
                    fiber_quality[str(fid)] = {'is_valid_donor': False, 'reason': f'parallel_sliced_at_{face_name}', 'length': total_len}
                    is_sliced = True
                    break

            if is_sliced:
                continue

            # 2. Endpoint check: entrance and exit must touch different boundary faces cleanly
            p0, p1 = curve[0], curve[-1]
            t0, t1 = tangents[0], tangents[-1]

            def check_end(p, t):
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
                    return best_f, norm_c
                return None, 0.0

            e0, norm0 = check_end(p0, t0)
            e1, norm1 = check_end(p1, t1)

            if not e0 or not e1 or e0 == e1:
                fiber_quality[str(fid)] = {'is_valid_donor': False, 'reason': f'invalid_endpoints_{e0}_{e1}', 'length': total_len}
                continue

            fiber_quality[str(fid)] = {
                'is_valid_donor': True,
                'length': round(total_len, 1),
                'entry_face': e0,
                'exit_face': e1,
                'entry_penetration': round(float(norm0), 3),
                'exit_penetration': round(float(norm1), 3)
            }
            intact_count += 1

        meta = {
            'patch_index': int(idx),
            'origin_zyx': [int(c) for c in self.current_patch_origin],
            'cube_size': int(self.cube_size),
            'num_fibers': int(self.current_resolution['num_fibers']),
            'intact_donor_fibers_count': int(intact_count),
            'density': float(np.mean(patch_bin)),
            'seeds': clean_seeds,
            'fiber_quality': fiber_quality,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        with open(f"{prefix}_meta.json", 'w', encoding='utf-8') as f:
            json.dump(meta, f, indent=2)

        self.patch_counter = len([f for f in os.listdir(self.curated_output_dir) if f.endswith('_meta.json')])
        print(f"Saved Curated Real Training Sample #{idx:04d} ({intact_count} intact donor fibers) to {self.curated_output_dir}/", flush=True)

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
        if len(vox_coords) > 24000:
            sub_idx = np.random.choice(len(vox_coords), size=24000, replace=False)
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
