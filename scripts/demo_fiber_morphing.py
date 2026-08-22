import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import json
import time
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from core.fiber_morpher import RealFiberLibrary, render_morphed_synthetic_patch

def extract_gad_curves_in_box(gad_path, box_origin, box_size=64):
    """Extracts continuous synthetic 3D curves from a GAD file within a specified bounding box."""
    with open(gad_path, 'r', encoding='utf-8') as f:
        gad = json.load(f)

    voxel_length = gad['Domain']['VoxelLength'][0]
    num_objects = gad['NumberOfObjects']
    z0, y0, x0 = box_origin
    S = box_size

    curves_in_box = []

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

        # Interpolate densely
        n_pts = len(pts_zyx)
        t_dense = np.linspace(0, 1, n_pts * 10)
        t_sparse = np.linspace(0, 1, n_pts)
        dense_z = np.interp(t_dense, t_sparse, pts_zyx[:, 0])
        dense_y = np.interp(t_dense, t_sparse, pts_zyx[:, 1])
        dense_x = np.interp(t_dense, t_sparse, pts_zyx[:, 2])
        dense_pts = np.column_stack([dense_z, dense_y, dense_x])

        # Shift relative to box origin
        rel_pts = dense_pts - np.array([z0, y0, x0], dtype=np.float32)

        # Check if points intersect the [0, S]^3 box
        in_box = (rel_pts[:, 0] >= -2) & (rel_pts[:, 0] <= S + 2) & \
                 (rel_pts[:, 1] >= -2) & (rel_pts[:, 1] <= S + 2) & \
                 (rel_pts[:, 2] >= -2) & (rel_pts[:, 2] <= S + 2)

        if np.sum(in_box) > 8:
            # Segment contiguous in-box sub-curves
            diff = np.diff(in_box.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            if in_box[0]:
                starts = np.insert(starts, 0, 0)
            if in_box[-1]:
                ends = np.append(ends, len(in_box))

            for s, e in zip(starts, ends):
                if e - s >= 8:
                    sub_curve = rel_pts[s:e]
                    curves_in_box.append(sub_curve.astype(np.float32))

    return curves_in_box


def render_standard_synthetic_patch(curves_list, patch_size=64, radius=2.5):
    """Renders a standard smooth-spherical synthetic patch for direct comparison."""
    S = patch_size
    vol = np.zeros((S, S, S), dtype=bool)
    grid_z, grid_y, grid_x = np.mgrid[0:S, 0:S, 0:S]
    voxels = np.column_stack([grid_z.ravel(), grid_y.ravel(), grid_x.ravel()]).astype(np.float32)

    for curve in curves_list:
        from scipy.spatial import cKDTree
        tree = cKDTree(curve)
        min_b = np.maximum(0, np.floor(curve.min(axis=0) - radius - 1).astype(int))
        max_b = np.minimum(S, np.ceil(curve.max(axis=0) + radius + 1).astype(int))
        if any(min_b >= max_b):
            continue
        gz, gy, gx = np.mgrid[min_b[0]:max_b[0], min_b[1]:max_b[1], min_b[2]:max_b[2]]
        cand = np.column_stack([gz.ravel(), gy.ravel(), gx.ravel()]).astype(np.float32)
        dists, _ = tree.query(cand, k=1)
        fg = cand[dists <= radius].astype(int)
        vol[fg[:, 0], fg[:, 1], fg[:, 2]] = True

    return vol


def main():
    print("=" * 80)
    print(" RUNNING REAL-TO-SYNTHETIC FIBER MORPHING DEMONSTRATION")
    print("=" * 80)

    os.makedirs('outputs', exist_ok=True)

    # 1. Initialize Real Fiber Library
    library = RealFiberLibrary(
        curated_dir='real_train_data/curated_patches',
        cache_path='real_train_data/fiber_library.pkl',
        min_length=45
    )

    # 2. Extract Synthetic Centerline Splines from AJ_model_1.gad
    gad_path = 'raw_data/AJ_model_1.gad'
    patch_size = 64
    box_origin = (120, 140, 130)

    print(f"Extracting synthetic centerlines from '{gad_path}' in box {box_origin} (size={patch_size}^3)...")
    curves = extract_gad_curves_in_box(gad_path, box_origin=box_origin, box_size=patch_size)
    print(f"Found {len(curves)} synthetic fiber curves traversing the box.")

    # 3. Render Standard Smooth Spherical Synthetic Volume
    t0 = time.time()
    std_vol = render_standard_synthetic_patch(curves, patch_size=patch_size, radius=2.6)
    print(f"Standard synthetic rendering took: {time.time() - t0:.2f}s (Volume: {std_vol.sum()} voxels)")

    # 4. Render Morphed Real Biological Fiber Volume
    t0 = time.time()
    morphed_vol, morphed_int, morphed_ori = render_morphed_synthetic_patch(curves, library, patch_size=patch_size)
    print(f"Morphed biological rendering took: {time.time() - t0:.2f}s (Volume: {morphed_vol.sum()} voxels)")

    # 5. Save 3D TIFFs
    tifffile.imwrite('outputs/demo_standard_synthetic.tif', (std_vol.astype(np.uint8) * 255), compression='zlib')
    tifffile.imwrite('outputs/demo_morphed_synthetic.tif', (morphed_vol.astype(np.uint8) * 255), compression='zlib')
    tifffile.imwrite('outputs/demo_morphed_intensity_gt.tif', ((morphed_int + 1.0) / 2.0 * 255).astype(np.uint8), compression='zlib')

    print("Saved 3D TIFFs to:")
    print("  - outputs/demo_standard_synthetic.tif")
    print("  - outputs/demo_morphed_synthetic.tif")
    print("  - outputs/demo_morphed_intensity_gt.tif")

    # 6. Generate Side-by-Side Visual Comparison Figure
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Row 1: Max Intensity Projections (MIP)
    std_mip_z = np.max(std_vol, axis=0)
    morph_mip_z = np.max(morphed_vol, axis=0)
    int_mip_z = np.max(morphed_int, axis=0)

    axes[0, 0].imshow(std_mip_z, cmap='gray')
    axes[0, 0].set_title("Standard Synthetic (Too Perfect Smooth MIP)", fontweight='bold')
    axes[0, 0].axis('off')

    axes[0, 1].imshow(morph_mip_z, cmap='gray')
    axes[0, 1].set_title("Morphed Real Fibers (Biological Texture MIP)", fontweight='bold', color='teal')
    axes[0, 1].axis('off')

    im_int = axes[0, 2].imshow(int_mip_z, cmap='viridis', vmin=-1.0, vmax=1.0)
    axes[0, 2].set_title("Exact Math Centerline GT Field I(x)", fontweight='bold')
    axes[0, 2].axis('off')
    plt.colorbar(im_int, ax=axes[0, 2], fraction=0.046, pad=0.04)

    # Row 2: 2D Mid-Slice Cross-Sections
    mid = patch_size // 2
    axes[1, 0].imshow(std_vol[mid], cmap='gray')
    axes[1, 0].set_title(f"Standard Synthetic Slice z={mid}", fontweight='bold')
    axes[1, 0].axis('off')

    axes[1, 1].imshow(morphed_vol[mid], cmap='gray')
    axes[1, 1].set_title(f"Morphed Real Fiber Slice z={mid}", fontweight='bold', color='teal')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(morphed_int[mid], cmap='viridis', vmin=-1.0, vmax=1.0)
    axes[1, 2].set_title(f"Ground Truth Slice z={mid}", fontweight='bold')
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/morphing_comparison.png', dpi=200)
    plt.close()

    print(f"Comparison figure saved to 'outputs/morphing_comparison.png'!")
    print("=" * 80)

if __name__ == '__main__':
    main()
