import os
import sys
import json
import time
import numpy as np
import tifffile
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.fiber_morpher import RealFiberLibrary, deform_real_fiber_along_spline, render_morphed_synthetic_patch


def extract_gad_curves_in_box(gad_path, box_origin, box_size=64, min_length=25.0):
    """
    Extracts continuous synthetic 3D curves from a GAD file within a specified bounding box,
    filtering out short grazing corner fragments and resampling to uniform 1-voxel spacing.
    """
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
        if len(pts) < 2:
            continue

        pts_zyx = (pts / voxel_length)[:, [2, 1, 0]]

        # Dense spline interpolation
        n_pts = len(pts_zyx)
        t_dense = np.linspace(0, 1, n_pts * 15)
        t_sparse = np.linspace(0, 1, n_pts)
        dense_z = np.interp(t_dense, t_sparse, pts_zyx[:, 0])
        dense_y = np.interp(t_dense, t_sparse, pts_zyx[:, 1])
        dense_x = np.interp(t_dense, t_sparse, pts_zyx[:, 2])
        dense_pts = np.column_stack([dense_z, dense_y, dense_x])

        # Shift relative to box origin
        rel_pts = dense_pts - np.array([z0, y0, x0], dtype=np.float32)

        # Check if points intersect the [0, S]^3 box
        in_box = (rel_pts[:, 0] >= -1) & (rel_pts[:, 0] <= S + 1) & \
                 (rel_pts[:, 1] >= -1) & (rel_pts[:, 1] <= S + 1) & \
                 (rel_pts[:, 2] >= -1) & (rel_pts[:, 2] <= S + 1)

        if np.sum(in_box) > 10:
            diff = np.diff(in_box.astype(int))
            starts = np.where(diff == 1)[0] + 1
            ends = np.where(diff == -1)[0] + 1
            if in_box[0]:
                starts = np.insert(starts, 0, 0)
            if in_box[-1]:
                ends = np.append(ends, len(in_box))

            for s, e in zip(starts, ends):
                sub_curve = rel_pts[s:e]
                segs = np.linalg.norm(np.diff(sub_curve, axis=0), axis=1)
                total_len = float(np.sum(segs))
                if total_len >= min_length:
                    # Resample to uniform 1.0-voxel step along spline arc length
                    cum = np.insert(np.cumsum(segs), 0, 0.0)
                    t_unif = np.linspace(0, cum[-1], int(np.ceil(cum[-1])) + 1)
                    resampled = np.column_stack([
                        np.interp(t_unif, cum, sub_curve[:, 0]),
                        np.interp(t_unif, cum, sub_curve[:, 1]),
                        np.interp(t_unif, cum, sub_curve[:, 2])
                    ])
                    curves_in_box.append(resampled.astype(np.float32))

    return curves_in_box


def render_standard_synthetic_patch(curves_list, patch_size=64, radius=2.6):
    """
    Renders a standard smooth-spherical synthetic patch for direct comparison.
    """
    S = patch_size
    vol = np.zeros((S, S, S), dtype=bool)

    for curve in curves_list:
        if len(curve) < 2:
            continue
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


def recompute_all_morphed_examples():
    print("=" * 80)
    print(" RECOMPUTING ALL MORPHED FIBER EXAMPLES AND FIGURES IN outputs/")
    print("=" * 80)

    os.makedirs('outputs', exist_ok=True)
    np.random.seed(42)

    # 1. Initialize Real Fiber Library with strict intact through-volume filter
    print("\n[Step 1/5] Loading Real Biological Fiber Library (Intact Through-Volume Only)...", flush=True)
    library = RealFiberLibrary(
        curated_dir='data/curated/patches',
        cache_path='data/curated/fiber_library.pkl',
        min_length=75
    )
    intact_count = sum(1 for f in library.fibers if f.is_boundary_continuous)
    print(f"Library loaded with {len(library.fibers)} fibers ({intact_count} verified intact donor fibers).", flush=True)

    gad_1 = 'data/synthetic/raw/AJ_model_1.gad'
    gad_2 = 'data/synthetic/raw/AJ_model_2.gad'
    gad_3 = 'data/synthetic/raw/AJ_model_3.gad'

    # -------------------------------------------------------------------------
    # 2. Recompute 64³ Demo Files & morphing_comparison.png
    # -------------------------------------------------------------------------
    print("\n[Step 2/5] Recomputing 64³ Demo Morphed Examples...", flush=True)
    box_origin_demo = (120, 140, 130)
    demo_curves = extract_gad_curves_in_box(gad_1, box_origin=box_origin_demo, box_size=64, min_length=25.0)
    print(f"  Extracted {len(demo_curves)} curves in demo 64³ box {box_origin_demo}")

    demo_std_vol = render_standard_synthetic_patch(demo_curves, patch_size=64, radius=2.6)
    demo_morph_vol, demo_morph_int, demo_morph_ori = render_morphed_synthetic_patch(demo_curves, library, patch_size=64)

    # Save TIFFs
    p_demo_std = 'outputs/demo_standard_synthetic.tif'
    p_demo_morph = 'outputs/demo_morphed_synthetic.tif'
    p_demo_int = 'outputs/demo_morphed_intensity_gt.tif'
    p_demo_side = 'outputs/demo_side_by_side_comparison_3d.tif'

    tifffile.imwrite(p_demo_std, (demo_std_vol.astype(np.uint8) * 255), compression='zlib')
    tifffile.imwrite(p_demo_morph, (demo_morph_vol.astype(np.uint8) * 255), compression='zlib')
    tifffile.imwrite(p_demo_int, ((demo_morph_int + 1.0) / 2.0 * 255).astype(np.uint8), compression='zlib')

    # Side-by-side composite: 64 + 2 (sep) + 64 + 2 (sep) + 64 = 196
    sep = np.full((64, 64, 2), 128, dtype=np.uint8)
    std_u8 = (demo_std_vol.astype(np.uint8) * 255)
    morph_u8 = (demo_morph_vol.astype(np.uint8) * 255)
    int_u8 = ((demo_morph_int + 1.0) / 2.0 * 255).astype(np.uint8)
    side_by_side_3d = np.concatenate([std_u8, sep, morph_u8, sep, int_u8], axis=2)
    tifffile.imwrite(p_demo_side, side_by_side_3d, compression='zlib')

    # morphing_comparison.png
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    std_mip_z = np.max(demo_std_vol, axis=0)
    morph_mip_z = np.max(demo_morph_vol, axis=0)
    int_mip_z = np.max(demo_morph_int, axis=0)

    axes[0, 0].imshow(std_mip_z, cmap='gray')
    axes[0, 0].set_title("Standard Synthetic (Too Perfect Smooth MIP)", fontweight='bold', fontsize=12)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(morph_mip_z, cmap='gray')
    axes[0, 1].set_title("Morphed Real Fibers (Biological Texture MIP)", fontweight='bold', fontsize=12, color='teal')
    axes[0, 1].axis('off')

    im_int = axes[0, 2].imshow(int_mip_z, cmap='viridis', vmin=-1.0, vmax=1.0)
    axes[0, 2].set_title("Exact Math Centerline GT Field I(x)", fontweight='bold', fontsize=12)
    axes[0, 2].axis('off')
    plt.colorbar(im_int, ax=axes[0, 2], fraction=0.046, pad=0.04)

    mid = 32
    axes[1, 0].imshow(demo_std_vol[mid], cmap='gray')
    axes[1, 0].set_title(f"Standard Synthetic Slice z={mid}", fontweight='bold', fontsize=12)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(demo_morph_vol[mid], cmap='gray')
    axes[1, 1].set_title(f"Morphed Real Fiber Slice z={mid}", fontweight='bold', fontsize=12, color='teal')
    axes[1, 1].axis('off')

    axes[1, 2].imshow(demo_morph_int[mid], cmap='viridis', vmin=-1.0, vmax=1.0)
    axes[1, 2].set_title(f"Ground Truth Slice z={mid}", fontweight='bold', fontsize=12)
    axes[1, 2].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/morphing_comparison.png', dpi=200)
    plt.close()
    print("  -> demo_standard_synthetic.tif, demo_morphed_synthetic.tif, demo_morphed_intensity_gt.tif, demo_side_by_side_comparison_3d.tif, morphing_comparison.png")

    # -------------------------------------------------------------------------
    # 3. Recompute 64³ Gallery: gallery_morphed_samples.png (4 diverse crops)
    # -------------------------------------------------------------------------
    print("\n[Step 3/5] Recomputing 64³ Multi-Crop Gallery (gallery_morphed_samples.png)...", flush=True)
    crop_origins = [
        (120, 140, 130),
        (200, 180, 220),
        (100, 300, 150),
        (260, 220, 280)
    ]
    gallery_crops = []
    for k, orig in enumerate(crop_origins, 1):
        gad_src = gad_1 if os.path.exists(gad_1) else 'data/synthetic/raw/AJ_model_1.gad'
        c_list = extract_gad_curves_in_box(gad_src, orig, box_size=64, min_length=25.0)
        if len(c_list) < 3 and os.path.exists(gad_2):
            c_list = extract_gad_curves_in_box(gad_2, orig, box_size=64, min_length=25.0)
        s_v = render_standard_synthetic_patch(c_list, patch_size=64, radius=2.6)
        m_v, m_i, m_o = render_morphed_synthetic_patch(c_list, library, patch_size=64)
        gallery_crops.append((s_v, m_v, m_i, m_o))

    fig_gal, axes_gal = plt.subplots(4, 3, figsize=(15, 18))
    for idx, (s_v, m_v, m_i, m_o) in enumerate(gallery_crops):
        row = idx
        s_mip = np.max(s_v, axis=0)
        m_mip = np.max(m_v, axis=0)
        i_mip = np.max(m_i, axis=0)

        axes_gal[row, 0].imshow(s_mip, cmap='gray')
        axes_gal[row, 0].set_title(f"Crop #{row+1} Standard Synthetic (Smooth)", fontweight='bold', fontsize=11)
        axes_gal[row, 0].axis('off')

        axes_gal[row, 1].imshow(m_mip, cmap='gray')
        axes_gal[row, 1].set_title(f"Crop #{row+1} Morphed Real Fibers (Biological)", fontweight='bold', fontsize=11, color='teal')
        axes_gal[row, 1].axis('off')

        axes_gal[row, 2].imshow(i_mip, cmap='viridis', vmin=-1.0, vmax=1.0)
        axes_gal[row, 2].set_title(f"Crop #{row+1} Exact Math Ground Truth I(x)", fontweight='bold', fontsize=11)
        axes_gal[row, 2].axis('off')

    plt.tight_layout()
    plt.savefig('outputs/gallery_morphed_samples.png', dpi=200)
    plt.close()
    print("  -> gallery_morphed_samples.png")

    # -------------------------------------------------------------------------
    # 4. Recompute 96³ Samples (sample_1, sample_2, sample_3) & 96³ Gallery + 3D TIFF
    # -------------------------------------------------------------------------
    print("\n[Step 4/5] Recomputing 96³ Samples & 96³ Multi-Block Gallery...", flush=True)
    block_configs = [
        ('data/synthetic/raw/AJ_model_1.gad', (100, 120, 110), 'sample_1'),
        ('data/synthetic/raw/AJ_model_2.gad' if os.path.exists('data/synthetic/raw/AJ_model_2.gad') else gad_1, (160, 180, 150), 'sample_2'),
        ('data/synthetic/raw/AJ_model_3.gad' if os.path.exists('data/synthetic/raw/AJ_model_3.gad') else gad_1, (220, 200, 240), 'sample_3'),
        ('data/synthetic/raw/AJ_model_1.gad', (180, 280, 200), 'sample_4_aux')
    ]

    blocks_data_96 = []

    for b_idx, (gad_file, b_orig, b_label) in enumerate(block_configs):
        print(f"  Processing 96³ block #{b_idx+1} from {gad_file} at {b_orig}...", flush=True)
        b_curves = extract_gad_curves_in_box(gad_file, b_orig, box_size=96, min_length=35.0)
        print(f"    Found {len(b_curves)} curves traversing 96³ volume.")

        b_std_vol = render_standard_synthetic_patch(b_curves, patch_size=96, radius=2.6)
        b_morph_vol, b_morph_int, b_morph_ori = render_morphed_synthetic_patch(b_curves, library, patch_size=96)

        blocks_data_96.append((b_std_vol, b_morph_vol, b_morph_int, b_morph_ori))

        # Save individual sample files for sample_1, sample_2, sample_3
        if b_idx < 3:
            s_num = b_idx + 1
            p_synth = f'outputs/sample_{s_num}_original_synth_96x96x96.tif'
            p_morph = f'outputs/sample_{s_num}_morphed_bio_96x96x96.tif'
            p_gt = f'outputs/sample_{s_num}_ground_truth_96x96x96.tif'
            p_over = f'outputs/sample_{s_num}_overlay_composite_96x96x96.tif'

            synth_u8 = (b_std_vol.astype(np.uint8) * 255)
            morph_u8 = (b_morph_vol.astype(np.uint8) * 255)
            overlay_stack = np.stack([synth_u8, morph_u8], axis=1)  # (96, 2, 96, 96) uint8

            tifffile.imwrite(p_synth, synth_u8, compression='zlib')
            tifffile.imwrite(p_morph, morph_u8, compression='zlib')
            tifffile.imwrite(p_gt, b_morph_int.astype(np.float32), compression='zlib')
            tifffile.imwrite(p_over, overlay_stack, compression='zlib')
            print(f"    Saved sample_{s_num} TIFFs: {p_synth}, {p_morph}, {p_gt}, {p_over}")

    # gallery_96_morphed_blocks.png (4 rows x 4 columns)
    fig_96, axes_96 = plt.subplots(4, 4, figsize=(18, 18))
    col_titles = [
        "1. Standard Synthetic (Smooth)",
        "2. Morphed Biological (Real Textures)",
        "3. Exact Ground-Truth I(x) Field",
        "4. Orientation Tangent Field O(x)"
    ]

    for c_idx, title in enumerate(col_titles):
        axes_96[0, c_idx].set_title(title, fontweight='bold', fontsize=12, pad=10)

    # Prepare 3D Comparison Stack (96, 384, 384)
    comp_grid_3d = np.zeros((96, 384, 384), dtype=np.float64)

    for r_idx, (std_v, morph_v, morph_i, morph_o) in enumerate(blocks_data_96):
        # 1. Standard Synthetic MIP
        std_mip = np.max(std_v, axis=0)
        axes_96[r_idx, 0].imshow(std_mip, cmap='gray')
        axes_96[r_idx, 0].set_ylabel(f"96³ Block #{r_idx+1}", fontweight='bold', fontsize=12)
        axes_96[r_idx, 0].set_xticks([])
        axes_96[r_idx, 0].set_yticks([])

        # 2. Morphed Biological MIP
        morph_mip = np.max(morph_v, axis=0)
        axes_96[r_idx, 1].imshow(morph_mip, cmap='gray')
        axes_96[r_idx, 1].set_xticks([])
        axes_96[r_idx, 1].set_yticks([])

        # 3. Exact GT I(x) Field MIP
        int_mip = np.max(morph_i, axis=0)
        axes_96[r_idx, 2].imshow(int_mip, cmap='viridis', vmin=-1.0, vmax=1.0)
        axes_96[r_idx, 2].set_xticks([])
        axes_96[r_idx, 2].set_yticks([])

        # 4. Orientation Tangent Field RGB MIP
        # Channel-first ori: (3, 96, 96, 96) -> (|V_z|, |V_y|, |V_x|) mapped to RGB
        rgb_vol = np.zeros((96, 96, 96, 3), dtype=np.float32)
        for c in range(3):
            rgb_vol[..., c] = np.abs(morph_o[c]) * morph_v

        rgb_mip = np.max(rgb_vol, axis=0)
        axes_96[r_idx, 3].imshow(np.clip(rgb_mip, 0.0, 1.0))
        axes_96[r_idx, 3].set_xticks([])
        axes_96[r_idx, 3].set_yticks([])

        # Populate 3D comparison stack
        r_start, r_end = r_idx * 96, (r_idx + 1) * 96
        comp_grid_3d[:, r_start:r_end, 0:96] = (std_v.astype(np.float64) * 255.0)
        comp_grid_3d[:, r_start:r_end, 96:192] = (morph_v.astype(np.float64) * 255.0)
        comp_grid_3d[:, r_start:r_end, 192:288] = ((morph_i.astype(np.float64) + 1.0) / 2.0 * 255.0)
        
        # Orientation encoded into float 0..255
        ori_gray = ((morph_o.mean(axis=0) + 1.0) / 2.0 * 255.0) * morph_v + 127.5 * (~morph_v)
        comp_grid_3d[:, r_start:r_end, 288:384] = ori_gray.astype(np.float64)

    plt.tight_layout()
    plt.savefig('outputs/gallery_96_morphed_blocks.png', dpi=200)
    plt.close()
    print("  -> gallery_96_morphed_blocks.png")

    tifffile.imwrite('outputs/morphed_96_blocks_3d_comparison.tif', comp_grid_3d, compression='zlib')
    print("  -> morphed_96_blocks_3d_comparison.tif")

    # -------------------------------------------------------------------------
    # 5. Recompute Full Model 1 Overview & Slice Visualizations
    # -------------------------------------------------------------------------
    print("\n[Step 5/5] Recomputing Model 1 Morphed Full-Scale Overview & Slices...", flush=True)
    m1_vol_path = 'data/synthetic/precomputed/train/model_1_morphed_vol.npy'
    m1_int_path = 'data/synthetic/precomputed/train/model_1_morphed_intensity.npy'

    if os.path.exists(m1_vol_path) and os.path.exists(m1_int_path):
        m1_vol = np.load(m1_vol_path, mmap_mode='r')
        m1_int = np.load(m1_int_path, mmap_mode='r')
        D, H, W = m1_vol.shape

        print(f"  Loaded Model 1 ({D}x{H}x{W}), computing full volume MIPs...", flush=True)
        m1_vol_mip = np.max(m1_vol[:], axis=0).astype(np.float32)
        m1_int_mip = np.max(m1_int[:], axis=0)

        # 1. model_1_morphed_overview.png
        fig_m1, axes_m1 = plt.subplots(1, 2, figsize=(16, 8))
        axes_m1[0].imshow(m1_vol_mip, cmap='gray', vmin=0.0, vmax=1.0)
        axes_m1[0].set_title(f"Full {D}x{H}x{W} Morphed Biological Volume (MIP)", fontweight='bold', fontsize=14, color='teal')
        axes_m1[0].axis('off')

        axes_m1[1].imshow(m1_int_mip, cmap='viridis', vmin=-1.0, vmax=1.0)
        axes_m1[1].set_title("Exact Ground-Truth Centerline Potential I(x) (MIP)", fontweight='bold', fontsize=14)
        axes_m1[1].axis('off')

        plt.tight_layout()
        plt.savefig('outputs/model_1_morphed_overview.png', dpi=200)
        plt.close()
        print("  -> model_1_morphed_overview.png")

        # 2. model_1_morphed_slices.png
        mid_z = D // 2
        z0, z1 = max(0, mid_z - 8), min(D, mid_z + 8)

        slab_vol_mip = np.max(m1_vol[z0:z1], axis=0).astype(np.float32)
        slab_int_mip = np.max(m1_int[z0:z1], axis=0)

        single_slice_vol = m1_vol[mid_z].astype(np.float32)
        single_slice_int = m1_int[mid_z]

        fig_sl, axes_sl = plt.subplots(2, 2, figsize=(14, 14))

        axes_sl[0, 0].imshow(slab_vol_mip, cmap='gray', vmin=0.0, vmax=1.0)
        axes_sl[0, 0].set_title(f"16-Slice Sub-Volume Slab MIP (z={z0}..{z1})", fontweight='bold', fontsize=13, color='teal')
        axes_sl[0, 0].axis('off')

        axes_sl[0, 1].imshow(slab_int_mip, cmap='viridis', vmin=-1.0, vmax=1.0)
        axes_sl[0, 1].set_title(f"Ground Truth I(x) 16-Slice Slab MIP", fontweight='bold', fontsize=13)
        axes_sl[0, 1].axis('off')

        axes_sl[1, 0].imshow(single_slice_vol, cmap='gray', vmin=0.0, vmax=1.0)
        axes_sl[1, 0].set_title(f"Single 2D Cross-Section (Slice z={mid_z})", fontweight='bold', fontsize=13, color='teal')
        axes_sl[1, 0].axis('off')

        axes_sl[1, 1].imshow(single_slice_int, cmap='viridis', vmin=-1.0, vmax=1.0)
        axes_sl[1, 1].set_title(f"Ground Truth I(x) Single Slice z={mid_z}", fontweight='bold', fontsize=13)
        axes_sl[1, 1].axis('off')

        plt.tight_layout()
        plt.savefig('outputs/model_1_morphed_slices.png', dpi=200)
        plt.close()
        print("  -> model_1_morphed_slices.png")
    else:
        print("  Warning: Model 1 full volume files not found in data/synthetic/precomputed/train/, skipping model_1 figures.")

    print("\n" + "=" * 80)
    print(" ALL MORPHED FIBER EXAMPLES AND FIGURES RECOMPUTED SUCCESSFULLY!")
    print("=" * 80)


if __name__ == '__main__':
    recompute_all_morphed_examples()
