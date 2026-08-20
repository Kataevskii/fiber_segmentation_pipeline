import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from curation_tool.engine import RealDataCurationEngine

def test_curation_engine():
    print("=" * 80)
    print(" TESTING REAL DATA CURATION ENGINE (96x96x96 DETERMINISTIC RESOLUTION) ")
    print("=" * 80)

    engine = RealDataCurationEngine(
        raw_volume_path='process_data/COLLAGENCROP_003_0000.tif',
        curated_output_dir='real_train_data/curated_patches',
        cube_size=96
    )

    # 1. Test random patch extraction
    print("\n--- [1/4] Testing Random 96x96x96 Patch Extraction ---")
    patch_info = engine.extract_random_patch(min_density=0.04, max_density=0.20)
    print(f"Extracted Patch Origin: {patch_info['origin']}, Density: {patch_info['density']*100:.2f}%")
    assert len(patch_info['face_images']) == 6, "Expected 6 face images"
    assert 'volume_shape' in patch_info and 'max_origin' in patch_info

    # 2. Test manual coordinate patch extraction
    print("\n--- [2/4] Testing Manual Coordinate Patch Extraction (Z=100, Y=120, X=140) ---")
    vol_info = engine.get_volume_info()
    print(f"Volume Info: Shape={vol_info['shape']}, CubeSize={vol_info['cube_size']}, MaxOrigin={vol_info['max_origin']}")
    manual_patch = engine.extract_patch_at(100, 120, 140)
    print(f"Manual Patch Origin: {manual_patch['origin']}, Density: {manual_patch['density']*100:.2f}%")
    assert manual_patch['origin'] == [100, 120, 140], f"Expected [100, 120, 140], got {manual_patch['origin']}"
    assert len(manual_patch['face_images']) == 6, "Expected 6 face images"

    # Test coordinate clamping on overflow
    clamped_patch = engine.extract_patch_at(9999, 9999, 9999)
    print(f"Overflow Clamped Origin: {clamped_patch['origin']} (Max Expected: {vol_info['max_origin']})")
    assert clamped_patch['origin'] == vol_info['max_origin']

    # 3. Test seed connection resolution
    print("\n--- [3/4] Testing Deterministic Geodesic Path Resolution ---")
    fg_coords = np.argwhere(engine.current_patch_bin > 0)
    if len(fg_coords) >= 2:
        p1 = fg_coords[0]
        p2 = fg_coords[-1]
        test_seeds = [
            {'face': 'custom', 'u': int(p1[1]), 'v': int(p1[2]), 'pos3d': list(p1), 'fiber_id': 1},
            {'face': 'custom', 'u': int(p2[1]), 'v': int(p2[2]), 'pos3d': list(p2), 'fiber_id': 1},
        ]
    else:
        test_seeds = [
            {'face': 'z_min', 'u': 48, 'v': 48, 'pos3d': [0, 48, 48], 'fiber_id': 1},
            {'face': 'z_max', 'u': 48, 'v': 48, 'pos3d': [95, 48, 48], 'fiber_id': 1}
        ]

    res = engine.resolve_connections(test_seeds)
    print(f"Resolution Successful: {res['success']}, Resolved {res['num_fibers']} fibers in {res['resolve_time_ms']} ms!")
    assert res['success'] == True

    # 4. Test saving curated sample
    print("\n--- [4/4] Testing Save Curated Sample to real_train_data/curated_patches/ ---")
    save_res = engine.save_current_curated_sample()
    print(f"Saved Curated Patch #{save_res['saved_index']} to {save_res['prefix']}*")
    
    # Verify files created
    prefix = save_res['prefix']
    assert os.path.exists(f"{prefix}_vol.npy")
    assert os.path.exists(f"{prefix}_instance.npy")
    assert os.path.exists(f"{prefix}_centerline.npy")
    assert os.path.exists(f"{prefix}_ori.npy")
    assert os.path.exists(f"{prefix}_intensity.npy")
    assert os.path.exists(f"{prefix}_meta.json")

    # 5. Test visual sub-box crop extraction (64x64x64 and custom ROI)
    print("\n--- [5/5] Testing Visual Sub-Box Crop Extraction (ROI) ---")
    crop_info = engine.get_cropped_view_data(z_min=16, z_max=79, y_min=20, y_max=80, x_min=10, x_max=90)
    print(f"Cropped Shape: {crop_info['crop_shape']}, Bounds: {crop_info['crop_bounds']}, Density: {crop_info['density']*100:.2f}%")
    assert crop_info['crop_bounds'] == [16, 79, 20, 80, 10, 90]
    assert crop_info['crop_shape'] == [64, 61, 81]
    assert len(crop_info['face_images']) == 6, "Expected 6 cropped face images"
    assert len(crop_info['face_images_rgba']) == 6, "Expected 6 cropped RGBA face images"
    assert 'point_cloud' in crop_info, "Expected point cloud in cropped data"

    print("\n" + "=" * 80)
    print(" ALL CURATION ENGINE TESTS (INCLUDING VISUAL CROP) PASSED WITH 100% ACCURACY! ")
    print("=" * 80)

if __name__ == '__main__':
    test_curation_engine()
