import os
import sys
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
from curation_tool.engine import RealDataCurationEngine

def test_curation_engine():
    print("=" * 80)
    print(" TESTING REAL DATA CURATION ENGINE (ZERO TEMPORARY FILES RETAINED) ")
    print("=" * 80)

    # Use an isolated temporary directory to ensure NO temporal files are kept on disk
    with tempfile.TemporaryDirectory() as tmp_curated_dir:
        print(f"Temporary test sandbox initialized at: {tmp_curated_dir}")

        raw_vol_path = 'process_data/COLLAGENCROP_003_0000.tif'
        if not os.path.exists(raw_vol_path):
            raw_vol_path = 'process_data/CROP_003_0000.tif'

        engine = RealDataCurationEngine(
            raw_volume_path=raw_vol_path,
            curated_output_dir=tmp_curated_dir,
            cube_size=96
        )

        # 1. Test random patch extraction
        print("\n--- [1/7] Testing Random 96x96x96 Patch Extraction ---")
        patch_info = engine.extract_random_patch(min_density=0.03, max_density=0.30)
        print(f"Extracted Patch Origin: {patch_info['origin']}, Density: {patch_info['density']*100:.2f}%")
        assert len(patch_info['face_images']) == 6, "Expected 6 face images"
        assert 'volume_shape' in patch_info and 'max_origin' in patch_info

        # 2. Test manual coordinate patch extraction & boundary clamping
        print("\n--- [2/7] Testing Manual Coordinate Extraction & Clamping ---")
        vol_info = engine.get_volume_info()
        print(f"Volume Info: Shape={vol_info['shape']}, CubeSize={vol_info['cube_size']}, MaxOrigin={vol_info['max_origin']}")
        manual_patch = engine.extract_patch_at(50, 60, 70)
        print(f"Manual Patch Origin: {manual_patch['origin']}, Density: {manual_patch['density']*100:.2f}%")
        assert manual_patch['origin'] == [50, 60, 70], f"Expected [50, 60, 70], got {manual_patch['origin']}"
        assert len(manual_patch['face_images']) == 6, "Expected 6 face images"

        clamped_patch = engine.extract_patch_at(9999, 9999, 9999)
        print(f"Overflow Clamped Origin: {clamped_patch['origin']} (Expected: {vol_info['max_origin']})")
        assert clamped_patch['origin'] == vol_info['max_origin']

        # 3. Test visual sub-box crop ROI extraction
        print("\n--- [3/7] Testing Visual Sub-Box Crop ROI Extraction ---")
        crop_info = engine.get_cropped_view_data(z_min=16, z_max=79, y_min=20, y_max=80, x_min=10, x_max=90)
        print(f"Cropped Shape: {crop_info['crop_shape']}, Bounds: {crop_info['crop_bounds']}, Density: {crop_info['density']*100:.2f}%")
        assert crop_info['crop_bounds'] == [16, 79, 20, 80, 10, 90]
        assert crop_info['crop_shape'] == [64, 61, 81]
        assert len(crop_info['face_images']) == 6
        assert len(crop_info['face_images_rgba']) == 6
        assert 'point_cloud' in crop_info

        # 4. Test deterministic geodesic path resolution
        print("\n--- [4/7] Testing Geodesic Path Resolution ---")
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
        print(f"Resolution Successful: {res['success']}, Resolved {res['num_fibers']} fibers in {res['resolve_time_ms']} ms")
        assert res['success'] == True

        # 5. Test saving curated sample to isolated temporary directory
        print("\n--- [5/7] Testing Save Curated Sample & Reloading in Sandbox ---")
        save_res = engine.save_current_curated_sample()
        prefix = save_res['prefix']
        print(f"Saved Sandbox Patch #{save_res['saved_index']} to {prefix}*")

        # Verify all 9 files created in the temporary sandbox
        for suffix in ['_vol.npy', '_vol.tif', '_instance.npy', '_instance.tif', '_centerline.npy', '_centerline.tif', '_ori.npy', '_intensity.npy', '_meta.json']:
            fpath = f"{prefix}{suffix}"
            assert os.path.exists(fpath), f"Missing expected output file: {fpath}"

        # Test listing saved patches
        saved_list = engine.list_saved_patches()
        assert len(saved_list) >= 1
        print(f"Saved Patches in Sandbox: {len(saved_list)} items listed correctly")

        # Test reloading curated patch
        loaded_patch = engine.load_curated_patch(save_res['saved_index'])
        assert loaded_patch['num_fibers'] == res['num_fibers']
        print(f"Reloaded Patch #{save_res['saved_index']} verified successfully")

        # 6. Test opening segmented instance TIFF (if present)
        seg_candidate = 'outputs/topology_resolved_morpho/instance_volume.tif'
        if os.path.exists(seg_candidate):
            print(f"\n--- [6/7] Testing Segmented Instance TIFF Opening ({seg_candidate}) ---")
            open_res = engine.open_volume_file(seg_candidate, z=100, y=100, x=100)
            assert open_res['is_segmented_source'] == True
            assert open_res['source_info']['type'] == 'segmented_instance'
            assert 'loaded_seeds' in open_res and len(open_res['loaded_seeds']) > 0
            print(f"Segmented TIFF Extraction: {open_res['num_fibers']} fibers, {len(open_res['loaded_seeds'])} seeds pre-populated")
        else:
            print("\n--- [6/7] Skipping Segmented Instance TIFF (file not found) ---")

        # 7. Test workspace file discovery
        print("\n--- [7/7] Testing Workspace File Discovery ---")
        files_dict = engine.list_available_files()
        assert 'segmented' in files_dict and 'raw' in files_dict
        print(f"Discovered: {len(files_dict['segmented'])} Segmented volumes, {len(files_dict['raw'])} Raw volumes")

    # Verify temporary directory is completely wiped
    assert not os.path.exists(tmp_curated_dir), "Temporary directory was not cleaned up!"
    print("\n" + "=" * 80)
    print(" ALL TESTS PASSED! ZERO TEMPORARY FILES RETAINED ON DISK. ")
    print("=" * 80)

if __name__ == '__main__':
    test_curation_engine()
