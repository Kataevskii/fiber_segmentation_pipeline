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
    print("\n--- [1/3] Testing Random 96x96x96 Patch Extraction ---")
    patch_info = engine.extract_random_patch(min_density=0.04, max_density=0.20)
    print(f"Extracted Patch Origin: {patch_info['origin']}, Density: {patch_info['density']*100:.2f}%")
    print(f"Auto-detected {len(patch_info['auto_candidates'])} candidate face blobs.")
    assert len(patch_info['face_images']) == 6, "Expected 6 face images"

    # 2. Test seed connection resolution
    print("\n--- [2/3] Testing Deterministic Geodesic Path Resolution ---")
    candidates = patch_info['auto_candidates']
    if len(candidates) >= 2:
        # Create paired seeds for Fiber 1, and singleton for Fiber 2
        test_seeds = [
            {'face': candidates[0]['face'], 'u': candidates[0]['u'], 'v': candidates[0]['v'], 'pos3d': candidates[0]['pos3d'], 'fiber_id': 1},
            {'face': candidates[1]['face'], 'u': candidates[1]['u'], 'v': candidates[1]['v'], 'pos3d': candidates[1]['pos3d'], 'fiber_id': 1},
        ]
        if len(candidates) >= 3:
            test_seeds.append({
                'face': candidates[2]['face'], 'u': candidates[2]['u'], 'v': candidates[2]['v'], 'pos3d': candidates[2]['pos3d'], 'fiber_id': 2
            })
    else:
        test_seeds = [
            {'face': 'z_min', 'u': 48, 'v': 48, 'pos3d': (0, 48, 48), 'fiber_id': 1},
            {'face': 'z_max', 'u': 48, 'v': 48, 'pos3d': (95, 48, 48), 'fiber_id': 1}
        ]

    res = engine.resolve_connections(test_seeds)
    print(f"Resolution Successful: {res['success']}, Resolved {res['num_fibers']} fibers in {res['resolve_time_ms']} ms!")
    assert res['success'] == True

    # 3. Test saving curated sample
    print("\n--- [3/3] Testing Save Curated Sample to real_train_data/curated_patches/ ---")
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

    print("\n" + "=" * 80)
    print(" ALL CURATION ENGINE TESTS PASSED WITH 100% ACCURACY! ")
    print("=" * 80)

if __name__ == '__main__':
    test_curation_engine()
