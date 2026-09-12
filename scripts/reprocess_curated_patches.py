"""
scripts/reprocess_curated_patches.py
====================================
Batch reprocesses and regenerates all curated patches in data/curated/patches/
using the updated RealDataCurationEngine.

Guarantees:
1. Colors NEVER propagate through empty space into unseeded connected components.
2. Centerline coordinates are hard-locked in the instance volume (0 mismatches).
3. Orientation vectors and distance fields are strictly synchronized with centered curves.
"""

import os
import sys
import json
import time
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from curation_tool.engine import RealDataCurationEngine


def main():
    curated_dir = os.path.join('data', 'curated', 'patches')
    raw_vol = os.path.join('data', 'fibers_to_segment', 'COLLAGENCROP_003_0000.tif')
    if not os.path.exists(raw_vol):
        cands = [
            os.path.join('data', 'fibers_to_segment', f)
            for f in sorted(os.listdir('data/fibers_to_segment'))
            if f.lower().endswith(('.tif', '.tiff', '.npy'))
        ]
        if cands:
            raw_vol = cands[0]

    print("=" * 80)
    print(" REPROCESSING ALL CURATED PATCHES WITH CC-GUARDED LABEL DIFFUSION ")
    print("=" * 80)
    print(f"Raw Volume Source : {raw_vol}")
    print(f"Curated Directory : {curated_dir}")

    engine = RealDataCurationEngine(
        raw_volume_path=raw_vol,
        curated_output_dir=curated_dir,
        cube_size=96
    )

    meta_files = sorted([
        f for f in os.listdir(curated_dir)
        if f.startswith('patch_') and f.endswith('_meta.json')
    ])

    if not meta_files:
        print("No curated patches found in directory.")
        return

    print(f"Found {len(meta_files)} patches to reprocess.\n")

    t_start = time.time()
    results = []

    for mf in meta_files:
        patch_idx = int(mf.split('_')[1])
        base_name = mf.replace('_meta.json', '')
        print(f"--> Processing {base_name} (Sample #{patch_idx:04d})...", flush=True)

        t0 = time.time()
        patch_data = engine.load_curated_patch(patch_idx)
        save_res = engine.save_current_curated_sample(overwrite=True)
        elapsed = time.time() - t0

        # Verification on saved arrays
        inst_vol = np.load(os.path.join(curated_dir, f"{base_name}_instance.npy"))
        inst_skel = np.load(os.path.join(curated_dir, f"{base_name}_centerline.npy"))

        skel_mask = inst_skel > 0
        total_skel_voxels = int(np.sum(skel_mask))
        mismatches = int(np.sum(inst_vol[skel_mask] != inst_skel[skel_mask]))
        total_vol_voxels = int(np.sum(inst_vol > 0))

        results.append({
            'patch': base_name,
            'fibers': patch_data['num_fibers'],
            'skel_voxels': total_skel_voxels,
            'vol_voxels': total_vol_voxels,
            'mismatches': mismatches,
            'time_s': elapsed
        })

        status = "OK" if mismatches == 0 else "FAIL"
        print(f"    [{status}] Fibers: {patch_data['num_fibers']}, Skel Vx: {total_skel_voxels}, Vol Vx: {total_vol_voxels}, Mismatches: {mismatches} ({elapsed:.2f}s)", flush=True)

    print("\n" + "=" * 80)
    print(" REPROCESSING SUMMARY ")
    print("=" * 80)
    print(f"{'Patch':<14} | {'Fibers':<7} | {'Skel Vx':<9} | {'Vol Vx':<9} | {'Mismatches':<10} | {'Status':<6}")
    print("-" * 65)

    all_clean = True
    for r in results:
        status_str = "CLEAN" if r['mismatches'] == 0 else "ERROR"
        if r['mismatches'] > 0:
            all_clean = False
        print(f"{r['patch']:<14} | {r['fibers']:<7} | {r['skel_voxels']:<9} | {r['vol_voxels']:<9} | {r['mismatches']:<10} | {status_str:<6}")

    total_elapsed = time.time() - t_start
    print("-" * 65)
    print(f"Total time: {total_elapsed:.2f}s | All Clean: {all_clean}\n")
    assert all_clean, "One or more patches had centerline mismatches!"


if __name__ == '__main__':
    main()

