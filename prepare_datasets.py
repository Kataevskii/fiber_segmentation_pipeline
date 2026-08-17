import os
import shutil
import numpy as np
import tifffile

from core.gt_generator import compute_analytical_orientation_from_gad

def prepare_all_datasets(overwrite=False):
    print("=" * 75, flush=True)
    print(" BATCH DATASET PRECOMPUTATION (MEMORY-MAPPED NPY FORMAT) ", flush=True)
    print("  - Signed Probability Targets with Subtracted Intersection Gaussians")
    print("  - Training Datasets: Models 1..8 -> augmented_data/")
    print("  - Test Dataset: Model 9 -> test_data/")
    print("  - Final Validation Dataset: Model 10 -> val_data/")
    print("=" * 75, flush=True)

    os.makedirs('augmented_data', exist_ok=True)
    os.makedirs('test_data', exist_ok=True)
    os.makedirs('val_data', exist_ok=True)
    os.makedirs('raw_data', exist_ok=True)

    # Move model 5 from test_data to augmented_data if present
    for ext in ['_base_vol.npy', '_base_vol.tif', '_base_intensity.npy', '_base_ori.npy', '_base_centerline.npy', '_base_centerline.tif', '_base_intensity.tif']:
        src_f = os.path.join('test_data', f'model_5{ext}')
        dst_f = os.path.join('augmented_data', f'model_5{ext}')
        if os.path.exists(src_f) and not os.path.exists(dst_f):
            print(f"Migrating {src_f} -> {dst_f}...", flush=True)
            shutil.move(src_f, dst_f)

    # Move model 6 from val_data to augmented_data if present
    for ext in ['_base_vol.npy', '_base_vol.tif', '_base_intensity.npy', '_base_ori.npy', '_base_centerline.npy', '_base_centerline.tif', '_base_intensity.tif']:
        src_f = os.path.join('val_data', f'model_6{ext}')
        dst_f = os.path.join('augmented_data', f'model_6{ext}')
        if os.path.exists(src_f) and not os.path.exists(dst_f):
            print(f"Migrating {src_f} -> {dst_f}...", flush=True)
            shutil.move(src_f, dst_f)

    # Clean up any non-model-9 files in test_data
    for f in os.listdir('test_data'):
        if not f.startswith('model_9_'):
            p = os.path.join('test_data', f)
            print(f"Cleaning obsolete test file: {p}", flush=True)
            os.remove(p)

    # Clean up any non-model-10 files in val_data
    for f in os.listdir('val_data'):
        if not f.startswith('model_10_'):
            p = os.path.join('val_data', f)
            print(f"Cleaning obsolete val file: {p}", flush=True)
            os.remove(p)

    def process_model(idx, out_dir):
        gad_path = os.path.join('raw_data', f'AJ_model_{idx}.gad')
        tif_path = os.path.join('raw_data', f'AJ_model_{idx}.tif')

        if not os.path.exists(gad_path) or not os.path.exists(tif_path):
            print(f"Warning: Raw dataset files missing for model {idx}: {gad_path} or {tif_path}, skipping...", flush=True)
            return

        base_vol_tif = os.path.join(out_dir, f'model_{idx}_base_vol.tif')
        base_vol_npy = os.path.join(out_dir, f'model_{idx}_base_vol.npy')
        base_intensity_npy = os.path.join(out_dir, f'model_{idx}_base_intensity.npy')
        base_ori_npy = os.path.join(out_dir, f'model_{idx}_base_ori.npy')

        if overwrite or not (os.path.exists(base_vol_npy) and os.path.exists(base_intensity_npy) and os.path.exists(base_ori_npy)):
            print(f"  Computing GT Signed Probability Target (std=1.0, Subtracted Intersections) & Orientation for Model {idx} in {out_dir}/...", flush=True)
            if not os.path.exists(base_vol_tif):
                shutil.copyfile(tif_path, base_vol_tif)
            
            if not os.path.exists(base_vol_npy):
                vol = tifffile.imread(tif_path).astype(np.float32)
                np.save(base_vol_npy, vol)

            compute_analytical_orientation_from_gad(
                gad_path=gad_path,
                tif_path=tif_path,
                out_vector_path=base_ori_npy,
                out_intensity_path=base_intensity_npy,
                sigma=1.0
            )
        else:
            print(f"  GT targets for Model {idx} already precomputed in {out_dir}/", flush=True)

    # 1. Training Datasets (Models 1..8)
    print("\n--- Processing Training Datasets (Models 1..8) ---", flush=True)
    for idx in range(1, 9):
        process_model(idx, 'augmented_data')

    # 2. Test Dataset (Model 9)
    print("\n--- Processing Test Dataset (Model 9) ---", flush=True)
    process_model(9, 'test_data')

    # 3. Final Validation Dataset (Model 10)
    print("\n--- Processing Final Validation Dataset (Model 10) ---", flush=True)
    process_model(10, 'val_data')

    print("\n" + "=" * 75, flush=True)
    print(" ALL DATASETS PREPARED & MEMORY-MAPPED SUCCESSFULLY! ", flush=True)
    print("=" * 75, flush=True)

if __name__ == '__main__':
    prepare_all_datasets()

