import os
import re
import shutil
import argparse
import numpy as np
import tifffile

from core.gt_generator import compute_analytical_orientation_from_gad


def natural_sort_key(s):
    """Sort strings containing numbers in human/natural order (e.g. model_2 before model_10)."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]


def find_raw_models(raw_dir='raw_data'):
    """
    Finds all matching pairs of (.tif / .tiff, .gad) in raw_dir.
    Returns sorted list of model dictionaries.
    """
    if not os.path.exists(raw_dir):
        return []

    files = os.listdir(raw_dir)
    tif_files = [f for f in files if f.lower().endswith(('.tif', '.tiff'))]

    models = []
    for tif_file in tif_files:
        base_name = os.path.splitext(tif_file)[0]
        gad_file = base_name + '.gad'
        tif_path = os.path.join(raw_dir, tif_file)
        gad_path = os.path.join(raw_dir, gad_file)

        if os.path.exists(gad_path):
            num_match = re.search(r'(\d+)', base_name)
            idx = int(num_match.group(1)) if num_match else len(models) + 1
            models.append({
                'idx': idx,
                'name': base_name,
                'tif': tif_path,
                'gad': gad_path
            })

    models.sort(key=lambda m: natural_sort_key(m['name']))
    return models


def process_model(model_info, out_dir, overwrite=False):
    """Processes a single model and saves memory-mapped arrays in out_dir."""
    idx = model_info['idx']
    name = model_info['name']
    gad_path = model_info['gad']
    tif_path = model_info['tif']

    base_vol_tif = os.path.join(out_dir, f'model_{idx}_base_vol.tif')
    base_vol_npy = os.path.join(out_dir, f'model_{idx}_base_vol.npy')
    base_intensity_npy = os.path.join(out_dir, f'model_{idx}_base_intensity.npy')
    base_ori_npy = os.path.join(out_dir, f'model_{idx}_base_ori.npy')

    if overwrite or not (os.path.exists(base_vol_npy) and os.path.exists(base_intensity_npy) and os.path.exists(base_ori_npy)):
        print(f"  Computing GT Signed Potential (std=1.0) & Orientation for {name} -> {out_dir}/...", flush=True)
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
        print(f"  GT targets for {name} already precomputed in {out_dir}/", flush=True)


def prepare_all_datasets(
    raw_dir: str = 'raw_data',
    train_dir: str = 'augmented_data',
    test_dir: str = 'test_data',
    test_split: float = 0.10,
    overwrite: bool = False
):
    print("=" * 75, flush=True)
    print(" BATCH DATASET PRECOMPUTATION (MEMORY-MAPPED NPY FORMAT) ", flush=True)
    print(f"  - Raw Data Directory: {raw_dir}/")
    print(f"  - Training Target:     {train_dir}/")
    print(f"  - Test Target:         {test_dir}/ (Default split: {test_split * 100:.0f}%)")
    print("=" * 75, flush=True)

    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    os.makedirs(raw_dir, exist_ok=True)

    models = find_raw_models(raw_dir)
    if not models:
        print(f"Warning: No matching (.tif, .gad) model pairs found in {raw_dir}/.", flush=True)
        print(f"Please place your raw synthetic files (e.g. AJ_model_*.tif, AJ_model_*.gad) into {raw_dir}/ first.", flush=True)
        return

    n_total = len(models)
    n_test = max(1, int(round(n_total * test_split))) if (n_total > 1 and test_split > 0.0) else 0
    n_train = n_total - n_test

    train_models = models[:n_train]
    test_models = models[n_train:]

    print(f"Discovered {n_total} raw synthetic models in {raw_dir}/:")
    print(f"  -> Train ({len(train_models)} models, {100.0 * len(train_models) / n_total:.0f}%): {[m['name'] for m in train_models]}")
    print(f"  -> Test  ({len(test_models)} models, {100.0 * len(test_models) / n_total:.0f}%): {[m['name'] for m in test_models]}\n", flush=True)

    print("--- [1/2] Processing Training Datasets ---", flush=True)
    for m in train_models:
        process_model(m, train_dir, overwrite=overwrite)

    if test_models:
        print("\n--- [2/2] Processing Test Datasets ---", flush=True)
        for m in test_models:
            process_model(m, test_dir, overwrite=overwrite)

    print("\n" + "=" * 75, flush=True)
    print(" ALL DATASETS PREPARED & MEMORY-MAPPED SUCCESSFULLY! ", flush=True)
    print("=" * 75, flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Precompute Memory-Mapped Continuous Ground-Truth Fields")
    parser.add_argument('--raw-dir', type=str, default='raw_data', help="Directory containing raw synthetic (.tif, .gad) models (default: raw_data)")
    parser.add_argument('--train-dir', type=str, default='augmented_data', help="Output directory for training datasets (default: augmented_data)")
    parser.add_argument('--test-dir', type=str, default='test_data', help="Output directory for test datasets (default: test_data)")
    parser.add_argument('--test-split', type=float, default=0.10, help="Fraction of raw models to allocate for test set (default: 0.10 / 10%%)")
    parser.add_argument('--overwrite', action='store_true', help="Force recomputation even if .npy files already exist")
    args = parser.parse_args()

    prepare_all_datasets(
        raw_dir=args.raw_dir,
        train_dir=args.train_dir,
        test_dir=args.test_dir,
        test_split=args.test_split,
        overwrite=args.overwrite
    )

