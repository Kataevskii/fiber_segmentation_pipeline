"""
scripts/extract_individual_fibers.py
====================================
Selective individual fiber extraction utility.

Extracts single-fiber sub-volume stamps (for stamping) and continuous
curvilinear fiber instances (for spline morphing) from user-specified
curated patches, leaving other patches exclusively as full 3D training volumes.

Usage:
  # List all curated patches and their extraction status:
  python scripts/extract_individual_fibers.py --list

  # Extract individual fibers from a specific patch:
  python scripts/extract_individual_fibers.py --patch patch_0001
  python scripts/extract_individual_fibers.py --patch 1

  # Extract individual fibers from multiple patches:
  python scripts/extract_individual_fibers.py --patches 1 2 3

  # Remove extracted fibers for a patch:
  python scripts/extract_individual_fibers.py --remove patch_0004

  # Clear all extracted individual fibers:
  python scripts/extract_individual_fibers.py --clear
"""

import os
import sys
import re
import json
import argparse
import pickle
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.fiber_morpher import RealFiberInstance, RealFiberLibrary


def normalize_patch_name(patch_arg):
    """Normalizes input like '1', 'patch_0001', 'patch_0001_vol.npy' -> 'patch_0001'."""
    s = str(patch_arg).strip()
    s = os.path.basename(s)
    for suffix in ['_vol.npy', '_vol.tif', '_instance.npy', '_meta.json', '.npy', '.tif']:
        if s.endswith(suffix):
            s = s[:-len(suffix)]
    if s.isdigit():
        return f"patch_{int(s):04d}"
    match = re.search(r'patch_(\d+)', s)
    if match:
        return f"patch_{int(match.group(1)):04d}"
    return s


def get_available_patches(curated_dir='data/curated/patches'):
    """Discovers all available curated patches in curated_dir."""
    if not os.path.exists(curated_dir):
        return []
    vols = [
        f.replace('_vol.npy', '')
        for f in os.listdir(curated_dir)
        if f.endswith('_vol.npy')
    ]
    return sorted(list(set(vols)))


def get_extracted_patch_counts(indiv_dir='data/curated/individual_fibers'):
    """Returns dict mapping patch_name -> count of extracted fibers."""
    if not os.path.exists(indiv_dir):
        return {}
    counts = {}
    for f in os.listdir(indiv_dir):
        if f.endswith('_vol.npy') and '_fiber_' in f:
            p_name = f.split('_fiber_')[0]
            counts[p_name] = counts.get(p_name, 0) + 1
    return counts


def list_patches_status(curated_dir='data/curated/patches', indiv_dir='data/curated/individual_fibers'):
    """Prints a summary of all available patches and their extraction status."""
    patches = get_available_patches(curated_dir)
    counts = get_extracted_patch_counts(indiv_dir)

    print("=" * 80)
    print(f" CURATED PATCHES & INDIVIDUAL FIBER EXTRACTION STATUS")
    print(f" Curated Patches Dir : {curated_dir}")
    print(f" Individual Fibers   : {indiv_dir}")
    print("=" * 80)

    if not patches:
        print("  No curated patches found in curated directory.")
        print("=" * 80)
        return

    print(f"{'Patch Name':<16} | {'Status':<18} | {'Extracted Stamps':<18} | {'Metadata'}")
    print("-" * 80)

    total_extracted_stamps = 0
    extracted_patches_count = 0

    for p in patches:
        n_stamps = counts.get(p, 0)
        meta_path = os.path.join(curated_dir, f"{p}_meta.json")
        meta_info = ""
        if os.path.exists(meta_path):
            try:
                with open(meta_path, 'r', encoding='utf-8') as fp:
                    m = json.load(fp)
                    meta_info = f"{m.get('num_fibers', '?')} fibers, density {m.get('density', 0.0)*100:.1f}%"
            except Exception:
                pass

        if n_stamps > 0:
            status_str = "EXTRACTED (Donor)"
            total_extracted_stamps += n_stamps
            extracted_patches_count += 1
        else:
            status_str = "Full Volume Only"

        print(f"{p:<16} | {status_str:<18} | {n_stamps:<18} | {meta_info}")

    print("-" * 80)
    print(f"Total: {len(patches)} patches ({extracted_patches_count} donor patches, {total_extracted_stamps} total stamps).")
    print("=" * 80 + "\n")


def extract_fibers_from_patch(patch_name, curated_dir='data/curated/patches', indiv_dir='data/curated/individual_fibers', padding=2, verbose=True):
    """Extracts all individual fibers from a specific curated patch."""
    p_name = normalize_patch_name(patch_name)
    p_vol_path = os.path.join(curated_dir, f"{p_name}_vol.npy")
    p_inst_path = os.path.join(curated_dir, f"{p_name}_instance.npy")
    p_int_path = os.path.join(curated_dir, f"{p_name}_intensity.npy")
    p_ori_path = os.path.join(curated_dir, f"{p_name}_ori.npy")

    if not (os.path.exists(p_vol_path) and os.path.exists(p_inst_path) and os.path.exists(p_int_path) and os.path.exists(p_ori_path)):
        if verbose:
            print(f"Error: Missing required arrays for '{p_name}' in '{curated_dir}'. Skipping.", flush=True)
        return 0

    os.makedirs(indiv_dir, exist_ok=True)
    vol = np.load(p_vol_path)
    inst = np.load(p_inst_path)
    intensity = np.load(p_int_path)
    ori = np.load(p_ori_path)

    unique_ids = np.unique(inst[inst > 0])
    extracted_count = 0

    for fib_id in unique_ids:
        mask = (inst == fib_id)
        coords = np.argwhere(mask)
        if len(coords) < 3:
            continue

        zmin, ymin, xmin = coords.min(axis=0)
        zmax, ymax, xmax = coords.max(axis=0) + 1

        z0, z1 = max(0, zmin - padding), min(vol.shape[0], zmax + padding)
        y0, y1 = max(0, ymin - padding), min(vol.shape[1], ymax + padding)
        x0, x1 = max(0, xmin - padding), min(vol.shape[2], xmax + padding)

        sub_mask = (inst[z0:z1, y0:y1, x0:x1] == fib_id)
        sub_vol = (vol[z0:z1, y0:y1, x0:x1] * sub_mask).astype(bool)
        sub_int = (intensity[z0:z1, y0:y1, x0:x1] * sub_mask).astype(np.float32)
        sub_ori = (ori[:, z0:z1, y0:y1, x0:x1] * sub_mask).astype(np.float32)

        out_prefix = os.path.join(indiv_dir, f"{p_name}_fiber_{fib_id:02d}")
        np.save(f"{out_prefix}_vol.npy", sub_vol)
        np.save(f"{out_prefix}_intensity.npy", sub_int)
        np.save(f"{out_prefix}_ori.npy", sub_ori)
        extracted_count += 1

    if verbose:
        print(f"Successfully extracted {extracted_count} individual fiber stamps from '{p_name}' into '{indiv_dir}'.", flush=True)

    return extracted_count


def remove_extracted_fibers_for_patch(patch_name, indiv_dir='data/curated/individual_fibers', verbose=True):
    """Removes all extracted fiber stamp files for a specified patch."""
    p_name = normalize_patch_name(patch_name)
    if not os.path.exists(indiv_dir):
        return 0

    prefix = f"{p_name}_fiber_"
    removed_count = 0
    for f in os.listdir(indiv_dir):
        if f.startswith(prefix) and f.endswith(('.npy', '.tif')):
            try:
                os.remove(os.path.join(indiv_dir, f))
                removed_count += 1
            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to delete {f}: {e}", flush=True)

    if verbose:
        print(f"Removed {removed_count} fiber files for patch '{p_name}' from '{indiv_dir}'.", flush=True)
    return removed_count


def clear_all_extracted_fibers(indiv_dir='data/curated/individual_fibers', verbose=True):
    """Removes all extracted fiber stamps from indiv_dir."""
    if not os.path.exists(indiv_dir):
        return 0
    removed_count = 0
    for f in os.listdir(indiv_dir):
        if f.endswith(('.npy', '.tif')) and '_fiber_' in f:
            try:
                os.remove(os.path.join(indiv_dir, f))
                removed_count += 1
            except Exception:
                pass
    if verbose:
        print(f"Cleared {removed_count} individual fiber files from '{indiv_dir}'.", flush=True)
    return removed_count


def sync_morpher_library(curated_dir='data/curated/patches', indiv_dir='data/curated/individual_fibers', cache_path='data/curated/fiber_library.pkl', verbose=True):
    """
    Rebuilds data/curated/fiber_library.pkl containing only fibers from patches
    that currently have stamps in indiv_dir.
    """
    extracted_counts = get_extracted_patch_counts(indiv_dir)
    extracted_patch_names = set(extracted_counts.keys())

    if verbose:
        print(f"Synchronizing morpher library '{cache_path}' with {len(extracted_patch_names)} donor patches...", flush=True)

    if not extracted_patch_names:
        os.makedirs(os.path.dirname(cache_path) if os.path.dirname(cache_path) else '.', exist_ok=True)
        with open(cache_path, 'wb') as fp:
            pickle.dump([], fp)
        if verbose:
            print("Morpher library synchronized: 0 donor fibers (no patches extracted).", flush=True)
        return 0

    lib = RealFiberLibrary(curated_dir=curated_dir, cache_path=cache_path, min_length=75, allowed_patches=extracted_patch_names, force_rebuild=True)
    if verbose:
        print(f"Morpher library synchronized: {len(lib.fibers)} continuous fibers across {len(extracted_patch_names)} donor patches.", flush=True)
    return len(lib.fibers)


def main():
    parser = argparse.ArgumentParser(description="Selective Individual Fiber Extraction Utility")
    parser.add_argument('--patch', type=str, default=None, help="Extract individual fibers from a specific patch (e.g. patch_0001 or 1)")
    parser.add_argument('--patches', nargs='+', default=None, help="Extract individual fibers from multiple patches (e.g. 1 2 3)")
    parser.add_argument('--all', action='store_true', help="Extract individual fibers from all curated patches")
    parser.add_argument('--list', action='store_true', help="List all curated patches and their extraction status")
    parser.add_argument('--remove', type=str, default=None, help="Remove extracted individual fibers for a patch (e.g. patch_0004 or 4)")
    parser.add_argument('--clear', action='store_true', help="Clear all extracted individual fibers from directory")
    parser.add_argument('--curated-dir', type=str, default='data/curated/patches', help="Curated patches directory (default: data/curated/patches)")
    parser.add_argument('--indiv-dir', type=str, default='data/curated/individual_fibers', help="Individual fibers output directory (default: data/curated/individual_fibers)")
    parser.add_argument('--library-cache', type=str, default='data/curated/fiber_library.pkl', help="Path to cached fiber morpher library (default: data/curated/fiber_library.pkl)")
    parser.add_argument('--padding', type=int, default=2, help="Voxel margin padding around extracted fiber stamp bounding boxes (default: 2)")
    parser.add_argument('--no-sync-library', action='store_true', help="Do not update data/curated/fiber_library.pkl")
    args = parser.parse_args()

    # If --list specified or no action given:
    if args.list or (args.patch is None and args.patches is None and not args.all and args.remove is None and not args.clear):
        list_patches_status(curated_dir=args.curated_dir, indiv_dir=args.indiv_dir)
        if args.patch is None and args.patches is None and not args.all and args.remove is None and not args.clear:
            print("Tip: Use --patch <id> (e.g. --patch 1) to extract fibers for a patch, or --help for all options.")
        return

    # Handle --clear
    if args.clear:
        clear_all_extracted_fibers(indiv_dir=args.indiv_dir, verbose=True)
        if not args.no_sync_library:
            sync_morpher_library(curated_dir=args.curated_dir, indiv_dir=args.indiv_dir, cache_path=args.library_cache, verbose=True)
        return

    # Handle --remove
    if args.remove:
        remove_extracted_fibers_for_patch(args.remove, indiv_dir=args.indiv_dir, verbose=True)
        if not args.no_sync_library:
            sync_morpher_library(curated_dir=args.curated_dir, indiv_dir=args.indiv_dir, cache_path=args.library_cache, verbose=True)
        return

    # Collect target patches for extraction
    target_patches = []
    if args.all:
        target_patches = get_available_patches(args.curated_dir)
    else:
        if args.patch:
            target_patches.append(normalize_patch_name(args.patch))
        if args.patches:
            for p in args.patches:
                target_patches.append(normalize_patch_name(p))

    # Deduplicate while preserving order
    seen = set()
    deduped = []
    for p in target_patches:
        if p not in seen:
            seen.add(p)
            deduped.append(p)
    target_patches = deduped

    if not target_patches:
        print("No valid patches specified for extraction.")
        return

    print(f"Extracting individual fibers from {len(target_patches)} patch(es): {', '.join(target_patches)}...")
    total_stamps = 0
    for p in target_patches:
        total_stamps += extract_fibers_from_patch(
            patch_name=p,
            curated_dir=args.curated_dir,
            indiv_dir=args.indiv_dir,
            padding=args.padding,
            verbose=True
        )

    print(f"\nExtraction complete: {total_stamps} total fiber stamps extracted.")

    if not args.no_sync_library:
        sync_morpher_library(curated_dir=args.curated_dir, indiv_dir=args.indiv_dir, cache_path=args.library_cache, verbose=True)


if __name__ == '__main__':
    main()

