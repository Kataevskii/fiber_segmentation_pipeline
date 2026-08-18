"""
run_inference.py
================
Runs GPU-accelerated 3D sliding-window inference using the Decoupled Specialist Models:
  1. Intensity Specialist (IntensityUNet3D) -> continuous radial intensity potential field I(x) in [-1.0, 1.0]
  2. Orientation Specialist (OrientationUNet3D) -> continuous 3D unit tangent vector field O(x) = (Vz, Vy, Vx)

Features:
  - Selectable inference modes: 'both' (default), 'intensity' (intensity-only), or 'orientation' (orientation-only)
  - Zero-RAM-overflow streamed architecture: uses disk-backed memory mapped accumulators (np.lib.format.open_memmap)
    and chunked normalization, preventing ArrayMemoryError even on huge >50 GB volumes.
  - Chunked BigTIFF export for memory-safe ImageJ visualization.
  - Exports float32 .npy arrays and standard .tif stacks.
"""

import os
import sys
import time
import argparse
import gc
import tempfile
import numpy as np
import torch
import tifffile
from tqdm import tqdm

# Ensure package root is in path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from core.models import IntensityUNet3D, OrientationUNet3D, infer_base_channels_from_checkpoint


def predict_sliding_window(
    model,
    volume,
    out_npy_path,
    patch_size=64,
    stride=32,
    device='cuda',
    out_channels=1,
    batch_size=4,
    desc="Inference",
    temp_dir=None,
    chunk_size=32
):
    """3D sliding window inference with batched GPU acceleration, smooth Gaussian patch blending,
    and disk-backed memory-mapped accumulators with chunked normalization to eliminate RAM overflows.
    """
    D, H, W = volume.shape
    pad_d = (stride - (D - patch_size) % stride) % stride if D >= patch_size else patch_size - D
    pad_h = (stride - (H - patch_size) % stride) % stride if H >= patch_size else patch_size - H
    pad_w = (stride - (W - patch_size) % stride) % stride if W >= patch_size else patch_size - W

    Dp, Hp, Wp = D + pad_d, H + pad_h, W + pad_w

    temp_dir = temp_dir or os.path.dirname(os.path.abspath(out_npy_path)) or tempfile.gettempdir()
    os.makedirs(temp_dir, exist_ok=True)

    tag = desc.lower().replace(' ', '_')
    timestamp = int(time.time() * 1000)
    temp_accum_file = os.path.join(temp_dir, f"_temp_accum_{tag}_{timestamp}.dat")
    temp_count_file = os.path.join(temp_dir, f"_temp_count_{tag}_{timestamp}.dat")

    # Disk-backed accumulators to guarantee zero RAM overflow
    output_sum = np.lib.format.open_memmap(
        temp_accum_file,
        mode='w+',
        dtype=np.float32,
        shape=(out_channels, Dp, Hp, Wp)
    )
    count_map = np.lib.format.open_memmap(
        temp_count_file,
        mode='w+',
        dtype=np.float32,
        shape=(1, Dp, Hp, Wp)
    )
    output_sum[:] = 0.0
    count_map[:] = 0.0

    # 3D Gaussian window for smooth patch boundary blending
    sigma = patch_size / 4.0
    grid = np.linspace(-patch_size/2 + 0.5, patch_size/2 - 0.5, patch_size)
    gz, gy, gx = np.meshgrid(grid, grid, grid, indexing='ij')
    gaussian_weight = np.exp(-(gz**2 + gy**2 + gx**2) / (2 * sigma**2)).astype(np.float32)
    gaussian_weight /= gaussian_weight.max()

    vol_padded = np.pad(volume, ((0, pad_d), (0, pad_h), (0, pad_w)), mode='reflect')

    z_steps = list(range(0, Dp - patch_size + 1, stride))
    y_steps = list(range(0, Hp - patch_size + 1, stride))
    x_steps = list(range(0, Wp - patch_size + 1, stride))

    coords = [(z, y, x) for z in z_steps for y in y_steps for x in x_steps]
    total_patches = len(coords)

    model.eval()
    pbar = tqdm(
        range(0, total_patches, batch_size),
        desc=f"  {desc} ({total_patches} patches)",
        unit="batch",
        ncols=95
    )

    with torch.no_grad():
        for b_idx in pbar:
            batch_coords = coords[b_idx:b_idx + batch_size]
            patches = [
                vol_padded[z:z+patch_size, y:y+patch_size, x:x+patch_size]
                for (z, y, x) in batch_coords
            ]
            batch_tensor = torch.from_numpy(np.stack(patches, axis=0)).unsqueeze(1).float().to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                preds = model(batch_tensor)  # (B, out_channels, D, H, W)

            preds_np = preds.float().cpu().numpy()

            for i, (z, y, x) in enumerate(batch_coords):
                output_sum[:, z:z+patch_size, y:y+patch_size, x:x+patch_size] += preds_np[i] * gaussian_weight
                count_map[:, z:z+patch_size, y:y+patch_size, x:x+patch_size] += gaussian_weight

            pbar.set_postfix({'Done': f"{min(b_idx + batch_size, total_patches)}/{total_patches} ({min(100.0, (b_idx + batch_size)/total_patches*100):.1f}%)"})

    del vol_padded
    gc.collect()

    output_sum.flush()
    count_map.flush()

    # Normalize chunk-by-chunk directly into destination .npy file
    final_shape = (D, H, W) if out_channels == 1 else (out_channels, D, H, W)
    final_mmap = np.lib.format.open_memmap(out_npy_path, mode='w+', dtype=np.float32, shape=final_shape)

    for z_start in range(0, D, chunk_size):
        z_end = min(z_start + chunk_size, D)
        chunk_out = np.array(output_sum[:, z_start:z_end, :H, :W], copy=True)
        chunk_cnt = np.maximum(count_map[0, z_start:z_end, :H, :W], 1e-6)

        if out_channels == 1:
            final_mmap[z_start:z_end, :, :] = chunk_out[0] / chunk_cnt
        else:
            for c in range(out_channels):
                chunk_out[c] /= chunk_cnt
            final_mmap[:, z_start:z_end, :, :] = chunk_out

    final_mmap.flush()

    del output_sum, count_map
    gc.collect()

    if os.path.exists(temp_accum_file):
        try:
            os.remove(temp_accum_file)
        except Exception:
            pass

    if os.path.exists(temp_count_file):
        try:
            os.remove(temp_count_file)
        except Exception:
            pass

    return final_mmap


def run_inference(
    input_path='process_data/COLLAGENCROP_003_0000.tif',
    intensity_ckpt='checkpoints/best_intensity_unet.pth',
    orientation_ckpt='checkpoints/best_orientation_unet.pth',
    patch_size=64,
    stride=32,
    batch_size=4,
    base_channels=32,
    output_prefix='outputs/dual_collagen',
    mode='both',
    chunk_size=32
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mode_normalized = mode.lower().replace('-', '_')

    if mode_normalized in ('intensity', 'intensity_only', 'int'):
        do_intensity = True
        do_orientation = False
        mode_label = "INTENSITY ONLY"
    elif mode_normalized in ('orientation', 'orientation_only', 'ori'):
        do_intensity = False
        do_orientation = True
        mode_label = "ORIENTATION ONLY"
    elif mode_normalized in ('both', 'all'):
        do_intensity = True
        do_orientation = True
        mode_label = "DUAL SPECIALISTS (INTENSITY + ORIENTATION)"
    else:
        raise ValueError(f"Unknown mode: '{mode}'. Choose from 'both', 'intensity', or 'orientation'.")

    print("=" * 85, flush=True)
    print(f" FAST NEURAL FIELD INFERENCE [{mode_label}] ON {device} ", flush=True)
    print("=" * 85, flush=True)

    out_dir = os.path.dirname(output_prefix) if os.path.dirname(output_prefix) else '.'
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input volume not found: {input_path}")

    # 1. Load Raw Volume
    print(f"Loading input volume: {input_path}...", flush=True)
    raw_vol = tifffile.imread(input_path)
    if raw_vol.dtype == bool:
        vol_float = raw_vol.astype(np.float32)
        raw_binary = raw_vol.astype(np.float32)
    elif raw_vol.max() > 1.0:
        vol_float = raw_vol.astype(np.float32) / 255.0
        raw_binary = (raw_vol > 127).astype(np.float32)
    else:
        vol_float = raw_vol.astype(np.float32)
        raw_binary = (raw_vol > 0.5).astype(np.float32)

    del raw_vol
    gc.collect()

    D, H, W = vol_float.shape
    fg_density = float(np.mean(vol_float > 0))
    print(f"Volume shape: {vol_float.shape} (Foreground density: {fg_density:.4f})", flush=True)

    # Save volume NPY early
    out_npy_vol = f"{output_prefix}_volume.npy"
    np.save(out_npy_vol, raw_binary)
    del raw_binary
    gc.collect()

    out_npy_int = f"{output_prefix}_intensity.npy"
    out_npy_ori = f"{output_prefix}_orientation.npy"

    # 2. Predict Intensity Potential Field with Specialist 1
    if do_intensity:
        print("\n--- Running Sliding Window Inference with IntensityUNet3D ---", flush=True)
        if os.path.exists(intensity_ckpt):
            ckpt_int = torch.load(intensity_ckpt, map_location=device)
            intensity_base_channels = infer_base_channels_from_checkpoint(ckpt_int, default=base_channels)
            intensity_model = IntensityUNet3D(in_channels=1, base_channels=intensity_base_channels).to(device)
            intensity_model.load_state_dict(ckpt_int['model_state_dict'])
            print(f"Loaded Intensity checkpoint: {intensity_ckpt}", flush=True)
        else:
            raise FileNotFoundError(f"Intensity checkpoint not found: {intensity_ckpt}")

        t_int = time.time()
        final_int_mmap = predict_sliding_window(
            intensity_model,
            vol_float,
            out_npy_path=out_npy_int,
            patch_size=patch_size,
            stride=stride,
            device=device,
            out_channels=1,
            batch_size=batch_size,
            desc="Intensity Specialist",
            temp_dir=out_dir,
            chunk_size=chunk_size
        )
        del intensity_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        int_min = float(final_int_mmap.min())
        int_max = float(final_int_mmap.max())
        print(f"Intensity prediction complete in {time.time()-t_int:.2f}s! Range: [{int_min:.3f}, {int_max:.3f}]", flush=True)

        # Streamed BigTIFF export in uint16
        tif_int_path = f"{output_prefix}_raw_probability.tif"
        temp_u16_path = f"{output_prefix}_u16.tmp"
        mmap_u16 = np.lib.format.open_memmap(temp_u16_path, mode='w+', dtype=np.uint16, shape=(D, H, W))
        for z_start in range(0, D, chunk_size):
            z_end = min(z_start + chunk_size, D)
            chunk = np.clip(final_int_mmap[z_start:z_end], 0.0, 1.0)
            mmap_u16[z_start:z_end] = (chunk * 65535.0).astype(np.uint16)
        mmap_u16.flush()
        del final_int_mmap
        gc.collect()

        tifffile.imwrite(tif_int_path, mmap_u16, compression='zlib', metadata={'axes': 'ZYX'}, bigtiff=True)
        del mmap_u16
        gc.collect()
        if os.path.exists(temp_u16_path):
            try:
                os.remove(temp_u16_path)
            except Exception:
                pass
    else:
        print("\n--- Skipping Intensity Specialist (Mode = Orientation Only) ---", flush=True)

    # 3. Predict 3D Orientation Vector Field with Specialist 2
    if do_orientation:
        print("\n--- Running Sliding Window Inference with OrientationUNet3D ---", flush=True)
        if os.path.exists(orientation_ckpt):
            ckpt_ori = torch.load(orientation_ckpt, map_location=device)
            orientation_base_channels = infer_base_channels_from_checkpoint(ckpt_ori, default=base_channels)
            orientation_model = OrientationUNet3D(in_channels=1, base_channels=orientation_base_channels).to(device)
            orientation_model.load_state_dict(ckpt_ori['model_state_dict'])
            print(f"Loaded Orientation checkpoint: {orientation_ckpt}", flush=True)
        else:
            raise FileNotFoundError(f"Orientation checkpoint not found: {orientation_ckpt}")

        t_ori = time.time()
        final_ori_mmap = predict_sliding_window(
            orientation_model,
            vol_float,
            out_npy_path=out_npy_ori,
            patch_size=patch_size,
            stride=stride,
            device=device,
            out_channels=3,
            batch_size=batch_size,
            desc="Orientation Specialist",
            temp_dir=out_dir,
            chunk_size=chunk_size
        )
        del orientation_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        print(f"Orientation prediction complete in {time.time()-t_ori:.2f}s! Shape: {final_ori_mmap.shape}", flush=True)

        # Streamed BigTIFF export in RGB uint8
        tif_ori_path = f"{output_prefix}_orientation_rgb.tif"
        temp_rgb_path = f"{output_prefix}_rgb.tmp"
        mmap_rgb = np.lib.format.open_memmap(temp_rgb_path, mode='w+', dtype=np.uint8, shape=(D, H, W, 3))
        for z_start in range(0, D, chunk_size):
            z_end = min(z_start + chunk_size, D)
            chunk_ori = np.abs(final_ori_mmap[:, z_start:z_end, :, :]) * 255.0
            mmap_rgb[z_start:z_end] = np.transpose(chunk_ori.astype(np.uint8), (1, 2, 3, 0))
        mmap_rgb.flush()
        del final_ori_mmap
        gc.collect()

        tifffile.imwrite(tif_ori_path, mmap_rgb, compression='zlib', photometric='rgb', metadata={'axes': 'ZYXC'}, bigtiff=True)
        del mmap_rgb
        gc.collect()
        if os.path.exists(temp_rgb_path):
            try:
                os.remove(temp_rgb_path)
            except Exception:
                pass
    else:
        print("\n--- Skipping Orientation Specialist (Mode = Intensity Only) ---", flush=True)

    del vol_float
    gc.collect()

    print(f"\n================================================================================")
    print(f" INFERENCE COMPLETE [{mode_label}]: Exports generated successfully!")
    if do_intensity:
        print(f"  - Intensity Potential Field:   {out_npy_int}")
        print(f"  - Raw Probability TIFF:        {output_prefix}_raw_probability.tif")
    if do_orientation:
        print(f"  - 3D Orientation Vector Field: {out_npy_ori}")
        print(f"  - Orientation RGB TIFF:        {output_prefix}_orientation_rgb.tif")
    print(f"  - Binary Volume:               {out_npy_vol}")
    print(f"================================================================================\n")

    return (
        out_npy_int if (do_intensity or os.path.exists(out_npy_int)) else None,
        out_npy_ori if (do_orientation or os.path.exists(out_npy_ori)) else None,
        out_npy_vol
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Sliding-Window Neural Field Inference")
    parser.add_argument('--input', type=str, default='process_data/COLLAGENCROP_003_0000.tif', help="Path to input .tif volume")
    parser.add_argument('--mode', type=str, default='both', choices=['both', 'intensity', 'orientation', 'intensity_only', 'orientation_only'],
                        help="Inference mode: 'both' (default), 'intensity', or 'orientation'")
    parser.add_argument('--intensity-only', action='store_true', help="Shortcut to run only Intensity Specialist")
    parser.add_argument('--orientation-only', action='store_true', help="Shortcut to run only Orientation Specialist")
    parser.add_argument('--skip-intensity', action='store_true', help="Skip Intensity Specialist (runs Orientation only)")
    parser.add_argument('--skip-orientation', action='store_true', help="Skip Orientation Specialist (runs Intensity only)")
    parser.add_argument('--intensity-ckpt', type=str, default='checkpoints/best_intensity_unet.pth', help="Path to Intensity model checkpoint")
    parser.add_argument('--orientation-ckpt', type=str, default='checkpoints/best_orientation_unet.pth', help="Path to Orientation model checkpoint")
    parser.add_argument('--patch-size', type=int, default=64, help="3D sliding window patch cube size")
    parser.add_argument('--stride', type=int, default=32, help="Sliding window stride step")
    parser.add_argument('--batch-size', type=int, default=4, help="GPU batch size")
    parser.add_argument('--base-channels', type=int, default=32, help="U-Net base channel capacity")
    parser.add_argument('--chunk-size', type=int, default=32, help="Slice chunk size for memory-safe streaming")
    parser.add_argument('--output-prefix', type=str, default='outputs/dual_collagen', help="Output path prefix")
    args = parser.parse_args()

    mode = args.mode
    if args.intensity_only or args.skip_orientation:
        mode = 'intensity'
    elif args.orientation_only or args.skip_intensity:
        mode = 'orientation'

    run_inference(
        input_path=args.input,
        intensity_ckpt=args.intensity_ckpt,
        orientation_ckpt=args.orientation_ckpt,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        base_channels=args.base_channels,
        output_prefix=args.output_prefix,
        mode=mode,
        chunk_size=args.chunk_size
    )
