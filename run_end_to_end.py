"""
run_end_to_end.py
=================
Master 1-Click End-to-End Pipeline:
  1. Sliding-Window Neural Field Inference (Intensity & Orientation U-Nets)
  2. 3D Medial Axis Extraction & Spur Pruning
  3. Orientation-Guided Transverse H-Severing
  4. Fragment Graph Construction with Robust Durable Endpoint Averaging
  5. Post-H-Sever Direction-Durable Bridge Candidate Search
  6. Global Minimum-Cost Topology Optimization
  7. Multi-Label Voronoi Instance Diffusion
  8. Short Fiber Pruning (< 5 vx) & Color Re-propagation

Usage:
  python fiber_resolution_pipeline/run_end_to_end.py
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from inference.run_inference import run_inference, get_default_input_volume
from inference.run_topology_optimization import run_topology_optimization, DEFAULT_PARAMS


def main():
    _default_input = get_default_input_volume()
    parser = argparse.ArgumentParser(description="Master End-to-End Fiber Resolution Pipeline")
    parser.add_argument('--input', type=str, default=_default_input, help=f"Input microscopy volume (.tif) (default: {_default_input})")
    parser.add_argument('--intensity-ckpt', type=str, default='checkpoints/best_intensity_unet.pth')
    parser.add_argument('--orientation-ckpt', type=str, default='checkpoints/best_orientation_unet.pth')
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--stride', type=int, default=32)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--cut-border', type=int, default=32, help="Margin in voxels (e.g. 32) to sever outer boundary loops")
    parser.add_argument('--out', type=str, default='outputs/fiber_resolution_final')
    parser.add_argument('--chunk-size', type=int, default=32, help="Slice chunk size for memory-safe streaming")
    parser.add_argument('--min-fiber-length', type=int, default=5, help="Minimum fiber centerline length in voxels (default: 5)")
    args = parser.parse_args()

    t_start = time.time()
    os.makedirs(args.out, exist_ok=True)
    temp_prefix = os.path.join(args.out, 'neural_fields')

    print("=" * 90)
    print(" EXECUTING END-TO-END FIBER RESOLUTION & TOPOLOGY OPTIMIZATION PIPELINE ")
    print("=" * 90)

    # 1. Neural Field Inference
    int_path, ori_path, vol_path = run_inference(
        input_path=args.input,
        intensity_ckpt=args.intensity_ckpt,
        orientation_ckpt=args.orientation_ckpt,
        patch_size=args.patch_size,
        stride=args.stride,
        batch_size=args.batch_size,
        output_prefix=temp_prefix,
        chunk_size=args.chunk_size
    )

    # 2. Global Topology Optimization
    opt_params = DEFAULT_PARAMS.copy()
    opt_params['cut_border'] = args.cut_border
    opt_params['min_fiber_length'] = args.min_fiber_length
    opt_params['min_chain_length'] = args.min_fiber_length

    metrics = run_topology_optimization(
        intensity_path=int_path,
        orientation_path=ori_path,
        volume_path=vol_path,
        out_dir=args.out,
        params=opt_params
    )

    print("=" * 90)
    print(f" ALL PIPELINE STAGES COMPLETED IN {time.time() - t_start:.2f}s!")
    print(f" Total Continuous Fibers Resolved: {metrics['N_final_fibers']}")
    print(f" Output Directory:                  {args.out}")
    print("=" * 90)


if __name__ == '__main__':
    main()
