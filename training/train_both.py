import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import argparse

from training.train_intensity import train_intensity_model
from training.train_orientation import train_orientation_model
from core.dataset import ensure_individual_fibers_extracted

def main():
    parser = argparse.ArgumentParser(description="Master Training Script for Dual Decoupled 3D Models")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs per model (default: 50)")
    parser.add_argument('--batch-size', type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument('--grad-accum-steps', type=int, default=1, help="Gradient accumulation steps (default: 1)")
    parser.add_argument('--patch-size', type=int, default=64, help="3D Patch size (default: 64)")
    parser.add_argument('--samples-per-epoch', type=int, default=200, help="Samples per epoch (default: 200)")
    parser.add_argument('--base-channels', type=int, default=32, help="Base U-Net channels (default: 32)")
    parser.add_argument('--lr', type=float, default=5e-4, help="Learning rate (default: 5e-4)")
    parser.add_argument('--pretrained', action='store_true', help="Warm-start from standard checkpoints/best_*.pth")
    parser.add_argument('--pretrained-intensity', type=str, default=None, help="Custom pretrained intensity checkpoint path")
    parser.add_argument('--pretrained-orientation', type=str, default=None, help="Custom pretrained orientation checkpoint path")
    parser.add_argument('--real-data-dir', type=str, default='data/curated', help="Directory containing curated real fibers for real stamping")
    parser.add_argument('--real-stamp-prob', type=float, default=0.25, help="Probability of stamping real fibers into synthetic training patches")
    parser.add_argument('--real-only', action='store_true', help="Train exclusively on real curated patches (no synthetic data)")
    parser.add_argument('--morph-on-the-fly', action='store_true', default=True, help="Augment GAD splines and morph real biological fibers on-the-fly (default: True)")
    parser.add_argument('--no-morph', action='store_false', dest='morph_on_the_fly', help="Disable on-the-fly morphing")
    parser.add_argument('--real-repeat', type=int, default=3, help="How many times to repeat real curated training triplets relative to synthetic data")
    parser.add_argument('--train-on-all-data', action='store_true', help="Use all real curated patches for training and skip held-out real validation")
    parser.add_argument('--mode', type=str, choices=['both', 'intensity', 'orientation'], default='both',
                        help="Which model(s) to train (default: both)")
    args = parser.parse_args()

    os.makedirs('checkpoints', exist_ok=True)
    ensure_individual_fibers_extracted(args.real_data_dir, verbose=True)

    int_pretrained = args.pretrained_intensity or ('checkpoints/best_intensity_unet.pth' if args.pretrained else None)
    ori_pretrained = args.pretrained_orientation or ('checkpoints/best_orientation_unet.pth' if args.pretrained else None)

    if args.mode in ['both', 'intensity']:
        print("\n" + "#" * 80)
        print(" [STAGE 1/2] TRAINING INDEPENDENT INTENSITY SPECIALIST (IntensityUNet3D)")
        print("#" * 80 + "\n")
        train_intensity_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            patch_size=args.patch_size,
            samples_per_epoch=args.samples_per_epoch,
            base_channels=args.base_channels,
            lr=args.lr,
            pretrained_path=int_pretrained,
            real_data_dir=args.real_data_dir,
            real_stamp_prob=args.real_stamp_prob,
            real_repeat=args.real_repeat,
            train_on_all_data=args.train_on_all_data,
            real_only=args.real_only,
            morph_on_the_fly=args.morph_on_the_fly
        )

    if args.mode in ['both', 'orientation']:
        print("\n" + "#" * 80)
        print(" [STAGE 2/2] TRAINING INDEPENDENT ORIENTATION SPECIALIST (OrientationUNet3D)")
        print("#" * 80 + "\n")
        train_orientation_model(
            epochs=args.epochs,
            batch_size=args.batch_size,
            grad_accum_steps=args.grad_accum_steps,
            patch_size=args.patch_size,
            samples_per_epoch=args.samples_per_epoch,
            base_channels=args.base_channels,
            lr=args.lr,
            pretrained_path=ori_pretrained,
            real_data_dir=args.real_data_dir,
            real_stamp_prob=args.real_stamp_prob,
            real_repeat=args.real_repeat,
            train_on_all_data=args.train_on_all_data,
            real_only=args.real_only,
            morph_on_the_fly=args.morph_on_the_fly
        )

    print("\n" + "=" * 80)
    print(" ALL REQUESTED DUAL MODEL TRAINING COMPLETE!")
    print(" Checkpoints saved to:")
    print("   - checkpoints/best_intensity_unet.pth")
    print("   - checkpoints/best_orientation_unet.pth")
    print("=" * 80 + "\n")

if __name__ == '__main__':
    main()
