import os
import sys
import tempfile
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.models import IntensityUNet3D, infer_base_channels_from_checkpoint
from core.losses import IntensityLoss
from core.dataset import Fiber3DPatchDataset, OnTheFlyMorphedDataset, collect_dataset_triplets, split_real_validation_triplets
from inference.run_inference import predict_sliding_window


def _load_array(path):
    if path.endswith('.npy'):
        return np.load(path, mmap_mode='r')

    import tifffile
    arr = tifffile.imread(path)
    if arr.dtype == np.uint16 and 'intensity' in path.lower():
        arr = arr.astype(np.float32) / 65535.0
    return arr


def _evaluate_full_volume_dice(model, loss_fn, volume_paths, intensity_paths, device, patch_size, batch_size):
    if not volume_paths:
        return None

    val_loss_sum = 0.0
    val_dice_sum = 0.0
    temp_dir = tempfile.gettempdir()

    with torch.no_grad():
        for index, (v_path, i_path) in enumerate(zip(volume_paths, intensity_paths)):
            volume = np.array(_load_array(v_path), dtype=np.float32, copy=True)
            gt_intensity = np.array(_load_array(i_path), dtype=np.float32, copy=True)

            out_path = os.path.join(temp_dir, f"_val_intensity_{index}_{os.getpid()}.npy")
            pred_mmap = predict_sliding_window(
                model,
                volume,
                out_npy_path=out_path,
                patch_size=patch_size,
                stride=max(1, patch_size // 2),
                device=device,
                out_channels=1,
                batch_size=batch_size,
                desc=f"Val Crop {index + 1}",
                temp_dir=temp_dir,
                chunk_size=patch_size,
                show_pbar=False
            )

            pred_volume = np.array(pred_mmap, dtype=np.float32, copy=True)
            pred_tensor = torch.from_numpy(pred_volume).unsqueeze(0).unsqueeze(0)
            gt_tensor = torch.from_numpy(gt_intensity).unsqueeze(0).unsqueeze(0)
            _, loss_dict = loss_fn(pred_tensor, gt_tensor)
            val_loss_sum += loss_dict['loss_total']
            val_dice_sum += loss_dict['dice_score']

            del pred_mmap
            del pred_volume
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass

    count = max(1, len(volume_paths))
    return val_loss_sum / count, val_dice_sum / count

def train_intensity_model(
    data_dir=['data/synthetic/precomputed/train'],
    real_train_dir='data/curated/patches',
    real_val_dir='data/curated/patches',
    real_data_dir=None,
    real_stamp_prob=0.25,
    real_repeat=3,
    test_dir='data/synthetic/precomputed/test',
    train_on_all_data=False,
    real_only=False,
    morph_on_the_fly=True,
    patch_size=64,
    batch_size=4,
    grad_accum_steps=1,
    epochs=50,
    samples_per_epoch=200,
    base_channels=32,
    lr=5e-4,
    pretrained_path=None,
    save_path='checkpoints/best_intensity_unet.pth',
    figure_path='outputs/intensity_training_curves.png'
):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("=" * 80, flush=True)
    print(f" TRAINING INDEPENDENT INTENSITY SPECIALIST (IntensityUNet3D) ON {device}", flush=True)
    effective_base_channels = base_channels
    pretrained_ckpt = None
    if pretrained_path and os.path.exists(pretrained_path):
        try:
            pretrained_ckpt = torch.load(pretrained_path, map_location=device, weights_only=False)
            effective_base_channels = infer_base_channels_from_checkpoint(pretrained_ckpt, default=base_channels)
        except Exception:
            effective_base_channels = base_channels

    print(f"  - Model Channels: {effective_base_channels} | Batch Size: {batch_size} | Epochs: {epochs}")
    print(f"  - Patch Size: {patch_size}x{patch_size}x{patch_size} | Samples/Epoch: {samples_per_epoch}")
    if morph_on_the_fly:
        print(f"  - Train Data: ON-THE-FLY DYNAMIC MORPHING (Augment GAD splines -> Morph Real Fibers)")
    elif real_only:
        if train_on_all_data:
            print(f"  - Train Data: ONLY Real Curated Patches from {real_val_dir} (all patches; no synthetic data, no validation split)")
        else:
            print(f"  - Train Data: ONLY Real Curated Patches from {real_train_dir} (held-out real validation from {real_val_dir}; no synthetic data)")
    elif train_on_all_data:
        print(f"  - Train Data: {data_dir} + {real_val_dir} (all real curated patches; no held-out validation)")
    else:
        print(f"  - Train Data: {data_dir} + {real_train_dir} x{real_repeat} (held-out real validation from {real_val_dir})")
    if real_data_dir and real_stamp_prob > 0:
        print(f"  - Real Fiber Stamping: {real_data_dir} (Prob: {real_stamp_prob:.0%})")
    else:
        print(f"  - Real Fiber Stamping: disabled")
    print(f"  - Warm-Start Pretrained Path: {pretrained_path}")
    print(f"  - Save Path: {save_path}")
    print("=" * 80, flush=True)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(figure_path) if os.path.dirname(figure_path) else 'outputs', exist_ok=True)

    if morph_on_the_fly:
        train_dataset = OnTheFlyMorphedDataset(
            raw_dir='data/synthetic/raw',
            real_data_dir=real_train_dir if os.path.exists(real_train_dir) else None,
            patch_size=patch_size,
            samples_per_epoch=samples_per_epoch,
            augment=True,
            real_patch_prob=0.25
        )
        if train_on_all_data:
            val_volume_paths, val_intensity_paths, val_orientation_paths = [], [], []
        else:
            _, real_val_triplets = split_real_validation_triplets(real_val_dir)
            val_volume_paths, val_intensity_paths, val_orientation_paths = real_val_triplets
    else:
        if real_only:
            train_volume_paths, train_intensity_paths, train_orientation_paths = [], [], []
        else:
            train_volume_paths, train_intensity_paths, train_orientation_paths = collect_dataset_triplets(data_dir)

        if train_on_all_data:
            real_triplets = collect_dataset_triplets(real_val_dir)
            train_volume_paths += real_triplets[0]
            train_intensity_paths += real_triplets[1]
            train_orientation_paths += real_triplets[2]
            val_volume_paths, val_intensity_paths, val_orientation_paths = [], [], []
        else:
            real_train_triplets, real_val_triplets = split_real_validation_triplets(real_val_dir)
            real_repeat = max(1, int(real_repeat))
            real_train_triplets = (
                real_train_triplets[0] * real_repeat,
                real_train_triplets[1] * real_repeat,
                real_train_triplets[2] * real_repeat,
            )
            train_volume_paths += real_train_triplets[0]
            train_intensity_paths += real_train_triplets[1]
            train_orientation_paths += real_train_triplets[2]
            val_volume_paths, val_intensity_paths, val_orientation_paths = real_val_triplets

        train_dataset = Fiber3DPatchDataset(
            volume_paths=train_volume_paths,
            intensity_paths=train_intensity_paths,
            orientation_paths=train_orientation_paths,
            real_data_dir=real_data_dir,
            real_stamp_prob=real_stamp_prob,
            patch_size=patch_size,
            samples_per_epoch=samples_per_epoch,
            augment=True,
            fg_prob=0.85,
            jitter_voxels=2
        )
    test_dataset = Fiber3DPatchDataset(
        data_dir=test_dir,
        real_data_dir=None,
        patch_size=patch_size,
        samples_per_epoch=50,
        augment=False,
        fg_prob=0.85
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda'))

    model = IntensityUNet3D(in_channels=1, base_channels=effective_base_channels).to(device)

    if pretrained_path and os.path.exists(pretrained_path):
        try:
            ckpt = pretrained_ckpt if pretrained_ckpt is not None else torch.load(pretrained_path, map_location=device, weights_only=False)
            if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
                model.load_state_dict(ckpt['model_state_dict'])
            elif isinstance(ckpt, dict):
                model.load_state_dict(ckpt)
            print(f"Successfully warm-started model weights from '{pretrained_path}'!", flush=True)
        except Exception as e:
            print(f"Could not load pretrained weights from '{pretrained_path}': {e}. Training from scratch.", flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    loss_fn = IntensityLoss(intensity_weight=1.0, dice_weight=1.0, neg_weight=0.5)
    scaler = torch.amp.GradScaler('cuda', enabled=(device.type == 'cuda'))

    best_val_dice = float('-inf')
    best_test_loss = float('inf')
    history = {'train_loss': [], 'train_dice': [], 'test_loss': [], 'test_dice': [], 'val_loss': [], 'val_dice': []}

    t0 = time.time()
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_dice = 0.0, 0.0

        pbar = tqdm(train_loader, desc=f"Epoch [{epoch:02d}/{epochs:02d}] (Train Intensity)", leave=False)
        for step, (sub_vol, gt_intensity, _) in enumerate(pbar):
            sub_vol = sub_vol.to(device)
            gt_intensity = gt_intensity.to(device)

            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                pred_intensity = model(sub_vol)
                loss, loss_dict = loss_fn(pred_intensity, gt_intensity)
                loss_scaled = loss / grad_accum_steps

            scaler.scale(loss_scaled).backward()

            if (step + 1) % grad_accum_steps == 0 or (step + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            running_loss += loss_dict['loss_total']
            running_dice += loss_dict['dice_score']
            pbar.set_postfix({'Loss': f"{loss_dict['loss_total']:.4f}", 'Dice': f"{loss_dict['dice_score']:.3f}"})

        scheduler.step()
        train_loss = running_loss / len(train_loader)
        train_dice = running_dice / len(train_loader)

        if hasattr(train_dataset, 'refresh_epoch_pool'):
            train_dataset.refresh_epoch_pool()

        if val_volume_paths:
            # -----------------------------------------------------------------
            # Validation Step (Held-Out Real Split, full-crop inference) - Used for checkpoint selection
            # -----------------------------------------------------------------
            model.eval()
            val_loss, val_dice = _evaluate_full_volume_dice(
                model,
                loss_fn,
                val_volume_paths,
                val_intensity_paths,
                device,
                patch_size,
                batch_size,
            )

            history['train_loss'].append(train_loss)
            history['train_dice'].append(train_dice)
            history['val_loss'].append(val_loss)
            history['val_dice'].append(val_dice)

            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} (Dice: {train_dice:.4f}) "
                f"|| Val (Full Real Crops) Loss: {val_loss:.4f} (Dice: {val_dice:.4f})",
                flush=True
            )

            if val_dice > best_val_dice:
                best_val_dice = val_dice
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'base_channels': effective_base_channels,
                    'patch_size': patch_size,
                    'val_loss': val_loss,
                    'val_dice': val_dice,
                    'train_on_all_data': train_on_all_data
                }, save_path)
                print(f"  --> Saved new best Intensity checkpoint to {save_path} (Val Dice: {val_dice:.4f} | Val Loss: {val_loss:.4f})", flush=True)
        else:
            history['train_loss'].append(train_loss)
            history['train_dice'].append(train_dice)
            print(
                f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} (Dice: {train_dice:.4f}) | No validation split",
                flush=True
            )

    if not val_volume_paths:
        torch.save({
            'epoch': epochs,
            'model_state_dict': model.state_dict(),
            'base_channels': effective_base_channels,
            'patch_size': patch_size,
            'train_on_all_data': train_on_all_data
        }, save_path)
        print(f"  --> Saved final Intensity checkpoint to {save_path} (all-data training mode)", flush=True)

    print("\n" + "=" * 80, flush=True)
    if val_volume_paths:
        print(f" INTENSITY TRAINING COMPLETE IN {(time.time()-t0)/60:.2f} MIN! BEST VAL DICE: {best_val_dice:.4f}", flush=True)
    else:
        print(f" INTENSITY TRAINING COMPLETE IN {(time.time()-t0)/60:.2f} MIN!", flush=True)
    print("=" * 80, flush=True)

    # -------------------------------------------------------------------------
    # Final Unbiased Test Evaluation on Held-Out Test Set (Model 9)
    # -------------------------------------------------------------------------
    if test_loader and os.path.exists(save_path):
        print(f"\n[FINAL BENCHMARK] Evaluating Best Checkpoint on Unseen Test Set ({test_dir} - Model 9)...", flush=True)
        ckpt = torch.load(save_path, map_location=device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()

        test_running_loss, test_running_dice = 0.0, 0.0
        with torch.no_grad():
            for sub_vol, gt_intensity, _ in test_loader:
                sub_vol = sub_vol.to(device)
                gt_intensity = gt_intensity.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    pred_intensity = model(sub_vol)
                    _, loss_dict = loss_fn(pred_intensity, gt_intensity)
                test_running_loss += loss_dict['loss_total']
                test_running_dice += loss_dict['dice_score']

        final_test_loss = test_running_loss / len(test_loader)
        final_test_dice = test_running_dice / len(test_loader)
        print("=" * 80, flush=True)
        print(f" FINAL HELD-OUT TEST RESULTS (Model 9):", flush=True)
        print(f"   - Test Total Loss:  {final_test_loss:.4f}", flush=True)
        print(f"   - Centerline Dice:  {final_test_dice:.4f} ({final_test_dice*100:.2f}%)", flush=True)
        print("=" * 80 + "\n", flush=True)

    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(range(1, epochs + 1), history['train_loss'], label='Train Loss', color='blue', lw=2)
    if history['val_loss']:
        axes[0].plot(range(1, epochs + 1), history['val_loss'], label='Val Loss (Real Split)', color='red', linestyle='--', lw=2)
    axes[0].set_title('Intensity Potential Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, epochs + 1), history['train_dice'], label='Train Dice', color='teal', lw=2)
    if history['val_dice']:
        axes[1].plot(range(1, epochs + 1), history['val_dice'], label='Val Dice (Real Split)', color='purple', linestyle='--', lw=2)
    axes[1].set_title('Centerline Dice Score', fontweight='bold')
    axes[1].set_xlabel('Epoch')
    axes[1].grid(True, alpha=0.3)
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(figure_path, dpi=200)
    plt.close()

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="Train Dedicated Intensity UNet3D Model")
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--grad-accum-steps', type=int, default=1)
    parser.add_argument('--patch-size', type=int, default=64)
    parser.add_argument('--samples-per-epoch', type=int, default=200)
    parser.add_argument('--base-channels', type=int, default=32)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--pretrained', type=str, default=None, help="Pretrained checkpoint path (default: None - train from scratch)")
    parser.add_argument('--real-only', action='store_true', help="Train exclusively on real curated patches (no synthetic data)")
    parser.add_argument('--morph-on-the-fly', action='store_true', default=True, help="Augment GAD splines and morph real biological fibers on-the-fly (default: True)")
    parser.add_argument('--no-morph', action='store_false', dest='morph_on_the_fly', help="Disable on-the-fly morphing")
    parser.add_argument('--train-on-all-data', action='store_true', help="Train on all real curated patches with no held-out validation")
    parser.add_argument('--real-stamp-prob', type=float, default=0.25, help="Probability of stamping real fibers")
    parser.add_argument('--save-path', type=str, default='checkpoints/best_intensity_unet.pth')
    args = parser.parse_args()

    train_intensity_model(
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum_steps=args.grad_accum_steps,
        patch_size=args.patch_size,
        samples_per_epoch=args.samples_per_epoch,
        base_channels=args.base_channels,
        lr=args.lr,
        pretrained_path=args.pretrained,
        train_on_all_data=args.train_on_all_data,
        real_only=args.real_only,
        morph_on_the_fly=args.morph_on_the_fly,
        real_stamp_prob=args.real_stamp_prob,
        save_path=args.save_path
    )

