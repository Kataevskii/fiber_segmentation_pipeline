import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt

from core.models import IntensityUNet3D, infer_base_channels_from_checkpoint
from core.losses import IntensityLoss
from core.dataset import Fiber3DPatchDataset

def train_intensity_model(
    data_dir=['augmented_data', 'real_train_data/curated_patches'],
    real_data_dir=None,
    real_stamp_prob=0.0,
    test_dir='test_data',
    val_dir='val_data',
    patch_size=64,
    batch_size=2,
    grad_accum_steps=2,
    epochs=10,
    samples_per_epoch=250,
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
    print(f"  - Train Data: {data_dir}")
    if real_data_dir and real_stamp_prob > 0:
        print(f"  - Real Fiber Stamping: {real_data_dir} (Prob: {real_stamp_prob:.0%})")
    else:
        print(f"  - Real Fiber Stamping: disabled")
    print(f"  - Warm-Start Pretrained Path: {pretrained_path}")
    print(f"  - Save Path: {save_path}")
    print("=" * 80, flush=True)

    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    os.makedirs(os.path.dirname(figure_path) if os.path.dirname(figure_path) else 'outputs', exist_ok=True)

    train_dataset = Fiber3DPatchDataset(
        data_dir=data_dir,
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
    val_dataset = Fiber3DPatchDataset(
        data_dir=val_dir,
        real_data_dir=None,
        patch_size=patch_size,
        samples_per_epoch=50,
        augment=False,
        fg_prob=0.85
    ) if os.path.exists(val_dir) else None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=(device.type == 'cuda'))
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda'))
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda')) if val_dataset else None

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

    best_val_loss = float('inf')
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

        # ---------------------------------------------------------------------
        # Validation Step (Model 10) - Used for checkpoint selection
        # ---------------------------------------------------------------------
        model.eval()
        val_running_loss, val_running_dice = 0.0, 0.0
        with torch.no_grad():
            for sub_vol, gt_intensity, _ in val_loader:
                sub_vol = sub_vol.to(device)
                gt_intensity = gt_intensity.to(device)
                with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                    pred_intensity = model(sub_vol)
                    _, loss_dict = loss_fn(pred_intensity, gt_intensity)
                val_running_loss += loss_dict['loss_total']
                val_running_dice += loss_dict['dice_score']

        val_loss = val_running_loss / len(val_loader)
        val_dice = val_running_dice / len(val_loader)

        history['train_loss'].append(train_loss)
        history['train_dice'].append(train_dice)
        history['val_loss'].append(val_loss)
        history['val_dice'].append(val_dice)

        print(
            f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} (Dice: {train_dice:.4f}) "
            f"|| Val (M10) Loss: {val_loss:.4f} (Dice: {val_dice:.4f})",
            flush=True
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'base_channels': effective_base_channels,
                'patch_size': patch_size,
                'val_loss': val_loss,
                'val_dice': val_dice
            }, save_path)
            print(f"  --> Saved new best Intensity checkpoint to {save_path} (Val Loss: {val_loss:.4f} | Val Dice: {val_dice:.4f})", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f" INTENSITY TRAINING COMPLETE IN {(time.time()-t0)/60:.2f} MIN! BEST VAL LOSS: {best_val_loss:.4f}", flush=True)
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
    axes[0].plot(range(1, epochs + 1), history['train_loss'], label='Train Loss (M1-8)', color='blue', lw=2)
    axes[0].plot(range(1, epochs + 1), history['val_loss'], label='Val Loss (M10)', color='red', linestyle='--', lw=2)
    axes[0].set_title('Intensity Potential Loss', fontweight='bold')
    axes[0].set_xlabel('Epoch')
    axes[0].grid(True, alpha=0.3)
    axes[0].legend()

    axes[1].plot(range(1, epochs + 1), history['train_dice'], label='Train Dice (M1-8)', color='teal', lw=2)
    axes[1].plot(range(1, epochs + 1), history['val_dice'], label='Val Dice (M10)', color='purple', linestyle='--', lw=2)
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
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--grad-accum-steps', type=int, default=2)
    parser.add_argument('--patch-size', type=int, default=96)
    parser.add_argument('--samples-per-epoch', type=int, default=250)
    parser.add_argument('--base-channels', type=int, default=24)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--pretrained', type=str, default=None, help="Pretrained checkpoint path (default: None - train from scratch)")
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
        save_path=args.save_path
    )

