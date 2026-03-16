#!/usr/bin/env python3
"""
🖼️ Image Denoising using Denoising Autoencoder (DAE)
Applied on CIFAR-10 Dataset

Pipeline:
  1. Dataset Preparation — Load, normalize, and split CIFAR-10
  2. Noise Injection — Gaussian & Salt-and-Pepper noise
  3. Model Architecture — Convolutional encoder-decoder with bottleneck
  4. Model Training — MSE reconstruction loss with Adam optimizer
  5. Evaluation & Visualization — MSE, PSNR, SSIM metrics + visual comparisons
  6. Experimental Study — Effect of noise levels and bottleneck sizes

Author : Muhammad Idrees (23i-0582)
Email  : i230582@isb.nu.edu.pk
Course : Generative AI — Spring 2026
"""

# ═══════════════════════════════════════════════════════════════════
# Step 0: Install & Import Dependencies
# ═══════════════════════════════════════════════════════════════════

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import time

warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
import torchvision.transforms as transforms

from skimage.metrics import peak_signal_noise_ratio as psnr_metric
from skimage.metrics import structural_similarity as ssim_metric

# Reproducibility
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Device configuration
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Using device: {DEVICE}')
print(f'📦 PyTorch version: {torch.__version__}')
if DEVICE.type == 'cuda':
    print(f'   GPU: {torch.cuda.get_device_name(0)}')
else:
    print()
    print('   CUDA not available — training will use CPU.')
    print('   To use GPU: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121')

# Output directories
OUTPUT_DIR      = os.getcwd()
FIGURES_DIR     = os.path.join(OUTPUT_DIR, 'report', 'figures')
CHECKPOINTS_DIR = OUTPUT_DIR
os.makedirs(FIGURES_DIR, exist_ok=True)
print(f'Outputs: figures -> {FIGURES_DIR}')


# ═══════════════════════════════════════════════════════════════════
# Step 1: Dataset Preparation
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('📂 Step 1: Dataset Preparation')
print('='*70)

# 1.1  Load CIFAR-10 via torchvision
transform = transforms.Compose([transforms.ToTensor()])      # → [0,1]

full_train_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=True, download=True, transform=transform)

test_dataset = torchvision.datasets.CIFAR10(
    root='./data', train=False, download=True, transform=transform)

CLASS_NAMES = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

print(f'CIFAR-10 loaded successfully.')
print(f'   Full training set : {len(full_train_dataset):,} images')
print(f'   Test set          : {len(test_dataset):,} images')

# 1.2  Split training → train (40 000) + validation (10 000)
TRAIN_SIZE = 40_000
VAL_SIZE   = 10_000

train_dataset, val_dataset = random_split(
    full_train_dataset,
    [TRAIN_SIZE, VAL_SIZE],
    generator=torch.Generator().manual_seed(SEED)
)

print('\nDataset Splits:')
print(f'   Train      : {len(train_dataset):,} samples')
print(f'   Validation : {len(val_dataset):,} samples')
print(f'   Test       : {len(test_dataset):,} samples')
print(f'   Image size : 32 × 32 × 3 (H × W × C)  →  tensor shape (3, 32, 32)')

# 1.3  Dataset Statistics
all_pixels = np.stack([np.array(img) for img, _ in full_train_dataset])  # (50000,3,32,32)

print('\nPixel Statistics (values in [0, 1] after ToTensor):')
print(f'   Global  — min={all_pixels.min():.4f}, max={all_pixels.max():.4f}, '
      f'mean={all_pixels.mean():.4f}, std={all_pixels.std():.4f}')
for c, ch in enumerate(['Red', 'Green', 'Blue']):
    print(f'   {ch:5s}   — mean={all_pixels[:, c].mean():.4f},  std={all_pixels[:, c].std():.4f}')

# 1.4  Visualise sample images from each class
samples_per_class = {i: None for i in range(10)}
for img, label in full_train_dataset:
    if samples_per_class[label] is None:
        samples_per_class[label] = img
    if all(v is not None for v in samples_per_class.values()):
        break

fig, axes = plt.subplots(2, 5, figsize=(14, 6))
fig.suptitle('CIFAR-10 — One Sample Image per Class', fontsize=16, fontweight='bold', y=1.02)
for idx, ax in enumerate(axes.flatten()):
    img_np = samples_per_class[idx].permute(1, 2, 0).numpy()
    ax.imshow(img_np)
    ax.set_title(CLASS_NAMES[idx], fontsize=12, fontweight='bold')
    ax.axis('off')
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'cifar10_samples.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Sample images displayed.')

# 1.5  Class distribution bar chart
labels_all = [label for _, label in full_train_dataset]
counts = np.bincount(labels_all)

fig, ax = plt.subplots(figsize=(10, 4))
bars = ax.bar(CLASS_NAMES, counts, color=plt.cm.tab10.colors)
ax.set_title('CIFAR-10 Training Set — Class Distribution', fontsize=14, fontweight='bold')
ax.set_xlabel('Class', fontsize=12)
ax.set_ylabel('Number of Samples', fontsize=12)
ax.set_ylim(0, max(counts) * 1.2)
for bar, count in zip(bars, counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50,
            str(count), ha='center', va='bottom', fontsize=9)
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'class_distribution.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f'Dataset is perfectly balanced: {counts[0]:,} samples per class.')


# ═══════════════════════════════════════════════════════════════════
# Step 2: Noise Injection
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('🔊 Step 2: Noise Injection')
print('='*70)


def add_gaussian_noise(images: torch.Tensor, std: float = 0.1) -> torch.Tensor:
    """
    Add zero-mean Gaussian noise to a batch of images.

    Args:
        images : Tensor of shape (N, C, H, W) or (C, H, W), values in [0,1]
        std    : Standard deviation of the Gaussian noise
    Returns:
        Noisy tensor clipped to [0, 1]
    """
    noise = torch.randn_like(images) * std
    return torch.clamp(images + noise, 0.0, 1.0)


def add_salt_pepper_noise(images: torch.Tensor, amount: float = 0.05) -> torch.Tensor:
    """
    Add salt-and-pepper noise to a batch of images.

    Args:
        images : Tensor of shape (N, C, H, W) or (C, H, W), values in [0,1]
        amount : Fraction of pixels to corrupt (half salt, half pepper)
    Returns:
        Noisy tensor with values in [0, 1]
    """
    noisy = images.clone()
    mask = torch.rand_like(images)
    noisy[mask < (amount / 2)] = 0.0                          # Pepper → black
    noisy[(mask >= amount / 2) & (mask < amount)] = 1.0       # Salt   → white
    return noisy


print('Noise functions defined:')
print('   • add_gaussian_noise(images, std=0.1)')
print('   • add_salt_pepper_noise(images, amount=0.05)')

# 2.2  Visualise noise at different levels
sample_imgs = torch.stack([test_dataset[i][0] for i in [0, 100, 200]])  # (3, 3, 32, 32)

gaussian_levels_viz = [0.05, 0.15, 0.30]
sp_levels_viz       = [0.02, 0.10, 0.25]
n_imgs = sample_imgs.shape[0]

fig = plt.figure(figsize=(18, 12))
fig.suptitle('Effect of Noise on CIFAR-10 Images', fontsize=16, fontweight='bold', y=1.01)

# Row 0: Clean images
for col, img in enumerate(sample_imgs):
    ax = fig.add_subplot(7, n_imgs, col + 1)
    ax.imshow(img.permute(1, 2, 0).numpy())
    ax.set_title('Clean', fontsize=10, fontweight='bold')
    ax.axis('off')
    if col == 0:
        ax.set_ylabel('CLEAN', rotation=0, labelpad=55, fontsize=9, fontweight='bold')

# Rows 1-3: Gaussian noise
for row, std in enumerate(gaussian_levels_viz):
    noisy_batch = add_gaussian_noise(sample_imgs, std=std)
    for col, noisy_img in enumerate(noisy_batch):
        ax = fig.add_subplot(7, n_imgs, (row + 1) * n_imgs + col + 1)
        ax.imshow(noisy_img.permute(1, 2, 0).numpy())
        ax.set_title(f'σ={std}', fontsize=9)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(f'Gaussian\nσ={std}', rotation=0, labelpad=55, fontsize=8)

# Rows 4-6: Salt-and-Pepper noise
for row, amt in enumerate(sp_levels_viz):
    noisy_batch = add_salt_pepper_noise(sample_imgs, amount=amt)
    for col, noisy_img in enumerate(noisy_batch):
        ax = fig.add_subplot(7, n_imgs, (row + 4) * n_imgs + col + 1)
        ax.imshow(noisy_img.permute(1, 2, 0).numpy())
        ax.set_title(f'amt={amt}', fontsize=9)
        ax.axis('off')
        if col == 0:
            ax.set_ylabel(f'Salt-Pepper\namt={amt}', rotation=0, labelpad=55, fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'noise_visualization.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Noise visualisation saved.')


# ═══════════════════════════════════════════════════════════════════
# Step 3: Model Architecture
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('🏗️ Step 3: Denoising Autoencoder Model Architecture')
print('='*70)


class DenoisingAutoencoder(nn.Module):
    """
    Convolutional Denoising Autoencoder for 32×32 RGB images.

    Parameters
    ----------
    bottleneck_channels : int
        Number of channels at the bottleneck layer (controls representation capacity).
        Default=128 → spatial size 4×4 → 2048 latent units.
    """

    def __init__(self, bottleneck_channels: int = 128):
        super(DenoisingAutoencoder, self).__init__()
        self.bottleneck_channels = bottleneck_channels

        # ENCODER: Conv → BN → ReLU → MaxPool  (spatial: 32→16→8→4)
        self.enc_block1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.enc_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        self.enc_block3 = nn.Sequential(
            nn.Conv2d(64, bottleneck_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(bottleneck_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

        # DECODER: ConvTranspose → Conv(refine) → BN → ReLU  (spatial: 4→8→16→32)
        self.dec_block1 = nn.Sequential(
            nn.ConvTranspose2d(bottleneck_channels, 64, kernel_size=2, stride=2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.dec_block2 = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.dec_block3 = nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=2, stride=2),
            nn.Conv2d(3, 3, kernel_size=3, padding=1),
            nn.Sigmoid()      # Output in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.enc_block1(x)
        x = self.enc_block2(x)
        x = self.enc_block3(x)
        return x

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        z = self.dec_block1(z)
        z = self.dec_block2(z)
        z = self.dec_block3(z)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encode(x)
        return self.decode(z)


# Instantiate default model and display
model = DenoisingAutoencoder(bottleneck_channels=128).to(DEVICE)
print(model)


# 3.2  Architecture summary table
def architecture_summary(model: nn.Module):
    """Print a table of all layers with their output shapes and parameter counts."""
    handles = []
    layer_info = []

    def hook(module, inp, out):
        layer_info.append({
            'name': type(module).__name__,
            'output_shape': tuple(out.shape[1:]),
            'params': sum(p.numel() for p in module.parameters())
        })

    for name, module in model.named_modules():
        if len(list(module.children())) == 0:
            h = module.register_forward_hook(hook)
            handles.append(h)

    dummy = torch.zeros(1, 3, 32, 32).to(DEVICE)
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()

    total_params = sum(p.numel() for p in model.parameters())
    trainable    = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print('=' * 65)
    print(f"{'Layer Type':<25} {'Output Shape':<20} {'Parameters':>12}")
    print('=' * 65)
    for info in layer_info:
        print(f"{info['name']:<25} {str(info['output_shape']):<20} {info['params']:>12,}")
    print('=' * 65)
    bn_ch = model.bottleneck_channels
    print(f"{'Total Parameters':<25} {'':20} {total_params:>12,}")
    print(f"{'Trainable Parameters':<25} {'':20} {trainable:>12,}")
    print(f"{'Bottleneck size':<25} {str((bn_ch, 4, 4)):<20} {f'{bn_ch*16} units':>12}")
    print('=' * 65)


architecture_summary(model)


# ═══════════════════════════════════════════════════════════════════
# Step 4: Model Training
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('🏋️ Step 4: Model Training')
print('='*70)


# 4.1  Custom Dataset wrapper that injects noise on-the-fly
class NoisyDataset(Dataset):
    """
    Wraps an existing dataset and returns (noisy_image, clean_image) pairs.

    Parameters
    ----------
    dataset    : base PyTorch dataset returning (img_tensor, label)
    noise_type : 'gaussian' | 'salt_pepper'
    noise_level: std for Gaussian, amount for S&P
    """

    def __init__(self, dataset, noise_type: str = 'gaussian', noise_level: float = 0.1):
        self.dataset     = dataset
        self.noise_type  = noise_type
        self.noise_level = noise_level

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        clean_img, _ = self.dataset[idx]
        if self.noise_type == 'gaussian':
            noisy_img = add_gaussian_noise(clean_img, std=self.noise_level)
        elif self.noise_type == 'salt_pepper':
            noisy_img = add_salt_pepper_noise(clean_img, amount=self.noise_level)
        else:
            raise ValueError(f'Unknown noise_type: {self.noise_type}')
        return noisy_img, clean_img


print('NoisyDataset class defined.')

# 4.2  Training configuration & DataLoaders
BATCH_SIZE  = 128
EPOCHS      = 30
LR          = 1e-3
NOISE_TYPE  = 'gaussian'
NOISE_LEVEL = 0.1

train_noisy = NoisyDataset(train_dataset, noise_type=NOISE_TYPE, noise_level=NOISE_LEVEL)
val_noisy   = NoisyDataset(val_dataset,   noise_type=NOISE_TYPE, noise_level=NOISE_LEVEL)
test_noisy  = NoisyDataset(test_dataset,  noise_type=NOISE_TYPE, noise_level=NOISE_LEVEL)

train_loader = DataLoader(train_noisy, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_noisy,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_noisy,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

print(f'Training configuration:')
print(f'   Noise type    : {NOISE_TYPE}  (level={NOISE_LEVEL})')
print(f'   Batch size    : {BATCH_SIZE}')
print(f'   Epochs        : {EPOCHS}')
print(f'   Learning rate : {LR}')
print(f'   Train batches : {len(train_loader)}')
print(f'   Val batches   : {len(val_loader)}')

# Re-instantiate model
model = DenoisingAutoencoder(bottleneck_channels=128).to(DEVICE)
print(f'Model on: {next(model.parameters()).device}')


# 4.3  Training loop with progress bars
def train_model(
    model, train_loader, val_loader,
    epochs: int = 30,
    lr: float = 1e-3,
    device=DEVICE,
    verbose: bool = True
):
    """
    Train the DAE with MSE loss, Adam optimizer, and ReduceLROnPlateau scheduler.

    Returns
    -------
    history : dict with 'train_loss' and 'val_loss' lists
    """
    criterion  = nn.MSELoss()
    optimizer  = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
                     optimizer, mode='min', factor=0.5, patience=5)

    model.to(device)
    history       = {'train_loss': [], 'val_loss': []}
    best_val_loss = float('inf')
    best_epoch    = -1
    BAR_W         = 35
    n_train       = len(train_loader)
    n_val         = len(val_loader)

    print(f'[Training] Device: {device} | Epochs: {epochs} | '
          f'Train batches: {n_train} | Val batches: {n_val}')
    print('-' * 72)

    total_start = time.time()

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        # ── TRAIN ──
        model.train()
        running_loss = 0.0
        batch_losses = []

        for batch_idx, (noisy_imgs, clean_imgs) in enumerate(train_loader, 1):
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)

            optimizer.zero_grad()
            loss = criterion(model(noisy_imgs), clean_imgs)
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())
            running_loss += loss.item() * noisy_imgs.size(0)

            if verbose:
                filled  = int(BAR_W * batch_idx / n_train)
                bar     = '█' * filled + '░' * (BAR_W - filled)
                elapsed = time.time() - epoch_start
                eta     = (elapsed / batch_idx) * (n_train - batch_idx)
                print(
                    f'\r  Epoch {epoch:3d}/{epochs} [TRAIN] |{bar}| {batch_idx:4d}/{n_train}'
                    f'  loss={loss.item():.5f}  avg={sum(batch_losses)/len(batch_losses):.5f}'
                    f'  {elapsed:5.1f}s ela  {eta:5.1f}s eta',
                    end='', flush=True
                )

        train_loss = running_loss / len(train_loader.dataset)
        print()

        # ── VAL ──
        model.eval()
        val_loss_sum = 0.0

        with torch.no_grad():
            for batch_idx, (noisy_imgs, clean_imgs) in enumerate(val_loader, 1):
                noisy_imgs = noisy_imgs.to(device)
                clean_imgs = clean_imgs.to(device)
                val_loss_sum += criterion(model(noisy_imgs), clean_imgs).item() * noisy_imgs.size(0)

                if verbose:
                    filled  = int(BAR_W * batch_idx / n_val)
                    bar     = '█' * filled + '░' * (BAR_W - filled)
                    elapsed = time.time() - epoch_start
                    print(
                        f'\r  Epoch {epoch:3d}/{epochs} [VAL  ] |{bar}| {batch_idx:4d}/{n_val}'
                        f'  {elapsed:5.1f}s ela',
                        end='', flush=True
                    )

        val_loss = val_loss_sum / len(val_loader.dataset)
        print()

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)

        # Checkpoint
        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            best_epoch    = epoch
            torch.save(model.state_dict(),
                       os.path.join(CHECKPOINTS_DIR, 'best_model.pth'))

        # Epoch summary
        lr_now        = optimizer.param_groups[0]['lr']
        epoch_time    = time.time() - epoch_start
        total_elapsed = time.time() - total_start
        total_eta     = (total_elapsed / epoch) * (epochs - epoch)
        best_tag      = ' [best]' if improved else ''

        print(
            f'  Epoch {epoch:3d}/{epochs}'
            f'  train={train_loss:.6f}  val={val_loss:.6f}{best_tag:<7}'
            f'  lr={lr_now:.2e}'
            f'  {epoch_time:5.1f}s/ep  ela={total_elapsed/60:5.1f}m  eta={total_eta/60:4.1f}m'
        )
        print('-' * 72)

    # Load best weights
    model.load_state_dict(
        torch.load(os.path.join(CHECKPOINTS_DIR, 'best_model.pth'),
                   map_location=device, weights_only=True))
    print(f'\nTraining complete. Best Val Loss: {best_val_loss:.6f} (Epoch {best_epoch})')
    return history


print('Starting training...\n')
history = train_model(model, train_loader, val_loader, epochs=EPOCHS, lr=LR)


# 4.4  Plot training & validation loss curves
fig, ax = plt.subplots(figsize=(10, 5))
epochs_range = range(1, len(history['train_loss']) + 1)
ax.plot(epochs_range, history['train_loss'], 'b-o', markersize=4, label='Training Loss')
ax.plot(epochs_range, history['val_loss'],   'r-s', markersize=4, label='Validation Loss')

best_epoch = np.argmin(history['val_loss']) + 1
ax.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7,
           label=f'Best Val (Epoch {best_epoch})')

ax.set_title('Training & Validation Reconstruction Loss (MSE)', fontsize=14, fontweight='bold')
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('MSE Loss', fontsize=12)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'training_curves.png'), dpi=150, bbox_inches='tight')
plt.show()

print(f'Final Training Loss   : {history["train_loss"][-1]:.6f}')
print(f'Final Validation Loss : {history["val_loss"][-1]:.6f}')
print(f'Best Validation Loss  : {min(history["val_loss"]):.6f}  (Epoch {best_epoch})')


# ═══════════════════════════════════════════════════════════════════
# Step 5: Evaluation and Visualization
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('📊 Step 5: Evaluation and Visualization')
print('='*70)


# 5.1  Compute evaluation metrics on the test set
def evaluate_model(model, loader, device=DEVICE):
    """
    Compute MSE, PSNR, and SSIM on the given dataloader.
    Also returns arrays of noisy, clean, and reconstructed images for visualisation.
    """
    model.eval()
    criterion = nn.MSELoss(reduction='sum')

    total_mse  = 0.0
    total_psnr = 0.0
    total_ssim = 0.0
    n_samples  = 0

    all_noisy = []
    all_clean = []
    all_recon = []

    with torch.no_grad():
        for noisy_imgs, clean_imgs in loader:
            noisy_imgs = noisy_imgs.to(device)
            clean_imgs = clean_imgs.to(device)
            recon_imgs = model(noisy_imgs)

            total_mse += criterion(recon_imgs, clean_imgs).item()
            n_samples += noisy_imgs.size(0)

            clean_np = clean_imgs.cpu().numpy().transpose(0, 2, 3, 1)
            recon_np = recon_imgs.cpu().numpy().transpose(0, 2, 3, 1)

            for c, r in zip(clean_np, recon_np):
                total_psnr += psnr_metric(c, r, data_range=1.0)
                total_ssim += ssim_metric(c, r, data_range=1.0, channel_axis=2)

            if len(all_noisy) == 0:
                all_noisy = noisy_imgs.cpu()
                all_clean = clean_imgs.cpu()
                all_recon = recon_imgs.cpu()

    avg_mse  = total_mse / (n_samples * 3 * 32 * 32)
    avg_psnr = total_psnr / n_samples
    avg_ssim = total_ssim / n_samples

    return avg_mse, avg_psnr, avg_ssim, all_noisy, all_clean, all_recon


mse, psnr_val, ssim_val, noisy_batch, clean_batch, recon_batch = evaluate_model(model, test_loader)

print('Test Set Reconstruction Metrics:')
print(f'   MSE  : {mse:.6f}')
print(f'   PSNR : {psnr_val:.4f} dB')
print(f'   SSIM : {ssim_val:.4f}')


# 5.2  Visualisation: Original | Noisy | Reconstructed
N_VIZ = 8
fig, axes = plt.subplots(3, N_VIZ, figsize=(18, 7))
fig.suptitle(
    f'Denoising Results — Gaussian Noise σ={NOISE_LEVEL}\n'
    f'MSE={mse:.5f}  PSNR={psnr_val:.2f}dB  SSIM={ssim_val:.4f}',
    fontsize=14, fontweight='bold'
)

row_labels = ['Clean (Original)', f'Noisy (σ={NOISE_LEVEL})', 'Reconstructed']
rows_data  = [clean_batch, noisy_batch, recon_batch]

for row, (row_data, row_label) in enumerate(zip(rows_data, row_labels)):
    for col in range(N_VIZ):
        img = row_data[col].permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)
        axes[row, col].imshow(img)
        axes[row, col].axis('off')
    axes[row, 0].set_ylabel(row_label, fontsize=10, fontweight='bold',
                             rotation=0, labelpad=120, va='center')

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'denoising_results.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Visualisation saved.')


# 5.3  Side-by-side per-image metric comparison
n_show = 5
fig, axes = plt.subplots(n_show, 3, figsize=(10, 3 * n_show))
fig.suptitle('Per-Image Metrics: Clean vs Reconstructed', fontsize=14, fontweight='bold')

col_titles = ['Clean Image', f'Noisy (σ={NOISE_LEVEL})', 'Reconstructed']
for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title, fontsize=11, fontweight='bold')

for i in range(n_show):
    imgs = {
        'clean': clean_batch[i].permute(1, 2, 0).numpy(),
        'noisy': noisy_batch[i].permute(1, 2, 0).numpy(),
        'recon': recon_batch[i].permute(1, 2, 0).numpy(),
    }

    img_psnr = psnr_metric(imgs['clean'], imgs['recon'], data_range=1.0)
    img_ssim = ssim_metric(imgs['clean'], imgs['recon'], data_range=1.0, channel_axis=2)
    img_mse  = np.mean((imgs['clean'] - imgs['recon'])**2)

    for col_idx, (key, img) in enumerate(imgs.items()):
        ax = axes[i, col_idx]
        ax.imshow(np.clip(img, 0, 1))
        ax.axis('off')
        if col_idx == 2:
            ax.set_xlabel(f'MSE={img_mse:.4f}  PSNR={img_psnr:.1f}dB  SSIM={img_ssim:.3f}',
                          fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'per_image_metrics.png'), dpi=150, bbox_inches='tight')
plt.show()


# ═══════════════════════════════════════════════════════════════════
# Step 6: Experimental Study
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('🔬 Step 6: Experimental Study')
print('='*70)

EXP_EPOCHS = 15   # Fewer epochs for the experimental sweep


def run_experiment(
    noise_type: str,
    noise_level: float,
    bottleneck_channels: int,
    epochs: int = EXP_EPOCHS,
    batch_size: int = 256
):
    """Train a DAE with given config and return (mse, psnr, ssim)."""
    tr_set = NoisyDataset(train_dataset, noise_type=noise_type, noise_level=noise_level)
    va_set = NoisyDataset(val_dataset,   noise_type=noise_type, noise_level=noise_level)
    te_set = NoisyDataset(test_dataset,  noise_type=noise_type, noise_level=noise_level)

    tr_ld = DataLoader(tr_set, batch_size=batch_size, shuffle=True,  num_workers=0, pin_memory=True)
    va_ld = DataLoader(va_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)
    te_ld = DataLoader(te_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

    mdl = DenoisingAutoencoder(bottleneck_channels=bottleneck_channels).to(DEVICE)
    train_model(mdl, tr_ld, va_ld, epochs=epochs, lr=1e-3, verbose=False)

    mse_val, psnr_v, ssim_v, _, _, _ = evaluate_model(mdl, te_ld)
    return mse_val, psnr_v, ssim_v


print(f'Experiment helper ready (EXP_EPOCHS={EXP_EPOCHS}).')

# ── 6.1  Experiment A: Effect of Noise Level ──
gaussian_levels = [0.05, 0.10, 0.20, 0.30, 0.40]
sp_levels       = [0.02, 0.05, 0.10, 0.20, 0.30]

noise_results = []

print('\n🔬 Experiment A: Noise Level Study')
print('   Fixed: bottleneck_channels=128')
print('-' * 70)

for std in gaussian_levels:
    m, p, s = run_experiment('gaussian', std, bottleneck_channels=128)
    noise_results.append({'Noise Type': 'Gaussian', 'Noise Level': std,
                          'MSE': m, 'PSNR (dB)': p, 'SSIM': s})
    print(f'   Gaussian σ={std:<5}  MSE={m:.6f}  PSNR={p:.2f}dB  SSIM={s:.4f}')

for amt in sp_levels:
    m, p, s = run_experiment('salt_pepper', amt, bottleneck_channels=128)
    noise_results.append({'Noise Type': 'Salt-Pepper', 'Noise Level': amt,
                          'MSE': m, 'PSNR (dB)': p, 'SSIM': s})
    print(f'   S&P   amt={amt:<5}  MSE={m:.6f}  PSNR={p:.2f}dB  SSIM={s:.4f}')

df_noise = pd.DataFrame(noise_results)
print('\nSummary Table:')
print(df_noise.to_string(index=False))


# 6.2  Plot Noise Level results
df_gauss = df_noise[df_noise['Noise Type'] == 'Gaussian']
df_sp    = df_noise[df_noise['Noise Type'] == 'Salt-Pepper']

metrics = [('MSE', 'MSE (lower=better)'),
           ('PSNR (dB)', 'PSNR dB (higher=better)'),
           ('SSIM', 'SSIM (higher=better)')]

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Experiment A: Reconstruction Metrics vs Noise Level',
             fontsize=14, fontweight='bold')

for ax, (metric, ylabel) in zip(axes, metrics):
    ax.plot(df_gauss['Noise Level'], df_gauss[metric], 'b-o',
            label='Gaussian', linewidth=2)
    ax.plot(df_sp['Noise Level'],    df_sp[metric],    'r-s',
            label='Salt-Pepper', linewidth=2)
    ax.set_title(metric, fontsize=12, fontweight='bold')
    ax.set_xlabel('Noise Level', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'experiment_noise_levels.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print('Noise level experiment plots saved.')


# ── 6.3  Experiment B: Effect of Bottleneck Size ──
bottleneck_sizes    = [16, 32, 64, 128, 256]
bottleneck_results  = []

print('\n🔬 Experiment B: Bottleneck Size Study')
print('   Fixed: Gaussian noise σ=0.1')
print('-' * 70)

for bn in bottleneck_sizes:
    latent_units = bn * 4 * 4
    m, p, s = run_experiment('gaussian', 0.1, bottleneck_channels=bn)
    n_params = sum(par.numel() for par in DenoisingAutoencoder(bn).parameters())
    bottleneck_results.append({
        'Bottleneck Channels': bn,
        'Latent Units': latent_units,
        'Model Params': n_params,
        'MSE': m,
        'PSNR (dB)': p,
        'SSIM': s
    })
    print(f'   BN={bn:3d}  Latent={latent_units:5d}  Params={n_params:,}  '
          f'MSE={m:.6f}  PSNR={p:.2f}dB  SSIM={s:.4f}')

df_bn = pd.DataFrame(bottleneck_results)
print('\nSummary Table:')
print(df_bn.to_string(index=False))


# 6.4  Plot Bottleneck Size results
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Experiment B: Reconstruction Metrics vs Bottleneck Size',
             fontsize=14, fontweight='bold')

colors  = ['royalblue', 'darkorange', 'green']
y_cols  = ['MSE', 'PSNR (dB)', 'SSIM']
y_labels = ['MSE (lower=better)', 'PSNR dB (higher=better)', 'SSIM (higher=better)']

for ax, col, ylabel, color in zip(axes, y_cols, y_labels, colors):
    ax.plot(df_bn['Bottleneck Channels'], df_bn[col], 'o-',
            color=color, linewidth=2, markersize=8)
    for _, row in df_bn.iterrows():
        ax.annotate(f'{row[col]:.3f}',
                    (row['Bottleneck Channels'], row[col]),
                    textcoords='offset points', xytext=(0, 10),
                    ha='center', fontsize=8)
    ax.set_title(col, fontsize=12, fontweight='bold')
    ax.set_xlabel('Bottleneck Channels', fontsize=11)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_xticks(df_bn['Bottleneck Channels'])
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'experiment_bottleneck.png'),
            dpi=150, bbox_inches='tight')
plt.show()
print('Bottleneck experiment plots saved.')


# ── 6.5  Heatmap: PSNR across noise levels × bottleneck sizes ──
gaussian_levels_hm = [0.05, 0.10, 0.20, 0.30]
bn_sizes_hm        = [32, 64, 128]

hm_results = np.zeros((len(gaussian_levels_hm), len(bn_sizes_hm)))

print('\n🔬 Grid Search: PSNR across Noise Levels × Bottleneck Sizes')
print('   (This may take several minutes...)\n')

for i, std in enumerate(gaussian_levels_hm):
    for j, bn in enumerate(bn_sizes_hm):
        _, psnr_v, _ = run_experiment('gaussian', std, bottleneck_channels=bn)
        hm_results[i, j] = psnr_v
        print(f'   σ={std}  BN={bn}  → PSNR={psnr_v:.2f} dB')

fig, ax = plt.subplots(figsize=(8, 5))
sns.heatmap(
    hm_results,
    annot=True, fmt='.2f', cmap='YlOrRd_r',
    xticklabels=[f'BN={b}' for b in bn_sizes_hm],
    yticklabels=[f'σ={s}' for s in gaussian_levels_hm],
    ax=ax, linewidths=0.5,
    cbar_kws={'label': 'PSNR (dB)'}
)
ax.set_title('PSNR (dB): Gaussian Noise Level × Bottleneck Size',
             fontsize=13, fontweight='bold')
ax.set_xlabel('Bottleneck Size', fontsize=11)
ax.set_ylabel('Gaussian Noise σ', fontsize=11)
plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'heatmap_psnr.png'), dpi=150, bbox_inches='tight')
plt.show()
print('PSNR heatmap saved.')


# 6.6  Combined summary tables
print('\n' + '=' * 75)
print('TABLE 1 — Experiment A: Effect of Noise Level (BN=128 channels)')
print('=' * 75)
print(df_noise.to_string(index=False, float_format='{:.4f}'.format))

print()
print('=' * 75)
print('TABLE 2 — Experiment B: Effect of Bottleneck Size (Gaussian σ=0.1)')
print('=' * 75)
print(df_bn.to_string(index=False, float_format='{:.4f}'.format))


# ═══════════════════════════════════════════════════════════════════
# Step 7: Discussion
# ═══════════════════════════════════════════════════════════════════

print('\n' + '='*70)
print('💬 Step 7: Discussion')
print('='*70)

discussion = """
7.1 Key Observations
─────────────────────────────
Noise Level Effects (Experiment A):
  • As Gaussian σ increases, MSE increases and PSNR/SSIM decrease monotonically.
  • Salt-and-Pepper noise is generally easier to denoise at moderate levels
    because S&P creates sparse, localized corruptions while Gaussian affects every pixel.
  • At very high noise (σ > 0.30 or S&P > 0.25), the model struggles —
    signal-to-noise ratio is too low to recover meaningful texture.

Bottleneck Size Effects (Experiment B):
  • Small bottlenecks (16–32 channels) act as aggressive information bottlenecks —
    they discard noise but also lose fine-grained texture detail.
  • Large bottlenecks (128–256 channels) preserve more structure but may retain noise.
  • Sweet spot: 64–128 channels for CIFAR-10 at moderate noise.

7.2 Model Strengths
  • Fully convolutional — preserves spatial structure, translation-equivariant.
  • BatchNorm — accelerates convergence, improves stability.
  • Lightweight (~182K params) — trains quickly, practical for deployment.
  • Works on both Gaussian and S&P noise without architectural changes.

7.3 Limitations
  • Low-resolution CIFAR-10 (32×32) limits pattern complexity.
  • The bottleneck is spatial (not FC), tying latent dim to spatial size.
  • Training with one noise type means the model is noise-specific.
  • MSE loss produces blurry reconstructions; doesn't capture perceptual quality.

7.4 Possible Improvements
  ┌───────────────────────────┬─────────────────────────────────────────┐
  │ Improvement               │ Expected Benefit                        │
  ├───────────────────────────┼─────────────────────────────────────────┤
  │ Perceptual loss (VGG)     │ Sharper, more visually appealing output │
  │ Skip connections (U-Net)  │ Better preservation of fine details     │
  │ Mixed noise training      │ Better generalization across noise types│
  │ Blind denoising           │ Single model handles multiple levels    │
  │ Attention mechanisms      │ Focus on spatially important regions    │
  │ Generative (VAE/Diffusion)│ Higher-quality at high noise levels     │
  │ Data augmentation         │ Improved generalization                 │
  └───────────────────────────┴─────────────────────────────────────────┘
"""
print(discussion)


# 7.5  Final visual comparison: both noise types, best model
model.eval()
clean_imgs_viz = torch.stack([test_dataset[i][0] for i in range(8)])

gauss_noisy = add_gaussian_noise(clean_imgs_viz, std=0.1)
with torch.no_grad():
    gauss_recon = model(gauss_noisy.to(DEVICE)).cpu()

sp_noisy = add_salt_pepper_noise(clean_imgs_viz, amount=0.05)
with torch.no_grad():
    sp_recon = model(sp_noisy.to(DEVICE)).cpu()

N = 4
fig, axes = plt.subplots(6, N, figsize=(14, 20))
fig.suptitle('Final Comparison: Gaussian & Salt-Pepper Denoising',
             fontsize=15, fontweight='bold')

rows = [
    (clean_imgs_viz, 'Clean Image', 'black'),
    (gauss_noisy,    'Gaussian Noisy (σ=0.1)', 'blue'),
    (gauss_recon,    'Gaussian Reconstructed', 'green'),
    (clean_imgs_viz, 'Clean Image', 'black'),
    (sp_noisy,       'S&P Noisy (amt=0.05)', 'red'),
    (sp_recon,       'S&P Reconstructed', 'green'),
]

for row_idx, (data, label, color) in enumerate(rows):
    for col_idx in range(N):
        img = data[col_idx].permute(1, 2, 0).numpy()
        axes[row_idx, col_idx].imshow(np.clip(img, 0, 1))
        axes[row_idx, col_idx].axis('off')
        for spine in axes[row_idx, col_idx].spines.values():
            spine.set_edgecolor(color)
            spine.set_linewidth(2)
    axes[row_idx, 0].set_ylabel(label, rotation=0, labelpad=130,
                                 fontsize=9, fontweight='bold', color=color)

plt.tight_layout()
plt.savefig(os.path.join(FIGURES_DIR, 'final_comparison.png'), dpi=150, bbox_inches='tight')
plt.show()
print('Final comparison saved.')


# 7.6  Final metrics summary
print('\n' + '=' * 65)
print('FINAL MODEL PERFORMANCE SUMMARY')
print('=' * 65)
print(f'  Architecture  : Convolutional Denoising Autoencoder')
print(f'  Bottleneck    : 128 channels × 4×4 spatial = 2,048 latent units')
print(f'  Train noise   : Gaussian (σ={NOISE_LEVEL})')
print(f'  Epochs        : {EPOCHS}')
print(f'  Optimizer     : Adam (lr={LR})')
print(f'  Loss function : MSE')
print()
print(f'  ── Test Set Metrics (Gaussian σ={NOISE_LEVEL}) ──')
print(f'  MSE  : {mse:.6f}')
print(f'  PSNR : {psnr_val:.4f} dB')
print(f'  SSIM : {ssim_val:.4f}')
print('=' * 65)
print()
print('Interpretation:')
print(f'  • PSNR > 25 dB is generally considered good denoising quality.')
print(f'  • SSIM > 0.80 indicates high structural similarity to the clean image.')
print(f'  • Our model achieves PSNR ≈ {psnr_val:.1f} dB and SSIM ≈ {ssim_val:.3f}.')

print('\n✅ All steps completed successfully!')
