import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import math # For PSNR calculation
from tqdm import tqdm # For progress bars
import logging # Import the logging module
import random # For reproducibility
from torchvision import transforms

# --- Import MS-SSIM ---
from pytorch_msssim import MS_SSIM

# --- Import your custom modules ---
from dataloader import ImageRestorationDataset # Assuming your DataLoader script is dataloader.py
from model import PromptIR # Assuming your model script is model.py (dim=48)

# --- Setup Logging ---
LOG_FILE = 'training_l1_msssim_aug_en.log' # New log file name for English version
if os.path.exists(LOG_FILE):
    os.remove(LOG_FILE)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration & Hyperparameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT_DIR = os.path.join(CURRENT_SCRIPT_DIR, 'data', 'train')

# --- Training Hyperparameters ---
LOAD_PRETRAINED_MODEL = True 
PRETRAINED_MODEL_PATH = 'best_promptir_l1_msssim.pth' 
MODEL_SAVE_PATH = 'best_promptir_l1_msssim_aug_en.pth' # New save path

LEARNING_RATE = 5e-5 
NUM_EPOCHS = 50      
BATCH_SIZE = 1
VAL_SPLIT_PERCENT = 0.1
SEED = 42

ALPHA_SSIM = 0.1

PROMPTIR_PARAMS = {
    'inp_channels': 3,
    'out_channels': 3,
    'dim': 48,
    'num_blocks': [4, 6, 6, 8],
    'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8],
    'ffn_expansion_factor': 2.66,
    'bias': False,
    'LayerNorm_type': 'WithBias',
    'decoder': True
}

torch.manual_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)
if DEVICE.type == 'cuda':
    torch.cuda.manual_seed_all(SEED)
logger.info(f"Seeds set to {SEED} for reproducibility.")

def calculate_psnr(img1, img2, max_pixel_val=1.0):
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0: return float('inf')
    return 20 * math.log10(max_pixel_val / math.sqrt(mse))

if __name__ == '__main__':
    logger.info(f"Loading dataset from: {DATASET_ROOT_DIR}")
    
    train_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=20), 
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), 
        transforms.ToTensor()
    ])

    val_transforms = transforms.Compose([
        transforms.ToTensor()
    ])
    
    base_dataset_scanner = ImageRestorationDataset(root_dir=DATASET_ROOT_DIR, transform=None) 
    all_image_file_pairs = base_dataset_scanner.image_pairs 

    if not all_image_file_pairs:
        logger.error(f"No image pairs found in {DATASET_ROOT_DIR}. Please check path and dataset structure. Exiting.")
        exit()

    random.Random(SEED).shuffle(all_image_file_pairs) 

    n_total = len(all_image_file_pairs)
    n_val = int(n_total * VAL_SPLIT_PERCENT)
    n_train = n_total - n_val

    if n_train <= 0 or n_val <= 0:
        logger.error(f"Not enough data for train/validation split. Dataset size: {n_total}. Exiting.")
        exit()

    train_file_pairs = all_image_file_pairs[:n_train]
    val_file_pairs = all_image_file_pairs[n_train:]

    train_dataset = ImageRestorationDataset(image_pairs=train_file_pairs, transform=train_transforms)
    val_dataset = ImageRestorationDataset(image_pairs=val_file_pairs, transform=val_transforms)
    
    logger.info(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")
    logger.info("Split file paths for train and validation sets and applied different transforms.")


    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True if BATCH_SIZE > 1 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = PromptIR(**PROMPTIR_PARAMS).to(DEVICE)
    logger.info(f"PromptIR model instantiated, decoder={model.decoder} (dim={PROMPTIR_PARAMS['dim']}).")

    if LOAD_PRETRAINED_MODEL:
        try:
            logger.info(f"Loading pretrained model from: {PRETRAINED_MODEL_PATH}")
            model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
            logger.info("Pretrained model loaded successfully.")
        except FileNotFoundError:
            logger.warning(f"Pretrained model file not found at {PRETRAINED_MODEL_PATH}. Will train from scratch (if this is the first run for this config).")
        except Exception as e:
            logger.error(f"Error loading pretrained model state_dict: {e}. Please ensure PROMPTIR_PARAMS match.")
            logger.warning("Will train from scratch (if this is the first run for this config).")


    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    criterion_l1 = nn.L1Loss().to(DEVICE)
    criterion_ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3, win_size=11).to(DEVICE)
    logger.info(f"Loss function: L1 + {ALPHA_SSIM} * (1 - MS_SSIM)")
    
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, min_lr=1e-7) 
    logger.info(f"Using ReduceLROnPlateau scheduler: factor=0.5, patience=5, min_lr={1e-7}")

    best_val_psnr = 0.0 
    
    logger.info(f"Starting training for {NUM_EPOCHS} epochs, initial learning rate={LEARNING_RATE}...")

    for epoch in range(NUM_EPOCHS):
        logger.info(f"--- Epoch {epoch+1}/{NUM_EPOCHS} --- Current LR: {optimizer.param_groups[0]['lr']:.2e} ---")
        
        model.train()
        running_train_loss_total = 0.0
        running_train_loss_l1 = 0.0
        running_train_loss_ssim_comp = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [Train]", unit="batch")

        for degraded_imgs, clean_imgs in train_pbar:
            degraded_imgs = degraded_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)

            optimizer.zero_grad()
            restored_imgs = model(degraded_imgs)
            restored_imgs_clamped = torch.clamp(restored_imgs, 0.0, 1.0)

            loss_l1 = criterion_l1(restored_imgs_clamped, clean_imgs)
            ms_ssim_val = criterion_ms_ssim(restored_imgs_clamped, clean_imgs)
            loss_ssim_component = 1.0 - ms_ssim_val
            
            total_loss = loss_l1 + ALPHA_SSIM * loss_ssim_component
            
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            running_train_loss_total += total_loss.item()
            running_train_loss_l1 += loss_l1.item()
            running_train_loss_ssim_comp += loss_ssim_component.item()
            train_pbar.set_postfix({'L1': f'{loss_l1.item():.4f}', 'SSIM_L': f'{loss_ssim_component.item():.4f}', 'Total': f'{total_loss.item():.4f}'})

        epoch_train_loss = running_train_loss_total / len(train_loader)
        epoch_train_l1 = running_train_loss_l1 / len(train_loader)
        epoch_train_ssim_comp = running_train_loss_ssim_comp / len(train_loader)
        logger.info(f"Epoch {epoch+1} Training Loss: {epoch_train_loss:.4f} (L1: {epoch_train_l1:.4f}, SSIM_Comp: {epoch_train_ssim_comp:.4f})")

        model.eval()
        running_val_psnr = 0.0
        running_val_loss_total = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [Validate]", unit="batch")

        with torch.no_grad():
            for degraded_imgs, clean_imgs in val_pbar:
                degraded_imgs = degraded_imgs.to(DEVICE)
                clean_imgs = clean_imgs.to(DEVICE)

                restored_imgs = model(degraded_imgs)
                restored_imgs_clamped = torch.clamp(restored_imgs, 0.0, 1.0)
                
                val_loss_l1 = criterion_l1(restored_imgs_clamped, clean_imgs)
                val_ms_ssim_val = criterion_ms_ssim(restored_imgs_clamped, clean_imgs)
                val_loss_ssim_component = 1.0 - val_ms_ssim_val
                val_total_loss = val_loss_l1 + ALPHA_SSIM * val_loss_ssim_component
                running_val_loss_total += val_total_loss.item()

                batch_psnr_sum = 0
                for i in range(restored_imgs_clamped.size(0)):
                    batch_psnr_sum += calculate_psnr(restored_imgs_clamped[i], clean_imgs[i])
                running_val_psnr += (batch_psnr_sum / restored_imgs_clamped.size(0))
                val_pbar.set_postfix({'val_loss': f'{val_total_loss.item():.4f}'})
        
        epoch_val_loss = running_val_loss_total / len(val_loader)
        epoch_val_psnr = running_val_psnr / len(val_loader)
        logger.info(f"Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f}, Validation PSNR: {epoch_val_psnr:.2f} dB")

        if scheduler:
            scheduler.step(epoch_val_psnr)
        
        if epoch_val_psnr > best_val_psnr:
            best_val_psnr = epoch_val_psnr
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Epoch {epoch+1}: New best model saved, PSNR: {best_val_psnr:.2f} dB, Path: {MODEL_SAVE_PATH}")
        
    logger.info("--- Training Finished ---")
    logger.info(f"Best validation PSNR: {best_val_psnr:.2f} dB")
    logger.info(f"Best model saved to: {MODEL_SAVE_PATH}")
    logger.info(f"Training log saved to: {LOG_FILE}")