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
LOG_FILE = 'finetuning_l1_msssim_aug_en.log' # Changed log file name for this version
if os.path.exists(LOG_FILE):
    pass 
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILE, mode='a'), # Append mode
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# --- Configuration & Hyperparameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT_DIR = os.path.join(CURRENT_SCRIPT_DIR, 'data', 'train')

# --- Fine-tuning Hyperparameters ---
PRETRAINED_MODEL_PATH = 'best_promptir_l1_msssim_aug_28.88.pth' 
MODEL_SAVE_PATH = 'best_promptir_finetuned_en.pth' 

FINETUNE_LEARNING_RATE = 1e-5 
FINETUNE_NUM_EPOCHS = 20      
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
    logger.info(f"--- Starting fine-tuning session ---")
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
        logger.error(f"No image pairs found in {DATASET_ROOT_DIR}. Exiting.")
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
    logger.info("Split file paths for train and validation sets with different transforms (fine-tuning stage).")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True, drop_last=True if BATCH_SIZE > 1 else False)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True)

    model = PromptIR(**PROMPTIR_PARAMS).to(DEVICE)
    logger.info(f"PromptIR model instantiated, decoder={model.decoder} (dim={PROMPTIR_PARAMS['dim']}).")

    try:
        logger.info(f"Loading model for fine-tuning from: {PRETRAINED_MODEL_PATH}")
        model.load_state_dict(torch.load(PRETRAINED_MODEL_PATH, map_location=DEVICE))
        logger.info("Pretrained model loaded successfully, ready for fine-tuning.")
    except FileNotFoundError:
        logger.error(f"Pretrained model file not found at {PRETRAINED_MODEL_PATH}! Cannot fine-tune. Please check the path.")
        exit()
    except Exception as e:
        logger.error(f"Error loading pretrained model state_dict: {e}. Please ensure PROMPTIR_PARAMS match. Cannot fine-tune.")
        exit()

    logger.info(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")

    criterion_l1 = nn.L1Loss().to(DEVICE)
    criterion_ms_ssim = MS_SSIM(data_range=1.0, size_average=True, channel=3, win_size=11).to(DEVICE)
    logger.info(f"Loss function: L1 + {ALPHA_SSIM} * (1 - MS_SSIM)")
    
    optimizer = optim.AdamW(model.parameters(), lr=FINETUNE_LEARNING_RATE, weight_decay=1e-2) 
    
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=3, min_lr=1e-7) 
    logger.info(f"Using ReduceLROnPlateau scheduler for fine-tuning: factor=0.5, patience=3, min_lr={1e-7}")
    
    logger.info("Evaluating initial validation PSNR of the loaded model before fine-tuning...")
    model.eval()
    initial_val_psnr = 0.0
    initial_val_loss = 0.0
    with torch.no_grad():
        for degraded_imgs, clean_imgs in val_loader: 
            degraded_imgs = degraded_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)
            restored_imgs = model(degraded_imgs)
            restored_imgs_clamped = torch.clamp(restored_imgs, 0.0, 1.0)
            val_loss_l1 = criterion_l1(restored_imgs_clamped, clean_imgs)
            val_ms_ssim_val = criterion_ms_ssim(restored_imgs_clamped, clean_imgs)
            val_loss_ssim_component = 1.0 - val_ms_ssim_val
            initial_val_loss += (val_loss_l1 + ALPHA_SSIM * val_loss_ssim_component).item()
            batch_psnr_sum = 0
            for i in range(restored_imgs_clamped.size(0)):
                batch_psnr_sum += calculate_psnr(restored_imgs_clamped[i], clean_imgs[i])
            initial_val_psnr += (batch_psnr_sum / restored_imgs_clamped.size(0))
    initial_val_psnr /= len(val_loader)
    initial_val_loss /= len(val_loader)
    best_val_psnr = initial_val_psnr 
    logger.info(f"Initial validation loss of loaded model: {initial_val_loss:.4f}, Initial validation PSNR: {best_val_psnr:.2f} dB")
    
    logger.info(f"Starting fine-tuning for {FINETUNE_NUM_EPOCHS} epochs, initial learning rate={FINETUNE_LEARNING_RATE}...")

    for epoch in range(FINETUNE_NUM_EPOCHS):
        current_epoch_num = epoch + 1 
        logger.info(f"--- Fine-tuning Epoch {current_epoch_num}/{FINETUNE_NUM_EPOCHS} --- Current LR: {optimizer.param_groups[0]['lr']:.2e} ---")
        
        model.train()
        running_train_loss_total = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {current_epoch_num} [Fine-tuning Train]", unit="batch")

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
            train_pbar.set_postfix({'TotalLoss': f'{total_loss.item():.4f}'})

        epoch_train_loss = running_train_loss_total / len(train_loader)
        logger.info(f"Epoch {current_epoch_num} Fine-tuning Training Loss: {epoch_train_loss:.4f}")

        model.eval()
        running_val_psnr = 0.0
        running_val_loss_total = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {current_epoch_num} [Fine-tuning Validate]", unit="batch")

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
        logger.info(f"Epoch {current_epoch_num} Fine-tuning Validation Loss: {epoch_val_loss:.4f}, Validation PSNR: {epoch_val_psnr:.2f} dB")

        if scheduler:
            scheduler.step(epoch_val_psnr)
        
        if epoch_val_psnr > best_val_psnr:
            best_val_psnr = epoch_val_psnr
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Epoch {current_epoch_num}: New best fine-tuned model saved, PSNR: {best_val_psnr:.2f} dB, Path: {MODEL_SAVE_PATH}")
        
    logger.info("--- Fine-tuning finished ---")
    logger.info(f"Best fine-tuned validation PSNR: {best_val_psnr:.2f} dB")
    logger.info(f"Best fine-tuned model saved to: {MODEL_SAVE_PATH}")
    logger.info(f"Fine-tuning log saved to: {LOG_FILE}")