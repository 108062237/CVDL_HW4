import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import os
import numpy as np
import math # For PSNR calculation
from tqdm import tqdm # For progress bars

# --- Import your custom modules ---
# Make sure these files are in the same directory or your PYTHONPATH is set correctly
from dataloader import ImageRestorationDataset # Assuming your DataLoader script is my_dataloader.py
from model import PromptIR # Assuming your model script is model.py

# --- Configuration & Hyperparameters ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {DEVICE}")

# Dataset path - MODIFY THIS TO YOUR ACTUAL DATASET PATH
# This should point to the directory containing 'degraded' and 'clean' folders
# e.g., "data/train/" as in the previous example
CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_ROOT_DIR = os.path.join(CURRENT_SCRIPT_DIR, 'data', 'train')


LEARNING_RATE = 1e-4
BATCH_SIZE = 1 # Adjust based on your GPU memory [cite: 59]
NUM_EPOCHS = 50 # Start with a reasonable number, you can increase this
VAL_SPLIT_PERCENT = 0.1 # Percentage of data to use for validation (e.g., 10%)
MODEL_SAVE_PATH = 'best_promptir_model.pth' # Path to save the best model

# PromptIR Model Parameters (Ensure these match your intended configuration)
# Remember you confirmed decoder=True for your model.py
PROMPTIR_PARAMS = {
    'inp_channels': 3,
    'out_channels': 3,
    'dim': 48,
    'num_blocks': [4, 6, 6, 8], # Example, adjust as per original/your modifications
    'num_refinement_blocks': 4,
    'heads': [1, 2, 4, 8],
    'ffn_expansion_factor': 2.66,
    'bias': False,
    'LayerNorm_type': 'WithBias',
    'decoder': True # CRITICAL: Ensure this enables the prompt mechanism
}

# --- Helper Function for PSNR ---
def calculate_psnr(img1, img2, max_pixel_val=1.0):
    """
    Calculates PSNR between two images.
    Assumes images are PyTorch tensors with values in [0, max_pixel_val].
    """
    mse = torch.mean((img1 - img2) ** 2)
    if mse == 0:
        return float('inf')
    psnr_val = 20 * math.log10(max_pixel_val / math.sqrt(mse))
    return psnr_val

# --- Main Training Script ---
if __name__ == '__main__':
    # 1. Dataset and DataLoaders
    print(f"Loading dataset from: {DATASET_ROOT_DIR}")
    # Define transformations (same as in your dataloader test script)
    from torchvision import transforms
    data_transforms = transforms.Compose([
        transforms.ToTensor()
    ])

    full_dataset = ImageRestorationDataset(root_dir=DATASET_ROOT_DIR, transform=data_transforms)
    
    if len(full_dataset) == 0:
        print(f"Error: Dataset at {DATASET_ROOT_DIR} is empty or could not be loaded. Please check the path and structure.")
        exit()

    n_val = int(len(full_dataset) * VAL_SPLIT_PERCENT)
    n_train = len(full_dataset) - n_val
    
    if n_train <= 0 or n_val <=0:
        print(f"Error: Not enough data for train/validation split. Dataset size: {len(full_dataset)}, Val split: {VAL_SPLIT_PERCENT}")
        print("Consider reducing VAL_SPLIT_PERCENT or adding more data.")
        # Fallback: Use full dataset for training if split is problematic, but validation is highly recommended
        train_dataset = full_dataset
        val_dataset = full_dataset # Not ideal, but a fallback for tiny datasets
        print("Warning: Using full dataset for both training and validation due to small dataset size / split issue.")
    else:
        train_dataset, val_dataset = random_split(full_dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42))

    print(f"Training samples: {len(train_dataset)}, Validation samples: {len(val_dataset)}")

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    # 2. Model Instantiation
    model = PromptIR(**PROMPTIR_PARAMS).to(DEVICE)
    print(f"PromptIR model instantiated with decoder={model.decoder}.") # Verify decoder status
    print(f"Model has {sum(p.numel() for p in model.parameters() if p.requires_grad):,} trainable parameters.")


    # 3. Loss Function and Optimizer
    criterion = nn.L1Loss() # L1 loss is common for image restoration
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    # Optional: Learning rate scheduler
    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=NUM_EPOCHS//3, gamma=0.5)

    # 4. Training Loop
    best_val_psnr = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        
        # Training Phase
        model.train()
        running_train_loss = 0.0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1} [TRAIN]")

        for degraded_imgs, clean_imgs in train_pbar:
            degraded_imgs = degraded_imgs.to(DEVICE)
            clean_imgs = clean_imgs.to(DEVICE)

            optimizer.zero_grad()

            # Forward pass: model output is the restored image
            # The PromptIR code you provided does: out_dec_level1 = self.output(out_dec_level1) + inp_img
            # So, 'restored_imgs' is the final prediction.
            restored_imgs = model(degraded_imgs)
            
            loss = criterion(restored_imgs, clean_imgs)
            
            loss.backward()
            optimizer.step()

            running_train_loss += loss.item() * degraded_imgs.size(0)
            train_pbar.set_postfix({'loss': loss.item()})

        epoch_train_loss = running_train_loss / len(train_dataset)
        print(f"Epoch {epoch+1} Training Loss: {epoch_train_loss:.4f}")

        # Validation Phase
        model.eval()
        running_val_psnr = 0.0
        running_val_loss = 0.0
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1} [VALIDATE]")

        with torch.no_grad():
            for degraded_imgs, clean_imgs in val_pbar:
                degraded_imgs = degraded_imgs.to(DEVICE)
                clean_imgs = clean_imgs.to(DEVICE)

                restored_imgs = model(degraded_imgs)
                
                # Clamp outputs to [0,1] before PSNR calculation if they are on that scale
                restored_imgs_clamped = torch.clamp(restored_imgs, 0.0, 1.0)
                
                val_loss = criterion(restored_imgs, clean_imgs) # Use original restored_imgs for loss
                running_val_loss += val_loss.item() * degraded_imgs.size(0)

                # Calculate PSNR for each image in batch and average
                # For simplicity, averaging PSNRs. More robust might be to calculate total MSE then one PSNR.
                batch_psnr = 0
                for i in range(degraded_imgs.size(0)):
                    batch_psnr += calculate_psnr(restored_imgs_clamped[i], clean_imgs[i])
                running_val_psnr += (batch_psnr / degraded_imgs.size(0))
                val_pbar.set_postfix({'val_loss': val_loss.item()})
        
        epoch_val_loss = running_val_loss / len(val_dataset)
        epoch_val_psnr = running_val_psnr / len(val_loader) # Avg PSNR over batches
        print(f"Epoch {epoch+1} Validation Loss: {epoch_val_loss:.4f}, Validation PSNR: {epoch_val_psnr:.2f} dB")

        # Save the best model
        if epoch_val_psnr > best_val_psnr:
            best_val_psnr = epoch_val_psnr
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"Epoch {epoch+1}: New best model saved with PSNR: {best_val_psnr:.2f} dB to {MODEL_SAVE_PATH}")
        
        # Optional: Step the scheduler
        # if 'scheduler' in locals():
        #     scheduler.step()

    print("\n--- Training Finished ---")
    print(f"Best Validation PSNR: {best_val_psnr:.2f} dB")
    print(f"Best model saved to: {MODEL_SAVE_PATH}")

    # To load the best model later:
    # model.load_state_dict(torch.load(MODEL_SAVE_PATH))
    # model.eval()