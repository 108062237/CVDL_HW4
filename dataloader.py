import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

import numpy as np
import random
# In your dataloader.py

class ImageRestorationDataset(Dataset):
    def __init__(self, root_dir=None, image_pairs=None, transform=None):
        self.transform = transform
        self.image_pairs = []

        if image_pairs: # If a list of pairs is provided, use it
            self.image_pairs = image_pairs
        elif root_dir: # Otherwise, load from root_dir (your existing logic)
            self.degraded_dir = os.path.join(root_dir, 'degraded')
            self.clean_dir = os.path.join(root_dir, 'clean')
            # 例如:
            if not os.path.isdir(self.degraded_dir) or not os.path.isdir(self.clean_dir):
                # Handle error: directory not found
                raise FileNotFoundError(f"Degraded or Clean directory not found in {root_dir}")
            degraded_files = sorted([f for f in os.listdir(self.degraded_dir) if f.endswith(('.png', '.jpg', '.jpeg'))])
            for degraded_file_name in degraded_files:
                clean_file_name = ""
                if degraded_file_name.startswith('rain-'):
                    clean_file_name = degraded_file_name.replace('rain-', 'rain_clean-')
                elif degraded_file_name.startswith('snow-'):
                    clean_file_name = degraded_file_name.replace('snow-', 'snow_clean-')
                if clean_file_name:
                    degraded_img_path = os.path.join(self.degraded_dir, degraded_file_name)
                    clean_img_path = os.path.join(self.clean_dir, clean_file_name)
                    if os.path.exists(degraded_img_path) and os.path.exists(clean_img_path):
                        self.image_pairs.append((degraded_img_path, clean_img_path))
        else:
            raise ValueError("Either root_dir or image_pairs must be provided to ImageRestorationDataset.")

    def __getitem__(self, idx):
        degraded_img_path, clean_img_path = self.image_pairs[idx]
        try:
            degraded_image_pil = Image.open(degraded_img_path).convert('RGB')
            clean_image_pil = Image.open(clean_img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image pair: {degraded_img_path}, {clean_img_path}. Error: {e}")
            # Return a dummy pair or raise error, depending on desired handling
            # For now, let's assume successful loading
            # You might want to return specific error handling here.
            # For robustness, one might skip this sample or return placeholder tensors.
            # This is a simplified error path.
            return torch.zeros(3, 64, 64), torch.zeros(3, 64, 64)


        # --- Crucial for paired geometric augmentations ---
        # If self.transform includes geometric transforms (like RandomRotation, Flip)
        # they need to be applied consistently to both images.
        # Simplest way if transform applies to one image:
        # 1. Get current random state
        # 2. Apply transform to image1
        # 3. Restore random state
        # 4. Apply transform to image2 (it will get the same geometric transform)
        # However, torchvision transforms like RandomHorizontalFlip handle this internally
        # if you pass them a PIL image. For sequences of transforms, it's more complex.
        # A common robust approach is to make self.transform expect a tuple/list of images
        # or to encapsulate this logic within the transform itself.

        # For most torchvision transforms that take PIL:
        if self.transform:
            # Create a seed for consistent geometric augmentations if any
            # This is a more robust way for paired transforms
            seed = np.random.randint(2147483647) # Make a seed

            random.seed(seed) # Apply to python's random module
            torch.manual_seed(seed) # Apply to torch's random module
            degraded_image = self.transform(degraded_image_pil)

            random.seed(seed) # Apply to python's random module
            torch.manual_seed(seed) # Apply to torch's random module
            # For clean image, usually only ToTensor is needed if it's for target
            # But if geometric transforms are in self.transform, they must match.
            # If self.transform is train_transform, clean_image also gets augmented.
            # This is okay if the loss can handle it (L1, SSIM can).
            clean_image = self.transform(clean_image_pil)
        else: # Fallback if no transform (should at least have ToTensor)
            degraded_image = transforms.ToTensor()(degraded_image_pil)
            clean_image = transforms.ToTensor()(clean_image_pil)

        return degraded_image, clean_image

    def __len__(self):
        return len(self.image_pairs)

if __name__ == '__main__':
    # Define transformations
    data_transforms = transforms.Compose([
        transforms.ToTensor()
        # transforms.RandomHorizontalFlip(), # Example augmentation
    ])

    # --- USAGE EXAMPLE ---
    # Path to your training data within the 'data' folder
    # Assumes 'data/train/' exists relative to this script
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    train_dataset_path = os.path.join(current_script_dir, 'data', 'train')
    
    print(f"Attempting to load dataset from: {train_dataset_path}")

    if not os.path.isdir(train_dataset_path):
        print(f"Error: Dataset directory not found at {train_dataset_path}")
        print("Please ensure you have a 'data/train/' directory structured correctly relative to your script,")
        print("or modify 'train_dataset_path' to the correct location.")
        print("Expected structure:")
        print(f"{train_dataset_path}/")
        print("  ├── degraded/")
        print("  │   ├── rain-1.png ...")
        print("  │   └── snow-1.png ...")
        print("  └── clean/")
        print("      ├── rain_clean-1.png ...")
        print("      └── snow_clean-1.png ...")
    else:
        image_dataset = ImageRestorationDataset(root_dir=train_dataset_path, transform=data_transforms)
        
        if len(image_dataset) == 0:
            print("\nWarning: The dataset is empty or no valid image pairs were found.")
            print(f"Please check the structure and content of '{train_dataset_path}'.")
            print(f"Ensure 'degraded' and 'clean' subfolders exist and contain correctly named PNG/JPG images.")
        else:
            print(f"Dataset size: {len(image_dataset)} samples.")

            batch_size = 4 
            dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=2)

            print(f"\nIterating through DataLoader (first few batches with batch_size={batch_size}):")
            for i, (degraded_batch, clean_batch) in enumerate(dataloader):
                print(f"Batch {i+1}:")
                print(f"  Degraded batch shape: {degraded_batch.shape}")
                print(f"  Clean batch shape: {clean_batch.shape}")
                print(f"  Degraded batch dtype: {degraded_batch.dtype}")
                print(f"  Clean batch dtype: {clean_batch.dtype}")
                print(f"  Degraded batch min/max: {degraded_batch.min():.2f}/{degraded_batch.max():.2f}")
                print(f"  Clean batch min/max: {clean_batch.min():.2f}/{clean_batch.max():.2f}")
                if i == 1: # Show 2 batches for example
                    break
            
            if i == 0 and len(image_dataset) > 0 and len(image_dataset) < batch_size:
                 print(f"Note: Only one batch was processed as dataset size ({len(image_dataset)}) is less than batch size ({batch_size}).")