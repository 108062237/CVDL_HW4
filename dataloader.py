import os
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

class ImageRestorationDataset(Dataset):
    """
    Custom Dataset for the Image Restoration task.
    It loads pairs of degraded (rain/snow) and corresponding clean images.
    """
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images (e.g., 'data/train/').
                               It should contain 'degraded' and 'clean' subfolders.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_dir = root_dir
        self.degraded_dir = os.path.join(root_dir, 'degraded')
        self.clean_dir = os.path.join(root_dir, 'clean')
        self.transform = transform

        self.image_pairs = []
        
        if not os.path.isdir(self.degraded_dir):
            print(f"Error: Degraded images directory not found: {self.degraded_dir}")
            return
        if not os.path.isdir(self.clean_dir):
            print(f"Error: Clean images directory not found: {self.clean_dir}")
            return

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
                    if not os.path.exists(degraded_img_path):
                        print(f"Warning: Degraded image file not found: {degraded_img_path}")
                    if not os.path.exists(clean_img_path):
                        print(f"Warning: Corresponding clean image file not found: {clean_img_path} for {degraded_file_name}")
            # else:
            #     print(f"Warning: Could not determine clean file name for {degraded_file_name}")


    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        degraded_img_path, clean_img_path = self.image_pairs[idx]

        degraded_image = Image.open(degraded_img_path).convert('RGB')
        clean_image = Image.open(clean_img_path).convert('RGB')

        if self.transform:
            degraded_image = self.transform(degraded_image)
            clean_image = self.transform(clean_image)

        return degraded_image, clean_image

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
            dataloader = DataLoader(image_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

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