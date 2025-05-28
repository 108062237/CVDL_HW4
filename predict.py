import torch
from torchvision import transforms
from PIL import Image
import os
import numpy as np
from tqdm import tqdm
import logging

# --- Import your custom modules ---
from model import PromptIR # Ensure this is your correct model.py (e.g., dim=48)

# --- Setup Logging ---
LOG_FILE = 'prediction_weighted_ensemble_en.log' # Changed log file name
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

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"Using device: {DEVICE}")

# --- IMPORTANT: Modify the model paths and corresponding weights below ---
# The sum of weights should ideally be 1.0, but it's not strictly necessary as the script normalizes them.
MODEL_PATHS_AND_WEIGHTS = [
    {'path': 'best_promptir_finetuned_29.05.pth', 'weight': 0.4},       # Example: Your best model, slightly higher weight
    {'path': 'best_promptir_l1_msssim_aug.pth', 'weight': 0.35},  # Example: Second best model
    {'path': 'best_promptir_model_29.4.pth', 'weight': 0.25}         # Example: Another model
    # {'path': 'path/to/your/model4.pth', 'weight': 0.1}, # If you have more models
]
# Ensure these models were trained with the same PROMPTIR_PARAMS

# Filter out non-existent model paths and re-calculate total weight for normalization
valid_models_info = []
for item in MODEL_PATHS_AND_WEIGHTS:
    if os.path.exists(item['path']):
        valid_models_info.append(item)
    else:
        logger.warning(f"Model path does not exist, removing from ensemble: {item['path']}")

if not valid_models_info:
    logger.error("List of valid model paths is empty! Please provide at least one valid model path and weight.")
    exit()

# Normalize weights (optional, but recommended)
total_weight = sum(item['weight'] for item in valid_models_info)
if total_weight > 0 and abs(total_weight - 1.0) > 1e-6: # Avoid division by zero and normalize if not already 1.0
    logger.info(f"Original sum of weights: {total_weight}. Normalizing weights...")
    for item in valid_models_info:
        item['weight'] /= total_weight
    normalized_total_weight = sum(item['weight'] for item in valid_models_info)
    logger.info(f"Normalized sum of weights: {normalized_total_weight:.4f}")


CURRENT_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_DATA_DIR = os.path.join(CURRENT_SCRIPT_DIR, 'data', 'test', 'degraded')
OUTPUT_NPZ_FILE = 'pred_weighted_ensemble.npz'

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

if __name__ == '__main__':
    base_model = PromptIR(**PROMPTIR_PARAMS).to(DEVICE)
    logger.info(f"PromptIR model instantiated (dim={PROMPTIR_PARAMS['dim']}).")

    test_transforms_to_tensor = transforms.Compose([
        transforms.ToTensor()
    ])

    results_dict = {}
    try:
        test_image_filenames = sorted([f for f in os.listdir(TEST_DATA_DIR) if f.endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        logger.error(f"Error: Test data directory not found at {TEST_DATA_DIR}.")
        exit()
    if not test_image_filenames:
        logger.error(f"Error: No images found in {TEST_DATA_DIR}.")
        exit()

    logger.info(f"Found {len(test_image_filenames)} images in {TEST_DATA_DIR} for weighted ensemble prediction.")
    logger.info(f"Ensembling {len(valid_models_info)} valid models.")

    with torch.no_grad():
        for filename in tqdm(test_image_filenames, desc="Weighted Ensemble Prediction", unit="image"):
            img_path = os.path.join(TEST_DATA_DIR, filename)
            try:
                degraded_img_pil = Image.open(img_path).convert('RGB')
            except Exception as e:
                logger.warning(f"Cannot open image {img_path}: {e}. Skipping this image.")
                continue

            degraded_img_tensor_bchw = test_transforms_to_tensor(degraded_img_pil).unsqueeze(0).to(DEVICE)

            ensembled_restored_tensor_chw_float = None
            actual_models_used_for_this_image = 0 

            for model_info in valid_models_info:
                model_path = model_info['path']
                model_weight = model_info['weight']
                logger.debug(f"  Processing image '{filename}' using model: {model_path} (weight: {model_weight:.4f})")

                try:
                    base_model.load_state_dict(torch.load(model_path, map_location=DEVICE))
                    base_model.eval()
                except Exception as e:
                    logger.warning(f"  Warning: Failed to load model {model_path}: {e}. Skipping this model for the current image.")
                    continue

                restored_single_model_tensor_bchw = base_model(degraded_img_tensor_bchw)
                restored_single_model_tensor_chw_float = torch.clamp(restored_single_model_tensor_bchw.squeeze(0).cpu(), 0.0, 1.0)

                if ensembled_restored_tensor_chw_float is None:
                    ensembled_restored_tensor_chw_float = restored_single_model_tensor_chw_float * model_weight
                else:
                    ensembled_restored_tensor_chw_float += restored_single_model_tensor_chw_float * model_weight
                
                actual_models_used_for_this_image +=1 
            
            if ensembled_restored_tensor_chw_float is not None and actual_models_used_for_this_image > 0:
                # Since weights were normalized at the beginning to sum to 1 (among valid models),
                # the ensembled_restored_tensor_chw_float is already the correct weighted average.
                # No further division by num_successful_models is needed if all weights for loaded models sum to 1.
                
                restored_output_np_uint8_chw = (ensembled_restored_tensor_chw_float.numpy() * 255).astype(np.uint8)
                results_dict[filename] = restored_output_np_uint8_chw
            elif actual_models_used_for_this_image == 0 : 
                 logger.warning(f"Warning: For image {filename}, no models were successfully loaded and predicted. Result for this image will be empty.")


    if results_dict:
        try:
            np.savez(OUTPUT_NPZ_FILE, **results_dict)
            logger.info(f"Successfully saved {len(results_dict)} weighted ensembled predicted images to {OUTPUT_NPZ_FILE}")
        except Exception as e:
            logger.error(f"Error saving .npz file: {e}")
    else:
        logger.warning("No successful predictions to save.")

    logger.info("Weighted model ensemble prediction process finished.")