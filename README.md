# CVDL Homework 4: Image Restoration

**Student ID:** 313551113

## Project Overview

This project implements an image restoration model based on PromptIR [1] to remove rain and snow degradations from images. The goal is to train a single model to handle both types of unknown degradations.

## File Descriptions

* `README.md`: This file.
* `data/`: Directory for storing datasets 
* `dataloader.py`: Contains the `ImageRestorationDataset` class for loading and preprocessing image pairs.
* `model.py`: Defines the PromptIR neural network architecture used for image restoration. 
* `train.py`: Script for the main training process of the PromptIR model. It handles data loading, model initialization, training loop, validation, and saving the best model.
* `finetune.py`: Script used for fine-tuning a previously trained model, potentially with different learning rates or loss functions.
* `predict.py`: Script for generating predictions on the test dataset using a trained model. It outputs a `pred.npz` file in the format required for submission. 
* `requirements.txt`: Lists Python package dependencies for setting up the environment.

## Basic Usage

1.  **Setup**:
    * Clone the repository.
    * Place the dataset in the `data/` directory as per the structure above.
    * Install required Python packages (e.g., PyTorch, torchvision, Pillow, NumPy, tqdm, pytorch-msssim).

2.  **Training**:
    * Configure hyperparameters and paths in `train.py`.
    * Run `python train.py`. The best model will be saved 

3.  **Fine-tuning (Optional)**:
    * Configure paths and hyperparameters in `finetune.py`.
    * Ensure `PRETRAINED_MODEL_PATH` points to an existing model.
    * Run `python finetune.py`.

4.  **Prediction**:
    * Set the `MODEL_PATH` in `predict.py` (or your specific prediction script) to your best trained model.
    * Ensure `TEST_DATA_DIR` points to `data/test/degraded/`.
    * Run `python predict.py`. This will generate `pred.npz`.
    * Zip `pred.npz` and submit to CodaBench.

