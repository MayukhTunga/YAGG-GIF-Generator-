# YAGG: Yet Another GIF Generator (Text-to-GIF with JAX/Flax)

[![JAX](https://img.shields.io/badge/Built%20with-JAX-orange)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Built%20with-Flax-blue)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A JAX/Flax implementation of a text-to-GIF generation model using a pre-trained CLIP text encoder and a custom 3D U-Net diffusion model. This project generates short, animated sequences conditioned on natural language descriptions.

**Current Status:** Experimental / Work-in-Progress. The model has been trained on a limited dataset (TGIF image-text pairs) and may require further training for high-quality, diverse outputs.

## Features

*   **Text-to-GIF Generation:** Creates short animated GIFs from text prompts.
*   **CLIP Conditioning:** Utilizes a pre-trained CLIP model ([`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)) for robust text-to-visual semantic understanding.
*   **3D Diffusion Model:** Employs a Denoising Diffusion Probabilistic Model (DDPM) with a custom 3D U-Net architecture (`UNetConditional3D`) to model spatio-temporal data.
*   **JAX/Flax Implementation:** Built entirely using JAX and the Flax neural network library for high-performance computation, especially on TPUs/GPUs.
*   **Factorized Attention:** Incorporates factorized spatial and temporal self-attention (`FactorizedAttentionBlock`) within the U-Net to manage memory usage, particularly on constrained hardware like Colab TPUs/GPUs.
*   **Data Preprocessing:** Includes scripts/notebook cells for downloading and preprocessing GIF data from URLs (e.g., from the TGIF dataset) into a suitable format for training.
*   **Training Loop:** Provides a complete training pipeline, including data loading, optimizer setup (AdamW), loss calculation, and periodic checkpointing.
*   **Inference/Sampling:** Demonstrates how to generate GIFs from text prompts using the trained model with a DDPM sampling loop.
*   **Google Colab Compatibility:** Designed with Colab in mind, including TPU utilization and Google Drive integration for saving/loading checkpoints.
*   **FastAPI Backend (Separate):** Note: The README mentions a FastAPI backend, but this is not present in the provided `YAGG.ipynb` file. This feature might be in a different part of the project.

## Architecture Overview

The model consists of two primary components:

1.  **Text Encoder (CLIP):**
    *   We use a **pre-trained and frozen** `FlaxCLIPTextModelWithProjection` (from `transformers`, specifically [`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)) to encode text prompts.
    *   It takes tokenized text input (max length 77 tokens) and outputs a fixed-size **semantic embedding vector** (512 dimensions in this implementation) that represents the meaning of the prompt. This embedding guides the diffusion process.

2.  **Frame Generator (Conditional 3D U-Net Diffusion Model):**
    *   Based on the DDPM framework, this component learns to reverse a noise-adding process.
    *   **Core Network:** A `UNetConditional3D` built with Flax.
        *   **Input:** Noisy frame sequence (e.g., `[Batch, Frames, Height, Width, Channels]`), diffusion timestep (`t`), and the CLIP text embedding.
        *   **Output:** Predicted noise that was added to the frames.
        *   **3D Convolutions:** Utilizes `flax.linen.Conv` with 3D kernels to process frame sequences, capturing temporal dependencies.
        *   **Conditioning:** Diffusion timestep embeddings (generated using `SinusoidalPosEmb` followed by an MLP) and text embeddings (projected via `flax.linen.Dense` layers) are incorporated into the `ResnetBlock3D` blocks of the U-Net.
        *   **Factorized Attention:** To manage the computational cost of full spatio-temporal attention, `FactorizedAttentionBlock` is used at specified resolutions (e.g., 2x and 4x downsampled from original). This block performs spatial attention within each frame, followed by temporal attention across frames for each spatial location.
        *   **U-Net Structure:** Standard encoder-decoder architecture with skip connections. The encoder downsamples spatio-temporally, and the decoder upsamples. ResNet blocks are used throughout.

## Demo / Example

The following GIF was generated using a model trained for a number of steps (e.g., 8000 steps as shown in the notebook) on a subset of the TGIF dataset. The quality can be improved with more extensive training and a larger dataset.

**Prompt (example from notebook):** "A ball falling down" (The actual demo GIF linked below is for "A Man Smiling")

![Generated GIF](https://github.com/user-attachments/assets/62669a48-cd15-4095-a42f-cfccbf95deec)

*(Note: The linked GIF shows "A Man Smiling." The notebook also contains code to generate a GIF for "A ball falling down" if a checkpoint is loaded).*

## Setup and Installation

This project is primarily designed to run in a Google Colab environment, preferably with a TPU runtime. However, it can be adapted for local execution with appropriate hardware (CPU, GPU, or TPU).

**1. Prerequisites:**
*   Python 3.9+ (as used in Colab environment, other 3.x versions might work but are untested).
*   Access to a JAX/Flax compatible environment (CPU, GPU, or TPU). The notebook is pre-configured for TPU.
*   Google Account (if using Colab and Google Drive for checkpoints).
**2. Environment Setup (Recommended for Local Development):**
*   It's highly recommended to use a virtual environment:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```
    Alternatively, you can use Conda:
    ```bash
    conda create -n yagg python=3.9
    conda activate yagg
    ```

**3. Install Dependencies:**
*   The core dependencies can be installed via pip. The `YAGG.ipynb` notebook includes the following installation commands:
    ```bash
    pip install --upgrade pip
    pip install --upgrade jax jaxlib flax optax
    pip install transformers datasets
    pip install imageio[ffmpeg] Pillow
    pip install gdown
    pip install tensorflow # Primarily for tf.data and tfds.as_numpy
    pip install einops
    ```
    It's good practice to put these into a `requirements.txt` file for local setup. Create a file named `requirements.txt` with the following content:
    ```text
    jax
    jaxlib
    flax
    optax
    transformers
    datasets
    imageio[ffmpeg]
    Pillow
    gdown
    tensorflow
    einops
    ```
    Then install with: `pip install -r requirements.txt`

**4. Google Drive (for Colab - Optional for Local):**
*   The notebook uses Google Drive to save and load model checkpoints. If running in Colab, you'll be prompted to mount your drive:
    ```python
    from google.colab import drive
    drive.mount('/content/drive')
    ```
    Ensure the `CHECKPOINT_DIR` variable in the notebook points to your desired Drive location. For local execution, you can change `CHECKPOINT_DIR` to a local path.

**5. Data Preprocessing:**
*   The notebook includes steps to download and preprocess data (e.g., from the TGIF dataset). You'll need to:
    *   Upload the `tgif-v1.0.tsv` file to your Colab environment (or provide a local path).
    *   Run the preprocessing cells to download GIFs, sample frames, tokenize text, and save them as `.npy` files. This data is saved to local Colab storage by default in the notebook (`/content/tgif_preprocessed_local_32px_16f`). **This local data will be lost if the Colab runtime restarts.** For persistent storage, modify the paths to save preprocessed data to Google Drive or another persistent location.

## YAGG.ipynb Notebook Guide

The `YAGG.ipynb` notebook is structured to guide you through the entire process from setup to GIF generation. It's recommended to run the cells sequentially.

1.  **Setup & Installations:** The initial cells install necessary libraries like JAX, Flax, Transformers, etc., and set up Google Drive integration.
2.  **Configuration & Preprocessing:**
    *   Defines paths for data and checkpoints (e.g., `TSV_FILE_PATH`, `PREPROCESSED_DATA_DIR`, `CHECKPOINT_DIR`). You might need to adjust these, especially `TSV_FILE_PATH` to point to your TGIF dataset file and `CHECKPOINT_DIR` to your preferred Google Drive location.
    *   Loads the CLIP tokenizer ([`openai/clip-vit-base-patch32`](https://huggingface.co/openai/clip-vit-base-patch32)).
    *   Processes the TGIF dataset: downloads GIFs, samples frames, tokenizes descriptions, and saves them as `.npy` files along with a `processed_manifest.csv`.
    *   **Expected Output:** A local directory (e.g., `/content/tgif_preprocessed_local_32px_16f`) containing `.npy` files for frames and tokens, and `processed_manifest.csv`.
3.  **Model Definition:**
    *   Loads the pre-trained CLIP text model (`FlaxCLIPTextModelWithProjection`).
    *   Defines the 3D U-Net architecture (`UNetConditional3D`), including `ResnetBlock3D`, `FactorizedAttentionBlock`, and `SinusoidalPosEmb`.
    *   Defines the DDPM diffusion schedule and parameters.
4.  **Training:**
    *   Initializes the optimizer (AdamW).
    *   Defines the `train_step` function, which computes the noise prediction loss (Mean Squared Error between predicted and actual noise) and updates model parameters using `jax.pmap` for multi-device training.
    *   The training loop iteratively fetches batches from the preprocessed data, executes the `train_step`, logs the loss, and saves model checkpoints periodically to `CHECKPOINT_DIR`.
    *   **Key Parameters to Modify:** `NUM_TRAIN_STEPS`, `LOG_EVERY_STEPS`, `CHECKPOINT_EVERY_STEPS` (formatted as code: `NUM_TRAIN_STEPS`, `LOG_EVERY_STEPS`, `CHECKPOINT_EVERY_STEPS`).
    *   **Expected Output:** Saved checkpoints in your `CHECKPOINT_DIR` on Google Drive.
    *   **Resuming Training:** To resume training, ensure `CHECKPOINT_DIR` points to your existing checkpoints. The notebook's training loop currently starts from scratch, but it could be adapted to load the latest `unet_params` and `opt_state` from a checkpoint to continue a previous run. The inference section already demonstrates loading `unet_params`.
5.  **Inference / GIF Generation:**
    *   Defines sampling functions (`p_sample`, `p_sample_loop`).
    *   The `generate_gif` function:
        *   Loads a trained U-Net model checkpoint from `CHECKPOINT_DIR`.
        *   Takes a text `prompt` as input.
        *   Uses the CLIP model to get text embeddings.
        *   Runs the DDPM sampling loop (`p_sample_loop`) to generate frames.
        *   Post-processes frames and saves them as a GIF.
    *   **Key Parameters to Modify:** `prompt` for generation, `checkpoint_path` (if not using the latest).
    *   **Expected Output:** A `generated_gif.gif` file.

## Key Components (Conceptual Flow)

The project revolves around these core ideas:

*   **Text Encoding:** Transforming textual prompts into meaningful numerical representations (embeddings) using CLIP.
*   **Diffusion Model (DDPM):**
    *   **Forward Process:** Gradually adding noise to image data until it becomes pure noise.
    *   **Reverse Process:** Training a model (the 3D U-Net) to predict and remove this noise step-by-step, conditioned on the text embedding and the current diffusion timestep.
*   **3D U-Net (`UNetConditional3D`):** A neural network architecture specifically designed to operate on 3D (spatio-temporal) data like video frames. It predicts the noise at each step of the reverse diffusion process.
*   **Sampling:** The iterative process of using the trained U-Net to denoise from pure noise to a clean sequence of frames, guided by the text prompt.

## Future Work

Based on the current project status and notebook contents, potential areas for future development include:

*   **More Extensive Training:** Training the model on larger and more diverse datasets for longer durations to improve GIF quality and variety.
*   **Hyperparameter Tuning:** Experimenting with different learning rates, batch sizes, U-Net configurations, and diffusion timesteps.
*   **Improved Sampling:** Implementing more advanced DDPM sampling techniques (e.g., DDIM) for faster and potentially higher-quality generation.
*   **User Interface:** Developing a more user-friendly interface for generation, such as a Gradio or Streamlit demo, or a simple web application.
*   **FastAPI Backend Integration:** Completing and documenting the FastAPI backend mentioned in the original feature list to serve the model as an API.
*   **Support for Different Input Modalities:** Extending the model to be conditioned on other inputs, like images or audio, or exploring unconditional generation.
*   **Model Scaling:** Exploring larger versions of the U-Net or different CLIP models.

## Contributing

Contributions are welcome! If you'd like to improve YAGG or add new features, please follow these steps:

1.  **Fork the Repository:** Create your own copy of the project.
2.  **Create a New Branch:** Make a branch in your fork for your specific feature or bug fix (e.g., `git checkout -b feature/my-new-feature` or `git checkout -b fix/issue-fix`).
3.  **Make Changes:** Implement your changes and additions.
4.  **Test Your Changes:** If applicable, add or update tests to ensure your changes work as expected and do not break existing functionality.
5.  **Commit Your Changes:** Use clear and descriptive commit messages.
6.  **Push to Your Branch:** Push your changes to your forked repository.
7.  **Submit a Pull Request:** Open a pull request from your branch to the main YAGG repository, providing a clear description of your changes and why they are being made.

## License

This project is licensed under the MIT License. See the badges at the top of this README. (Note: A formal `LICENSE` file is not currently present in the repository but the intention is MIT License).
