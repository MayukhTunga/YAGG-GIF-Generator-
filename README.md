# YAGG: Yet Another GIF Generator (Text-to-GIF with JAX/Flax)

[![JAX](https://img.shields.io/badge/Built%20with-JAX-orange)](https://github.com/google/jax)
[![Flax](https://img.shields.io/badge/Built%20with-Flax-blue)](https://github.com/google/flax)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) 

A JAX/Flax implementation of a text-to-GIF generation model using a pre-trained CLIP text encoder and a custom 3D U-Net diffusion model. This project explores generating short, animated sequences conditioned on natural language descriptions.

**Current Status:** Experimental / Work-in-Progress

## Features

*   **Text-to-GIF Generation:** Creates short animated GIFs from text prompts.
*   **CLIP Conditioning:** Utilizes OpenAI's CLIP model (`openai/clip-vit-base-patch32`) for powerful text-to-visual semantic understanding.
*   **3D Diffusion Model:** Employs a Denoising Diffusion Probabilistic Model (DDPM) with a custom 3D U-Net architecture to model spatio-temporal data.
*   **JAX/Flax Implementation:** Built entirely using JAX and the Flax neural network library for high performance, especially on TPUs/GPUs.
*   **Factorized Attention:** Incorporates factorized spatial and temporal self-attention within the U-Net to manage memory usage on constrained hardware (like Colab TPUs/GPUs).
*   **FastAPI Backend (Seperate):** Includes code to serve the trained model via a FastAPI web API for inference.

## Architecture Overview

The model consists of two primary components:

1.  **Text Encoder (CLIP):**
    *   We use a **pre-trained and frozen** `FlaxCLIPTextModelWithProjection` from the Hugging Face `transformers` library.
    *   It takes tokenized text input and outputs a fixed-size **semantic embedding vector** (e.g., 512 dimensions) that represents the meaning of the prompt. This embedding guides the diffusion process.

2.  **Frame Generator (Conditional 3D U-Net Diffusion Model):**
    *   Based on the DDPM framework, this component learns to reverse a noise-adding process.
    *   **Core Network:** A `UNetConditional3D` built with Flax.
        *   **Input:** Noisy frame sequence (`[B, F, H, W, C]`), diffusion timestep (`t`), CLIP text embedding (`[B, E]`).
        *   **Output:** Predicted noise added to the frames (`[B, F, H, W, C]`).
        *   **3D Convolutions:** Utilizes `nn.Conv` with 3D kernels to process frames simultaneously, capturing temporal dependencies.
        *   **Conditioning:** Time embeddings (via `SinusoidalPosEmb` + MLP) and text embeddings (via learned projections) are added within the network's `ResnetBlock3D` blocks.
        *   **Factorized Attention:** To handle the high memory cost of full spatio-temporal attention, especially at higher resolutions, `FactorizedAttentionBlock` is used at specified resolutions. It performs spatial attention within each frame followed by temporal attention across frames for each spatial location.
        *   **U-Net Structure:** Standard encoder-decoder structure with skip connections.


## Demo / Examples (This is just trained on 8k GIFs, More training to be done)

**Prompt:** "A Man Smiling" :

![8k Train](https://github.com/user-attachments/assets/62669a48-cd15-4095-a42f-cfccbf95deec)


## Setup and Installation

This project is primarily designed to run in a Google Colab environment with a GPU or TPU runtime.

**1. Prerequisites:**
*   Google Account (for Colab and Drive)
*   Python 3.9+
