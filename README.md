# 👗 Clothes Virtual Try-On (VITON-HD Based)

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SwayamInSync/clothes-virtual-try-on/blob/main/setup_gradio.ipynb)

A deep learning pipeline that lets you virtually "try on" a clothing item onto a person's photo. Given an image of a person and an image of a garment, the system generates a realistic image of that person wearing the new garment — preserving their pose, body shape, and identity.

---

## 📑 Table of Contents

- [Overview](#overview)
- [How it Works — High Level](#how-it-works--high-level)
- [Full Pipeline — Step by Step](#full-pipeline--step-by-step)
- [Architecture & Deep Learning Models](#architecture--deep-learning-models)
- [Project Structure & File-by-File Breakdown](#project-structure--file-by-file-breakdown)
- [Data Flow Diagram](#data-flow-diagram)
- [Required Model Weights](#required-model-weights)
- [Local Setup (Windows)](#local-setup-windows)
- [Local Setup (Google Colab — Original)](#local-setup-google-colab--original)
- [Client-Server Architecture](#client-server-architecture)
- [Troubleshooting & Common Issues](#troubleshooting--common-issues)
- [Credits & Citation](#credits--citation)

---

## Overview

This project implements a **virtual try-on** system based on the [VITON-HD](https://github.com/shadow2496/VITON-HD) framework. The goal is to solve a common e-commerce problem: customers cannot try on clothes before buying online. This system takes:

| Input | Description |
|---|---|
| **Person image** | A photo of a person (full/half body, front-facing) |
| **Cloth image** | A flat-lay photo of a garment (upper-body clothing) |

And produces:

| Output | Description |
|---|---|
| **Try-on result** | A photorealistic image of the person wearing the new garment |

### Key Technologies Used

- **PyTorch** — Deep learning framework for all neural networks
- **U²-Net** — Cloth segmentation (generating cloth masks)
- **OpenPose** — Human pose estimation (body keypoints)
- **Self-Correction Human Parsing (SCHP)** — Body part semantic segmentation
- **rembg** — Background removal from person images
- **TPS (Thin Plate Spline)** — Geometric warping of cloth to match body shape
- **ALIAS (Appearance flow-based Layered Inpainting and Synthesis)** — Final image generation via GAN

---

## How it Works — High Level

```
Person Image + Cloth Image
        │
        ▼
┌─────────────────────────────────────────────┐
│           PREPROCESSING PIPELINE            │
│                                             │
│  1. Resize images to 768×1024               │
│  2. Remove background from person image     │
│  3. Generate cloth segmentation mask        │
│  4. Parse human body parts (semantic map)   │
│  5. Detect body pose keypoints (OpenPose)   │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────────┐
│            INFERENCE PIPELINE               │
│                                             │
│  Stage 1: Segmentation Generation (SegGen)  │
│  Stage 2: Geometric Matching (GMM/TPS)      │
│  Stage 3: Try-On Synthesis (ALIASGen)       │
└──────────────────┬──────────────────────────┘
                   │
                   ▼
          Final Try-On Image
```

---

## Full Pipeline — Step by Step

### Stage 0: Preprocessing (run.py orchestrates all of this)

#### Step 0.1 — Image Resizing
All input images (person + cloth) are resized to **768×1024** pixels. This is the resolution the models were trained on.

#### Step 0.2 — Background Removal (`remove_bg.py`)
The person image goes through **rembg** (which uses U²-Net internally) to remove the background. The result is then composited onto a pure white background and saved as a JPEG. This ensures the models don't get confused by complex backgrounds.

#### Step 0.3 — Cloth Mask Generation (`cloth-mask.py`)
The flat-lay cloth image is fed through a **U²-Net** segmentation model (`cloth_segm_u2net_latest.pth`) to produce a binary mask. This mask separates the garment from its background — the model needs to know exactly which pixels are the cloth.

#### Step 0.4 — Human Parsing (External: Self-Correction-Human-Parsing)
The person image is processed by the **SCHP** model to create a semantic segmentation map of body parts. Each pixel is classified into one of 20 categories:

| ID | Body Part | ID | Body Part |
|----|-----------|-----|-----------|
| 0 | Background | 10 | Neck |
| 1 | Hat | 11 | Scarf |
| 2 | Hair | 12 | Skirt |
| 3 | Glove | 13 | Face |
| 4 | Sunglasses | 14 | Left arm |
| 5 | Upper-clothes | 15 | Right arm |
| 6 | Dress | 16 | Left leg |
| 7 | Coat | 17 | Right leg |
| 8 | Socks | 18 | Left shoe |
| 9 | Pants | 19 | Right shoe |

#### Step 0.5 — Pose Estimation (External: OpenPose)
OpenPose detects body keypoints (shoulders, elbows, hips, etc.) and outputs:
- **Rendered pose image** (`_rendered.png`) — a visualization of the skeleton
- **Keypoints JSON** (`_keypoints.json`) — numeric (x, y) coordinates for each joint

These are essential for the model to understand the person's body configuration.

#### Step 0.6 — Test Pairs File
A `test_pairs.txt` file is generated that maps each person image to its corresponding cloth image, e.g.:
```
model.jpg cloth.jpg
```

---

### Stage 1: Segmentation Generation (SegGenerator)

**Model:** `seg_final.pth` (~138 MB)  
**Architecture:** U-Net with InstanceNorm  
**Input (concatenated):** cloth mask (downscaled) + masked cloth + agnostic parse map + pose + noise = **21 channels**  
**Output:** 13-channel segmentation prediction

This model predicts **what the person's segmentation map should look like** after putting on the new garment. It essentially "imagines" where the new cloth would appear on the body, replacing the existing upper-body clothing region. The 13-channel output is then collapsed into 7 semantic regions:

| Channel | Region | Source Parse IDs |
|---------|--------|------------------|
| 0 | Background | 0 |
| 1 | Paste (keep as-is) | 2, 4, 7, 8, 9, 10, 11 |
| 2 | Upper body (new cloth) | 3 |
| 3 | Hair | 1 |
| 4 | Left arm | 5 |
| 5 | Right arm | 6 |
| 6 | Noise | 12 |

---

### Stage 2: Geometric Matching Module (GMM)

**Model:** `gmm_final.pth` (~76 MB)  
**Architecture:** Dual-stream Feature Extraction → Feature Correlation → TPS Grid Regression  
**Input A:** predicted cloth region parse + pose + agnostic person image (7 channels)  
**Input B:** cloth image (3 channels)  
**Output:** TPS transformation grid

This stage spatially **warps (deforms) the flat-lay cloth image** to match the person's body shape and pose. It uses Thin Plate Spline (TPS) transformation with a 5×5 grid of control points to create smooth, realistic deformations. The process:

1. **Feature Extraction:** Two separate CNNs extract features from the body representation and the cloth image
2. **Feature Correlation:** Computes a correlation map between body and cloth features (tells which cloth parts correspond to which body parts)
3. **Regression:** Predicts TPS parameters (25 control point offsets)
4. **Grid Generation:** Creates a sampling grid using TPS
5. **Warping:** `F.grid_sample` warps both the cloth image and its mask using the predicted grid

---

### Stage 3: Try-On Synthesis (ALIASGenerator)

**Model:** `alias_final.pth` (~402 MB)  
**Architecture:** SPADE-like generator with ALIAS normalization + spectral norm  
**Input:** agnostic person image + pose + warped cloth (9 channels)  
**Conditioning:** predicted segmentation map + misalignment mask  
**Output:** Final 3-channel RGB try-on image

This is the final generation step. It takes the agnostic (clothes-removed) person, their pose, and the warped cloth, then synthesizes a photorealistic output. Key innovations:

- **ALIAS Normalization:** A variant of SPADE that handles the misalignment between the warped cloth and the predicted segmentation. It uses separate normalization for foreground/background regions (MaskNorm).
- **Multi-scale feature injection:** The input is sampled at 8 different resolutions and features are injected at each decoder level.
- **Spectral normalization:** Stabilizes GAN training.
- **Residual blocks:** Ensures gradient flow and stable synthesis.

---

## Architecture & Deep Learning Models

### Neural Network Summary

| Model | File | Checkpoint | Params | Purpose |
|-------|------|------------|--------|---------|
| **U²-Net** | `networks/u2net.py` | `cloth_segm_u2net_latest.pth` (177 MB) | ~44M | Cloth segmentation mask generation |
| **SegGenerator** | `network.py` | `seg_final.pth` (138 MB) | ~36M | Predict body segmentation with new cloth |
| **GMM** | `network.py` | `gmm_final.pth` (76 MB) | ~19M | Warp cloth to match body shape |
| **ALIASGenerator** | `network.py` | `alias_final.pth` (402 MB) | ~103M | Final photorealistic try-on synthesis |
| **SCHP** | External repo | `final.pth` (267 MB) | ~65M | Human body part parsing |
| **OpenPose** | External binary | Multiple `.caffemodel` files | N/A | Body pose keypoint detection |

---

## Project Structure & File-by-File Breakdown

```
clothes-virtual-try-on/
│
├── run.py                  # 🎯 Main orchestrator script
├── test.py                 # 🧪 Inference/prediction script (3-stage model pipeline)
├── cloth-mask.py           # 🎭 Cloth segmentation mask generator
├── remove_bg.py            # 🖼️ Background removal from person images
├── datasets.py             # 📦 PyTorch Dataset & DataLoader for VITON format
├── network.py              # 🧠 All neural network architectures (SegGen, GMM, ALIAS)
├── utils.py                # 🔧 Utility functions (noise, save images, load checkpoints)
│
├── networks/               # 📁 Additional network architectures
│   ├── __init__.py         #     Exports U2NET class
│   └── u2net.py            #     U²-Net architecture for cloth segmentation
│
├── client-side/            # 🌐 Flask web application (client)
│   ├── app.py              #     Flask server with upload + API proxy
│   ├── templates/
│   │   └── index.html      #     Web UI with TailwindCSS
│   └── static/
│       ├── css/style.css   #     Custom styles
│       ├── images/logo.png #     Logo asset
│       └── output/         #     Placeholder for results
│
├── assets/                 # 📸 Sample images for testing
│   ├── cloth/              #     12 sample clothing images (768×1024 JPG)
│   └── image/              #     6 sample person images (768×1024 JPG)
│
├── checkpoints/            # 💾 Model weights (NOT in repo — must download)
│   ├── seg_final.pth       #     Segmentation generator weights
│   ├── gmm_final.pth       #     Geometric matching weights
│   └── alias_final.pth     #     ALIAS generator weights
│
├── cloth_segm_u2net_latest.pth  # 💾 U²-Net cloth segmentation weights (must download)
│
├── setup_gradio.ipynb      # 📓 Colab notebook with Gradio UI (recommended)
├── setup_ngrok.ipynb       # 📓 Colab notebook with ngrok API server
├── .gitignore              # Git ignore rules
└── README.md               # This file
```

### Detailed File Descriptions

---

#### `run.py` — Main Orchestrator
**What it does:** This is the master script that coordinates the entire preprocessing + inference pipeline. It is designed to run on Google Colab (paths like `/content/...`).

**Step-by-step:**
1. Resizes all cloth images in `inputs/test/cloth/` to 768×1024
2. Runs `cloth-mask.py` to generate cloth segmentation masks
3. Runs `remove_bg.py` to remove background from person images
4. Runs **Self-Correction-Human-Parsing** to create body parsing maps
5. Runs **OpenPose** to extract pose keypoints + render pose images
6. Generates `test_pairs.txt` mapping person↔cloth
7. Runs `test.py` to perform the 3-stage try-on inference

> ⚠️ **Note:** This file has hardcoded Colab paths (`/content/...`). For local use, you must modify these paths.

---

#### `test.py` — Inference Script
**What it does:** Loads the three trained models (SegGenerator, GMM, ALIASGenerator), processes the inputs through the 3-stage pipeline, and saves the output images.

**Command-line arguments:**

| Argument | Default | Description |
|----------|---------|-------------|
| `--name` | (required) | Output folder name |
| `--batch_size` | 1 | Batch size |
| `--load_height` | 1024 | Input image height |
| `--load_width` | 768 | Input image width |
| `--dataset_dir` | `./datasets/` | Root data directory |
| `--checkpoint_dir` | `./checkpoints/` | Model weights directory |
| `--save_dir` | `./results/` | Output directory |
| `--seg_checkpoint` | `seg_final.pth` | Segmentation model filename |
| `--gmm_checkpoint` | `gmm_final.pth` | GMM model filename |
| `--alias_checkpoint` | `alias_final.pth` | ALIAS model filename |
| `--grid_size` | 5 | TPS grid size (5×5 = 25 control points) |
| `--semantic_nc` | 13 | Number of human parsing classes |
| `--ngf` | 64 | Generator base filter count |
| `--num_upsampling_layers` | `most` | How many upsampling layers in ALIAS |

**Usage:**
```bash
python test.py --name output --dataset_dir ./inputs --checkpoint_dir ./checkpoints --save_dir ./results/
```

---

#### `cloth-mask.py` — Cloth Segmentation
**What it does:** Uses a pre-trained **U²-Net** to segment clothing items from their background, producing black/white masks.

**Process:**
1. Loads the U²-Net model from `cloth_segm_u2net_latest.pth`
2. For each cloth image: resizes to 768×768, normalizes, runs through the network
3. Takes the argmax of the 4-class output (background, upper body, lower body, full body)
4. Applies a binary palette (white for cloth, black for background)
5. Saves the mask to `cloth-mask/` directory

**Requires:** CUDA GPU, U²-Net checkpoint file.

---

#### `remove_bg.py` — Background Removal
**What it does:** Removes the background from person images using the `rembg` library and places the person on a clean white background.

**Class: `preprcessInput`**
- `remove_bg(file_path)` — Opens image, removes background via `rembg`, returns RGBA numpy array
- `transform(width, height)` — Resizes to target dimensions, composites onto white background, saves as JPG

---

#### `datasets.py` — Data Loading
**What it does:** Implements the PyTorch `Dataset` and `DataLoader` for the VITON data format.

**Class: `VITONDataset`**
- Reads `test_pairs.txt` to get image-cloth pairs
- For each sample, loads and preprocesses:
  - Cloth image + cloth mask → normalized tensors
  - Pose rendering (OpenPose) → tensor
  - Pose keypoints (JSON) → numpy array
  - Human parsing map → agnostic segmentation tensor
  - Person image → agnostic (cloth-removed) version
- `get_parse_agnostic()` — Creates the "agnostic" parse by removing clothing, arms, and neck regions
- `get_img_agnostic()` — Creates the "agnostic" person image by painting over the torso/arms with gray

**Class: `VITONDataLoader`**
- Wraps PyTorch DataLoader with convenient `next_batch()` method

**Expected folder structure for dataset:**
```
dataset_dir/
├── test/
│   ├── cloth/            # Flat-lay garment images (.jpg)
│   ├── cloth-mask/       # Binary cloth masks (.jpg)
│   ├── image/            # Person photos (.jpg)
│   ├── image-parse/      # Human parsing maps (.png)
│   ├── openpose-img/     # Rendered pose images (_rendered.png)
│   └── openpose-json/    # Pose keypoints (_keypoints.json)
└── test_pairs.txt        # "person_img.jpg cloth_img.jpg" per line
```

---

#### `network.py` — Neural Network Architectures
**What it does:** Defines all three main neural networks used in the try-on pipeline.

**Models defined:**

1. **`BaseNetwork`** — Base class with weight initialization and parameter counting
2. **`SegGenerator`** — U-Net encoder-decoder for segmentation prediction
   - 5 encoder blocks (conv + pool) → 4 decoder blocks (upsample + skip connections)
   - Uses InstanceNorm and dropout
3. **`FeatureExtraction`** — CNN for extracting visual features (used by GMM)
4. **`FeatureCorrelation`** — Computes correlation maps between two feature tensors
5. **`FeatureRegression`** — Regresses TPS parameters from correlation maps
6. **`TpsGridGen`** — Generates TPS (Thin Plate Spline) transformation grids
7. **`GMM`** — Geometric Matching Module combining extraction, correlation, regression, and TPS
8. **`MaskNorm`** — Region-aware instance normalization
9. **`ALIASNorm`** — ALIAS normalization layer (parameter-free norm + learned affine from segmentation)
10. **`ALIASResBlock`** — Residual block with ALIAS normalization
11. **`ALIASGenerator`** — Full image generator with multi-scale feature injection

---

#### `utils.py` — Utilities
**What it does:** Three helper functions used across the project.

| Function | Purpose |
|----------|---------|
| `gen_noise(shape)` | Generates random noise tensor (used as input to SegGenerator) |
| `save_images(img_tensors, img_names, save_dir)` | Converts model output tensors to PIL images and saves as JPEG |
| `load_checkpoint(model, checkpoint_path)` | Loads PyTorch state dict into a model |

---

#### `networks/u2net.py` — U²-Net Architecture
**What it does:** Implements the **U²-Net (U-square Net)** architecture, a nested U-structure network designed for salient object detection. Used here specifically for cloth segmentation.

**Key components:**
- **REBNCONV** — Basic convolution block (Conv2d + BatchNorm + ReLU) with dilated convolutions
- **RSU-7, RSU-6, RSU-5, RSU-4** — Residual U-blocks with varying depths (7 to 4 levels)
- **RSU-4F** — Flat version of RSU-4 using dilated convolutions instead of pooling
- **U2NET** — Full model: 6 encoder stages + 5 decoder stages with side outputs
- **U2NETP** — Lightweight version (not used in this project)

---

#### `client-side/app.py` — Flask Web Client
**What it does:** A simple Flask web app that provides a UI for uploading person and cloth images. It sends these to a backend server (running on Colab via ngrok) and displays the result.

**Routes:**
- `GET /` — Renders the upload page
- `POST /preds` — Accepts cloth + model image uploads, proxies to the ngrok backend, displays the result

> ⚠️ The backend URL is hardcoded to an ngrok URL that changes every session.

---

#### `client-side/templates/index.html` — Web UI
**What it does:** A responsive, dark-themed landing page using TailwindCSS with:
- Hero section explaining the problem, solution, and approach
- Two file upload zones (one for cloth, one for person)
- Submit button that triggers the try-on
- Result display area (base64-encoded output image)

---

#### `setup_gradio.ipynb` — Colab Notebook (Gradio UI)
**What it does:** The **recommended** way to run this project. Sets up the entire environment on Google Colab with a Gradio interface. Handles all dependency installation, model downloads, and provides an interactive UI.

---

#### `setup_ngrok.ipynb` — Colab Notebook (API Server)
**What it does:** Alternative Colab setup that creates an ngrok-tunneled Flask API server. The client-side Flask app connects to this server. Includes all environment setup steps:
1. Install CMake, OpenCV, system dependencies
2. Build OpenPose from source
3. Download model weights (via gdown)
4. Clone and set up Self-Correction-Human-Parsing
5. Set up Flask + ngrok server

---

## Data Flow Diagram

```
┌──────────────┐    ┌──────────────┐
│ Person Image │    │  Cloth Image │
│ (768×1024)   │    │ (768×1024)   │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│  rembg       │    │  U²-Net      │
│  (remove bg) │    │  (cloth seg) │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ White-bg     │    │ Cloth Mask   │
│ Person Image │    │ (binary)     │
└──────┬───────┘    └──────┴───────┘
       │                   │
       ├───────┐           │
       ▼       ▼           │
┌─────────┐ ┌─────────┐   │
│ SCHP    │ │OpenPose │   │
│ (parse) │ │ (pose)  │   │
└────┬────┘ └────┬────┘   │
     │           │         │
     ▼           ▼         ▼
┌─────────────────────────────────┐
│        datasets.py              │
│  (Assembles all inputs into     │
│   tensors for the model)        │
│                                 │
│  Outputs:                       │
│  • img_agnostic (person w/o     │
│    torso/arms, gray filled)     │
│  • parse_agnostic (semantic     │
│    map w/o clothing regions)    │
│  • pose (rendered skeleton)     │
│  • cloth tensor                 │
│  • cloth_mask tensor            │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│    Stage 1: SegGenerator        │
│                                 │
│  Input: cloth_mask + cloth +    │
│    parse_agnostic + pose + noise│
│  Output: predicted segmentation │
│    (where new cloth goes)       │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│    Stage 2: GMM (Warping)       │
│                                 │
│  Input: parse_cloth + pose +    │
│    agnostic_person / cloth      │
│  Output: warped cloth + mask    │
│    (deformed to match body)     │
└─────────────┬───────────────────┘
              │
              ▼
┌─────────────────────────────────┐
│  Stage 3: ALIASGenerator        │
│                                 │
│  Input: agnostic + pose +       │
│    warped cloth                 │
│  Conditioning: segmentation +   │
│    misalignment mask            │
│  Output: Final try-on image     │
└─────────────┬───────────────────┘
              │
              ▼
       ┌──────────────┐
       │  Result JPEG  │
       │  (768×1024)   │
       └──────────────┘
```

---

## Required Model Weights

You need to download **4 checkpoint files** (~793 MB total) and place them in the correct locations:

### VITON-HD Checkpoints (go in `checkpoints/` folder)

| File | Size | Google Drive ID | Download Command |
|------|------|-----------------|------------------|
| `alias_final.pth` | 402 MB | `18q4lS7cNt1_X8ewCgya1fq0dSk93jTL6` | `gdown --id 18q4lS7cNt1_X8ewCgya1fq0dSk93jTL6` |
| `gmm_final.pth` | 76 MB | `1uDRPY8gh9sHb3UDonq6ZrINqDOd7pmTz` | `gdown --id 1uDRPY8gh9sHb3UDonq6ZrINqDOd7pmTz` |
| `seg_final.pth` | 138 MB | `1d7lZNLh51Qt5Mi1lXqyi6Asb2ncLrEdC` | `gdown --id 1d7lZNLh51Qt5Mi1lXqyi6Asb2ncLrEdC` |

### U²-Net Cloth Segmentation (goes in project root)

| File | Size | Google Drive ID | Download Command |
|------|------|-----------------|------------------|
| `cloth_segm_u2net_latest.pth` | 177 MB | `1ysEoAJNxou7RNuT9iKOxRhjVRNY5RLjx` | `gdown --id 1ysEoAJNxou7RNuT9iKOxRhjVRNY5RLjx` |

### External Model: Self-Correction Human Parsing

| File | Size | Google Drive ID | Notes |
|------|------|-----------------|-------|
| `exp-schp-201908261155-lip.pth` | 267 MB | `1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH` | Rename to `final.pth`, place in SCHP `checkpoints/` |

### External: OpenPose Models

| File | Download |
|------|----------|
| OpenPose model weights (ZIP) | Google Drive ID: `1QCSxJZpnWvM00hx49CJ2zky7PWGzpcEh` |

---

## Local Setup (Windows)

> ⚠️ **IMPORTANT:** This project was originally designed for **Google Colab (Linux with GPU)**. Running locally on Windows requires significant modifications. The instructions below guide you through adapting it.

### Prerequisites

- **Python 3.8–3.10** (recommended: 3.10)
- **NVIDIA GPU** with CUDA support (required — the models call `.cuda()`)
  - Minimum ~6 GB VRAM recommended
- **CUDA Toolkit 11.x** installed
- **Git** installed

### Step 1: Create a Virtual Environment

```powershell
cd "d:\Office Projects\Virtual\clothes-virtual-try-on"
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Step 2: Install Python Dependencies

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install torchgeometry
pip install opencv-python
pip install Pillow
pip install numpy
pip install rembg[gpu]
pip install gdown
pip install flask
pip install requests
```

### Step 3: Download Model Weights

```powershell
# Create checkpoints directory
mkdir checkpoints

# Download VITON-HD model weights
gdown --id 18q4lS7cNt1_X8ewCgya1fq0dSk93jTL6 --output checkpoints/alias_final.pth
gdown --id 1uDRPY8gh9sHb3UDonq6ZrINqDOd7pmTz --output checkpoints/gmm_final.pth
gdown --id 1d7lZNLh51Qt5Mi1lXqyi6Asb2ncLrEdC --output checkpoints/seg_final.pth

# Download U2-Net cloth segmentation weights  
gdown --id 1ysEoAJNxou7RNuT9iKOxRhjVRNY5RLjx --output cloth_segm_u2net_latest.pth
```

### Step 4: Set Up External Dependencies

#### Self-Correction Human Parsing
```powershell
cd "d:\Office Projects\Virtual"
git clone https://github.com/PeikeLi/Self-Correction-Human-Parsing.git
cd Self-Correction-Human-Parsing
mkdir checkpoints
gdown --id 1k4dllHpu0bdx38J7H28rVVLpU-kOHmnH --output checkpoints/final.pth
```

#### OpenPose (Hardest Part on Windows)
OpenPose must be built from source on Windows with CUDA. Alternatives:
- **Option A:** Use pre-built Windows binaries from the [OpenPose releases page](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases)
- **Option B:** Use a Python wrapper like `openpose-python` or `mediapipe` as a simpler alternative
- **Option C:** Use the Gradio Colab notebook instead (recommended for simplicity)

### Step 5: Prepare Input Data

```powershell
cd "d:\Office Projects\Virtual\clothes-virtual-try-on"

# Create the expected folder structure
mkdir inputs\test\cloth
mkdir inputs\test\cloth-mask
mkdir inputs\test\image
mkdir inputs\test\image-parse
mkdir inputs\test\openpose-img
mkdir inputs\test\openpose-json
```

Place your person image(s) in `inputs/test/image/` and cloth image(s) in `inputs/test/cloth/`.

### Step 6: Run the Preprocessing Steps Individually

Since `run.py` uses Colab-specific paths, run each step manually with adapted paths:

```powershell
# 1. Resize cloth images (do manually or script it)

# 2. Generate cloth masks
python cloth-mask.py  
# (You'll need to edit the hardcoded paths in this file first)

# 3. Remove backgrounds
python remove_bg.py
# (You'll need to edit the hardcoded paths in this file first)

# 4. Run human parsing
python Self-Correction-Human-Parsing/simple_extractor.py --dataset lip --model-restore Self-Correction-Human-Parsing/checkpoints/final.pth --input-dir inputs/test/image --output-dir inputs/test/image-parse

# 5. Run OpenPose (if you have it installed)
# openpose.exe --image_dir inputs/test/image/ --write_json inputs/test/openpose-json/ --display 0 --render_pose 0 --hand
# openpose.exe --image_dir inputs/test/image/ --display 0 --write_images inputs/test/openpose-img/ --hand --render_pose 1 --disable_blending true

# 6. Create test_pairs.txt
# Create a file at inputs/test_pairs.txt with: "person_image.jpg cloth_image.jpg"
```

### Step 7: Run Inference

```powershell
python test.py --name output --dataset_dir ./inputs --checkpoint_dir ./checkpoints --save_dir ./results/
```

The output images will be saved in `./results/output/`.

---

## Local Setup (Google Colab — Original)

The easiest way to run this project is via Google Colab:

1. Click the **"Open in Colab"** badge at the top of this README
2. The notebook (`setup_gradio.ipynb`) handles everything:
   - Installs all dependencies
   - Downloads all model weights
   - Builds OpenPose from source
   - Sets up Self-Correction-Human-Parsing
   - Launches a Gradio UI for easy image upload

> **Requires:** Google account with access to a GPU runtime (T4 or better)

---

## Client-Server Architecture

The project supports a client-server mode (via `setup_ngrok.ipynb`):

```
┌──────────────────────────┐         ┌──────────────────────────┐
│     CLIENT (Local PC)    │         │   SERVER (Google Colab)   │
│                          │         │                           │
│  Flask App (app.py)      │  HTTP   │  Flask + ngrok            │
│  Port 5000               ├────────►│  /api/transform endpoint  │
│                          │         │                           │
│  • Upload UI             │         │  • Receives images        │
│  • Sends cloth + model   │         │  • Runs full pipeline     │
│  • Displays result       │◄────────┤  • Returns try-on result  │
│                          │         │                           │
└──────────────────────────┘         └──────────────────────────┘
```

To use this mode:
1. Run `setup_ngrok.ipynb` on Colab (provides the backend + ngrok tunnel)
2. Copy the ngrok URL from Colab
3. Update the `url` in `client-side/app.py` with your ngrok URL
4. Run `python client-side/app.py` on your local machine
5. Open `http://localhost:5000` in your browser

---

## Troubleshooting & Common Issues

### ❌ CUDA out of memory
The models are large (~600 MB total VRAM). Try:
- Reduce `--batch_size` to 1
- Use a GPU with more VRAM (6 GB+)
- Add `torch.cuda.empty_cache()` between stages

### ❌ `torchgeometry` installation fails
```bash
pip install kornia  # Modern replacement for torchgeometry
```
You may need to update the `import torchgeometry as tgm` line in `test.py` to use `kornia` instead.

### ❌ OpenPose won't build on Windows
Use [pre-built binaries](https://github.com/CMU-Perceptual-Computing-Lab/openpose/releases) or switch to Colab.

### ❌ Hardcoded `/content/` paths
The scripts `run.py`, `cloth-mask.py`, and `remove_bg.py` have Colab-specific paths. You must update them to your local directory structure before running.

### ❌ `gdown` download fails (quota exceeded)
Google Drive has download quotas. Try:
- Wait and retry later
- Use a different Google account
- Download manually from the Google Drive links

### ❌ Output quality is poor
- Ensure person image is front-facing with visible upper body
- Use clean, well-lit photos
- Cloth images should be flat-lay (not worn)
- Both images should be in the correct resolution (768×1024)

---

## Credits & Citation

- **VITON-HD:** [High-Resolution Virtual Try-On with Misalignment and Occlusion-Free Conditions](https://github.com/shadow2496/VITON-HD) by Choi et al.
- **U²-Net:** [U2-Net: Going Deeper with Nested U-Structure for Salient Object Detection](https://github.com/xuebinqin/U-2-Net) by Qin et al.
- **Self-Correction Human Parsing:** [SCHP](https://github.com/PeikeLi/Self-Correction-Human-Parsing) by Li et al.
- **OpenPose:** [CMU Perceptual Computing Lab](https://github.com/CMU-Perceptual-Computing-Lab/openpose)

### Original Authors
This project was created as part of a **Crework community project** by Swayam, Parth, Keerthi, and Navaneth.

---

## License

Please refer to the original [VITON-HD repository](https://github.com/shadow2496/VITON-HD) for licensing terms. This project uses multiple open-source components, each with their own licenses.
