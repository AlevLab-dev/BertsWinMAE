# BertsWinMAE: 3D Masked Autoencoder with BERT-style Swin Transformer

**BertsWinMAE** is a hybrid 3D deep learning architecture designed for self-supervised pre-training on volumetric medical data (e.g., CT, CBCT, MRI).

It combines the local feature extraction capabilities of a **3D CNN Stem**, the long-range context modeling of a **BERT-style (non-hierarchical) Swin Transformer Encoder**, and a lightweight **3D CNN Decoder** for reconstruction.

## Key Features

  * **Hybrid Architecture:** Leverages a CNN for patch embedding to capture local structural details before passing tokens to the Transformer.
  * **BERT-style Encoding:** Unlike standard Hierarchical Swin Transformers, this model processes a full grid of tokens at a constant resolution (similar to BERT or ViT), making it ideal for Masked Autoencoding (MAE).
  * **Dynamic CNN Stem:** Automatically adapts the depth of the convolutional stem based on the provided `patch_size`.
  * **Lightweight Reconstruction:** Uses an efficient CNN decoder to upsample latent representations back to the voxel space.
  * **3D Native:** Built from the ground up for volumetric data (D, H, W).
  * **Strict Non-Overlapping Stem (Optional):** Offers an alternative non-overlapping CNN stem (strict_stem=True) designed to extract clean hierarchical skip-connections. This is critical for dense prediction downstream tasks (e.g., U-Net decoders) and Multi-Task Learning (MTL) with dynamic masking strategies.

## Installation

The model depends on standard PyTorch and Torchvision libraries.

```bash
pip install torch torchvision numpy

```

## Usage

### 1. Initialization

You can initialize the model with custom parameters or use the factory function for a standard base configuration.

```python
import torch
from bertswin_mae import BertsWinMAE, bertswin_mae_base_patch16_224

# Option A: Custom Initialization
model = BertsWinMAE(
    img_size=(224, 224, 224),
    patch_size=16,
    in_chans=1,
    encoder_embed_dim=768,
    encoder_depths=[12],
    encoder_num_heads=[12],
    mask_ratio=0.75,
    strict_stem=False # Set to True for dense prediction tasks and skip-connections
)

# Option B: Factory Function (Base Config)
model = bertswin_mae_base_patch16_224(in_chans=1)

if torch.cuda.is_available():
    model = model.cuda()

```

### 2. Pre-training (Masked Autoencoding)

During pre-training, the model randomly masks a portion of the input patches and attempts to reconstruct the full volume using isolated patch processing in the stem.

```python
# Create a dummy 3D volume batch: (Batch, Channel, Depth, Height, Width)
input_volume = torch.randn(2, 1, 224, 224, 224).cuda()

# Forward pass
output = model(input_volume, valid_for_masking_mask=None)

reconstructed_cube = output['reconstructed_cube']   # (B, 1, 224, 224, 224)
mae_mask = output['mae_mask_1d']                    # (B, N_patches) - Binary mask (1=masked)

```

### 3. Volumetric Inference Adaptation (Stem Tuning)

During MAE pre-training, the model processes heavily masked, isolated patches. However, for inference and downstream tasks, processing thousands of isolated patches creates massive kernel launch overhead. To maximize performance, we switch to **Volumetric Mode**, feeding the entire 3D volume into the model continuously.

Because moving from isolated patches to a continuous volume introduces boundary and padding shifts, the CNN Stem must be adapted to avoid out-of-distribution (OOD) degradation. We freeze the Transformer and fine-tune *only* the stem using `tbj_experiments/tune_cnnstem.py`.

Choose your workflow based on your downstream task:

#### Workflow A: Standard Extraction (Classification / No Skips)
*Best performance and simplest pipeline.*
1. **Pre-train:** Train the standard model (default CNN stem with padding).
2. **Tune:** Run the stem tuning script using the default stem.
3. **Inference:** Use `extract_volumetric_features`. Tests show the standard stem's padding yields slightly better representation quality when skip-connections are not required.

#### Workflow B: Dense Prediction (Segmentation / Requires Skips)
*Fastest path to segmentation without pre-training from scratch.*
1. **Pre-train:** Train the standard model (default CNN stem) for optimal SSL convergence.
2. **Tune:** Swap the stem by initializing the tuning script with `strict_stem=True`. Fine-tune this new non-overlapping stem on full volumes.
3. **Inference:** The model is now ready to output both global features and hierarchical skip-connections using `extract_volumetric_features(..., return_skips=True)`.

#### Workflow C: Dense Pre-training / MTL (Advanced)
*Use only if your pre-training phase actively requires processing a full 100% visible patch grid (e.g., localized pathology detection combined with SSL).*
1. **Pre-train:** Initialize the model with `strict_stem=True` from the start and train on dense volumetric grids.
2. **Inference:** Direct feature extraction using `extract_volumetric_features(..., return_skips=True)`.

**Running the Tuning Script:**
The tuning logic, including Gradient Conductor (GCond) optimization and patch-vs-volumetric baseline validation, is provided in the experiments folder.

```bash
# Configure your paths and target checkpoint inside the script before running
python tbj_experiments/tune_cnnstem.py

### 4. Feature Extraction

After stem tuning, the model is fully adapted for volumetric inputs. Use `extract_volumetric_features` for efficient downstream tasks (segmentation, classification).

```python
# Extract features from the encoder using the adapted volumetric stem
# Input: Full 3D Volume (B, C, D, H, W)
# Processing: Continuous convolution (no patch splitting)
features = model.extract_volumetric_features(input_volume)

# Output shape: (B, Embed_Dim, Grid_D, Grid_H, Grid_W)
# For 224^3 input and patch_size=16, grid is 14^3
print(f"Volumetric features: {features.shape}") 
# Expected: torch.Size([2, 768, 14, 14, 14])

```

### 5. Dense Prediction and Multi-Task Learning (MTL)

The original CNN stem excels at processing heavily masked isolated patches during SSL pre-training. However, for dense prediction tasks (like segmentation) or MTL setups requiring dynamic masking (e.g., 100% visible patches for fine pathologies, 25% for SSL), the model supports a strict non-overlapping stem (`strict_stem=True`).

This mode allows the extraction of hierarchical skip-connections and raw patch tokens before the Transformer encoder.

```python
# 1. Extracting Volumetric Features with Skip-Connections
# Requires initializing the model with strict_stem=True
features, skips = model.extract_volumetric_features(input_volume, return_skips=True)

print(f"Main features: {features.shape}") # (B, 768, 14, 14, 14)
print(f"Number of skip levels: {len(skips)}")

# 2. Extracting Raw Tokens for Dynamic Masking (MTL)
# Bypasses Positional Embeddings and Transformer for custom masking logic
tokens, skips = model.extract_tokens(input_volume, return_skips=True)

print(f"Raw tokens: {tokens.shape}") # (B, 2744, 768)

```

## Configuration Parameters

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `img_size` | `tuple` | `(224, 224, 224)` | Input volume dimensions (Depth, Height, Width). |
| `patch_size` | `int` | `16` | The size of the cubic patch (P). Must be a power of 2. |
| `in_chans` | `int` | `1` | Number of input channels (e.g., 1 for grayscale CT). |
| `encoder_embed_dim` | `int` | `768` | Embedding dimension for the Transformer encoder. |
| `encoder_depths` | `list` | `[12]` | Number of Swin blocks. Must be a single-element list for BERT-style. |
| `encoder_num_heads` | `list` | `[12]` | Number of attention heads. Must be a single-element list. |
| `swin_window_size` | `tuple` | `(7, 7, 7)` | The window size for Swin Attention. |
| `decoder_embed_dim` | `int` | `512` | Feature dimension before the decoder CNN. |
| `stem_base_dim` | `int` | `48` | Base channel width for the CNN patch embedding stem. |
| `mask_ratio` | `float` | `0.75` | Percentage of patches to mask during training (0.0 to 1.0). |
| `strict_stem` | `bool` | `False` | Activates the non-overlapping CNN stem, enabling `return_skips` for dense downstream tasks. |

## Limitations & Constraints

1. **Input Size Divisibility:** The dimensions of `img_size` must be perfectly divisible by `patch_size`.
2. **Power of 2 Patch Size:** The `patch_size` must be a power of 2 (e.g., 8, 16, 32) to ensure the dynamic CNN stem and decoder function correctly.
3. **Window Size:** The `swin_window_size` cannot be larger than the calculated grid size (Input Size // Patch Size).
4. **Non-Hierarchical:** This model is designed as a columnar (BERT-style) transformer. It does not support multi-stage downsampling within the Transformer encoder itself; all downsampling happens in the CNN stem.

## Citation

If you use this code or models in your research, please cite our preprint:

**BertsWin: Resolving Topological Sparsity in 3D Masked Autoencoders via Component-Balanced Structural Optimization** *Evgeny Alves Limarenko, Anastasiia Studenikina* [Preprint](https://www.arxiv.org/abs/2512.21769)

```bibtex
@article{bertswinmae2025,
  title={BertsWin: Resolving Topological Sparsity in 3D Masked Autoencoders via Component-Balanced Structural Optimization},
  author={Evgeny Alves Limarenko, Anastasiia Studenikina},
  journal={arXiv preprint arXiv:2512.21769},
  year={2025}
}