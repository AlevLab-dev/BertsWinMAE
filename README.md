# BertsWinMAE: 3D Masked Autoencoder with BERT-style Swin Transformer

**BertsWinMAE** is a hybrid 3D deep learning architecture designed for self-supervised pre-training on volumetric medical data (e.g., CT, CBCT, MRI).

It combines the local feature extraction capabilities of a **3D CNN Stem**, the long-range context modeling of a **BERT-style (non-hierarchical) Swin Transformer Encoder**, and a lightweight **3D CNN Decoder** for reconstruction.

## Key Features

  * **Hybrid Architecture:** Leverages a CNN for patch embedding to capture local structural details before passing tokens to the Transformer.
  * **BERT-style Encoding:** Unlike standard Hierarchical Swin Transformers, this model processes a full grid of tokens at a constant resolution (similar to BERT or ViT), making it ideal for Masked Autoencoding (MAE).
  * **Dynamic CNN Stem:** Automatically adapts the depth of the convolutional stem based on the provided `patch_size`.
  * **Lightweight Reconstruction:** Uses an efficient CNN decoder to upsample latent representations back to the voxel space.
  * **3D Native:** Built from the ground up for volumetric data (D, H, W).

## Installation

The model depends on standard PyTorch and Torchvision libraries.

```bash
pip install torch torchvision numpy
```

## Usage

### 1\. Initialization

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
    mask_ratio=0.75
)

# Option B: Factory Function (Base Config)
model = bertswin_mae_base_patch16_224(in_chans=1)

if torch.cuda.is_available():
    model = model.cuda()
```

### 2\. Pre-training (Masked Autoencoding)

During pre-training, the model randomly masks a portion of the input patches and attempts to reconstruct the full volume.

```python
# Create a dummy 3D volume batch: (Batch, Channel, Depth, Height, Width)
input_volume = torch.randn(2, 1, 224, 224, 224).cuda()

# Optional: Provide a mask for valid regions (e.g., Field of View)
# valid_mask = ... (Batch, N_patches) or None

# Forward pass
output = model(input_volume, valid_for_masking_mask=None)

reconstructed_cube = output['reconstructed_cube']   # (B, 1, 224, 224, 224)
mae_mask = output['mae_mask_1d']                    # (B, N_patches) - Binary mask (1=masked)

print(f"Reconstruction shape: {reconstructed_cube.shape}")
```

### 3\. Feature Extraction

For downstream tasks (segmentation, classification), you can extract dense feature representations from the encoder. This skips the masking and decoding steps.

```python
# Extract features from the encoder
features = model.extract_features(input_volume)

# Output shape: (B, Embed_Dim, Grid_D, Grid_H, Grid_W)
# For 224^3 input and patch_size=16, grid is 14^3
print(f"Encoded features: {features.shape}") 
# Expected: torch.Size([2, 768, 14, 14, 14])
```

## Configuration Parameters

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
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

## Limitations & Constraints

1.  **Input Size Divisibility:** The dimensions of `img_size` must be perfectly divisible by `patch_size`.
2.  **Power of 2 Patch Size:** The `patch_size` must be a power of 2 (e.g., 8, 16, 32) to ensure the dynamic CNN stem and decoder function correctly.
3.  **Window Size:** The `swin_window_size` cannot be larger than the calculated grid size (Input Size // Patch Size).
4.  **Non-Hierarchical:** This model is designed as a columnar (BERT-style) transformer. It does not support multi-stage downsampling within the Transformer encoder itself; all downsampling happens in the CNN stem.

