import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, List, Any

class PhysicsEngine:
    """
    Handles GPU-based 3D volumetric augmentations and Siamese preprocessing.
    
    This engine ensures anatomical consistency for bilateral TMJ analysis by mapping 
    both joints into a unified coordinate system and applying rigorous physical 
    transformations for regularization.
    """

    @staticmethod
    def get_affine_matrix(
        batch_size: int,
        device: torch.device,
        rotation_range: List[float],
        scale_range: List[float],
        translate_range_pct: List[float]
    ) -> torch.Tensor:
        """
        Generates the [B, 3, 4] affine matrix for 3D spatial manipulation via grid_sample.
        
        Mathematical Formulation:
        The affine transformation matrix M is constructed as M = R^T @ S^-1, 
        where R is the combined rotation matrix (Z @ Y @ X) and S is the scaling matrix.
        The inversion (S^-1 and R^T) is required because PyTorch's grid_sample performs 
        backward mapping (target -> source).
        
        Coordinate System:
        PyTorch grid_sample uses (x, y, z) coordinates corresponding to (Width, Height, Depth).
        """
        # 1. Rotation (Euler angles in radians)
        # We apply rotations in Z -> Y -> X order to preserve spatial coherence.
        # angles shape: [Batch, 3] -> [Z(Depth), Y(Height), X(Width)]
        angles = (torch.rand(batch_size, 3, device=device) * (rotation_range[1] - rotation_range[0]) + rotation_range[0]) * (np.pi / 180.0)
        
        cos_a, sin_a = torch.cos(angles[:, 0]), torch.sin(angles[:, 0]) # Z-axis (Depth)
        cos_b, sin_b = torch.cos(angles[:, 1]), torch.sin(angles[:, 1]) # Y-axis (Height)
        cos_g, sin_g = torch.cos(angles[:, 2]), torch.sin(angles[:, 2]) # X-axis (Width)

        # Construct Rotation Matrices
        # Rx (Rotation around X-axis)
        Rx = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        Rx[:, 1, 1], Rx[:, 1, 2] = cos_g, -sin_g
        Rx[:, 2, 1], Rx[:, 2, 2] = sin_g, cos_g

        # Ry (Rotation around Y-axis)
        Ry = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        Ry[:, 0, 0], Ry[:, 0, 2] = cos_b, sin_b
        Ry[:, 2, 0], Ry[:, 2, 2] = -sin_b, cos_b

        # Rz (Rotation around Z-axis)
        Rz = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        Rz[:, 0, 0], Rz[:, 0, 1] = cos_a, -sin_a
        Rz[:, 1, 0], Rz[:, 1, 1] = sin_a, cos_a
        
        # Combined Rotation: R = Rz @ Ry @ Rx
        R = Rz @ Ry @ Rx 

        # 2. Scaling
        # We construct the Inverse Scaling Matrix (S^-1) directly.
        # scale shape: [Batch, 3] -> [Sz, Sy, Sx]
        # Note: Single scale factor is broadcasted to 3 dimensions to maintain aspect ratio,
        # or separate scales can be used for anisotropic scaling. Here we assume isotropic range.
        single_scale = (torch.rand(batch_size, 1, device=device) * (scale_range[1] - scale_range[0]) + scale_range[0])
        scales = single_scale.repeat(1, 3)
        
        # Construct diagonal S^-1. Order corresponds to (x, y, z) -> (Width, Height, Depth)
        inv_S_diag = torch.stack([
            1.0 / scales[:, 2], # 1/Sx (Width)
            1.0 / scales[:, 1], # 1/Sy (Height)
            1.0 / scales[:, 0]  # 1/Sz (Depth)
        ], dim=1)
        inv_S = torch.diag_embed(inv_S_diag)

        # 3. Translation
        # Shift is calculated as a percentage of the dimension, then mapped to [-1, 1] grid space.
        # PyTorch grid coordinates: -1 is left/top/front, +1 is right/bottom/back.
        min_pct, max_pct = translate_range_pct[0] / 100.0, translate_range_pct[1] / 100.0
        shift_pct = torch.rand(batch_size, 3, device=device) * (max_pct - min_pct) + min_pct
        
        # Convert percentage shift to normalized grid shift (multiply by 2.0)
        translations_theta = shift_pct * 2.0 

        # 4. Combine: M = R^T @ S^-1
        inv_R = R.transpose(1, 2)
        M = inv_R @ inv_S 

        # 5. Assemble Affine Matrix [B, 3, 4]
        theta = torch.zeros(batch_size, 3, 4, device=device)
        theta[:, :3, :3] = M
        
        # Translation vector column (tx, ty, tz) -> (Width, Height, Depth)
        theta[:, 0, 3] = translations_theta[:, 2] # Tx (Width)
        theta[:, 1, 3] = translations_theta[:, 1] # Ty (Height)
        theta[:, 2, 3] = translations_theta[:, 0] # Tz (Depth)
        
        return theta

    @staticmethod
    def pad_and_crop(
        image: torch.Tensor, 
        target_size: Tuple[int, int, int], 
        pad_value: float
    ) -> torch.Tensor:
        """
        Center-crops or pads input 5D tensor [B, C, D, H, W] to target_size.
        """
        _, _, D, H, W = image.shape
        tD, tH, tW = target_size
        
        # 1. Padding (if input dimension < target dimension)
        pad_D = max(0, tD - D)
        pad_H = max(0, tH - H)
        pad_W = max(0, tW - W)
        
        # Padding tuple format for F.pad: (W_left, W_right, H_top, H_bottom, D_front, D_back)
        pad_W_pre = pad_W // 2
        pad_W_post = pad_W - pad_W_pre
        pad_H_pre = pad_H // 2
        pad_H_post = pad_H - pad_H_pre
        pad_D_pre = pad_D // 2
        pad_D_post = pad_D - pad_D_pre
        
        pads = (pad_W_pre, pad_W_post, pad_H_pre, pad_H_post, pad_D_pre, pad_D_post)
        
        if any(p > 0 for p in pads):
            image = F.pad(image, pads, mode='constant', value=pad_value)
        
        # 2. Cropping (if input dimension > target dimension)
        # Note: Use the shape of the *padded* image, as it might have grown
        _, _, pD, pH, pW = image.shape
        
        crop_D_pre = (pD - tD) // 2
        crop_H_pre = (pH - tH) // 2
        crop_W_pre = (pW - tW) // 2
        
        cropped_image = image[
            :, :, 
            crop_D_pre : crop_D_pre + tD, 
            crop_H_pre : crop_H_pre + tH, 
            crop_W_pre : crop_W_pre + tW
        ]
        
        return cropped_image

    @staticmethod
    def process_siamese_batch(
        batch: dict, 
        config: Any, 
        device: torch.device, 
        augment: bool
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Standardized preprocessing pipeline for TMJ Siamese inputs.
        
        ACADEMIC NOTE: ANATOMICAL MIRRORING & COORDINATE SYSTEMS
        1. Mirroring Strategy: The backbone encoder was pre-trained exclusively on Right TMJs.
           To leverage these learned features, all Left TMJs are mirrored along the Sagittal 
           plane (W-axis flip) to map them into the Right anatomical coordinate system.
           The model effectively "sees" two Right joints per patient.
        
        2. Masking Threshold Logic: The dataset uses a specific minimum HU value (approx -3400 HU)
           to denote background/padding, calculated via rigorous FOV analysis during dataset creation.
           However, strictly checking `img > min_hu` is unsafe due to interpolation artifacts
           (e.g., bicubic overshoot) introduced during resampling.
           We use a robust threshold `img > (min_hu * 0.95)` to guarantee the mask covers 
           all valid tissue while excluding the synthetic background.
        """
        

        def _proc_side(img_tensor: torch.Tensor, min_hu_tensor: torch.Tensor, flip_sagittal: bool):
            # 1. Move to Device and Add Channel Dim
            # Input: [B, D, H, W] -> Output: [B, 1, D, H, W]
            img = img_tensor.to(device).float().unsqueeze(1) 
            min_hu = min_hu_tensor.to(device).float().view(-1, 1, 1, 1, 1)
            
            # 2. Anatomical Mirroring (Left -> Right Mapping)
            # We flip along the last dimension (Width/Sagittal axis)
            if flip_sagittal: 
                img = torch.flip(img, dims=[-1])
            
            # 3. Dynamic Background Masking
            # We calculate a binary mask where data exists. 
            # Threshold is set to 95% of the recorded air/background HU value to be robust.
            mask = img > (min_hu * 0.95)
            
            # 4. Augmentation (Rotate, Scale, Translate)
            if augment:
                theta = PhysicsEngine.get_affine_matrix(
                    batch_size=img.shape[0], 
                    device=device,
                    rotation_range=config.AUG_ROTATION_DEGREES,
                    scale_range=config.AUG_SCALE_RANGE, # Ensure config uses [0.9, 1.1] style list
                    translate_range_pct=config.AUG_TRANSLATE_PERCENT_RANGE
                )
                
                # Create sampling grid
                grid = F.affine_grid(theta, img.size(), align_corners=False)
                
                # Apply transformation to Image (Bilinear interpolation)
                img = F.grid_sample(img, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
                
                # Apply transformation to Mask (Nearest neighbor to preserve boolean nature)
                mask_float = F.grid_sample(mask.float(), grid, mode='nearest', padding_mode='zeros', align_corners=False)
                
                # 5. Background Cleaning
                # After rotation/translation, zero-padding enters the frame. 
                # We must ensure these new areas are set to the minimum HU (air), not 0 (which is water/tissue).
                mask_bool = (mask_float > 0.5)
                
                # Where the mask is valid, keep the image. Elsewhere, set to HU_CLIP_MIN.
                img = torch.where(mask_bool, img, torch.tensor(config.HU_CLIP_MIN, device=device))

                mask = mask_bool.float()
            else:
                mask = mask.float()
            
            # 6. Geometric Standardization (Pad/Crop to fixed Cube)
            img = PhysicsEngine.pad_and_crop(img, config.TARGET_IMAGE_SIZE, config.HU_CLIP_MIN)
            mask = PhysicsEngine.pad_and_crop(mask, config.TARGET_IMAGE_SIZE, 0.0)
            
            # 7. Intensity Normalization (Z-Score)
            # Clamp outliers first, then normalize
            img_clamped = img.clamp(min=config.HU_CLIP_MIN, max=config.HU_CLIP_MAX)
            norm_img = (img_clamped - config.DATASET_MEAN) / config.DATASET_STD
            
            return norm_img, mask

        # Process Left Joint (Flip = True)
        x, mask = _proc_side(batch['img_L'], batch['min_hu_L'], flip_sagittal=True)
        
        # Process Right Joint (Flip = False)
        x_pair, mask_pair = _proc_side(batch['img_R'], batch['min_hu_R'], flip_sagittal=False)
        
        return x, mask, x_pair, mask_pair