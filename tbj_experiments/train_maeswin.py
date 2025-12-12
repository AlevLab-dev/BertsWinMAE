import os
import sys
import h5py
import hdf5plugin
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import numpy as np
import logging
import shutil
import math
import json
import random
from pathlib import Path
from tqdm import tqdm
from typing import Optional, Tuple, Dict, Any, List
from collections import defaultdict

from scipy.ndimage import affine_transform
from scipy.spatial.transform import Rotation

import torchvision
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset, DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler

from functools import partial

try:
    parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    if parent_dir not in sys.path:
        sys.path.append(parent_dir)
    from bertswin_mae import BertsWinMAE, patchify_image #BertsWinMAE.
except ImportError:
    raise ImportError("Warning: Failed to import BertsWinMAE")


from monai.networks.nets import MaskedAutoEncoderViT
from monai.networks.nets import ViT as MonaiViT

# Set NCCL environment variables for stable DDP performance
os.environ.setdefault("TORCH_NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_NCCL_ASYNC_ERROR_HANDLING", "1")

# =================================================================================
# Config
# =================================================================================

class Config:
    """
    Configuration class for the MAE-Swin training pipeline.
    
    This class centralizes all hyperparameters, paths, and settings for the 
    experiment. Parameters are grouped by function (e.g., paths, data, model 
    architecture) for clarity.
    """

    ## --------------------------------------------------------------------
    ## 1. Path and Experiment Settings
    ## --------------------------------------------------------------------
    # Path to the main HDF5 dataset file
    #DATA_HDF5_PATH = './tbj_cbct.hdf5'
    DATA_HDF5_PATH = '/data/datasets/tbj_cbct.hdf5'
    # Base directory for all experiment outputs
    EXPERIMENTS_DIR = Path('./')
    # Unique name for this experiment run (monai_baseline/bertswin_adam)
    RUN_NAME = 'bertswin_adam_l2_nofov'

    # --- Data Processing and Augmentation ---
    
    # Target input cube dimensions (Z, Y, X) after preprocessing
    INPUT_CUBE_SIZE = (224, 224, 224)
    # Validation split ratio
    VAL_SPLIT_RATIO = 0.10
    # Global random seed for reproducibility
    SEED = 42

    ## --------------------------------------------------------------------
    ## 2. Dataset and Preprocessing Parameters
    ## --------------------------------------------------------------------
    
    # Radius of the CBCT Field of View (FOV) in voxels. Used for FOV masking.
    FOV_RADIUS_VOXELS = 440.0
    # Expected shape (Z, Y, X) of the ROI loaded from the HDF5 file.
    ROI_SHAPE = (400, 400, 400)
    # Pre-calculated Hounsfield Unit (HU) clipping bounds for the dataset
    HU_CLIP_MIN = -1011.
    HU_CLIP_MAX = 1992.0
    # Z-normalization statistics, pre-calculated from the dataset
    DATASET_MEAN = -21.14
    DATASET_STD = 578.02

    # --- Geometric Augmentation Settings ---

    # "Safe zone" (Z,Y,X) around the joint center for augmentation cropping
    CORE_SAFE_ZONE_VOXELS = np.array([140, 140, 140])
    # Min/max rotation angle (in degrees) for augmentation
    AUG_ROTATION_DEGREES = [-12, 12]
    # Min/max random scaling factor for augmentation
    AUG_SCALE = [0.9, 1.1]

    ## --------------------------------------------------------------------
    ## 3. Model Architecture Settings
    ## --------------------------------------------------------------------

    # Specifies the model architecture: 'bertsWin' (our proposed model) 
    # or 'monai_vit' (for baseline comparison).
    MODEL_TYPE = 'bertsWin'

    # --- MAE Masking Strategy ---
    # The effective patch size (voxel dimensions) processed by the encoder (3.2mm)
    EFFECTIVE_PATCH_SIZE = 16
    # Flag to enable/disable FOV-aware masking.
    # When True, the model avoids masking patches that are mostly outside
    # the cylindrical FOV.
    # NOTE: This must be set to False for a fair comparison against the
    # MONAI baseline (which is FOV-agnostic)
    USE_FOV_AWARE_MASKING = False
    # Percentage of voxels a patch must have inside the FOV to be
    # considered "valid" for masking (if USE_FOV_AWARE_MASKING is True)
    PATCH_MASK_FOV_THRESHOLD = 0.95
    # Percentage of voxels a patch must contain (e.g., tissue) to be
    # included in physics-informed loss calculations
    PATCH_MASK_TISSUE_THRESHOLD = 0.5
    # The ratio of patches to be masked during MAE pre-training
    MAE_MASK_RATIO = 0.75 # 75%

    # --- 3.1. MONAI ViT-Base Baseline Parameters ---
    # Patch size for the MONAI ViT
    MONAI_VIT_PATCH_SIZE = (16, 16, 16)
    # ViT-Base embedding dimension
    MONAI_VIT_EMBED_DIM = 768
    # ViT-Base number of transformer layers
    MONAI_VIT_DEPTH = 12
    # ViT-Base number of attention heads
    MONAI_VIT_NUM_HEADS = 12
    # ViT-Base MLP dimension
    MONAI_VIT_MLP_DIM = 3072

    # --- 3.2. Custom Swin-MAE (Our Model) Parameters ---
    # Embedding dimension for the Swin encoder
    ENCODER_EMBED_DIM = 768
    # Window size (Z, Y, X) for 3D shifted window attention
    SWIN_WINDOW_SIZE = (7,7,7)
    # Number of Swin blocks in the encoder stage
    SWIN_DEPTHS = [12]
    # Number of attention heads in the encoder stage
    SWIN_NUM_HEADS = [12]
    # Embedding dimension for the lightweight CNN decoder
    DECODER_EMBED_DIM = 512
    STEM_BASE_DIM = 48

    ## --------------------------------------------------------------------
    ## 4. Training and Optimization Parameters
    ## --------------------------------------------------------------------
    LEARNING_RATE = 1.5e-4
    EPOCHS = 1000

    # Number of *optimizer steps* per epoch. This creates a "virtual" epoch
    # of a fixed length, regardless of the true dataset size
    TRAIN_STEPS_PER_EPOCH = 300

    # Batch size per GPU
    GPU_BATCH_SIZE_TRAIN = 10
    GPU_BATCH_SIZE_VAL = 10
    # Gradient accumulation steps.
    # Effective global batch size = GPU_BATCH_SIZE_TRAIN * WORLD_SIZE * ACCUMULATION_STEPS
    ACCUMULATION_STEPS = 5
    # Max gradient norm for gradient clipping
    GRADIENT_CLIP_VAL = 1.0
    # Dataloader workers
    NUM_WORKERS_TRAIN = GPU_BATCH_SIZE_TRAIN
    NUM_WORKERS_VAL = GPU_BATCH_SIZE_VAL

    # --- AdamW Optimizer Settings ---
    ADAMW_WEIGHT_DECAY = 0.05
    ADAMW_BETA1 = 0.9
    ADAMW_BETA2 = 0.95
    # --- Learning Rate Scheduler Settings ---
    SCHEDULER_TYPE = 'Cosine'
    # Warmup duration in epochs
    WARMUP_EPOCHS = 20
    COSINE_ETA_MIN = 1e-6
    # Early stopping patience (in "virtual" epochs)
    PATIENCE_EPOCHS = 50

    ## --------------------------------------------------------------------
    ## 5. Loss Function Settings
    ## --------------------------------------------------------------------
    # Loss function for training: 'l2' (MSE) or 'custom_ssim' (our LCS loss)
    TRAIN_LOSS_TYPE = 'l2'
    # The specific metric key from the validation output dict used for
    # checkpointing and early stopping (e.g., 'l2_final_loss')
    VALIDATION_PRIMARY_LOSS = 'l2_final_loss'

    # Weights for the 'custom_ssim' (LCS) loss components: (Brightness, Contrast, Structure)
    LCS_LOSS_WEIGHTS = (0.3, 0.2, 0.5)
    # Size of the sub-patches used within the LCS loss calculation
    LOSS_SUB_PATCH_SIZE = 8
    # Weights for the physics-informed loss (if used): (Total, Soft Tissue, Surface)
    PHY_LOSS_WEIGHTS = (0.3,0.5, 0.2)

    ## --------------------------------------------------------------------
    ## 6. Intensity Augmentation Settings (HU Warping and CT Noise)
    ## --------------------------------------------------------------------
    USE_HU_AUG = False
    USE_NOISE_AUG = False
    HU_AUG_PROBABILITY = 0.8
    NOISE_AUG_PROBABILITY = 0.5
    HU_AUG_START_EPOCH = 0
    NOISE_AUG_START_EPOCH = 0
    # --- 6.1. CT Noise Simulation Parameters ---
    DOSE_LEVEL_RANGE = (8e2, 9e3)
    KERNEL_SIGMA_SHARP_RANGE = (0.7, 1.2)
    KERNEL_SIGMA_SMOOTH_RANGE = (1.8, 2.5)
    MIX_RATIO_RANGE = (0.4, 0.8)
    NOISE_DENSITY_SENSITIVITY = 0.0015
    ELECTRONIC_NOISE_Z_STD = 0.0015
    # --- 6.2. HU Warping Parameters ---
    # Defines anchor points (HU value: max_shift_in_HU) for
    # piecewise-linear intensity warping
    HU_AUG_ANCHORS = {
        -500: 0,
        -80:  25,
        15:   10,
        45:   8,
        90:   6,
        1000: 20,
        1500: 0
    }

    ## --------------------------------------------------------------------
    ## 7. Physics-Informed Mask Parameters
    ## --------------------------------------------------------------------
    # Voxel erosion kernel size for the inner bone surface boundary
    SURFACE_SHELL_INNER_VOXELS = 2 
    # Voxel dilation kernel size for the outer bone surface boundary
    SURFACE_SHELL_OUTER_VOXELS = 4 

    ## --------------------------------------------------------------------
    ## 9. Derived Parameters (Calculated, Do Not Edit Manually)
    ## --------------------------------------------------------------------
    # These parameters are derived from the base settings defined above.
    # --- Derived Paths ---
    
    CHECKPOINT_DIR = EXPERIMENTS_DIR / Path(RUN_NAME) / Path('checkpoints')
    LOG_DIR = EXPERIMENTS_DIR / Path(RUN_NAME) / Path('logs')
    # NumPy array representation of the input cube size.
    CUBE_SIZE_VOXELS = np.array(INPUT_CUBE_SIZE)

    # The 3D grid size of patches/tokens fed to the transformer.
    SWIN_GRID_SIZE = (
        INPUT_CUBE_SIZE[0] // EFFECTIVE_PATCH_SIZE,
        INPUT_CUBE_SIZE[1] // EFFECTIVE_PATCH_SIZE,
        INPUT_CUBE_SIZE[2] // EFFECTIVE_PATCH_SIZE
    )
    # Z-normalized threshold for soft tissue mask (lower bound).
    Z_NORM_SOFT_TISSUE_MIN = (-300 - DATASET_MEAN) / DATASET_STD
    
    # Z-normalized threshold for soft tissue mask (upper bound).
    Z_NORM_SOFT_TISSUE_MAX = (300 - DATASET_MEAN) / DATASET_STD
    
    ENABLE_DEBUG = False

# =================================================================================
# Utils
# =================================================================================

def setup_ddp():
    """Initializes the Distributed Data Parallel (DDP) process group"""
    if 'RANK' not in os.environ:
        return 0, 1
    dist.init_process_group("nccl")
    return int(os.environ["RANK"]), int(os.environ["WORLD_SIZE"])

def set_seed(seed: int, rank: int = 0):
    """
    Sets random seeds for reproducibility across all relevant libraries.
    
    Ensures that Python's `random`, `numpy`, and `torch` all use a
    consistent seed. In a distributed setup, each rank receives a
    unique seed (`seed + rank`) to ensure different augmentations
    for each process.
    """
    seed_for_rank = seed + rank
    random.seed(seed_for_rank)
    np.random.seed(seed_for_rank)
    torch.manual_seed(seed_for_rank)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_for_rank)
        torch.cuda.manual_seed_all(seed_for_rank)
        # Note: Using deterministic algorithms can impact performance.
        # These are disabled by default but can be enabled for debugging
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

def setup_logging(rank: int, log_dir: Path):
    """
    Configures logging for the experiment.
    
    Only the main process (rank 0) will log to stdout and a file.
    All other processes will have logging suppressed to avoid clutter
    """
    log_dir.mkdir(exist_ok=True, parents=True)
    if rank == 0:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s [%(levelname)s] %(message)s",
            handlers=[logging.FileHandler(log_dir / "training.log", mode='a'), logging.StreamHandler(sys.stdout)]
        )
    else:
        logging.basicConfig(level=logging.WARNING, handlers=[logging.NullHandler()])

def log_config(config: Config):
    """
    Logs the effective configuration, showing default and overridden parameters.
    """
    logging.info("="*25 + " EFFECTIVE CONFIGURATION " + "="*25)
    
    # Get all class attributes (default config)
    default_config = {
        key: value for key, value in Config.__dict__.items()
        if not key.startswith('__') and not callable(value)
    }

    # Get instance-specific overrides
    overrides = config.__dict__
    
    # Combine and sort keys for consistent output
    all_keys = sorted(default_config.keys())
    max_key_length = max(len(key) for key in all_keys) if all_keys else 0

    for key in all_keys:
        value = getattr(config, key)
        marker = "[overridden]" if key in overrides else "[default]   "
        padding = " " * (max_key_length - len(key))
        logging.info(f"{key}:{padding} = {value} {marker}")
        
    logging.info("="*75)

def atomic_save(checkpoint_data: dict, path: Path):
    """
    Atomically saves a checkpoint to a file to prevent corruption.

    This function first saves the data to a temporary file. If the save is
    successful, it then moves the temporary file to the final destination path.
    This ensures that the original checkpoint file is not corrupted if the
    program crashes during the save operation.

    Args:
        checkpoint_data (dict): The dictionary containing the checkpoint data.
        path (Path): The final destination path for the checkpoint.
    """
    tmp_path = path.with_suffix(".tmp")
    torch.save(checkpoint_data, tmp_path)
    shutil.move(str(tmp_path), str(path))

class EarlyStopping:
    """
    Implements early stopping to halt training when validation loss stops improving.

    Monitors the validation loss and stops the training process if the loss does
    not improve (decrease) by more than `delta` for a specified number of
    `patience` epochs.
    """
    def __init__(self, patience: int = 7, delta: float = 0):
        """
        Args:
            patience (int): How many epochs to wait for improvement before stopping.
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
        """
        self.patience, self.delta = patience, delta
        self.counter, self.best_score = 0, None
        self.early_stop, self.val_loss_min = False, float('inf')

    def __call__(self, val_loss: float):
        """
        Updates the state of the early stopper based on the current validation loss.

        Args:
            val_loss (float): The current validation loss.
        """
        score = -val_loss
        if self.best_score is None:
            self.best_score, self.val_loss_min = score, val_loss
        elif score < self.best_score + self.delta:
            self.counter += 1
            logging.info(f'[EarlyStopping] Counter: {self.counter} of {self.patience}')
            if self.counter >= self.patience: self.early_stop = True
        else:
            self.best_score, self.val_loss_min = score, val_loss
            self.counter = 0

    def state_dict(self): return self.__dict__
    def load_state_dict(self, state_dict): self.__dict__.update(state_dict)


def reduce_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """
    Averages a tensor across all DDP processes.
    
    Args:
        tensor (torch.Tensor): The tensor to average.

    Returns:
        torch.Tensor: The averaged tensor (same shape as input).
    """
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= dist.get_world_size()
    return rt

class HounsfieldUnitInterpolator(nn.Module):
    """
    Applies piecewise-linear warping to Hounsfield Unit (HU) values on the GPU.

    This module performs an intensity-based augmentation by randomly shifting
    pre-defined HU anchor points. It then uses vectorized piecewise-linear
    interpolation to create realistic, non-linear variations in tissue
    intensity across the entire 3D volume. All operations are performed
    on Z-normalized data.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.p = cfg.HU_AUG_PROBABILITY

        def hu_to_z_norm(hu_value: float) -> float:
            hu_clipped = max(cfg.HU_CLIP_MIN, min(hu_value, cfg.HU_CLIP_MAX))
            val_01 = (hu_clipped - cfg.HU_CLIP_MIN) / (cfg.HU_CLIP_MAX - cfg.HU_CLIP_MIN)
            return (val_01 - cfg.DATASET_MEAN) / cfg.DATASET_STD

        def hu_shift_to_z_range(hu_shift_amount: float) -> float:
            full_range = cfg.HU_CLIP_MAX - cfg.HU_CLIP_MIN
            return (hu_shift_amount / full_range) / cfg.DATASET_STD

        # Read anchor points and shift ranges from the config
        # Sort keys to ensure anchors are in ascending order
        sorted_anchors = sorted(cfg.HU_AUG_ANCHORS.items())

        # Convert original HU anchor points to the Z-normalized space
        original_hu_points = [float(k) for k, v in sorted_anchors]
        self.register_buffer(
            "original_z_anchors",
            torch.tensor([hu_to_z_norm(hu) for hu in original_hu_points], dtype=torch.float32)
        )
        
        # Convert HU shifts to Z-normalized shift ranges
        hu_shifts = [float(v) for k, v in sorted_anchors]
        self.register_buffer(
            "shift_ranges",
            torch.tensor([hu_shift_to_z_range(shift) for shift in hu_shifts], dtype=torch.float32)
        )

        logging.info(f"   [HU Aug] Original Z-norm anchors: {self.original_z_anchors.numpy().round(3)}")
        logging.info(f"   [HU Aug] Z-norm shift ranges: {self.shift_ranges.numpy().round(3)}")


    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the random HU warping to an input batch.

        Args:
            batch (torch.Tensor): The input batch of 3D volumes,
                                  shape (B, C, D, H, W).

        Returns:
            torch.Tensor: The augmented batch, same shape as input.
        """
        # Check if augmentation should be applied
        if torch.rand(1).item() > self.p:
            return batch

        B, C, D, H, W = batch.shape
        device = batch.device
        
        # 1. Generate random shifts for the anchor points
        # shifts are in range [-shift_range, +shift_range]
        shifts = (torch.rand(B, len(self.shift_ranges), device=device) * 2 - 1) * self.shift_ranges
        
        # 2. Create the "new" (shifted) anchor points for each item in the batch
        new_z_anchors = self.original_z_anchors + shifts
        # Ensure the new anchors remain monotonically increasing
        new_z_anchors, _ = torch.sort(new_z_anchors, dim=1)

        # 3. Perform fully vectorized interpolation
        
        # Find which interval each voxel falls into indices will have the same shape as batch
        indices = torch.searchsorted(
            self.original_z_anchors, 
            batch.contiguous(), 
            right=True
        ).clamp(min=1, max=len(self.original_z_anchors)-1)

        # Expand anchor tensors from (B, N_anchors) to (B, N_anchors, D, H, W)
        # to match the shape of indices for use with torch.gather
        expanded_new_anchors = new_z_anchors.view(B, -1, 1, 1, 1).expand(B, -1, D, H, W)
        expanded_orig_anchors = self.original_z_anchors.view(1, -1, 1, 1, 1).expand(B, -1, D, H, W)

        # Use gather to select the correct "start" and "end" anchor points
        # for every single voxel in the batch simultaneously
        idx1 = indices - 1
        idx2 = indices
        
        x1 = torch.gather(expanded_orig_anchors, 1, idx1)
        x2 = torch.gather(expanded_orig_anchors, 1, idx2)
        y1 = torch.gather(expanded_new_anchors, 1, idx1)
        y2 = torch.gather(expanded_new_anchors, 1, idx2)
        
        # 4. Calculate the interpolation coefficient t
        # t = (value - start) / (end - start)
        t = (batch - x1) / (x2 - x1 + 1e-8)
        
        # 5. Apply linear interpolation
        augmented_batch = y1 + t * (y2 - y1)

        return augmented_batch

class CTNoiseSimulatorGPU(nn.Module):
    """
    Applies a physics-informed CT noise model to 3D volumes on the GPU.

    This module simulates realistic CBCT noise by combining three components:
    1.  **Amplitude Model:** A spatially-varying noise magnitude that depends on
        local tissue density (HU value) and a simulated X-ray dose level.
        Noise is higher in dense regions (e.g., bone) and lower at high doses.
    2.  **Texture Model:** A dual-component correlated noise field, created by
        mixing two 3D Gaussian-blurred noise fields (a "sharp" high-frequency
        component and a "smooth" low-frequency component) to replicate
        complex scanner-specific noise textures.
    3.  **Electronic Noise:** A simple, spatially-invariant Gaussian noise
        to simulate baseline sensor noise.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.p = cfg.NOISE_AUG_PROBABILITY
        self.debug = cfg.ENABLE_DEBUG

        # Store log-space dose ranges for uniform sampling in log-space
        self.log_dose_min = math.log(cfg.DOSE_LEVEL_RANGE[0])
        self.log_dose_max = math.log(cfg.DOSE_LEVEL_RANGE[1])
        
        # Store texture model parameters
        self.sigma_sharp_min, self.sigma_sharp_max = cfg.KERNEL_SIGMA_SHARP_RANGE
        self.sigma_smooth_min, self.sigma_smooth_max = cfg.KERNEL_SIGMA_SMOOTH_RANGE
        self.mix_min, self.mix_max = cfg.MIX_RATIO_RANGE
        
        self.electronic_z_std = cfg.ELECTRONIC_NOISE_Z_STD

        # --- Pre-calculate coefficients for the amplitude model ---
        # This avoids redundant calculations in the forward pass.
        # The model is: Noise_HU = exp(B * (HU_raw)) / sqrt(Dose)
        # We operate in Z-normed space, so we must convert
        B = cfg.NOISE_DENSITY_SENSITIVITY
        self.std = cfg.DATASET_STD
        mean = cfg.DATASET_MEAN
        hu_range = cfg.HU_CLIP_MAX - cfg.HU_CLIP_MIN
        hu_min = cfg.HU_CLIP_MIN
        z_sensitivity = B * self.std * hu_range
        self.register_buffer("z_sensitivity", torch.tensor(z_sensitivity, dtype=torch.float32))
        base_exponent = B * (mean * hu_range + hu_min)
        base_scale = math.exp(base_exponent)
        self.register_buffer("base_scale", torch.tensor(base_scale, dtype=torch.float32))
    
    def gaussian_blur_3d(self, tensor: torch.Tensor, kernel_size: int, sigma: float) -> torch.Tensor:
        """
        Applies a separable 3D Gaussian blur to a (B, C, D, H, W) tensor.

        This is more efficient than a full 3D convolution by applying
        three 1D convolutions sequentially.
        """
        # Create a 1D Gaussian kernel
        ax = torch.arange(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0, device=tensor.device)
        xx = torch.exp(-0.5 * (ax**2 / sigma**2))
        kernel_1d = xx / xx.sum()
        
        # Create 3D kernels for each axis (as 1D kernels)
        kernel_x = kernel_1d.view(1, 1, 1, 1, -1)
        kernel_y = kernel_1d.view(1, 1, 1, -1, 1)
        kernel_z = kernel_1d.view(1, 1, -1, 1, 1)
        
        # Apply blur separable-ly
        padding = kernel_size // 2
        
        # Get channel count for grouped convolution
        C = tensor.shape[1]
        kernel_x = kernel_x.repeat(C, 1, 1, 1, 1)
        kernel_y = kernel_y.repeat(C, 1, 1, 1, 1)
        kernel_z = kernel_z.repeat(C, 1, 1, 1, 1)

        blurred = F.conv3d(tensor, kernel_z, padding=(padding, 0, 0), groups=C)
        blurred = F.conv3d(blurred, kernel_y, padding=(0, padding, 0), groups=C)
        blurred = F.conv3d(blurred, kernel_x, padding=(0, 0, padding), groups=C)
        
        return blurred

    @torch.no_grad()
    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        """
        Applies the physics-informed noise model to the input batch.

        Args:
            batch (torch.Tensor): The input batch of clean 3D volumes,
                                  shape (B, C, D, H, W).

        Returns:
            torch.Tensor: The noisy batch, same shape as input.
        """
        if torch.rand(1).item() > self.p:
            return batch

        B, C, D, H, W = batch.shape
        device = batch.device

        # Sample a random dose level for each item in the batch
        log_dose = torch.rand(B, 1, 1, 1, 1, device=device) * (self.log_dose_max - self.log_dose_min) + self.log_dose_min
        dose_level = torch.exp(log_dose)

        # Sample texture parameters
        sigma_sharp = torch.rand(1, device=device).item() * (self.sigma_sharp_max - self.sigma_sharp_min) + self.sigma_sharp_min
        sigma_smooth = torch.rand(1, device=device).item() * (self.sigma_smooth_max - self.sigma_smooth_min) + self.sigma_smooth_min
        mix_ratio = torch.rand(1, device=device).item() * (self.mix_max - self.mix_min) + self.mix_min

        # --- 1. Amplitude Model Calculation ---
        # Calculate the spatially-varying noise amplitude based on density
        exponent = torch.clamp(batch * self.z_sensitivity, max=15.0) 
        noise_magnitude_hu = self.base_scale * torch.exp(exponent)
        scaled_magnitude_hu = noise_magnitude_hu / torch.sqrt(dose_level)
        final_magnitude_z = scaled_magnitude_hu / self.std

        # --- 2. Complex Noise Texture Generation ---
        # Create two independent base noise fields
        base_noise_hf = torch.randn_like(batch)
        base_noise_lf = torch.randn_like(batch)

        # Create the "sharp" (high-frequency) component
        kernel_size_sharp = int(2 * round(2.5 * sigma_sharp) + 1)
        kernel_size_sharp += 1 if kernel_size_sharp % 2 == 0 else 0
        correlated_noise_sharp = self.gaussian_blur_3d(
            base_noise_hf, kernel_size=kernel_size_sharp, sigma=sigma_sharp
        )

        # Create the "smooth" (low-frequency) component
        kernel_size_smooth = int(2 * round(2.5 * sigma_smooth) + 1)
        kernel_size_smooth += 1 if kernel_size_smooth % 2 == 0 else 0
        correlated_noise_smooth = self.gaussian_blur_3d(
            base_noise_lf, kernel_size=kernel_size_smooth, sigma=sigma_smooth
        )
        # Linearly mix the two texture components
        final_noise_texture = mix_ratio * correlated_noise_sharp + (1 - mix_ratio) * correlated_noise_smooth
        
        # --- 3. Final Noise Assembly ---
        # Modulate the noise texture by the spatial amplitude map
        quantum_noise = final_noise_texture * final_magnitude_z
        # Add constant-magnitude electronic noise
        electronic_noise = torch.randn_like(batch) * self.electronic_z_std
        total_noise = quantum_noise + electronic_noise
        
        noisy_batch = batch + total_noise

        if self.debug:
            params = {
                'dose': dose_level.squeeze().item(),
                'sigma_sharp': sigma_sharp,
                'sigma_smooth': sigma_smooth,
                'mix_ratio': mix_ratio
            }
            return noisy_batch, params

        return noisy_batch

def hdf5_worker_init_fn(worker_id: int):
    """
    Initializer function for DataLoader workers to handle HDF5 files.

    The `h5py` library is not fork-safe, which causes issues when using
    multiple workers in a PyTorch DataLoader. This function is called
    for each worker process. It ensures that:
    1. Each worker opens its own independent file handle to the HDF5 dataset.
    2. The `numpy` random seed is correctly initialized for this worker,
       which is crucial for any CPU-based augmentations in `__getitem__`.
       
    Args:
        worker_id (int): The ID of the worker process.
    """
    try:
        worker_info = torch.utils.data.get_worker_info()
        dataset = worker_info.dataset  # Get the Dataset instance for this worker
        
        # Each worker opens its own HDF5 file handle
        dataset.h5_file = h5py.File(dataset.hdf5_path, 'r')
        
        # Set the numpy seed for this worker based on the seed
        # provided by the DataLoader. This is essential for
        # reproducible CPU-based augmentations (if any).
        np.random.seed(worker_info.seed % (2**32))
        
    except Exception as e:
        print(f"Error in hdf5_worker_init_fn (worker {worker_id}): {e}")

# =================================================================================
# Dataset
# =================================================================================

class TMJDataset(Dataset):
    """
    Optimized and fault-tolerant dataset for TMJ (Temporomandibular Joint) CBCT volumes.

    Features:
        - Loads data from HDF5 efficiently.
        - Applies tricubic interpolation for high-quality geometric transformations.
        - Generates FOV (Field of View) masks efficiently using affine transformations.
        - Handles bounding box calculations for rotation/scaling augmentations.
    """
    def __init__(self, hdf5_path: str, joint_identifiers: List[Tuple[str, str]], cfg: object):
        self.hdf5_path = hdf5_path
        self.joint_identifiers = joint_identifiers
        self.cfg = cfg
        self.h5_file = None

        self.max_bbox_size = self._calculate_max_bounding_box()

    def __len__(self):
        return len(self.joint_identifiers)
    
    def _get_rotated_bounding_box_size(self, rotation_matrix: np.ndarray, base_size: np.ndarray) -> np.ndarray:
        """Calculates the bounding box size required to contain a rotated volume."""
        half_size = base_size / 2.0
        corners = np.array([
            [-1, -1, -1], [-1, -1, 1], [-1, 1, -1], [-1, 1, 1],
            [1, -1, -1], [1, -1, 1], [1, 1, -1], [1, 1, 1]
        ]) * half_size
        
        rotated_corners = rotation_matrix @ corners.T
        min_coords = rotated_corners.min(axis=1)
        max_coords = rotated_corners.max(axis=1)
        
        # Round up to the next even number for symmetric padding
        bbox_size = np.ceil(max_coords - min_coords).astype(int)
        bbox_size += bbox_size % 2 
        return bbox_size
    
    def _calculate_max_bounding_box(self) -> np.ndarray:
        """Calculates the maximum possible bounding box size across all augmentations."""
        max_abs_angle_deg = max(abs(x) for x in self.cfg.AUG_ROTATION_DEGREES)

        min_scale = self.cfg.AUG_SCALE[0]
        base_size_scaled = np.ceil(self.cfg.CUBE_SIZE_VOXELS / min_scale).astype(int)
        base_size_scaled += base_size_scaled % 2
        
        # Consider rotation around each axis independently to find the upper bound
        r_x = Rotation.from_euler('x', max_abs_angle_deg, degrees=True).as_matrix()
        r_y = Rotation.from_euler('y', max_abs_angle_deg, degrees=True).as_matrix()
        r_z = Rotation.from_euler('z', max_abs_angle_deg, degrees=True).as_matrix()

        size_x = self._get_rotated_bounding_box_size(r_x, base_size_scaled)
        size_y = self._get_rotated_bounding_box_size(r_y, base_size_scaled)
        size_z = self._get_rotated_bounding_box_size(r_z, base_size_scaled)
        
        # The final size is the maximum along each axis
        max_size = np.maximum.reduce([base_size_scaled, size_x, size_y, size_z])
        max_size += max_size % 2
        return max_size.astype(int)

    def _get_transform_matrices(self, roi_shape: np.ndarray, tmj_center: np.ndarray, roi_offset: np.ndarray):
        # 1. Calculate valid sampling space
        half_cube = self.cfg.CUBE_SIZE_VOXELS / 2.0
        min_by_roi = half_cube
        max_by_roi = roi_shape - half_cube
        
        fov_center_in_roi = roi_shape / 2.0 - roi_offset
        # Use full FOV radius instead of inscribed square for flexibility
        min_by_fov_yx = fov_center_in_roi[1:] - self.cfg.FOV_RADIUS_VOXELS
        max_by_fov_yx = fov_center_in_roi[1:] + self.cfg.FOV_RADIUS_VOXELS
        
        wiggle_room = (self.cfg.CUBE_SIZE_VOXELS - self.cfg.CORE_SAFE_ZONE_VOXELS) / 2.0
        min_by_core = tmj_center - wiggle_room
        max_by_core = tmj_center + wiggle_room
        
        final_min = np.maximum(np.maximum(min_by_roi, min_by_core), 
                               np.array([-np.inf, min_by_fov_yx[0], min_by_fov_yx[1]]))
        final_max = np.minimum(np.minimum(max_by_roi, max_by_core),
                               np.array([np.inf, max_by_fov_yx[0], max_by_fov_yx[1]]))

        if np.any(final_min >= final_max):
            target_cube_center = tmj_center
        else:
            target_cube_center = np.random.uniform(final_min, final_max)

        # 2. Generate random augmentation parameters
        random_angles = np.random.uniform(self.cfg.AUG_ROTATION_DEGREES[0], self.cfg.AUG_ROTATION_DEGREES[1], 3)
        rotation = Rotation.from_euler('zyx', random_angles, degrees=True)
        scale_factor = np.random.uniform(self.cfg.AUG_SCALE[0], self.cfg.AUG_SCALE[1])

        return rotation, target_cube_center, scale_factor
    

    def __getitem__(self, index):
        
        patient_id, joint_side = self.joint_identifiers[index]
        study_group = self.h5_file[patient_id]
        
        if joint_side not in study_group:
            # Skip sample if the required joint is missing
            logging.warning(f"Warning: Joint {joint_side} not found for patient {patient_id}. Skipping.")
            return None 

        joint_group = study_group[joint_side]
        
        roi_image = joint_group['roi_image'][:]
        tmj_center = joint_group.attrs['relative_tmj_center']
        roi_offset = joint_group.attrs['roi_offset_from_center']
        background_value = joint_group.attrs['background_value']
        segmentation_mask = joint_group['segmentation_mask'][:]
        roi_shape = np.array(roi_image.shape)
        background_value = max(self.cfg.HU_CLIP_MIN, background_value)

        # Verify that actual ROI size matches config expectation.
        # This protects against "dirty" data.
        expected_shape = np.array(self.cfg.ROI_SHAPE)
        if not np.array_equal(roi_shape, expected_shape):
            logging.warning(
                f"Skipping sample ({patient_id}/{joint_side}): "
                f"unexpected ROI shape. Expected {expected_shape}, got {roi_shape}."
            )
            return None
        
        # 1. Get inverse transformation
        rotation, target_cube_center, scale_factor = self._get_transform_matrices(
            roi_shape=np.array(roi_shape),
            tmj_center=tmj_center,
            roi_offset=roi_offset
        )
        # 2. Cut sub-volume handling boundaries
        half_bbox = self.max_bbox_size / 2.0
        bb_origin = np.round(target_cube_center - half_bbox).astype(int)
        bb_end = bb_origin + self.max_bbox_size

        # "Safe" crop from roi_image
        slice_origin = np.maximum(0, bb_origin)
        slice_end = np.minimum(roi_shape, bb_end)
        
        sub_volume_cropped = roi_image[
            slice_origin[0]:slice_end[0], 
            slice_origin[1]:slice_end[1], 
            slice_origin[2]:slice_end[2]
        ]
        
        # 3. Padding to max_bbox_size if near edges
        padded_sub_volume = np.full(self.max_bbox_size, fill_value=background_value, dtype=np.int16)
        paste_origin = slice_origin - bb_origin
        paste_end = paste_origin + sub_volume_cropped.shape
        
        padded_sub_volume[
            paste_origin[0]:paste_end[0],
            paste_origin[1]:paste_end[1],
            paste_origin[2]:paste_end[2]
        ] = sub_volume_cropped
        
        sub_mask_cropped = segmentation_mask[
            slice_origin[0]:slice_end[0], 
            slice_origin[1]:slice_end[1], 
            slice_origin[2]:slice_end[2]
        ]
        padded_sub_mask = np.zeros(self.max_bbox_size, dtype=np.uint8)
        padded_sub_mask[
            paste_origin[0]:paste_end[0],
            paste_origin[1]:paste_end[1],
            paste_origin[2]:paste_end[2]
        ] = sub_mask_cropped

        # 4. Prepare affine matrix for torch.affine_grid
        # We need a matrix mapping coordinates from target cube (256^3) 
        # to sub_volume coordinates (max_bbox_size^3)
        inv_rot_matrix = rotation.as_matrix().T

        # Target cube center should map to (target_cube_center) relative to roi_image
        translation_vec = target_cube_center - bb_origin

        # 5. Prepare FOV data
        fov_center_in_roi = roi_shape / 2.0 - roi_offset

        return {
            "joint_side": joint_side,
            "sub_volume": torch.from_numpy(padded_sub_volume),
            "sub_mask": torch.from_numpy(padded_sub_mask),
            "inv_rotation": torch.from_numpy(inv_rot_matrix).float(),
            "translation_vox": torch.from_numpy(translation_vec).float(),
            "scale_factor": torch.tensor(scale_factor, dtype=torch.float32),
            "bb_origin_in_roi": torch.from_numpy(bb_origin).float(),
            "fov_center_in_roi": torch.from_numpy(fov_center_in_roi).float(),
            "fov_radius": torch.tensor(self.cfg.FOV_RADIUS_VOXELS, dtype=torch.float32),
            #"tmj_center": torch.from_numpy(tmj_center).float()
        }
        
def collate_fn(batch):
    # 1. Filter out None values returned from __getitem__
    batch = list(filter(lambda x: x is not None, batch))
    if not batch:
        return None

    collated = {}
    keys = batch[0].keys()
    
    for key in keys:
        if isinstance(batch[0][key], torch.Tensor):
            collated[key] = torch.stack([d[key] for d in batch], 0)
        else:
            collated[key] = [d[key] for d in batch]
            
    return collated

def prepare_dataloaders(rank: int, world_size: int, cfg: Config, checkpoint: Optional[Dict] = None):
    """
    Prepares data loaders with patient-level splitting.
    """
    # 1. Get list of all patients from HDF5
    with h5py.File(cfg.DATA_HDF5_PATH, 'r') as f:
        all_patient_ids = sorted(list(f.keys()))

    # 2. Train/Val split logic
    split_json_path = cfg.EXPERIMENTS_DIR / 'patient_split.json'
    if checkpoint and 'train_patient_ids' in checkpoint:
        if rank == 0: 
            logging.info("Restoring patient split from checkpoint.")
        train_ids = checkpoint['train_patient_ids']
        val_ids = checkpoint['val_patient_ids']
    elif split_json_path.exists():
        if rank == 0: 
            logging.info(f"Loading patient split from {split_json_path}")
        with open(split_json_path, 'r') as f:
            split_data = json.load(f)
        train_ids, val_ids = split_data['train'], split_data['val']
    else:
        if rank == 0: 
            logging.warning(f"Split file {split_json_path} not found. Generating new split.")
        random.shuffle(all_patient_ids)
        split_idx = int(len(all_patient_ids) * (1.0 - cfg.VAL_SPLIT_RATIO))
        train_ids, val_ids = all_patient_ids[:split_idx], all_patient_ids[split_idx:]
        if rank == 0:
            with open(split_json_path, 'w') as f:
                json.dump({'train': train_ids, 'val': val_ids}, f, indent=4)
            logging.info(f"New split saved. Train: {len(train_ids)}, Val: {len(val_ids)}")

    def expand_patient_ids_to_joints(patient_ids: List[str]) -> List[Tuple[str, str]]:
        joint_identifiers = []
        with h5py.File(cfg.DATA_HDF5_PATH, 'r') as f:
            for patient_id in patient_ids:
                study_group = f[patient_id]
                if 'right_joint' in study_group:
                    joint_identifiers.append((patient_id, 'right_joint'))
                if 'left_joint' in study_group:
                    joint_identifiers.append((patient_id, 'left_joint'))
        return joint_identifiers
    
    train_joint_ids = expand_patient_ids_to_joints(train_ids)
    val_joint_ids = expand_patient_ids_to_joints(val_ids)

    if rank == 0:
        logging.info(f"Found {len(train_joint_ids)} training joints and {len(val_joint_ids)} validation joints.")
    
    train_dataset = TMJDataset(cfg.DATA_HDF5_PATH, train_joint_ids, cfg)
    val_dataset = TMJDataset(cfg.DATA_HDF5_PATH, val_joint_ids, cfg)

    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    
    train_loader = DataLoader(
        train_dataset, batch_size=cfg.GPU_BATCH_SIZE_TRAIN, sampler=train_sampler,
        num_workers=cfg.NUM_WORKERS_TRAIN, pin_memory=True, drop_last=True,
        persistent_workers=(cfg.NUM_WORKERS_TRAIN > 0),
        collate_fn=collate_fn,
        worker_init_fn=hdf5_worker_init_fn
    )
    val_loader = DataLoader(
        val_dataset, batch_size=cfg.GPU_BATCH_SIZE_VAL, sampler=val_sampler,
        num_workers=cfg.NUM_WORKERS_VAL, pin_memory=True,
        persistent_workers=(cfg.NUM_WORKERS_VAL > 0),
        collate_fn=collate_fn,
        worker_init_fn=hdf5_worker_init_fn
    )
    
    return train_loader, val_loader, train_ids, val_ids

# =================================================================================
# 5. Model Architecture
# =================================================================================

class MonaiMAEWrapper(nn.Module):
    """
    Adapter class for MONAI MaskedAutoEncoderViT.
    
    - Internally uses monai.nets.MaskedAutoEncoderViT (SOTA ViT-Base).
    - Accepts same inputs as the custom HybridMAE.
    - Returns the same output dictionary format.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg

        self.mae = MaskedAutoEncoderViT(
            in_channels=1,
            img_size=cfg.INPUT_CUBE_SIZE,
            patch_size=cfg.MONAI_VIT_PATCH_SIZE,
            
            hidden_size=cfg.MONAI_VIT_EMBED_DIM,
            mlp_dim=cfg.MONAI_VIT_MLP_DIM,
            num_layers=cfg.MONAI_VIT_DEPTH,
            num_heads=cfg.MONAI_VIT_NUM_HEADS,
            
            masking_ratio=cfg.MAE_MASK_RATIO,
            pos_embed_type='sincos', 
            
            # MONAI SOTA defaults
            # decoder_hidden_size=384,
            # decoder_mlp_dim=512,
            # decoder_num_layers=4,
            # decoder_num_heads=12,
            
            spatial_dims=3,
        )
        
        self.patchifier = lambda x: patchify_image(x, cfg.EFFECTIVE_PATCH_SIZE)
        
        self.p = cfg.EFFECTIVE_PATCH_SIZE
        self.grid_size = cfg.SWIN_GRID_SIZE # (n_d, n_h, n_w)
        self.C = 1
        self.D, self.H, self.W = cfg.INPUT_CUBE_SIZE

    def _unpatchify_image(self, patches: torch.Tensor) -> torch.Tensor:
        """
        Reconstructs full volume from patches.
        Input: (B, N, P*P*P*C) -> Output: (B, C, D, H, W)
        """
        B, N, P_dim = patches.shape
        n_d, n_h, n_w = self.grid_size
        p = self.p
        C = self.C

        if N != (n_d * n_h * n_w):
             raise ValueError("Patch count does not match grid_size")

        # 1. (B, N, P_dim) -> (B, C, n_d, n_h, n_w, p, p, p)
        # (B, 2744, 4096) -> (B, 1, 14, 14, 14, 16, 16, 16)
        x = patches.view(B, n_d, n_h, n_w, C, p, p, p)
        x = x.permute(0, 4, 1, 5, 2, 6, 3, 7) # (B, C, n_d, p, n_h, p, n_w, p)

        # 2. (B, C, n_d, p, n_h, p, n_w, p) -> (B, C, D, H, W)
        # (B, 1, 14, 16, 14, 16, 14, 16) -> (B, 1, 224, 224, 224)
        reconstructed_cube = x.contiguous().view(B, C, self.D, self.H, self.W)
        return reconstructed_cube

    def forward(self, x: torch.Tensor, valid_for_masking_mask: Optional[torch.Tensor] = None):
        """
        Mimics HybridMAE.forward() interface.
        Note: valid_for_masking_mask is ignored for the baseline ViT.
        """
        
        reconstructed_patches, mae_mask_1d = self.mae(x) 
        
        # (B, N, P*P*P*C) -> (B, C, D, H, W)
        reconstructed_cube = self._unpatchify_image(reconstructed_patches)

        output = {
            'reconstructed_patches': reconstructed_patches,
            'reconstructed_cube': reconstructed_cube,
            'mae_mask_1d': mae_mask_1d
        }
        return output


# =================================================================================
# 6. Loss Functions
# =================================================================================

class L2MAELoss(nn.Module):
    """
    Standard L2 (MSE) loss for MAE.
    Calculates MSE only on masked patches.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.mse_loss = nn.MSELoss()

    def forward(self,
                model_output: Dict[str, torch.Tensor],
                processed_batch: Dict[str, torch.Tensor]
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        pred_patches_all = model_output['reconstructed_patches']  # (B, N, P*P*P)
        target_patches_all = processed_batch['target_patches']      # (B, N, P*P*P)
        mae_mask_1d = model_output['mae_mask_1d'].bool()           # (B, N)
        
        pred_final = pred_patches_all[mae_mask_1d]      # -> (N_masked, P*P*P)
        target_final = target_patches_all[mae_mask_1d]  # -> (N_masked, P*P*P)

        if pred_final.numel() == 0:
            zero_loss = torch.tensor(0.0, device=pred_final.device, dtype=torch.float32)
            loss_components = {
                "l2_loss": zero_loss.detach()
            }
            return zero_loss, loss_components

        final_loss = self.mse_loss(pred_final, target_final) 

        loss_components = {
            "l2_loss": final_loss.detach(),
        }

        return final_loss, loss_components

class PatchLossEngine:
    """
    Engine for calculating patch-oriented 'ssim_a' (LCS) loss.
    """
    def __init__(self, cfg: Config):
        self.lcs_weights = torch.tensor(cfg.LCS_LOSS_WEIGHTS, dtype=torch.float32)
        self.sp_size = cfg.LOSS_SUB_PATCH_SIZE
        self.p_size = cfg.EFFECTIVE_PATCH_SIZE

    def _get_patch_stats(self, patches: torch.Tensor):
        """Calculate statistics along the last axis (voxels inside patch)."""
        with torch.amp.autocast(patches.device.type, enabled=False):
            patches_f32 = patches.float()
            mean = patches_f32.mean(dim=-1, keepdim=True)
            std = patches_f32.std(dim=-1, keepdim=True, unbiased=False)
        return mean, std

    def __call__(self, pred_patches: torch.Tensor, target_patches: torch.Tensor) -> torch.Tensor:
        """
        Calculates loss for a batch of filtered patches.
        Input: (num_valid_patches, P*P*P).
        """
        if pred_patches.numel() < 2:
            return torch.tensor(0.0, device=pred_patches.device)
        
        SP =  self.sp_size # Sub-Patch size
        P = self.p_size # Original Patch size
        
        # (N, 16*16*16) -> (N, 1, 16, 16, 16)
        num_valid_patches = pred_patches.shape[0]
        pred_patches_vol = pred_patches.view(num_valid_patches, 1, P, P, P)
        target_patches_vol = target_patches.view(num_valid_patches, 1, P, P, P)

        pred_sub_patches = pred_patches_vol.unfold(2, SP, SP).unfold(3, SP, SP).unfold(4, SP, SP)
        target_sub_patches = target_patches_vol.unfold(2, SP, SP).unfold(3, SP, SP).unfold(4, SP, SP)

        pred_sub_patches = pred_sub_patches.contiguous().view(-1, SP*SP*SP)
        target_sub_patches = target_sub_patches.contiguous().view(-1, SP*SP*SP)

        if self.lcs_weights.device != pred_patches.device:
            self.lcs_weights = self.lcs_weights.to(pred_patches.device)

        mean_pred, std_pred = self._get_patch_stats(pred_sub_patches)
        mean_target, std_target = self._get_patch_stats(target_sub_patches.detach())

        # 1. Brightness Loss
        loss_br = F.mse_loss(mean_pred, mean_target)

        # 2. Contrast Loss
        loss_cntr = F.mse_loss(std_pred, std_target)

        # 3. Structure Loss
        with torch.amp.autocast(pred_sub_patches.device.type, enabled=False):
            pred_norm = (pred_sub_patches.float() - mean_pred) / (std_pred + 1e-8)
            target_norm = (target_sub_patches.float() - mean_target) / (std_target + 1e-8)
        loss_str = F.mse_loss(pred_norm, target_norm)

        final_loss = (self.lcs_weights[0] * loss_br +
                      self.lcs_weights[1] * loss_cntr +
                      self.lcs_weights[2] * loss_str)
        return final_loss

class PhysicsInformedMAELoss(nn.Module):
    """
    Main loss class operating in patch space with semantic awareness.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_loss_engine = PatchLossEngine(cfg)
        self.register_buffer('loss_weights', torch.tensor(cfg.PHY_LOSS_WEIGHTS, dtype=torch.float32))

    def forward(self,
                model_output: Dict[str, torch.Tensor],
                processed_batch: Dict[str, torch.Tensor]
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        pred_patches_all = model_output['reconstructed_patches'] 
        target_patches_all = processed_batch['target_patches'] 
        mae_mask_1d = model_output['mae_mask_1d'].bool() 
        semantic_patches_all = processed_batch['combined_semantic_mask_patches']

        pred_final = pred_patches_all[mae_mask_1d] 
        target_final = target_patches_all[mae_mask_1d] 
        semantic_final = semantic_patches_all[mae_mask_1d]

        if pred_final.numel() == 0:
            zero_loss = torch.tensor(0.0, device=pred_final.device, dtype=torch.float32)
            loss_components = {
                "total_loss": zero_loss.detach(),
                "soft_tissue_loss": zero_loss.detach(),
                "surface_loss": zero_loss.detach()
            }
            return zero_loss, loss_components

        FOV_FLAG, SOFT_TISSUE_FLAG, SURFACE_FLAG = 1, 2, 4
        patch_volume = semantic_final.shape[-1]

        soft_tissue_votes = (semantic_final & SOFT_TISSUE_FLAG).count_nonzero(dim=-1)
        surface_votes = (semantic_final & SURFACE_FLAG).count_nonzero(dim=-1)

        soft_tissue_mask = soft_tissue_votes > (patch_volume * self.cfg.PATCH_MASK_TISSUE_THRESHOLD)
        surface_mask = surface_votes > (patch_volume * self.cfg.PATCH_MASK_TISSUE_THRESHOLD)

        loss_total = self.patch_loss_engine(pred_final, target_final)

        loss_soft_tissue = self.patch_loss_engine(
            pred_final[soft_tissue_mask],
            target_final[soft_tissue_mask]
        )
        loss_surface = self.patch_loss_engine(
            pred_final[surface_mask],
            target_final[surface_mask]
        )

        final_loss = (self.loss_weights[0] * loss_total +
                    self.loss_weights[1] * loss_soft_tissue +
                    self.loss_weights[2] * loss_surface)

        loss_components = {
            "total_loss": loss_total.detach(),
            "soft_tissue_loss": loss_soft_tissue.detach(),
            "surface_loss": loss_surface.detach()
        }

        return final_loss, loss_components
class SubpatchSSIMLoss(nn.Module):
    """
    Implementation of the MVC (LCS) loss calculated on sub-patches.
    
    Works similarly to L2MAELoss but uses the PatchLossEngine to compute
    structural similarity components (Luminance, Contrast, Structure).
    Only calculates loss on masked patches.
    """
    def __init__(self, cfg: Config):
        super().__init__()
        self.cfg = cfg
        self.patch_loss_engine = PatchLossEngine(cfg)

    def forward(self,
                model_output: Dict[str, torch.Tensor],
                processed_batch: Dict[str, torch.Tensor]
            ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:

        pred_patches_all = model_output['reconstructed_patches']  # (B, N, P*P*P)
        target_patches_all = processed_batch['target_patches']      # (B, N, P*P*P)
        mae_mask_1d = model_output['mae_mask_1d'].bool()           # (B, N)
        
        pred_final = pred_patches_all[mae_mask_1d]      # -> (N_masked, P*P*P)
        target_final = target_patches_all[mae_mask_1d]  # -> (N_masked, P*P*P)

        if pred_final.numel() == 0:
            zero_loss = torch.tensor(0.0, device=pred_final.device, dtype=torch.float32)
            loss_components = {
                "ssim_a_loss": zero_loss.detach()
            }
            return zero_loss, loss_components

        final_loss = self.patch_loss_engine(pred_final, target_final) 

        loss_components = {
            "ssim_a_loss": final_loss.detach(),
        }

        return final_loss, loss_components
# =================================================================================
# 7. GPU Processing
# =================================================================================

@torch.no_grad()
def morphology_op_3d(tensor: torch.Tensor, op: str, kernel_size: int) -> torch.Tensor:
    """
    Performs 3D morphological operations (dilation or erosion) on binary tensors
    using efficient MaxPool3d.
    
    Args:
        tensor (torch.Tensor): Input binary tensor (B, 1, D, H, W), float.
        op (str): Operation 'dilation' or 'erosion'.
        kernel_size (int): Kernel size (e.g., 3 for 3x3x3 neighborhood).
    """
    padding = kernel_size // 2
    if op == 'dilation':
        return F.max_pool3d(tensor, kernel_size=kernel_size, stride=1, padding=padding)
    elif op == 'erosion':
        return -F.max_pool3d(-tensor, kernel_size=kernel_size, stride=1, padding=padding)
    else:
        raise ValueError("Operation must be 'dilation' or 'erosion'")

@torch.no_grad()
def process_batch_on_gpu(
    batch: Dict[str, Any], 
    cfg: Config, 
    epoch: int,
    hu_augmenter: Optional[nn.Module] = None, 
    noise_simulator: Optional[nn.Module] = None
) -> Dict[str, torch.Tensor]:
    """
    Executes the full batch preparation pipeline on GPU:
    1. Affine transformation from BB to target cube.
    2. On-the-fly FOV mask generation.
    3. Physics-based mask creation.
    4. Application of HU/Noise augmentations.
    """
    # ========================================================================
    # STEP 1: Construct Affine Matrix for grid_sample
    # ========================================================================
    # Goal: Construct matrix theta (B, 3, 4) transforming coordinates from 
    # OUTPUT cube grid (target_cube, [-1, 1]) to INPUT sampling coordinates
    # (sub_volume, [-1, 1]).

    sub_volume_int16 = batch['sub_volume'] # (B, D, H, W), int16
    device = sub_volume_int16.device
    
    dtype = torch.bfloat16

    sub_volume_normalized = (torch.clamp(
        sub_volume_int16.float(),
        min=cfg.HU_CLIP_MIN, 
        max=cfg.HU_CLIP_MAX
    ) - cfg.DATASET_MEAN) / cfg.DATASET_STD
    B = sub_volume_normalized.shape[0]
    
    sub_volume_normalized = sub_volume_normalized.to(dtype)

    target_size_zyx = torch.tensor(cfg.INPUT_CUBE_SIZE, device=device, dtype=dtype)
    sub_vol_size_zyx = torch.tensor(batch['sub_volume'].shape[1:], device=device, dtype=dtype)

    # Z,Y,X -> X,Y,Z
    target_size_xyz = torch.flip(target_size_zyx, dims=[0])
    sub_vol_size_xyz = torch.flip(sub_vol_size_zyx, dims=[0])

    output_center_xyz = ((target_size_xyz - 1) / 2.0)
    
    inv_R = batch['inv_rotation'].to(dtype) 
    t_in_zyx = batch['translation_vox'].to(dtype)
    t_in_xyz = torch.flip(t_in_zyx, dims=[1]) 
    scale = batch['scale_factor'].to(dtype)

    inv_scale_val = 1.0 / scale

    inv_scale_val_expanded = inv_scale_val.unsqueeze(1).repeat(1, 3) 

    inv_S = torch.diag_embed(inv_scale_val_expanded)

    M_rs_inv = torch.bmm(inv_R, inv_S)

    rotation_offset = torch.einsum('bij,j->bi', M_rs_inv, output_center_xyz)
    final_translation_xyz = t_in_xyz - rotation_offset

    M_vox = torch.eye(4, device=device, dtype=dtype).unsqueeze(0).repeat(B, 1, 1)
    M_vox[:, :3, :3] = M_rs_inv
    M_vox[:, :3, 3] = final_translation_xyz

    # T_out: grid [-1,1] -> vox [0, size-1]
    T_out = torch.eye(4, device=device, dtype=dtype)
    T_out.diagonal(0, -2, -1)[:3] = (target_size_xyz - 1) / 2.0
    T_out[:3, 3] = (target_size_xyz - 1) / 2.0

    # T_in_inv: vox [0, size-1] -> grid [-1,1]
    T_in_inv = torch.eye(4, device=device, dtype=dtype)
    T_in_inv.diagonal(0, -2, -1)[:3] = 2.0 / (sub_vol_size_xyz - 1)
    T_in_inv[:3, 3] = -1.0
    
    final_transformation_matrix = T_in_inv @ M_vox @ T_out
    theta = final_transformation_matrix[:, :3, :]

    # ========================================================================
    # STEP 2: Apply Affine Transformation
    # ========================================================================
    sub_volume_gpu = sub_volume_normalized.unsqueeze(1) 
    sub_mask_gpu = batch['sub_mask'].unsqueeze(1).to(dtype) 
    target_size_torch = (B, 1, *cfg.INPUT_CUBE_SIZE)

    grid = F.affine_grid(theta, target_size_torch, align_corners=False)

    target_cube = F.grid_sample(
        sub_volume_gpu, grid, mode='bilinear', padding_mode='zeros', align_corners=False
    ) # will actually be trilinear
    seg_mask = F.grid_sample(
        sub_mask_gpu, grid, mode='nearest', padding_mode='zeros', align_corners=False
    ).short()

    del sub_volume_gpu, sub_mask_gpu

    # ========================================================================
    # STEP 3: On-the-fly FOV Mask Generation
    # ========================================================================
    coords_in_sub_vol_xyz = (grid + 1) / 2.0 * (sub_vol_size_xyz - 1).view(1, 1, 1, 1, 3)

    bb_origin_zyx = batch['bb_origin_in_roi'].view(B, 1, 1, 1, 3)
    bb_origin_xyz = torch.flip(bb_origin_zyx, dims=[-1])
    
    fov_center_zyx = batch['fov_center_in_roi'].view(B, 1, 1, 1, 3)
    fov_center_xyz = torch.flip(fov_center_zyx, dims=[-1])

    coords_in_roi_xyz = coords_in_sub_vol_xyz + bb_origin_xyz

    fov_radius_sq_gpu = (batch['fov_radius']**2).view(B, 1, 1, 1)
    
    dist_sq = (
        (coords_in_roi_xyz[..., 0] - fov_center_xyz[..., 0])**2 +
        (coords_in_roi_xyz[..., 1] - fov_center_xyz[..., 1])**2
    )
    fov_mask = (dist_sq <= fov_radius_sq_gpu).unsqueeze(1)
    del grid, coords_in_sub_vol_xyz

    sides = batch['joint_side']
    for i in range(len(sides)):
        if sides[i] == 'left_joint':
            target_cube[i] = torch.flip(target_cube[i], dims=[-1])
            seg_mask[i] = torch.flip(seg_mask[i], dims=[-1])
            fov_mask[i] = torch.flip(fov_mask[i], dims=[-1])

    soft_tissue_mask = (target_cube > cfg.Z_NORM_SOFT_TISSUE_MIN) & (target_cube < cfg.Z_NORM_SOFT_TISSUE_MAX)

    bone_mask_bool = (seg_mask == 1) | (seg_mask == 2)
    bone_mask_float = bone_mask_bool.float() # (B, 1, D, H, W)

    inner_boundary_base = bone_mask_float
    for _ in range(cfg.SURFACE_SHELL_INNER_VOXELS):
        inner_boundary_base = morphology_op_3d(inner_boundary_base, 'erosion', 3)
    
    outer_boundary_base = bone_mask_float
    for _ in range(cfg.SURFACE_SHELL_OUTER_VOXELS):
        outer_boundary_base = morphology_op_3d(outer_boundary_base, 'dilation', 3)

    surface_mask = (outer_boundary_base - inner_boundary_base).bool()

    input_cube = target_cube.clone()

    if hu_augmenter and cfg.USE_HU_AUG and epoch >= cfg.HU_AUG_START_EPOCH:
        input_cube = hu_augmenter(input_cube)
    
    if noise_simulator and cfg.USE_NOISE_AUG and epoch >= cfg.NOISE_AUG_START_EPOCH:
        input_cube = noise_simulator(input_cube)

    FOV_FLAG, SOFT_TISSUE_FLAG, SURFACE_FLAG = 1, 2, 4
    
    combined_mask_3d = torch.zeros_like(target_cube, dtype=torch.uint8)

    combined_mask_3d[fov_mask] |= FOV_FLAG
    combined_mask_3d[soft_tissue_mask] |= SOFT_TISSUE_FLAG
    combined_mask_3d[surface_mask] |= SURFACE_FLAG

    fov_mask_patches = patchify_image(fov_mask.float(), cfg.EFFECTIVE_PATCH_SIZE)
    patch_fov_ratio = fov_mask_patches.mean(dim=-1)
    valid_for_masking_mask = patch_fov_ratio > cfg.PATCH_MASK_FOV_THRESHOLD

    processed_batch = {
        'input_cube': input_cube,
        'target_patches': patchify_image(target_cube, cfg.EFFECTIVE_PATCH_SIZE), 
        'combined_semantic_mask_patches': patchify_image(combined_mask_3d.short(), cfg.EFFECTIVE_PATCH_SIZE),
        'valid_for_masking_mask': valid_for_masking_mask,
        'dense_mask': seg_mask,
        #'transformed_tmj_center': transformed_tmj_center
    }
    
    return processed_batch

# =================================================================================
# 8. Training Loop
# =================================================================================

@torch.no_grad()
def log_mae_visualizations_to_tensorboard(
    writer: SummaryWriter,
    originals: torch.Tensor,
    reconstructions: torch.Tensor,
    mae_mask_1d: torch.Tensor,
    cfg: Config,
    epoch: int,
    num_samples_to_log: int = 4
):
    """
    Creates and logs a 3x3 grid visualization to TensorBoard.
    - Row 1: Original Orthographic Projections (XY, ZY, ZX).
    - Row 2: Visible area (non-masked patches).
    - Row 3: Reconstructed Orthographic Projections.
    """
    originals = originals.cpu().float()
    reconstructions = reconstructions.cpu().float()
    mae_mask_1d = mae_mask_1d.cpu().float()

    z_min_hu_0 = (0.0 - cfg.DATASET_MEAN) / cfg.DATASET_STD
    z_max_hu_1500 = (1500.0 - cfg.DATASET_MEAN) / cfg.DATASET_STD
    
    z_range = (z_max_hu_1500 - z_min_hu_0) + 1e-6
    
    B = originals.shape[0]
    num_samples = min(num_samples_to_log, B)

    h = cfg.INPUT_CUBE_SIZE[0] // cfg.EFFECTIVE_PATCH_SIZE 

    visible_mask_patches = (1 - mae_mask_1d).view(B, h, h, h)

    P = cfg.EFFECTIVE_PATCH_SIZE
    mask_3d = visible_mask_patches.repeat_interleave(P, dim=1) \
                                .repeat_interleave(P, dim=2) \
                                .repeat_interleave(P, dim=3)

    mask_3d = mask_3d.unsqueeze(1)

    for i in range(num_samples):
        orig_cube = originals[i, 0]
        recon_cube = reconstructions[i, 0]
        
        visible_cube = orig_cube * mask_3d[i, 0]
        
        d, h, w = orig_cube.shape
        center_slices = {
            'orig': [orig_cube[d//2, :, :], orig_cube[:, h//2, :], orig_cube[:, :, w//2]],
            'visible': [visible_cube[d//2, :, :], visible_cube[:, h//2, :], visible_cube[:, :, w//2]],
            'recon': [recon_cube[d//2, :, :], recon_cube[:, h//2, :], recon_cube[:, :, w//2]]
        }

        grid_slices = []
        
        grid_slices.extend(center_slices['orig'])
        grid_slices.extend(center_slices['visible'])
        grid_slices.extend(center_slices['recon'])

        processed_slices = []
        for s in grid_slices:
            s_norm = (s - z_min_hu_0) / z_range
            s_norm = torch.clamp(s_norm, 0.0, 1.0)
            processed_slices.append(s_norm.unsqueeze(0))
            
        grid = torchvision.utils.make_grid(processed_slices, nrow=3, padding=2, pad_value=0.5)
        
        writer.add_image(
            f'validation_sample_{i}',
            grid,
            global_step=epoch
        )


def train_one_epoch(model: nn.Module, train_loader: DataLoader, optimizer: torch.optim.Optimizer, 
                    scaler: GradScaler, lr_scheduler, loss_fn: nn.Module, cfg: Config, 
                    device: torch.device, epoch: int, rank: int, world_size: int, writer: Optional[SummaryWriter],
                    hu_augmenter: Optional[nn.Module], noise_simulator: Optional[nn.Module],
                    train_iter_state: Dict[str, Any],
                ) -> Dict[str, float]:
    """Single training epoch with logging and DDP support."""
    model.train()
    
    epoch_losses = defaultdict(float)

    num_steps_per_epoch = cfg.TRAIN_STEPS_PER_EPOCH * cfg.ACCUMULATION_STEPS 
    
    pbar = tqdm(total=cfg.TRAIN_STEPS_PER_EPOCH, desc=f"Epoch {epoch} [Train]", disable=(rank != 0), leave=False) 


    def get_next_batch():
        nonlocal train_iter_state
        try:
            batch = next(train_iter_state['iterator'])
        except StopIteration:
            train_iter_state['real_epoch'] += 1
            train_loader.sampler.set_epoch(train_iter_state['real_epoch'])
            train_iter_state['iterator'] = iter(train_loader)
            batch = next(train_iter_state['iterator'])
        
        if batch is None:
            logging.warning(f"Epoch {epoch}: Skipped empty/corrupt batch.")
            return get_next_batch() 
        
        return batch
    
    for step in range(num_steps_per_epoch):
        #if (step > 20):
        #    break
        batch = get_next_batch()
        if batch is None:
            logging.warning(f"Epoch {epoch}, step {step}: Skipped empty/corrupt batch.")
            continue

        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device, non_blocking=True)
            
        processed_batch = process_batch_on_gpu(
            batch, cfg, epoch, hu_augmenter, noise_simulator
        )
        
        
        mask_input = None
        if cfg.USE_FOV_AWARE_MASKING:
            mask_input = processed_batch['valid_for_masking_mask']

        with autocast('cuda', dtype=torch.bfloat16):
            model_output = model(
                processed_batch['input_cube'],
                valid_for_masking_mask=mask_input
            )
            loss, loss_components = loss_fn(model_output, processed_batch)
            loss = loss / cfg.ACCUMULATION_STEPS 
        
        loss_dict_step = {k: v.detach() for k, v in loss_components.items()}
        loss_dict_step['final_loss'] = loss.detach() * cfg.ACCUMULATION_STEPS
        
        scaler.scale(loss).backward()
        
        if (step + 1) % cfg.ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRADIENT_CLIP_VAL)
            if writer and rank == 0:
                global_optimizer_step = (epoch * cfg.TRAIN_STEPS_PER_EPOCH) + (step // cfg.ACCUMULATION_STEPS)
                writer.add_scalar('train_step/grad_norm', total_norm.item(), global_optimizer_step)

            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            if rank == 0:
                pbar.update(1)
                current_lr = optimizer.param_groups[0]['lr']
                pbar.set_postfix(loss=loss_dict_step['final_loss'].item(), lr=f"{current_lr:.2e}")
        

        if world_size > 1:
            for k, v in loss_dict_step.items():
                loss_dict_step[k] = reduce_tensor(v)

        for k, v in loss_dict_step.items():
            epoch_losses[k] += v.item()

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']

            if writer and (step % 20 == 0): 
                global_step = (epoch * cfg.TRAIN_STEPS_PER_EPOCH) + (step // cfg.ACCUMULATION_STEPS)
                writer.add_scalar('train_step/learning_rate', current_lr, global_step)
                for k, v in loss_dict_step.items():
                    writer.add_scalar(f'train_step/{k}', v.item(), global_step)
    if rank == 0:
        pbar.close()

    avg_epoch_losses = {k: v / num_steps_per_epoch for k, v in epoch_losses.items()}
    return avg_epoch_losses

@torch.no_grad()
def validate(model: nn.Module, val_loader: DataLoader, 
             validation_loss_fns: Dict[str, nn.Module],
             cfg: Config, 
             device: torch.device, epoch: int, rank: int, world_size: int, writer: Optional[SummaryWriter]) -> Dict[str, float]:
    """
    Performs a single validation pass.
    """
    model.eval()
    
    total_losses_sum = defaultdict(lambda: torch.tensor(0.0, device=device))
    
    valid_batches_count = torch.tensor(0.0, device=device)
    pbar = tqdm(val_loader, desc=f"Epoch {epoch} [Val]", disable=(rank != 0), leave=False)
    
    for step, batch in enumerate(pbar):
        
        if batch is None:
            logging.warning(f"Epoch {epoch} [Val]: Skipped empty/corrupt batch.")
            continue
            
        valid_batches_count += 1
            
        for key, val in batch.items():
            if isinstance(val, torch.Tensor):
                batch[key] = val.to(device, non_blocking=True)
                
        processed_batch = process_batch_on_gpu(
            batch, cfg, epoch, hu_augmenter=None, noise_simulator=None
        )
        
        mask_input = None
        if cfg.USE_FOV_AWARE_MASKING:
            mask_input = processed_batch['valid_for_masking_mask']
        with autocast('cuda', dtype=torch.bfloat16):
            model_output = model(
                processed_batch['input_cube'],
                valid_for_masking_mask=mask_input
            )

            for loss_name, loss_fn in validation_loss_fns.items():
                loss_value, loss_components = loss_fn(model_output, processed_batch)
                for key, value in loss_components.items():
                    total_losses_sum[f'{loss_name}_{key}'] += value.detach()

                total_losses_sum[f'{loss_name}_final_loss'] += loss_value.detach()
        
        if step == 0 and rank == 0 and writer:
            log_mae_visualizations_to_tensorboard(
                writer=writer,
                originals=processed_batch['input_cube'],
                reconstructions=model_output['reconstructed_cube'],
                mae_mask_1d=model_output['mae_mask_1d'],
                cfg=cfg,
                epoch=epoch,
                num_samples_to_log=min(4, processed_batch['input_cube'].shape[0])
            )
            
    
    if world_size > 1:
        sorted_keys = sorted(total_losses_sum.keys())
        tensor_list_sorted = [total_losses_sum[k] for k in sorted_keys]
        
        if not tensor_list_sorted:
             pass
        else:
            all_losses_tensor = torch.stack(tensor_list_sorted)
            dist.all_reduce(all_losses_tensor, op=dist.ReduceOp.SUM)
            
            total_losses_sum = {
                key: all_losses_tensor[i] for i, key in enumerate(sorted_keys)
            }
        dist.all_reduce(valid_batches_count, op=dist.ReduceOp.SUM)

    if valid_batches_count.item() == 0:
        logging.warning(f"Epoch {epoch} [Val]: No valid batches for evaluation")
        return {k: 0.0 for k in total_losses_sum.keys()}

    avg_losses_tensors = {k: v / valid_batches_count.item() for k, v in total_losses_sum.items()}
     
    avg_val_losses = {key: val.item() for key, val in avg_losses_tensors.items()}
     
    return avg_val_losses

# =================================================================================
# 9. Main Function
# =================================================================================

def main():
    cfg = Config()
    rank, world_size = setup_ddp()
    set_seed(cfg.SEED, rank)
    
    device = torch.device(f"cuda:{os.environ.get('LOCAL_RANK', 0)}")
    setup_logging(rank, cfg.LOG_DIR)
    if rank == 0:
        cfg.CHECKPOINT_DIR.mkdir(exist_ok=True, parents=True)
        writer = SummaryWriter(log_dir=str(cfg.LOG_DIR / "tensorboard"))
        log_config(cfg)
    else:
        writer = None

    latest_checkpoint_path = cfg.CHECKPOINT_DIR / "latest_checkpoint.pth"
    checkpoint = None
    if latest_checkpoint_path.exists():
        checkpoint = torch.load(latest_checkpoint_path, map_location=device, weights_only=False)
        if rank == 0: 
            logging.info(f"Resuming from epoch {checkpoint['epoch']}")
    
    train_loader, val_loader, train_ids, val_ids = prepare_dataloaders(rank, world_size, cfg, checkpoint)
    
    if cfg.MODEL_TYPE == 'bertsWin':
        if rank == 0: 
            logging.info(f"Initializing BertsWinMAE with USE_FOV_AWARE_MASKING={cfg.USE_FOV_AWARE_MASKING}")
        
        model = BertsWinMAE(
            img_size=cfg.INPUT_CUBE_SIZE,              # (224, 224, 224)
            patch_size=cfg.EFFECTIVE_PATCH_SIZE,       # 16
            in_chans=1,
            encoder_embed_dim=cfg.ENCODER_EMBED_DIM,   # 768
            encoder_depths=cfg.SWIN_DEPTHS,            # [12]
            encoder_num_heads=cfg.SWIN_NUM_HEADS,      # [12]
            swin_window_size=cfg.SWIN_WINDOW_SIZE,     # (7, 7, 7)
            decoder_embed_dim=cfg.DECODER_EMBED_DIM,   # 512
            stem_base_dim=cfg.STEM_BASE_DIM,           # 48
            mask_ratio=cfg.MAE_MASK_RATIO              # 0.75
        ).to(device)
    
    elif cfg.MODEL_TYPE == 'monai_vit':
        if rank == 0: logging.info("Initializing MONAI MAE (ViT) baseline")
        model = MonaiMAEWrapper(cfg).to(device)
    else:
        raise ValueError(f"Unknown MODEL_TYPE: {cfg.MODEL_TYPE}")
    
    
    if world_size > 1:
        model = DDP(
            model, 
            device_ids=[int(os.environ["LOCAL_RANK"])], 
            find_unused_parameters=False
        )

    model = torch.compile(model)

    loss_factory = {
        'custom_ssim': PhysicsInformedMAELoss(cfg).to(device),
        'l2': L2MAELoss(cfg).to(device)
    }

    if cfg.TRAIN_LOSS_TYPE not in loss_factory:
        raise ValueError(
            f"Unknown loss type: {cfg.TRAIN_LOSS_TYPE}. "
            f"Available: {list(loss_factory.keys())}"
        )
    try:
        primary_loss_prefix = cfg.VALIDATION_PRIMARY_LOSS.split('_')[0]
    except Exception:
        raise ValueError(f"VALIDATION_PRIMARY_LOSS ('{cfg.VALIDATION_PRIMARY_LOSS}') has invalid format.")
    
    if primary_loss_prefix not in loss_factory:
        raise ValueError(
            f"Prefix '{primary_loss_prefix}' from VALIDATION_PRIMARY_LOSS ('{cfg.VALIDATION_PRIMARY_LOSS}') "
            f"not found in loss_factory. Available prefixes: {list(loss_factory.keys())}"
        )
    
    loss_fn_train = loss_factory[cfg.TRAIN_LOSS_TYPE]
    if rank == 0: 
        logging.info(f"Training with loss: {cfg.TRAIN_LOSS_TYPE}")

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.LEARNING_RATE,
        weight_decay=cfg.ADAMW_WEIGHT_DECAY,
        betas=(cfg.ADAMW_BETA1, cfg.ADAMW_BETA2)
    )

    hu_augmenter = HounsfieldUnitInterpolator(cfg).to(device) if cfg.USE_HU_AUG else None
    noise_simulator = CTNoiseSimulatorGPU(cfg).to(device) if cfg.USE_NOISE_AUG else None
    if rank == 0:
        if hu_augmenter: logging.info("HU augmentation enabled.")
        if noise_simulator: logging.info("CT noise simulation enabled.")
    
    num_steps_per_epoch_virtual = cfg.TRAIN_STEPS_PER_EPOCH
    warmup_steps = cfg.WARMUP_EPOCHS * num_steps_per_epoch_virtual
    total_steps = cfg.EPOCHS * num_steps_per_epoch_virtual

    if cfg.SCHEDULER_TYPE == 'Cosine':
        if rank == 0:
            logging.info(f"Using CosineAnnealingLR: Warmup={cfg.WARMUP_EPOCHS} epochs ({warmup_steps} steps), T_max={total_steps - warmup_steps} steps")

        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
        main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=(total_steps - warmup_steps), eta_min=cfg.COSINE_ETA_MIN
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup_scheduler, main_scheduler], milestones=[warmup_steps]
        )
    
    elif cfg.SCHEDULER_TYPE == 'Linear':
        if rank == 0:
            logging.info(f"Using LinearLR (for testing): Warmup={cfg.WARMUP_EPOCHS} epochs ({warmup_steps} steps)")
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1e-6, end_factor=1.0, total_iters=warmup_steps
        )
    else:
        raise ValueError(f"Unknown SCHEDULER_TYPE: {cfg.SCHEDULER_TYPE}")
    
    scaler = GradScaler('cuda')
    
    start_epoch = 0
    real_epoch_start = 0
    best_val_loss = float('inf')
    early_stopper = EarlyStopping(patience=cfg.PATIENCE_EPOCHS)
    
    if checkpoint:
        (model.module if world_size > 1 else model).load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        if 'scaler_state_dict' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        real_epoch_start = checkpoint.get('real_epoch', start_epoch)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        if 'early_stopper_state_dict' in checkpoint:
            early_stopper.load_state_dict(checkpoint['early_stopper_state_dict'])

    train_loader.sampler.set_epoch(real_epoch_start)
    
    train_iter_state = {
        'iterator': iter(train_loader),
        'real_epoch': real_epoch_start 
    }

    if rank == 0:
        logging.info("Starting training...")
        
    for epoch in range(start_epoch, cfg.EPOCHS):
        #train_loader.sampler.set_epoch(epoch)
        if world_size > 1:
            val_loader.sampler.set_epoch(epoch)
        
        train_metrics = train_one_epoch(
            model, train_loader, optimizer, scaler, lr_scheduler, loss_fn_train, cfg, 
            device, epoch, rank, world_size, writer,
            hu_augmenter, noise_simulator,
            train_iter_state=train_iter_state
        )
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        val_metrics = validate(
             model, val_loader, loss_factory, cfg, device, epoch, rank, world_size, writer
        )
        
        if rank == 0:
            main_train_loss = train_metrics['final_loss']
            main_val_loss = val_metrics[cfg.VALIDATION_PRIMARY_LOSS]
            
            log_msg = (
                f"Epoch {epoch:03d} | "
                f"Train Loss ({cfg.TRAIN_LOSS_TYPE}): {main_train_loss:.5f} | "
                f"Val Loss ({cfg.VALIDATION_PRIMARY_LOSS.split('_')[0]}): {main_val_loss:.5f} | "
                f"LR: {optimizer.param_groups[0]['lr']:.2e}"
            )
            logging.info(log_msg)
            if writer:
                for key, value in train_metrics.items():
                    writer.add_scalar(f'Train/{key}', value, epoch)
                for key, value in val_metrics.items():
                    writer.add_scalar(f'Validation/{key}', value, epoch)
               
                writer.add_scalar('Meta/Learning_Rate', optimizer.param_groups[0]['lr'], epoch)

            
            is_best = main_val_loss < best_val_loss
            if is_best:
                best_val_loss = main_val_loss
            
            checkpoint_data = {
                'epoch': epoch,
                'real_epoch': train_iter_state['real_epoch'],
                'model_state_dict': (model.module if world_size > 1 else model).state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': lr_scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'early_stopper_state_dict': early_stopper.state_dict(),
                'train_patient_ids': train_ids,
                'val_patient_ids': val_ids,
            }
            
            atomic_save(checkpoint_data, cfg.CHECKPOINT_DIR / "latest_checkpoint.pth")
            if is_best:
                logging.info(f"New best model saved with val loss: {best_val_loss:.6f}")
                atomic_save(checkpoint_data, cfg.CHECKPOINT_DIR / "best_model.pth")
                
            early_stopper(main_val_loss)
            if early_stopper.early_stop:
                logging.info("Early stopping. Training finished.")
                break

        if world_size > 1:
            dist.barrier()
            
    if rank == 0 and writer:
        writer.close()
        logging.info("Training completed.")

if __name__ == '__main__':
    main()