import os
from pathlib import Path
from typing import Dict, List, Any

class EvalConfig:
    """
    Configuration for Downstream Evaluation.
    Designed to be scientifically reproducible and compatible with pre-trained configs.
    """
    
    # --- Paths ---
    BASE_OUTPUT_DIR = Path('/med_data/cbct-ct/eval_BertsWinMAE/2layer/')
    EXTERNAL_HDF5_PATH = '/med_data/datasets/tmj_canonical.hdf5'
    PRETRAIN_HDF5_PATH = '/data/datasets/tbj_cbct.hdf5'
    PRETRAIN_SCRIPT_PATH = os.getenv('PRETRAIN_SCRIPT', '../train_maeswin.py')
    PRETRAIN_SPLIT_FILE = '../patient_split.json'
    MRI_HDF5_PATH = '/med_data/datasets/ct_mri_tmj.hdf5'
    MRI_CSV_PATH = '/med_data/datasets/mri_data.csv'
    
    # --- Model Registry ---
    # Defines architecture types, checkpoint paths, and specific inference modes.
    # 'arch_type' must match keys in ModelFactory.
    MODEL_REGISTRY = {
        'bertswin_ssimgcond_mask': {
            'path': '/med_data/cbct-ct/dntum_mae/checkpoints_bertswin_gcond_phy/best_model.pth',
            'inference_mode': 'background_mask',
            'arch_type': 'bertswin_mae',
        },
        'bertswin_gcond_l2': {
            'path': '/med_data/cbct-ct/dntum_mae/checkpoints_bertswin_gcond_l2/best_model.pth',
            'inference_mode': 'background_mask',
            'arch_type': 'bertswin_mae',
        },
        'bertswin_gcond_assim': {
            'path': '/med_data/cbct-ct/dntum_mae/checkpoints_bertswin_gcond_ssim_a/best_model.pth',
            'inference_mode': 'background_mask',
            'arch_type': 'bertswin_mae',
        },
        'bertswin_l2adam_full': {
            'path': '/med_data/cbct-ct/dntum_mae/checkpoints_bertswin_adam_l2/best_model.pth',
            'inference_mode': 'full_image',
            'arch_type': 'bertswin_mae',
        },
        'bertswin_random': {
            'path': None,
            'inference_mode': 'full_image',
            'arch_type': 'bertswin_mae',
        },
        'monai_mae_pretrained': {
            'path': '/med_data/cbct-ct/dntum_mae/checkpoints_monai_baseline/best_model.pth',
            'inference_mode': 'full_image',
            'arch_type': 'monai_vit',
        },
        'monai_mae_random': {
            'path': None, # Random initialization
            'inference_mode': 'full_image',
            'arch_type': 'monai_vit',
        },
        'monai_swin_unetr_btcv': {
            'path': '/med_data/cbct-ct/dntum_mae/MONAI_swin_unetr_btcv_segmentation.pt', 
            'inference_mode': 'full_image',
            'arch_type': 'monai_swin_unetr_btcv',
        },
        'sam_med3d_vit_b_ori': {
            'path': '/med_data/cbct-ct/dntum_mae/sam_med3d_turbo.pth', 
            'inference_mode': 'full_image',
            'arch_type': 'sam_med3d_vit_b',
        },
    }
    
    # --- Cross-Validation Settings ---
    K_FOLDS = 5
    CV_REPEATS = 3#10
    SEED = 42
    TTA_RUNS = 4
    
    # --- Training Hyperparameters ---
    NUM_EPOCHS = 100#50
    PATIENCE_EPOCHS = 20
    EARLYSTOP_WINDOW = 7
    GPU_BATCH_SIZE = 6
    NUM_WORKERS = 6
    
    # --- Optimization ---
    LR_HEAD_ONLY = 1e-4
    LR_FINETUNE = 5e-6
    WEIGHT_DECAY = 0.05
    
    # --- Data Augmentation (GPU) ---
    # Exact values from the original experiment to preserve integrity
    AUG_ROTATION_DEGREES = [-15, 15]
    AUG_SCALE_RANGE = [0.9, 1.1]
    AUG_TRANSLATE_PERCENT_RANGE = [-10., 10.]

    # --- Segmentation Specifics ---
    # 'linear': Upsample -> Conv1x1
    # '2layer': Conv3x3 -> BN -> ReLU -> Upsample -> Conv1x1
    SEG_HEAD_TYPE = '2layer'#linear' 
    
    # --- Dataset Statistics (Will be overwritten by hydration from Pretrain Config) ---
    TARGET_IMAGE_SIZE = None
    HU_CLIP_MIN = None
    HU_CLIP_MAX = None
    DATASET_MEAN = None
    DATASET_STD = None
    EFFECTIVE_PATCH_SIZE = None