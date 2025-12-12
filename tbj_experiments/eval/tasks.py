import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss, HuberLoss
from torch.utils.data import Dataset
import torch.nn.functional as F
import numpy as np
import h5py
import logging
import json
import random
import pandas as pd
import copy
from pathlib import Path
from contextlib import contextmanager
from collections import defaultdict
from abc import ABC, abstractmethod
from typing import Dict, List, Any, Tuple, Optional
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, r2_score, mean_absolute_error, mean_squared_error, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold

from monai.losses import DiceLoss
from monai.metrics import compute_hausdorff_distance, compute_average_surface_distance
from monai.transforms import AsDiscrete

# Internal imports
from transforms import PhysicsEngine
from models import TextureHead, SpatialAttentionHead, VolumetricPyramidHead

def worker_init_fn(worker_id):
    """
    Ensures unique random seeding for each DataLoader worker.
    Essential for scientific reproducibility when using NumPy transforms.
    """
    worker_info = torch.utils.data.get_worker_info()
    dataset = worker_info.dataset
    # If dataset is a Subset, unwrap it to access the HDF5 logic if needed
    
    # Reseed NumPy
    np.random.seed(worker_info.seed % (2**32))

class LazyHDF5Dataset(Dataset):
    """
    A robust Dataset implementation for large-scale HDF5 volumetric data.
    
    Design Rationale:
    PyTorch DataLoader workers use multiprocessing. Opening an HDF5 file handle 
    in the main process and sharing it causes locking issues and data corruption.
    This class implements 'Lazy Loading': the file handle is opened exclusively 
    within the worker process during the first call to __getitem__.
    
    Attributes:
        hdf5_path (str): Path to the HDF5 dataset.
        manifest (List[Dict]): A pre-loaded list of metadata/targets to keep RAM usage low 
                               while avoiding repeated I/O for non-image data.
    """
    def __init__(self, hdf5_path: str, manifest: List[Dict]):
        self.hdf5_path = hdf5_path
        self.manifest = manifest
        self.h5_file = None

    def __len__(self): 
        return len(self.manifest)

    def _open_file(self):
        """Safely opens the HDF5 file handle for the current process."""
        if self.h5_file is None:
            try:
                self.h5_file = h5py.File(self.hdf5_path, 'r')
            except Exception as e:
                logging.error(f"Failed to open HDF5 file at {self.hdf5_path}: {e}")
                raise e

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        if self.h5_file is None: 
            self._open_file()
        
        # Retrieve pre-loaded metadata from RAM
        meta = self.manifest[idx]
        pid = meta['pid']
        
        try:
            # Access HDF5 groups
            patient_grp = self.h5_file[pid]
            baseline_grp = patient_grp['baseline']
            
            # Lazy Load Images (Heavy I/O)
            # Note: Data is kept as raw tensors here. Anatomical flipping (Left->Right)
            # and normalization are handled later by the PhysicsEngine to ensure
            # consistency with the GPU processing pipeline.
            img_L = torch.from_numpy(baseline_grp['left']['image'][:])
            min_L = baseline_grp['left'].attrs['min_hu_background']
            
            img_R = torch.from_numpy(baseline_grp['right']['image'][:])
            min_R = baseline_grp['right'].attrs['min_hu_background']
            
            # Construct the sample dictionary
            item = {
                'pid': pid,
                'img_L': img_L, 
                'min_hu_L': min_L,
                'img_R': img_R, 
                'min_hu_R': min_R,
            }
            
            # Merge metadata (targets/labels) into the item
            # We exclude 'pid' from meta merge as it's already added
            item.update({k: v for k, v in meta.items() if k != 'pid'})
            
            return item
            
        except KeyError as e:
            logging.error(f"KeyError accessing data for PID {pid}: {e}")
            raise e
        except Exception as e:
            logging.error(f"Unexpected error for PID {pid}: {e}")
            raise e

class LabeledTMJDataset(Dataset):
    """
    Dataset wrapper that reuses the robust Pre-training logic for loading and cropping,
    but injects supervised labels from a lookup dictionary.
    """
    def __init__(self, pt_dataset_cls, hdf5_path: str, identifiers: List[Tuple[str, str]], 
                 cfg: Any, targets_lookup: Dict[Tuple[str, str], torch.Tensor]):
        """
        Args:
            pt_dataset_cls: The TMJDataset class from train_maeswin.py
            identifiers: List of (doc_id, side) tuples.
            targets_lookup: Mapping {(doc_id, side) -> Tensor[4]}
        """
        # Instantiate the pre-training dataset to handle geometry/IO
        self.pt_dataset = pt_dataset_cls(hdf5_path, identifiers, cfg)
        self.targets_lookup = targets_lookup
        self.hdf5_path = hdf5_path
    
    @property
    def h5_file(self):
        return self.pt_dataset.h5_file

    @h5_file.setter
    def h5_file(self, value):
        # When worker_init_fn sets dataset.h5_file, it actually sets it on the internal pt_dataset
        self.pt_dataset.h5_file = value
        
    def __len__(self):
        return len(self.pt_dataset)
        
    def __getitem__(self, idx):
        # 1. Load Image and Geometry via Pre-train logic
        # Returns dict: {'sub_volume', 'inv_rotation', ...} or None
        data = self.pt_dataset[idx]
        
        if data is None:
            return None
            
        # 2. Inject Labels
        # pt_dataset.joint_identifiers stores the keys used for fetching
        doc_id, side = self.pt_dataset.joint_identifiers[idx]
        
        # Retrieve targets: [Bone, Disc, Inflam, Space]
        labels = self.targets_lookup.get((doc_id, side))
        
        if labels is None:
            logging.warning(f"Labels missing for {doc_id} {side}")
            return None
            
        data['labels'] = labels
        return data


class BaseTask(ABC):
    """
    Abstract Base Class defining the contract for Downstream Tasks.
    Ensures consistent handling of Data Loading, Loss computation, and Metric evaluation
    across different clinical objectives (Classification vs. Regression).
    """
    def __init__(self, config: Any):
        self.config = config
        self.target_key: str = ""
        self.type: str = "" # 'classification' or 'regression'

    @abstractmethod
    def load_manifest(self) -> Tuple[List[Any], List[Any]]:
        """
        Returns the full cohort manifest and stratification labels.
        """
        pass

    @abstractmethod
    def get_datasets(self, train_idx: List[int], val_idx: List[int], manifest: List[Any]) -> Tuple[Dataset, Dataset]:
        """
        Factory method to instantiate task-specific datasets based on fold indices.
        """
        pass

    def get_fold_iterator(self, manifest: List[Any], stratify_labels: List[Any], num_folds: int, seed: int):
        """
        Returns an iterator yielding (train_indices, val_indices).
        Default implementation is Stratified K-Fold. Override for fixed splits.
        """
        from sklearn.model_selection import StratifiedKFold
        skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
        return skf.split(manifest, stratify_labels)

    @abstractmethod
    def get_head(self, in_features: int) -> nn.Module:
        """Constructs the task-specific prediction head."""
        pass
    
    @abstractmethod
    def get_loss_fn(self, device: torch.device) -> nn.Module:
        """Returns the loss function appropriate for the task."""
        pass

    @abstractmethod
    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Computes task-specific metrics (e.g., AUC for classification, R2 for regression)."""
        pass

    def process_batch(self, batch: Dict[str, Any], device: torch.device, augment: bool = False) -> Dict[str, Any]:
        """
        Orchestrates the transition of data from CPU (DataLoader) to GPU (Model).
        Delegates geometric transformations to the PhysicsEngine.
        
        Args:
            batch: Raw dictionary from LazyHDF5Dataset.
            device: Computation device.
            augment: Whether to apply stochastic augmentations.
            
        Returns:
            x: Processed Left TMJ tensor.
            x_pair: Processed Right TMJ tensor.
            labels: Target labels on the correct device.
        """
        # Delegate to PhysicsEngine for:
        # 1. Anatomical Mirroring (Left -> Right)
        # 2. Affine Augmentation (if augment=True)
        # 3. Normalization (Z-Score)
        x, mask, x_pair, mask_pair = PhysicsEngine.process_siamese_batch(batch, self.config, device, augment)
        
        # Prepare labels
        labels = batch[self.target_key].to(device, non_blocking=True)
        
        return {
            'x': x, 'mask': mask,
            'x_pair': x_pair, 'mask_pair': mask_pair,
            'labels': labels
        }


class TMJCoreTask(BaseTask):
    """
    Concrete implementation for the Core Cohort (N=72).
    
    Supported Tasks:
    1. 'health_classification': Binary classification (Healthy vs Pathology).
    2. 'radiomics_baseline_regression': Regression of baseline condyle/fossa features.
    3. 'radiomics_prognosis_regression': Regression of follow-up (prognosis) features.
    
    Stratification:
    All tasks use 'Baseline_Health_status' for Stratified K-Fold to ensure 
    balanced distribution of pathological cases across folds.
    """
    def __init__(self, task_name: str, config: Any):
        super().__init__(config)
        self.task_name = task_name
        
        # Configure Task Specifics
        if task_name == 'health_classification':
            self.type = 'classification'
            self.target_key = 'target_health_status'
            self.out_dim = 2
            
        elif task_name == 'radiomics_baseline_regression':
            self.type = 'regression'
            self.target_key = 'target_rad_baseline'
            # Dimensions will be determined dynamically during manifest loading
            self.out_dim = None 
            
        elif task_name == 'radiomics_prognosis_regression':
            self.type = 'regression'
            self.target_key = 'target_rad_prognosis'
            self.out_dim = None
            
        else:
            raise ValueError(f"Unsupported task name: {task_name}")

    def load_manifest(self) -> Tuple[List[Dict], List[Any]]:
        logging.info(f"Scanning HDF5 ({self.config.EXTERNAL_HDF5_PATH}) to build manifest for '{self.task_name}'...")
        
        manifest = []
        stratify_labels = []
        
        with h5py.File(self.config.EXTERNAL_HDF5_PATH, 'r') as f:
            all_pids = sorted(list(f.keys()))
            
            for pid in all_pids:
                try:
                    # 1. Integrity Check: Ensure image data exists
                    _ = f[f"{pid}/baseline/left/image"]
                    _ = f[f"{pid}/baseline/right/image"]
                    
                    # 2. Load Metadata & Targets
                    # We load these into RAM immediately to avoid random seeking during training.
                    rad_group = f[f"{pid}/radiomics_selected_side"]
                    
                    # Health Status (Classification Target & Stratification Key)
                    health_status = int(rad_group.attrs['Baseline_Health_status'])
                    
                    # Radiomics Vectors (Regression Targets)
                    # Concatenate Condyle + Fossa features for a holistic morphological descriptor
                    rad_base_vec = np.concatenate([
                        rad_group['condyle_baseline'][:],
                        rad_group['fossa_baseline'][:]
                    ]).astype(np.float32)
                    
                    rad_prog_vec = np.concatenate([
                        rad_group['condyle_followup'][:],
                        rad_group['fossa_followup'][:]
                    ]).astype(np.float32)
                    
                    # 3. Build Patient Entry
                    patient_entry = {
                        'pid': pid,
                        'target_health_status': torch.tensor(health_status, dtype=torch.long),
                        'target_rad_baseline': torch.from_numpy(rad_base_vec),
                        'target_rad_prognosis': torch.from_numpy(rad_prog_vec)
                    }
                    
                    manifest.append(patient_entry)
                    stratify_labels.append(health_status)
                    
                except KeyError:
                    # Skip incomplete patients quietly (standard filtering)
                    continue
                except Exception as e:
                    logging.warning(f"Skipping PID {pid} due to error: {e}")
                    continue
        
        if not manifest:
            raise RuntimeError("Manifest generation failed: No valid patients found.")

        logging.info(f"Manifest built successfully. Valid cohort size N={len(manifest)}.")
        
        # Dynamically set output dimensions for regression tasks based on data
        if self.type == 'regression':
            sample_target = manifest[0][self.target_key]
            self.out_dim = sample_target.shape[0]
            logging.info(f"Regression output dimension set to {self.out_dim}.")
            
        return manifest, stratify_labels

    def get_datasets(self, train_idx: List[int], val_idx: List[int], manifest: List[Any]) -> Tuple[Dataset, Dataset]:
        full_ds = LazyHDF5Dataset(self.config.EXTERNAL_HDF5_PATH, manifest)
        
        from torch.utils.data import Subset
        return Subset(full_ds, train_idx), Subset(full_ds, val_idx)
    
    def get_head(self, in_features: int) -> nn.Module:
        """
        Returns the prediction head. 
        Uses a standard MLP design: LayerNorm -> Linear -> ReLU -> Linear.
        """
        if self.type == 'classification':
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, 256),
                nn.ReLU(),
                nn.Linear(256, self.out_dim)
            )
        elif self.type == 'regression':
            return nn.Sequential(
                nn.LayerNorm(in_features),
                nn.Linear(in_features, 128),
                nn.ReLU(),
                nn.Linear(128, self.out_dim)
            )
        else:
             raise ValueError(f"Unknown task type: {self.type}")

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        if self.type == 'classification':
            return nn.CrossEntropyLoss().to(device)
        elif self.type == 'regression':
            return nn.MSELoss().to(device)
        else:
            raise ValueError(f"Unknown task type: {self.type}")

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes scientifically relevant metrics.
        Handles edge cases like single-class batches in classification.
        """
        metrics = {}
        
        if self.type == 'classification':
            # Apply Softmax to get probabilities
            # y_pred is expected to be raw logits
            probs = torch.softmax(torch.from_numpy(y_pred).float(), dim=1).numpy()
            preds_labels = np.argmax(probs, axis=1)
            
            # Robust AUC calculation
            # If only one class is present in y_true (e.g., small batch), AUC is undefined.
            try:
                if len(np.unique(y_true)) > 1:
                    auc_score = roc_auc_score(y_true, probs[:, 1])
                else:
                    auc_score = np.nan
            except ValueError:
                auc_score = np.nan
                
            metrics = {
                'auc': auc_score,
                'acc': accuracy_score(y_true, preds_labels),
                'f1': f1_score(y_true, preds_labels, zero_division=0)
            }
            
        elif self.type == 'regression':
            # Compute R2 (Variance Weighted), MAE, and MSE
            # Note: y_true and y_pred must be in the original physical scale (not Z-scored)
            try:
                r2 = r2_score(y_true, y_pred, multioutput='variance_weighted')
                mae = mean_absolute_error(y_true, y_pred)
                mse = mean_squared_error(y_true, y_pred)
                
                metrics = {
                    'r2': r2,
                    'mae': mae,
                    'mse': mse,
                    'rmse': np.sqrt(mse)
                }
            except Exception as e:
                logging.warning(f"Metric calculation failed: {e}")
                metrics = {'r2': np.nan, 'mae': np.nan, 'mse': np.nan}
                
        return {k: float(v) for k, v in metrics.items()}
  

class TMJSegmentationTask(BaseTask):
    """
    Downstream Task: 3D Anatomical Segmentation (Multi-class).

    **Objective:**
    Probes the spatial semantic quality of the frozen backbone features. 
    Unlike U-Net, this task uses a lightweight 'Linear Probe' decoder to ensure 
    performance is driven by the encoder's representational power, not the decoder's depth.

    **Classes:**
    0: Background
    1: Mandible (Condyle)
    2: Skull (Temporal Bone/Fossa)

    **Data Source:**
    Strictly uses the 'val' split from the pre-training dataset to prevent data leakage.
    """

    def __init__(self, task_name: str, config: Any, tmj_dataset_cls: Any, gpu_process_fn: Any, pt_config: Any = None):
        super().__init__(config)
        self.task_name = task_name
        self.type = 'segmentation'
        # Expects the modified train_maeswin.py to return this key
        self.target_key = 'dense_mask' 
        self.out_dim = 3 
        
        # Injected dependencies from the pre-training script (train_maeswin.py)
        self.tmj_dataset_cls = tmj_dataset_cls
        self.gpu_process_fn = gpu_process_fn

        self.pt_config = pt_config 
        if self.pt_config is None:
             raise ValueError("pt_config is required for TMJSegmentationTask to run gpu_process_fn")
        
        self._cached_train_idx = None
        self._cached_val_idx = None
    
    def load_manifest(self) -> Tuple[List[Any], List[Any]]:
        """
        Loads or Creates a persistent split for the segmentation task.
        
        Algorithm:
        1. Check for 'segmentation_downstream_split.json' alongside the pretrain split file.
        2. If found -> Load manifest and indices directly.
        3. If missing -> Parse HDF5, split by Patient ID (to prevent leakage), 
           save the JSON artifact, and then return.
        """
        seg_split_path = Path('segmentation_downstream_split.json')
        base_split_path = Path(self.config.PRETRAIN_SPLIT_FILE)
        
        if seg_split_path.exists():
            return self._load_existing_split(seg_split_path)
        else:
            return self._generate_and_save_split(seg_split_path, base_split_path)
        
    def _load_existing_split(self, json_path: Path) -> Tuple[List[Any], List[Any]]:
        logging.info(f"Loading persistent segmentation split from: {json_path}")
        
        with open(json_path, 'r') as f:
            data = json.load(f)
            
        train_identifiers = data['train'] # List of [study_id, side]
        val_identifiers = data['val']     # List of [study_id, side]
        
        manifest = train_identifiers + val_identifiers
        
        self._cached_train_idx = list(range(len(train_identifiers)))
        self._cached_val_idx = list(range(len(train_identifiers), len(manifest)))
        
        logging.info(f"Loaded Cohort: {len(manifest)} joints (Train: {len(train_identifiers)}, Val: {len(val_identifiers)})")
        
        # Dummy stratify labels (not used for fixed split)
        return manifest, [0] * len(manifest)

    def _generate_and_save_split(self, output_path: Path, pretrain_split_path: Path) -> Tuple[List[Any], List[Any]]:
        logging.info("Generating NEW segmentation split (preserving patient isolation)...")
        
        with open(pretrain_split_path, 'r') as f:
            allowed_study_ids = set(json.load(f).get('val', []))
            
        if not allowed_study_ids:
            raise RuntimeError("Pre-train split file has no 'val' entries.")

        patient_to_joints = defaultdict(list)
        hdf5_path = self.config.PRETRAIN_HDF5_PATH
        
        with h5py.File(hdf5_path, 'r') as f:
            for study_id in allowed_study_ids:
                if study_id not in f: continue
                
                grp = f[study_id]
                raw_pid = grp.attrs.get('patient_id')
                if raw_pid is None: continue
                
                pid = raw_pid.decode('utf-8') if isinstance(raw_pid, bytes) else str(raw_pid)
                
                if 'right_joint' in grp: patient_to_joints[pid].append([study_id, 'right_joint'])
                if 'left_joint' in grp: patient_to_joints[pid].append([study_id, 'left_joint'])

        unique_pids = sorted(list(patient_to_joints.keys()))
        rng = random.Random(self.config.SEED)
        rng.shuffle(unique_pids)
        
        split_idx = int(len(unique_pids) * 0.8) # 80% Train
        train_pids = unique_pids[:split_idx]
        val_pids = unique_pids[split_idx:]
        
        train_identifiers = []
        for pid in train_pids:
            train_identifiers.extend(patient_to_joints[pid])
            
        val_identifiers = []
        for pid in val_pids:
            val_identifiers.extend(patient_to_joints[pid])
            
        payload = {
            'meta': {
                'source': str(pretrain_split_path),
                'seed': self.config.SEED,
                'train_ratio': 0.8
            },
            'train': train_identifiers,
            'val': val_identifiers
        }
        
        with open(output_path, 'w') as f:
            json.dump(payload, f, indent=4)
        logging.info(f"Split artifact saved to: {output_path}")

        return self._load_existing_split(output_path)
    
    def get_fold_iterator(self, manifest: List[Any], stratify_labels: List[Any], num_folds: int, seed: int):
        """
        Override standard K-Fold. Returns a single generator yielding the fixed split.
        """
        # Yields exactly one tuple (train, val)
        yield (self._cached_train_idx, self._cached_val_idx)

    def get_datasets(self, train_idx: List[int], val_idx: List[int], manifest: List[Any]) -> Tuple[Dataset, Dataset]:
        """
        Instantiates the injected TMJDataset using the resolved identifiers.
        """
        # Extract sub-lists of identifiers
        train_ids = [manifest[i] for i in train_idx]
        val_ids = [manifest[i] for i in val_idx]
        
        # Config for Val needs to disable augmentation
        pt_cfg_val = copy.deepcopy(self.pt_config)
        pt_cfg_val.AUG_ROTATION_DEGREES = [0.0, 0.0]
        pt_cfg_val.AUG_SCALE = [1.0, 1.0]
        
        ds_train = self.tmj_dataset_cls(self.pt_config.DATA_HDF5_PATH, train_ids, self.pt_config)
        ds_val = self.tmj_dataset_cls(self.pt_config.DATA_HDF5_PATH, val_ids, pt_cfg_val)
        
        return ds_train, ds_val

    def get_head(self, in_features: int) -> nn.Module:
        """
        Constructs the Segmentation Probe.
        
        Architecture follows the 'Project-then-Upsample' paradigm:
        1. Linear Projection: (C_in) -> (Num_Classes) in low-res latent space.
        2. Trilinear Upsample: Low-res logits -> Target Image Size.
        """
        target_shape = self.config.TARGET_IMAGE_SIZE # e.g., (224, 224, 224)
        head_type = getattr(self.config, 'SEG_HEAD_TYPE', 'linear')
        
        class SegmentationProbe(nn.Module):
            def __init__(self, dim, num_classes, mode):
                super().__init__()
                self.target_shape = target_shape
                
                if mode == 'linear':
                    # Strict Linear Probe: 1x1 Conv only. 
                    # Tests if features are linearly separable.
                    self.net = nn.Conv3d(dim, num_classes, kernel_size=1)
                    
                elif mode == '2layer':
                    # Non-linear Probe: Allows minimal local context aggregation (3x3).
                    inter_dim = max(32, dim // 4)
                    self.net = nn.Sequential(
                        nn.Conv3d(dim, inter_dim, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm3d(inter_dim),
                        nn.ReLU(inplace=True),
                        nn.Conv3d(inter_dim, num_classes, kernel_size=1)
                    )
                else:
                    raise ValueError(f"Unknown SEG_HEAD_TYPE: {mode}")

            def forward(self, x_tuple):
                # Support output from TMJAnalysisModel which might return a tuple (L, R) or single tensor
                # In this task, we process joints independently, so we expect a single tensor or take the first.
                x_in = x_tuple[0] if isinstance(x_tuple, (tuple, list)) else x_tuple
                
                # Safety Check: Ensure input is 5D (B, C, D, H, W)
                if x_in.dim() != 5:
                    raise RuntimeError(
                        f"Segmentation Head expects 5D tensor, got {x_in.dim()}D. "
                        "Ensure TMJAnalysisModel is in 'segmentation' inference_mode."
                    )

                # 1. Projection (Low Resolution)
                logits_low_res = self.net(x_in)
                
                # 2. Upsampling (High Resolution)
                # We upsample LOGITS, not features, to save memory.
                logits_high_res = F.interpolate(
                    logits_low_res, 
                    size=self.target_shape, 
                    mode='trilinear', 
                    align_corners=False
                )
                return logits_high_res

        return SegmentationProbe(in_features, self.out_dim, mode=head_type)

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        """
        Returns the composite loss: Dice + CrossEntropy.
        """
        # Dice Loss: Ignores background to focus on bone structures
        dice_loss = DiceLoss(
            to_onehot_y=True, 
            softmax=True, 
            include_background=False 
        )
        
        # CE Loss: Weights balanced slightly against background dominance
        # Background (0): 0.1, Mandible (1): 1.0, Skull (2): 1.0
        class_weights = torch.tensor([0.1, 1.0, 1.0], device=device)
        ce_loss = CrossEntropyLoss(weight=class_weights)
        
        def composite_loss(preds, targets):
            # preds: (B, 3, D, H, W) logits
            # targets: (B, D, H, W) long indices
            
            # Dice expects channel dim in targets for internal one-hot conversion
            l_dice = dice_loss(preds, targets.unsqueeze(1))
            l_ce = ce_loss(preds, targets.long())
            
            return l_dice + l_ce
            
        return composite_loss

    @contextmanager
    def _augmentation_lock(self, augment: bool):
        """
        Context manager to temporarily override PRE-TRAIN config for deterministic validation.
        Must operate on self.pt_config because that is what gpu_process_fn consumes.
        """
        # 1. Save original values from pt_config
        # Note: Attribute names must match train_maeswin.Config exactly
        orig_rot = self.pt_config.AUG_ROTATION_DEGREES
        orig_scale = self.pt_config.AUG_SCALE 
        orig_trans = getattr(self.pt_config, 'AUG_TRANSLATE_PERCENT_RANGE', [0.0, 0.0])
        
        try:
            if not augment:
                # Force Identity Transform
                self.pt_config.AUG_ROTATION_DEGREES = [0.0, 0.0]
                self.pt_config.AUG_SCALE = [1.0, 1.0]
                self.pt_config.AUG_TRANSLATE_PERCENT_RANGE = [0.0, 0.0]
            
            # Yield control back to process_batch
            yield
            
        finally:
            # 2. Restore original values strictly
            self.pt_config.AUG_ROTATION_DEGREES = orig_rot
            self.pt_config.AUG_SCALE = orig_scale
            self.pt_config.AUG_TRANSLATE_PERCENT_RANGE = orig_trans

    def process_batch(self, batch: Dict[str, Any], device: torch.device, augment: bool = False) -> Dict[str, Any]:
        """
        Processes a batch from TMJDataset using the pre-training GPU pipeline.
        Returns:
            x: The processed image tensor (B, 1, D, H, W).
            labels: The dense semantic mask (B, D, H, W).
        """
        # 1. Transfer to GPU
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                # Stack lists of tensors if collate didn't (safety net)
                batch[k] = torch.stack(v).to(device, non_blocking=True)

        # 2. Process with Locked Config
        with self._augmentation_lock(augment):
            processed = self.gpu_process_fn(
                batch, self.pt_config, epoch=0, hu_augmenter=None, noise_simulator=None
            )

        # 3. Extract Tensors
        inp = processed['input_cube'] # (B, 1, D, H, W)
        
        # Check for updated return signature from train_maeswin
        if 'dense_mask' in processed:
            mask_full = processed['dense_mask'] # (B, 1, D, H, W)
        else:
             raise RuntimeError(
                 "Incompatible 'process_batch_on_gpu' signature. "
                 "Update train_maeswin.py to return 'dense_mask'."
             )

        # Prepare Label: (B, D, H, W) Long Tensor
        labels = mask_full.long().squeeze(1)
        
        # Note: We treat every joint in the batch as an independent sample.
        # L/R splitting is irrelevant for segmentation unless comparing side-symmetry,
        # which is not the goal of this probe.
        return {
            'x': inp,
            'x_pair': None, 
            'mask': None, 
            'mask_pair': None,
            'labels': labels
        }

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes Dice Similarity Coefficient (DSC) for each anatomical structure.
        
        Args:
            y_true: (B, D, H, W) Ground Truth indices {0, 1, 2}
            y_pred: (B, 3, D, H, W) Logits from model
        """
        # Get hard class predictions
        preds = np.argmax(y_pred, axis=1) # (B, D, H, W)
        
        # Flatten for efficient computation
        preds_flat = preds.ravel()
        true_flat = y_true.ravel()
        
        metrics = {}
        
        def calc_dsc_class(p, t, c_idx):
            # Standard DSC formula: 2*|A n B| / (|A| + |B|)
            inter = np.logical_and(p == c_idx, t == c_idx).sum()
            union = (p == c_idx).sum() + (t == c_idx).sum()
            
            if union == 0: 
                return 1.0 # Correctly predicted empty space
            return (2. * inter) / (union + 1e-8)

        # Class 1: Mandible (Condyle)
        d_man = calc_dsc_class(preds_flat, true_flat, 1)
        metrics['dsc_mandible'] = d_man
        
        # Class 2: Skull (Fossa)
        d_skull = calc_dsc_class(preds_flat, true_flat, 2)
        metrics['dsc_skull'] = d_skull
        
        # Mean Dice (Excluding background)
        metrics['dsc_mean'] = (d_man + d_skull) / 2.0
        
        return {k: float(v) for k, v in metrics.items()}

    def calculate_geometric_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """
        Computes geometric metrics (HD95, ASSD) for the best model evaluation.
        
        ACADEMIC NOTE:
        These metrics evaluate surface alignment accuracy, which is critical for 
        surgical planning reliability. We use MONAI's implementation for robust 
        voxel-wise distance calculation.
        
        Args:
            y_true: (B, D, H, W) Ground Truth indices.
            y_pred: (B, C, D, H, W) Logits from model.
        
        Returns:
            Dictionary with HD95 and ASSD for relevant classes.
        """
        # 1. Convert inputs to One-Hot format (B, C, D, H, W) required by MONAI
        # y_pred is logits -> argmax -> one_hot
        n_classes = self.out_dim
        
        # Convert to Tensor for MONAI utilities
        pred_tensor = torch.from_numpy(y_pred)
        true_tensor = torch.from_numpy(y_true).long().unsqueeze(1) # (B, 1, D, H, W)
        
        # Post-processing transforms
        post_pred = AsDiscrete(argmax=True, to_onehot=n_classes, dim=1)
        post_label = AsDiscrete(to_onehot=n_classes, dim=1)
        
        pred_onehot = post_pred(pred_tensor)
        true_onehot = post_label(true_tensor)
        
        metrics = {}
        
        # 2. Compute Metrics (excluding background channel 0)
        # Percentile 95 is standard to eliminate outlier artifacts
        hd95 = compute_hausdorff_distance(
            pred_onehot, true_onehot, 
            include_background=False, 
            percentile=95.0,
            spacing=None # Assume voxel space (1.0, 1.0, 1.0) if physical spacing unavailable
        )
        
        assd = compute_average_surface_distance(
            pred_onehot, true_onehot, 
            include_background=False,
            spacing=None
        )
        
        # 3. Aggregate (Handle NaNs for empty masks safely)
        # MONAI returns (B, C-1). We average across batch and define specific class metrics.
        
        def safe_mean(tensor, idx):
            # Extract column for specific class
            vals = tensor[:, idx]
            # We explicitly exclude Inf (empty mask distance) and NaN from the mean calculation.
            # This is the "manual handling" of the empty class issue.
            valid_mask = torch.isfinite(vals)
            if valid_mask.sum() == 0:
                return np.nan
            return vals[valid_mask].mean().item()

        # Class indices in result (since background is removed): 0 -> Mandible, 1 -> Skull
        metrics['hd95_mandible'] = safe_mean(hd95, 0)
        metrics['hd95_skull'] = safe_mean(hd95, 1)
        metrics['hd95_mean'] = np.nanmean([metrics['hd95_mandible'], metrics['hd95_skull']])
        
        metrics['assd_mandible'] = safe_mean(assd, 0)
        metrics['assd_skull'] = safe_mean(assd, 1)
        metrics['assd_mean'] = np.nanmean([metrics['assd_mandible'], metrics['assd_skull']])
        
        return metrics


# class TMJMRITask(BaseTask):
#     """
#     Joint-Level MRI Analysis Task (Multi-Label/Multi-Task).
    
#     Academic Rigor:
#     1. Group Split: Splitting is strictly by 'patient_id' to avoid leakage from multiple scans per patient.
#     2. Masked Loss: Handles missing values (NaN) in the dataset without discarding samples.
#     3. Rebalancing: Calculates positive weights for BCE loss based on training fold statistics.
#     4. Normalization: Joint Space regression targets are Z-scored using Train-only statistics.
#     """
#     def __init__(self, task_name: str, config: Any, tmj_dataset_cls: Any, gpu_process_fn: Any, pt_config: Any):
#         super().__init__(config)
#         self.task_name = task_name
#         self.type = 'mri_analysis'
#         self.target_key = 'labels'
        
#         self.tmj_dataset_cls = tmj_dataset_cls
#         self.gpu_process_fn = gpu_process_fn
#         self.pt_config = pt_config
        
#         # Statistics placeholders
#         self.train_pos_weights = None # For classification rebalancing
#         self.train_space_stats = None # For regression normalization

#     def load_manifest(self) -> Tuple[List[Any], List[Any]]:
#         """
#         Scans HDF5 attributes directly to build the cohort.
#         Groups scans by Patient ID.
#         """
#         logging.info(f"Scanning HDF5 {self.config.MRI_HDF5_PATH} for MRI labels...")
        
#         self.targets_lookup = {}
#         patient_groups = defaultdict(list) # patient_id -> list of (doc_id, side, inflam_status)
        
#         with h5py.File(self.config.MRI_HDF5_PATH, 'r') as f:
#             doc_ids = list(f.keys())
            
#             for doc_id in doc_ids:
#                 grp = f[doc_id]
                
#                 # Robust extraction of Patient ID (critical for grouping)
#                 p_id_attr = grp.attrs.get('patient_id', 'unknown')
#                 p_id = str(p_id_attr).strip()
#                 if not p_id or p_id == 'nan':
#                     # Fallback: if patient_id missing, assume doc_id is unique patient (risky but necessary fallback)
#                     p_id = f"doc_{doc_id}"

#                 # Iterate sides
#                 for side_key, side_code in [('left_joint', 'L'), ('right_joint', 'R')]:
#                     if side_key not in grp:
#                         continue
                        
#                     sub = grp[side_key]
#                     attrs = sub.attrs
                    
#                     # 1. Parse Targets
#                     # Helper to safely parse float or string with comma
#                     def parse_val(key):
#                         val = attrs.get(key, float('nan'))
#                         if isinstance(val, bytes):
#                             val = val.decode('utf-8')
#                         if isinstance(val, str):
#                             val = val.replace(',', '.').strip()
#                             if val == '': return float('nan')
#                         return float(val)

#                     bone = parse_val('BoneStructure')
#                     disc = parse_val('DiscPosition')
#                     inflam = parse_val('Inflammation')
#                     space = parse_val('JointSpace')
                    
#                     # Store tensor [4]
#                     target_vec = torch.tensor([bone, disc, inflam, space], dtype=torch.float32)
#                     self.targets_lookup[(doc_id, side_key)] = target_vec
                    
#                     # Store metadata for stratification
#                     # We use Inflammation as the primary stratifier (most severe condition)
#                     strat_label = int(inflam) if not np.isnan(inflam) else 0
#                     patient_groups[p_id].append({
#                         'doc_id': doc_id,
#                         'side': side_key,
#                         'strat_label': strat_label
#                     })

#         # Convert groups to flat list for K-Fold
#         # We need parallel lists: groups (p_ids) and stratification targets (max inflam per patient)
#         unique_pids = sorted(list(patient_groups.keys()))
#         group_strat_labels = []
        
#         for pid in unique_pids:
#             # Determine patient-level label: Max inflammation across all their scans/sides
#             scans = patient_groups[pid]
#             max_inflam = max([s['strat_label'] for s in scans])
#             group_strat_labels.append(max_inflam)
            
#         logging.info(f"Manifest loaded: {len(unique_pids)} patients, {len(self.targets_lookup)} joints.")
        
#         # Return unique PIDs and their labels. Detailed joint list is reconstructed in get_datasets.
#         return unique_pids, group_strat_labels

#     def get_fold_iterator(self, manifest: List[str], stratify_labels: List[int], num_folds: int, seed: int):
#         # manifest is a list of Patient IDs (strings)
#         split_artifact_path = self.config.BASE_OUTPUT_DIR / f"mri_split_seed{seed}_k{num_folds}.json"
        
#         # Helper: Map PID -> Index for efficient lookup during restoration
#         pid_to_idx = {pid: i for i, pid in enumerate(manifest)}
        
#         # 1. Load Existing Split (Reproducibility Mode)
#         if split_artifact_path.exists():
#             logging.info(f"Loading persistent data split from {split_artifact_path}")
#             with open(split_artifact_path, 'r') as f:
#                 splits_data = json.load(f)
                
#             for fold_key in sorted(splits_data.keys()):
#                 fold_data = splits_data[fold_key]
#                 # Reconstruct indices from saved PIDs
#                 try:
#                     train_idx = [pid_to_idx[pid] for pid in fold_data['train']]
#                     val_idx = [pid_to_idx[pid] for pid in fold_data['val']]
#                     yield np.array(train_idx), np.array(val_idx)
#                 except KeyError as e:
#                     raise RuntimeError(f"Patient ID {e} from saved split not found in current HDF5 manifest!")

#         # 2. Generate New Split (Discovery Mode)
#         else:
#             logging.info(f"Generating NEW split (Stratified K-Fold) and saving to {split_artifact_path}...")
#             from sklearn.model_selection import StratifiedKFold
#             skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            
#             # We must materialize the generator to save it
#             all_folds_indices = list(skf.split(manifest, stratify_labels))
            
#             serializable_splits = {}
#             for fold_i, (train_idx, val_idx) in enumerate(all_folds_indices):
#                 # Save Patient IDs (Strings), not indices (Integers), for robustness
#                 serializable_splits[f"fold_{fold_i}"] = {
#                     'train': [manifest[i] for i in train_idx],
#                     'val': [manifest[i] for i in val_idx]
#                 }
            
#             # Atomic Write
#             with open(split_artifact_path, 'w') as f:
#                 json.dump(serializable_splits, f, indent=4)
            
#             # Yield the indices we just generated
#             for train_idx, val_idx in all_folds_indices:
#                 yield train_idx, val_idx

#     def get_datasets(self, train_idx: List[int], val_idx: List[int], manifest: List[str]) -> Tuple[Dataset, Dataset]:
#         """
#         Reconstructs joint-level lists from patient-level indices and calculates stats.
#         """
#         # 1. Resolve indices to Patient IDs
#         train_pids = set([manifest[i] for i in train_idx])
#         val_pids = set([manifest[i] for i in val_idx])
        
#         # 2. Flatten to Joint List [(doc_id, side), ...]
#         train_identifiers = []
#         val_identifiers = []
        
#         # We iterate over the lookup keys to assign them to splits
#         train_targets_accum = [] # For stats calculation
        
#         for (doc_id, side), target_vec in self.targets_lookup.items():
#             # Find which patient this doc_id belongs to
#             # This requires a reverse lookup or re-opening file. 
#             # Optimization: Embed pid in targets_lookup key or store a map in load_manifest.
#             # To stay efficient, we re-read the attribute cached in the lookup if we stored it,
#             # OR we simply iterate the `patient_groups` constructed earlier (which we should persist).
#             pass 
        
#         # REVISION: Let's assume we stored patient_groups in `self` during load_manifest
#         # Since BaseTask structure might not persist state between load_manifest and here easily without modification,
#         # we will reconstruct the map efficiently.
        
#         # To make this robust:
#         # We need to map (doc_id) -> p_id. 
#         # Let's perform a quick pass on self.targets_lookup keys against the file or a cached map.
#         # Ideally, load_manifest should populate a self.doc_to_pid map.
        
#         # Let's rebuild the split lists using the HDF5 structure again (fast enough for metadata)
#         with h5py.File(self.config.MRI_HDF5_PATH, 'r') as f:
#             for doc_id in f.keys():
#                 p_id_attr = f[doc_id].attrs.get('patient_id', f"doc_{doc_id}")
#                 p_id = str(p_id_attr).strip()
#                 if not p_id or p_id == 'nan': p_id = f"doc_{doc_id}"
                
#                 is_train = p_id in train_pids
                
#                 for side in ['left_joint', 'right_joint']:
#                     if side in f[doc_id]:
#                         ident = (doc_id, side)
#                         if ident not in self.targets_lookup: continue # Skip if no targets
                        
#                         if is_train:
#                             train_identifiers.append(ident)
#                             train_targets_accum.append(self.targets_lookup[ident])
#                         elif p_id in val_pids:
#                             val_identifiers.append(ident)

#         # 3. Compute Train Statistics for Normalization/Rebalancing
#         if train_targets_accum:
#             t_tensor = torch.stack(train_targets_accum) # [N_train, 4]
            
#             # A. Regression Stats (JointSpace - idx 3)
#             space_vals = t_tensor[:, 3]
#             valid_space = space_vals[~torch.isnan(space_vals)]
            
#             self.train_space_stats = {
#                 'mean': valid_space.mean().item(),
#                 'std': valid_space.std().item() + 1e-6
#             }
#             logging.info(f"Regression Norm Stats: {self.train_space_stats}")
            
#             # B. Classification Pos Weights (Bone, Disc, Inflam - idx 0,1,2)
#             # pos_weight = (num_neg) / (num_pos)
#             pos_weights = []
#             for i in range(3):
#                 vals = t_tensor[:, i]
#                 valid = vals[~torch.isnan(vals)]
#                 n_pos = (valid == 1).sum()
#                 n_neg = (valid == 0).sum()
                
#                 if n_pos > 0:
#                     w = n_neg / float(n_pos)
#                 else:
#                     w = 1.0 # Fallback
#                 pos_weights.append(w)
            
#             self.train_pos_weights = torch.tensor(pos_weights)
#             logging.info(f"Class Pos Weights (Bone, Disc, Inflam): {self.train_pos_weights}")
            
#         else:
#             logging.warning("No training data found! Stats will be default.")
#             self.train_space_stats = {'mean': 0.0, 'std': 1.0}
#             self.train_pos_weights = torch.ones(3)

#         # 4. Instantiate Datasets
#         # Val config needs augmentation disabled
#         pt_cfg_val = copy.deepcopy(self.pt_config)
#         pt_cfg_val.AUG_ROTATION_DEGREES = [0.0, 0.0]
#         pt_cfg_val.AUG_SCALE = [1.0, 1.0]
        
#         ds_train = LabeledTMJDataset(self.tmj_dataset_cls, self.config.MRI_HDF5_PATH, 
#                                      train_identifiers, self.pt_config, self.targets_lookup)
#         ds_val = LabeledTMJDataset(self.tmj_dataset_cls, self.config.MRI_HDF5_PATH, 
#                                    val_identifiers, pt_cfg_val, self.targets_lookup)
        
#         return ds_train, ds_val

#     def get_head(self, in_features: int) -> nn.Module:
#         from models import MRIAnalysisHead
#         return MRIAnalysisHead(in_features)

#     def process_batch(self, batch: Dict[str, Any], device: torch.device, augment: bool = False) -> Dict[str, Any]:
#         """
#         Processing for Independent Joint Analysis.
#         Since LabeledTMJDataset yields single joints (doc_id, side), the batch
#         coming from DataLoader is already [B, D, H, W]. We don't need to explode it.
#         We just need to ensure the TMJAnalysisModel treats it correctly.
#         """
#         for k, v in batch.items():
#             if isinstance(v, torch.Tensor):
#                 batch[k] = v.to(device, non_blocking=True)
#             elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
#                 # Handle lists of tensors if collate didn't stack them
#                 batch[k] = torch.stack(v).to(device, non_blocking=True)
#         # 1. Use GPU Process Function (from Pre-train)
#         # This handles Affine Transforms, Intensity Norm, Masking
#         # Note: gpu_process_fn expects keys 'sub_volume', 'inv_rotation' etc. which are in 'batch'
        
#         # Temporarily disable augment via config lock if needed, or pass epoch param
#         # Assuming gpu_process_fn respects config in batch or global config passed
        
        
#         processed = self.gpu_process_fn(
#             batch, self.pt_config, epoch=0 if not augment else 100, # Hack to enable/disable aug via epoch check
#             hu_augmenter=None, noise_simulator=None
#         )
        
#         inp = processed['input_cube'] # [B, 1, D, H, W]
#         # We don't need masks for the model input in 'full_image' mode usually, 
#         # but if using 'background_mask' inference mode:
#         mask_valid = processed.get('valid_for_masking_mask')

#         # 2. Process Labels
#         # Raw: [B, 4]
#         targets = batch['labels'].to(device).float()
        
#         # Normalize Regression Target (Index 3) for Training Loss
#         # We clone to avoid modifying the original batch which might be used for metrics later
#         targets_norm = targets.clone()
        
#         mask_space = ~torch.isnan(targets[:, 3])
#         if self.train_space_stats and mask_space.any():
#             mu = self.train_space_stats['mean']
#             sigma = self.train_space_stats['std']
#             targets_norm[mask_space, 3] = (targets[mask_space, 3] - mu) / sigma

#         # 3. Structure for Model
#         # TMJAnalysisModel with x_pair=None triggers Independent Mode
#         return {
#             'x': inp,
#             'x_pair': None,
#             'mask': mask_valid, # Optional
#             'mask_pair': None,
#             'labels': targets_norm, # Normalized for Loss
#             'labels_raw': targets   # Original for Metrics
#         }

#     def get_loss_fn(self, device: torch.device) -> nn.Module:
#         """
#         Masked Multi-Task Loss with dynamic rebalancing.
#         """
#         # Weights
#         pos_weights = self.train_pos_weights.to(device) if self.train_pos_weights is not None else torch.ones(3, device=device)
        
#         # Loss definitions (reduction='none' is mandatory for masking)
#         crit_bone = BCEWithLogitsLoss(pos_weight=pos_weights[0], reduction='none')
#         crit_disc = BCEWithLogitsLoss(pos_weight=pos_weights[1], reduction='none')
#         crit_inflam = BCEWithLogitsLoss(pos_weight=pos_weights[2], reduction='none')
#         crit_space = nn.L1Loss(reduction='none') # MAE for regression is robust
        
#         def masked_loss(preds, targets):
#             """
#             preds: [B, 4]
#             targets: [B, 4] (Normalized, contains NaNs)
#             """
#             loss_accum = 0.0
            
#             # Helper to compute masked mean
#             def apply(crit, idx):
#                 p = preds[:, idx]
#                 t = targets[:, idx]
#                 mask = ~torch.isnan(t)
#                 if mask.sum() == 0: return 0.0
                
#                 loss_elem = crit(p[mask], t[mask])
#                 return loss_elem.mean()

#             l_bone = apply(crit_bone, 0)
#             l_disc = apply(crit_disc, 1)
#             l_inflam = apply(crit_inflam, 2)
#             l_space = apply(crit_space, 3)
            
#             # Simple weighting, can be tuned
#             return l_bone + l_disc + l_inflam + l_space

#         return masked_loss

#     def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
#         """
#         Args:
#             y_true: [N, 4] Raw physical values
#             y_pred: [N, 4] Raw logits / Normalized Z-scores
#         """
#         metrics = {}
        
#         # Classification (0, 1, 2)
#         names = ['bone', 'disc', 'inflam']
#         for i, name in enumerate(names):
#             # Filter NaNs
#             mask = ~np.isnan(y_true[:, i])
#             if mask.sum() == 0: continue
            
#             yt = y_true[mask, i]
#             yp_logits = y_pred[mask, i]
#             yp_probs = 1 / (1 + np.exp(-yp_logits))
#             yp_labels = (yp_probs > 0.5).astype(int)
            
#             metrics[f'{name}_acc'] = balanced_accuracy_score(yt, yp_labels)
#             metrics[f'{name}_f1'] = f1_score(yt, yp_labels, average='binary')
#             try:
#                 metrics[f'{name}_auc'] = roc_auc_score(yt, yp_probs)
#             except:
#                 metrics[f'{name}_auc'] = 0.0

#         # Regression (3)
#         mask_reg = ~np.isnan(y_true[:, 3])
#         if mask_reg.sum() > 0:
#             yt_mm = y_true[mask_reg, 3]
#             yp_z = y_pred[mask_reg, 3]
            
#             # Denormalize
#             mu = self.train_space_stats['mean']
#             sigma = self.train_space_stats['std']
#             yp_mm = (yp_z * sigma) + mu
            
#             metrics['space_mae'] = mean_absolute_error(yt_mm, yp_mm)
            
#         return metrics


class BaseClinicalTask(BaseTask):
    """
    Abstract Base Class for Clinical MRI Analysis Tasks.
    
    This class orchestrates the lifecycle of a specific clinical downstream task
    (e.g., Inflammation Detection, Bone Structure Analysis). It standardizes:
    1. HDF5 Data Scanning & Label Filtering (removing NaNs).
    2. Patient-Level Stratification (preventing data leakage).
    3. Cross-Validation Split Management (Reproducibility).
    4. Fold-Specific Statistics Calculation (Normalization/Rebalancing).
    5. GPU Data Processing & Preprocessing.

    Attributes:
        task_name (str): Unique identifier for the task (e.g., 'mri_inflam').
        target_key (str): Key used in the batch dictionary (default: 'labels').
        targets_lookup (Dict): Mapping from (doc_id, side) -> target_tensor.
        train_stats (Dict): Stores mean/std for regression normalization.
        train_pos_weight (Tensor): Stores class weights for binary classification.
    """

    def __init__(
        self, 
        task_name: str, 
        config: Any, 
        tmj_dataset_cls: Any, 
        gpu_process_fn: Any, 
        pt_config: Any
    ):
        """
        Args:
            task_name: Unique string ID for the task (used for saving splits).
            config: The global EvalConfig object.
            tmj_dataset_cls: The Dataset class definition from pre-training logic.
            gpu_process_fn: The GPU-based preprocessing function from pre-training.
            pt_config: Configuration object from the pre-training script.
        """
        super().__init__(config)
        self.task_name = task_name
        self.tmj_dataset_cls = tmj_dataset_cls
        self.gpu_process_fn = gpu_process_fn
        self.pt_config = pt_config
        
        # Standard key for the targets in the batch dictionary
        self.target_key = 'labels' 
        
        # In-memory storage for targets to avoid repeated HDF5 attribute access
        self.targets_lookup: Dict[Tuple[str, str], torch.Tensor] = {}
        
        # Mapping from Patient ID to list of associated Joint Identifiers
        # Essential for reconstructing joint-level datasets from patient-level splits
        self.patient_to_joints: Dict[str, List[Tuple[str, str]]] = defaultdict(list)
        
        # Placeholders for fold-specific statistics
        self.train_pos_weight: Optional[torch.Tensor] = None 
        self.train_stats: Optional[Dict[str, float]] = None 

    @property
    @abstractmethod
    def target_column_key(self) -> str:
        """
        Abstract Property: Must return the exact attribute key string 
        used in the HDF5 file (e.g., 'Inflammation', 'JointSpace').
        """
        pass

    def load_manifest(self) -> Tuple[List[str], List[int]]:
        """
        Scans the HDF5 dataset to build a cohort of valid patients.
        
        Filters:
        1. Patients with missing IDs.
        2. Joints with NaN values for the specific task target.
        
        Returns:
            unique_pids (List[str]): List of valid Patient IDs.
            group_strat_labels (List[int]): Stratification labels corresponding to unique_pids.
        """
        logging.info(f"[{self.task_name}] Scanning HDF5 for valid targets (Key: '{self.target_column_key}')...")
        
        self.targets_lookup = {}
        self.patient_to_joints = defaultdict(list)
        
        # Temporary storage for stratification calculation
        patient_strat_values = defaultdict(list)
        
        with h5py.File(self.config.MRI_HDF5_PATH, 'r') as f:
            all_doc_ids = list(f.keys())
            
            for doc_id in all_doc_ids:
                grp = f[doc_id]
                
                # Robust Patient ID Extraction
                p_id_attr = grp.attrs.get('patient_id', f"doc_{doc_id}")
                p_id = str(p_id_attr).strip()
                if not p_id or p_id.lower() == 'nan':
                    p_id = f"doc_{doc_id}"

                # Iterate over both joints
                for side_key in ['left_joint', 'right_joint']:
                    if side_key not in grp:
                        continue
                    
                    # Access Attributes
                    attrs = grp[side_key].attrs
                    raw_val = attrs.get(self.target_column_key, float('nan'))
                    
                    # Robust Parsing of Values (handling strings with commas, bytes, etc.)
                    val = float('nan')
                    if isinstance(raw_val, (bytes, str)):
                        try:
                            val_str = str(raw_val)
                            if isinstance(raw_val, bytes):
                                val_str = raw_val.decode('utf-8')
                            val = float(val_str.replace(',', '.').strip())
                        except ValueError:
                            val = float('nan')
                    else:
                        val = float(raw_val)

                    # CRITICAL FILTER: Skip if Target is NaN
                    if np.isnan(val):
                        continue

                    # Store Valid Target
                    ident = (doc_id, side_key)
                    # We store as a 1D tensor [1] to maintain consistency
                    self.targets_lookup[ident] = torch.tensor([val], dtype=torch.float32)
                    self.patient_to_joints[p_id].append(ident)
                    
                    # Accumulate value for stratification
                    # For classification: 0/1. For regression: the raw value (will be binned later if needed)
                    strat_val = int(val > 0) if self.type == 'classification' else int(val)
                    patient_strat_values[p_id].append(strat_val)

        # Build Final Patient List
        unique_pids = sorted(list(self.patient_to_joints.keys()))
        group_strat_labels = []
        
        for pid in unique_pids:
            # Aggregation Strategy for Stratification:
            # We use the MAX value (worst condition) for the patient.
            # This ensures that patients with pathology are evenly distributed across folds.
            labels = patient_strat_values[pid]
            group_strat_labels.append(max(labels) if labels else 0)

        logging.info(
            f"[{self.task_name}] Manifest built. "
            f"Valid Joints: {len(self.targets_lookup)}. Valid Patients: {len(unique_pids)}."
        )
        
        if len(unique_pids) == 0:
            raise RuntimeError(f"No valid data found for task {self.task_name} in {self.config.MRI_HDF5_PATH}")
            
        return unique_pids, group_strat_labels

    def get_fold_iterator(self, manifest: List[str], stratify_labels: List[int], num_folds: int, seed: int):
        """
        Manages Cross-Validation Splitting with Persistence.
        
        Checks for an existing JSON split file to ensure reproducibility.
        If not found, generates a new Stratified K-Fold split and saves it.
        """
        split_filename = f"split_{self.task_name}_seed{seed}_k{num_folds}.json"
        split_artifact_path = self.config.BASE_OUTPUT_DIR / split_filename
        
        # 1. Load Existing Split
        if split_artifact_path.exists():
            logging.info(f"[{self.task_name}] Loading persistent split from {split_artifact_path}")
            with open(split_artifact_path, 'r') as f:
                splits_data = json.load(f)
            
            # Helper to map PIDs back to indices
            pid_to_idx = {pid: i for i, pid in enumerate(manifest)}
            
            for fold_name in sorted(splits_data.keys()):
                fold_data = splits_data[fold_name]
                try:
                    train_indices = np.array([pid_to_idx[p] for p in fold_data['train']])
                    val_indices = np.array([pid_to_idx[p] for p in fold_data['val']])
                    yield train_indices, val_indices
                except KeyError as e:
                    logging.warning(f"Patient ID {e} in saved split not found in current manifest. Ignoring fold.")
                    continue
                    
        # 2. Generate New Split
        else:
            logging.info(f"[{self.task_name}] Generating NEW Stratified K-Fold split...")
            skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
            folds_generator = list(skf.split(manifest, stratify_labels))
            
            serializable_splits = {}
            for i, (train_idx, val_idx) in enumerate(folds_generator):
                fold_key = f"fold_{i}"
                serializable_splits[fold_key] = {
                    'train': [manifest[x] for x in train_idx],
                    'val': [manifest[x] for x in val_idx]
                }
            
            # Save artifact
            try:
                with open(split_artifact_path, 'w') as f:
                    json.dump(serializable_splits, f, indent=4)
                logging.info(f"[{self.task_name}] Split saved to {split_artifact_path}")
            except Exception as e:
                logging.error(f"Failed to save split artifact: {e}")
            
            # Yield splits
            for train_idx, val_idx in folds_generator:
                yield train_idx, val_idx

    def get_datasets(self, train_idx: List[int], val_idx: List[int], manifest: List[str]) -> Tuple[Any, Any]:
        """
        Constructs Train and Validation Datasets from patient indices.
        Computes training statistics (Mean/Std or Class Weights) to avoid data leakage.
        """
        # Resolve Patient IDs
        train_pids = set([manifest[i] for i in train_idx])
        val_pids = set([manifest[i] for i in val_idx])
        
        train_identifiers = []
        val_identifiers = []
        train_targets_accum = []
        
        # Reconstruct Joint Lists using the pre-computed patient map
        for pid in train_pids:
            joints = self.patient_to_joints.get(pid, [])
            for ident in joints:
                train_identifiers.append(ident)
                train_targets_accum.append(self.targets_lookup[ident])
                
        for pid in val_pids:
            joints = self.patient_to_joints.get(pid, [])
            val_identifiers.extend(joints)

        # --- Statistics Calculation (Strictly on Training Data) ---
        if self.type == 'classification':
            if train_targets_accum:
                all_targets = torch.cat(train_targets_accum).view(-1)
                n_pos = (all_targets == 1).sum()
                n_neg = (all_targets == 0).sum()
                
                # Calculate Positive Weight for BCE
                if n_pos > 0:
                    weight = n_neg / float(n_pos)
                else:
                    weight = 1.0
                
                self.train_pos_weight = torch.tensor([weight], dtype=torch.float32)
                logging.info(f"[{self.task_name}] Calculated Pos Weight: {self.train_pos_weight.item():.4f}")
            else:
                self.train_pos_weight = torch.tensor([1.0])

        elif self.type == 'regression':
            if train_targets_accum:
                all_targets = torch.cat(train_targets_accum).view(-1)
                self.train_stats = {
                    'mean': all_targets.mean().item(),
                    'std': all_targets.std().item() + 1e-6
                }
                logging.info(f"[{self.task_name}] Regression Stats: {self.train_stats}")
            else:
                self.train_stats = {'mean': 0.0, 'std': 1.0}

        # --- Dataset Instantiation ---
        # Create a deep copy of config for validation to strictly disable augmentation
        pt_cfg_val = copy.deepcopy(self.pt_config)
        pt_cfg_val.AUG_ROTATION_DEGREES = [0.0, 0.0]
        pt_cfg_val.AUG_SCALE_RANGE = [1.0, 1.0] # Ensure naming consistency with config
        pt_cfg_val.AUG_TRANSLATE_PERCENT_RANGE = [0.0, 0.0]

        ds_train = LabeledTMJDataset(
            self.tmj_dataset_cls, 
            self.config.MRI_HDF5_PATH, 
            train_identifiers, 
            self.pt_config, 
            self.targets_lookup
        )
        
        ds_val = LabeledTMJDataset(
            self.tmj_dataset_cls, 
            self.config.MRI_HDF5_PATH, 
            val_identifiers, 
            pt_cfg_val, 
            self.targets_lookup
        )
        
        return ds_train, ds_val

    def process_batch(self, batch: Dict[str, Any], device: torch.device, augment: bool = False) -> Dict[str, Any]:
        """
        Orchestrates GPU-based processing using the PhysicsEngine logic.
        
        1. Moves data to GPU.
        2. Applies Affine Transforms (via gpu_process_fn).
        3. Normalizes Targets (if Regression).
        """
        # 1. Transfer to Device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device, non_blocking=True)
            elif isinstance(v, list) and isinstance(v[0], torch.Tensor):
                batch[k] = torch.stack(v).to(device, non_blocking=True)

        # 2. Geometric & Intensity Transformations
        # We use 'epoch=100' to force augmentation ON in gpu_process_fn if augment=True,
        # and 'epoch=0' to force it OFF if augment=False. 
        # This assumes gpu_process_fn uses a curriculum schedule or simple threshold.
        dummy_epoch = 100 if augment else 0
        
        processed = self.gpu_process_fn(
            batch, 
            self.pt_config, 
            epoch=dummy_epoch,
            hu_augmenter=None, 
            noise_simulator=None
        )
        
        input_tensor = processed['input_cube'] # Shape: [B, 1, D, H, W]
        
        # 3. Target Preparation
        targets_raw = batch['labels'].float() # Shape: [B, 1]
        labels_for_loss = targets_raw.clone()

        # Apply Z-score normalization for regression tasks using Train Fold Stats
        if self.type == 'regression' and self.train_stats is not None:
            mu = self.train_stats['mean']
            sigma = self.train_stats['std']
            labels_for_loss = (targets_raw - mu) / sigma
            
        return {
            'x': input_tensor,
            'x_pair': None,  # Independent processing, so no pair
            'mask': processed.get('valid_for_masking_mask'), # Optional mask
            'mask_pair': None,
            'labels': labels_for_loss, # Normalized (if reg)
            'labels_raw': targets_raw  # Original (for metrics)
        }

class TMJInflammationTask(BaseClinicalTask):
    def __init__(self, *args, **kwargs):
        super().__init__('mri_inflam', *args, **kwargs)
        self.type = 'classification'
    
    @property
    def target_column_key(self): return 'Inflammation'

    def get_head(self, in_features: int) -> nn.Module:
        return TextureHead(in_features)

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        # Weighted BCE
        weight = self.train_pos_weight.to(device) if self.train_pos_weight is not None else None
        return BCEWithLogitsLoss(pos_weight=weight)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # y_true: [N, 1], y_pred: [N, 1] logits
        probs = 1 / (1 + np.exp(-y_pred))
        preds = (probs > 0.5).astype(int)
        return {
            'auc': roc_auc_score(y_true, probs),
            'f1': f1_score(y_true, preds),
            'acc': balanced_accuracy_score(y_true, preds)
        }

class TMJBoneTask(BaseClinicalTask):
    def __init__(self, *args, **kwargs):
        super().__init__('mri_bone', *args, **kwargs)
        self.type = 'classification'

    @property
    def target_column_key(self): return 'BoneStructure'

    def get_head(self, in_features: int) -> nn.Module:
        #return SpatialAttentionHead(in_features)
        return TextureHead(in_features)

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        weight = self.train_pos_weight.to(device) if self.train_pos_weight is not None else None
        return BCEWithLogitsLoss(pos_weight=weight)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        probs = 1 / (1 + np.exp(-y_pred))
        preds = (probs > 0.5).astype(int)
        return {
            'auc': roc_auc_score(y_true, probs),
            'f1': f1_score(y_true, preds)
        }

class TMJDiscTask(BaseClinicalTask):
    def __init__(self, *args, **kwargs):
        super().__init__('mri_disc', *args, **kwargs)
        self.type = 'classification'

    @property
    def target_column_key(self): return 'DiscPosition'

    def get_head(self, in_features: int) -> nn.Module:
        return SpatialAttentionHead(in_features)

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        weight = self.train_pos_weight.to(device) if self.train_pos_weight is not None else None
        return BCEWithLogitsLoss(pos_weight=weight)

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        probs = 1 / (1 + np.exp(-y_pred))
        preds = (probs > 0.5).astype(int)
        return {
            'auc': roc_auc_score(y_true, probs),
            'f1': f1_score(y_true, preds)
        }

class TMJSpaceTask(BaseClinicalTask):
    def __init__(self, *args, **kwargs):
        super().__init__('mri_space', *args, **kwargs)
        self.type = 'regression'

    @property
    def target_column_key(self): return 'JointSpace'

    def get_head(self, in_features: int) -> nn.Module:
        return VolumetricPyramidHead(in_features)

    def get_loss_fn(self, device: torch.device) -> nn.Module:
        # Huber Loss is robust to outliers in measurements
        return HuberLoss(delta=1.0)

    def process_batch(self, batch: Dict[str, Any], device: torch.device, augment: bool = False) -> Dict[str, Any]:
        """
        Override to strictly disable Scaling (Zoom) augmentation.
        Geometry must be preserved for regression.
        """
        # Save original config state
        orig_scale = self.pt_config.AUG_SCALE_RANGE
        try:
            # Force Zoom OFF
            self.pt_config.AUG_SCALE_RANGE = [1.0, 1.0]
            # Delegate to parent process
            return super().process_batch(batch, device, augment)
        finally:
            # Restore state
            self.pt_config.AUG_SCALE_RANGE = orig_scale

    def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        # y_true: [N, 1] raw mm, y_pred: [N, 1] raw mm (denormalized in validate_epoch)
        return {
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred)
        }