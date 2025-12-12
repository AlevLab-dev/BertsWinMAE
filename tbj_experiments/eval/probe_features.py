"""
TMJ Feature Space Analysis Suite
-------------------------------------------------------------------------
A comprehensive evaluation framework for analyzing the latent space properties
of self-supervised models trained on TMJ CBCT volumes.

Experiments included:
1. Geometric Invariance: Robustness to affine transformations.
2. Inter-Patient Separability: Baseline distinctiveness of patient features.
3. Bilateral Symmetry: Feature similarity between contralateral joints.
4. Longitudinal Stability: Feature consistency across temporal scans.
"""

import argparse
import copy
import gc
import json
import logging
import os
import random
import sys
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import wilcoxon
from torch.utils.data import Dataset
from tqdm import tqdm

# Internal Project Imports
from configs import EvalConfig
from main import load_pretrain_modules
from models import ModelFactory, TMJAnalysisModel

# ==============================================================================
# 1. Environment & Utilities
# ==============================================================================

def setup_reproducibility(seed: int, output_dir: Path) -> torch.device:
    """Configures the execution environment, logging, and random seeds."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "analysis_log.txt", mode='w'),
            logging.StreamHandler(sys.stdout)
        ]
    )

    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        return torch.device("cuda")
    return torch.device("cpu")


def compute_descriptive_statistics(values: List[float]) -> Dict[str, float]:
    """Computes standard descriptive statistics for a distribution."""
    if not values:
        return {}
    return {
        'n': len(values),
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'median': float(np.median(values)),
        'min': float(np.min(values)),
        'max': float(np.max(values))
    }


# ==============================================================================
# 2. Data Management
# ==============================================================================

class ProbeDatasetAdapter(Dataset):
    """
    Wraps the core TMJDataset to inject metadata and handle HDF5 lazy loading.
    Allows dynamic configuration updates for augmentation experiments.
    """
    def __init__(self, tmj_dataset_cls: Any, hdf5_path: str, 
                 identifiers: List[Tuple[str, str]], config: Any, meta_map: Dict):
        self.dataset = tmj_dataset_cls(hdf5_path, identifiers, config)
        self.identifiers = identifiers
        self.meta_map = meta_map
        self.hdf5_path = hdf5_path

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Optional[Dict[str, Any]]:
        # Handle HDF5 file handle initialization for the worker process
        if getattr(self.dataset, 'h5_file', None) is None:
            try:
                self.dataset.h5_file = h5py.File(self.hdf5_path, 'r')
            except Exception as e:
                logging.error(f"HDF5 Access Error: {e}")
                return None

        # Fetch data (triggers augmentation logic in self.dataset.__getitem__)
        data = self.dataset[idx]
        if data is None:
            return None

        scan_id, side = self.identifiers[idx]
        meta = self.meta_map.get(scan_id, {})

        return {
            'raw_data': data,
            'meta': {
                'hdf5_key': scan_id,
                'joint_side': side,
                'patient_id': meta.get('patient_id', scan_id),
                'date': meta.get('date', '00000000')
            }
        }


class CohortManager:
    """
    Manages cohort construction, metadata parsing, and sample stratification.
    """
    @staticmethod
    def build(config: EvalConfig, pt_config: Any, tmj_dataset_cls: Any) -> Dict[str, Any]:
        with open(config.PRETRAIN_SPLIT_FILE, 'r') as f:
            val_ids = json.load(f).get('val', [])

        identifiers = []
        meta_map = {}
        paired_map = defaultdict(dict)
        patient_history = defaultdict(list)

        with h5py.File(config.PRETRAIN_HDF5_PATH, 'r') as f:
            for scan_id in tqdm(val_ids, desc="Indexing Cohort"):
                if scan_id not in f:
                    continue
                
                grp = f[scan_id]
                pid_raw = grp.attrs.get('patient_id', scan_id)
                pid = pid_raw.decode('utf-8') if isinstance(pid_raw, bytes) else str(pid_raw)
                
                date_matches = re.findall(r'\.(20\d{6})', scan_id)
                if date_matches:
                    date = date_matches[-1]
                else:
                    date = "00000000"
                
                meta_map[scan_id] = {'patient_id': pid, 'date': date}

                # Index joints
                for side in ['left_joint', 'right_joint']:
                    if side in grp:
                        identifiers.append((scan_id, side))
                        paired_map[scan_id][side] = {'hdf5_key': scan_id, 'joint_side': side}

                # Build longitudinal history
                if pid != scan_id:
                    entry = {'hdf5_key': scan_id, 'date': date, 'patient_id': pid}
                    # Avoid duplicates
                    if not any(x['hdf5_key'] == scan_id for x in patient_history[pid]):
                        patient_history[pid].append(entry)

        # Filter valid subsets
        valid_pairs = {k: v for k, v in paired_map.items() 
                       if 'left_joint' in v and 'right_joint' in v}
        
        valid_longitudinal = {k: sorted(v, key=lambda x: x['date']) 
                              for k, v in patient_history.items() if len(v) > 1}

        dataset = ProbeDatasetAdapter(tmj_dataset_cls, config.PRETRAIN_HDF5_PATH, 
                                      identifiers, pt_config, meta_map)

        logging.info(f"Cohort Indexing Complete: N={len(dataset)} joints. "
                     f"{len(valid_pairs)} bilateral subjects. "
                     f"{len(valid_longitudinal)} longitudinal subjects.")

        return {
            'dataset': dataset,
            'paired': valid_pairs,
            'longitudinal': valid_longitudinal
        }


# ==============================================================================
# 3. Statistical Probe (Feature Space Analysis)
# ==============================================================================

class FeatureSpaceProbe:
    """
    Encapsulates model inference, feature whitening, and similarity metrics.
    """
    def __init__(self, model: TMJAnalysisModel, device: torch.device, 
                 gpu_process_fn: callable, collate_fn: callable, pt_config: Any,
                 cache_dir: Path):
        self.model = model.eval().to(device)
        self.device = device
        self.gpu_process_fn = gpu_process_fn
        self.collate_fn = collate_fn
        self.pt_config = pt_config
        self.cache_dir = cache_dir
        
        # Whitening parameters
        self.mean: Optional[torch.Tensor] = None
        self.std: Optional[torch.Tensor] = None
        self.is_fitted = False
        
        # Diagnostics
        self.feature_dim = 0
        self.effective_rank = 0.0

    def fit_whitening(self, dataset: Dataset, num_samples: int = 2000) -> None:
        """
        Computes Global Mean for Centering (Pearson Correlation Proxy).
        We AVOID full PCA/ZCA whitening to prevent noise amplification in 
        low-rank manifolds of self-supervised models.
        """
        stats_path = self.cache_dir / "centering_stats.pt"
        
        if stats_path.exists():
            logging.info(f"Loading stats from {stats_path}")
            saved = torch.load(stats_path, map_location=self.device)
            self.mean = saved['mean']
            self.effective_rank = saved.get('effective_rank', 0.0)
            self.is_fitted = True
            return

        logging.info(f"Fitting Centering...")
        
        embeddings_buffer = []
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        with torch.no_grad():
            for idx in tqdm(indices, desc="Accumulating features"):
                item = dataset[idx]
                if item:
                    emb = self._extract_raw_embedding(item).cpu()
                    embeddings_buffer.append(emb)
        
        X = torch.cat(embeddings_buffer, dim=0).float().to(self.device)
        self.feature_dim = X.shape[1]
        
        # --- 1. Compute Mean Only (Centering) ---
        self.mean = torch.mean(X, dim=0, keepdim=True)
        
        # --- 2. Diagnostics (Effective Rank) - DO NOT USE FOR TRANSFORM ---
        # We calculate this purely for the paper's "Feature Space Analysis" table
        try:
            X_centered = X - self.mean
            # Normalize for SVD stability only (diagnostic)
            X_norm = F.normalize(X_centered, p=2, dim=1)
            # Covariance of normalized data
            cov = (X_norm.T @ X_norm) / (X.shape[0] - 1)
            _, S, _ = torch.linalg.svd(cov)
            
            eigenvalues = S + 1e-10
            p = eigenvalues / eigenvalues.sum()
            entropy = -torch.sum(p * torch.log(p))
            self.effective_rank = torch.exp(entropy).item()
        except Exception as e:
            logging.warning(f"Rank calculation failed: {e}")
            self.effective_rank = 1.0

        logging.info(f"--- Feature Space Diagnostics ---")
        logging.info(f"Effective Rank: {self.effective_rank:.2f}")

        self.is_fitted = True
        torch.save({
            'mean': self.mean, 
            'effective_rank': self.effective_rank
        }, stats_path)

    def get_embedding(self, item: Dict, force_side_override: Optional[str] = None) -> torch.Tensor:
        """
        Returns the Centered & Normalized embedding.
        Equivalent to Pearson Correlation when dot product is applied later.
        """
        if not self.is_fitted:
            raise RuntimeError("Probe must be fitted before inference.")
        
        raw_emb = self._extract_raw_embedding(item, force_side_override).float().to(self.device)
        
        # 1. Centering (Remove Anisotropy/Cone Effect)
        centered = raw_emb - self.mean
        
        # 2. L2 Normalization (Project to Hypersphere)
        # This ensures Dot Product == Cosine Similarity
        normalized = F.normalize(centered, p=2, dim=1)
        
        return normalized.cpu()

    def compute_similarity(self, emb_a: torch.Tensor, emb_b: torch.Tensor) -> float:
        """Computes Cosine Similarity between two embeddings."""
        sim = F.cosine_similarity(emb_a, emb_b, eps=1e-8).item()
        return max(-1.0, min(1.0, sim))

    def _extract_raw_embedding(self, item: Dict, force_side: Optional[str] = None) -> torch.Tensor:
        """Internal extraction pipeline using the GPU process function."""
        # Deepcopy to prevent mutation of the dataset cache
        raw_data = copy.deepcopy(item['raw_data'])
        
        if force_side:
            raw_data['joint_side'] = force_side
            
        batch = self.collate_fn([raw_data])
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(self.device, non_blocking=True)
        
        with torch.amp.autocast('cuda', dtype=torch.bfloat16):
            processed = self.gpu_process_fn(
                batch, self.pt_config, epoch=0, hu_augmenter=None, noise_simulator=None
            )
            embedding = self.model.get_embedding(processed['input_cube'])
            
        return embedding.detach()


# ==============================================================================
# 4. Experiment Suite
# ==============================================================================

class ExperimentSuite:
    def __init__(self, probe: FeatureSpaceProbe, output_dir: Path):
        self.probe = probe
        self.out_dir = output_dir

    @torch.no_grad()
    def run_geometric_invariance(self, dataset: Dataset, aug_config: Any, num_samples: int = 500):
        """
        Experiment 1: Geometric Invariance.
        Measures cosine similarity between two independently augmented views of the same volume.
        """
        logging.info("--- Experiment 1: Geometric Invariance ---")
        
        # Save original config
        original_config = self.probe.pt_config
        
        # Inject augmentation config into both probe and dataset
        self.probe.pt_config = aug_config
        dataset.dataset.cfg = aug_config 
        
        results = []
        indices = np.random.choice(len(dataset), min(num_samples, len(dataset)), replace=False)
        
        try:
            for i, idx in enumerate(tqdm(indices, desc="Computing Invariance")):
                # Double-fetch triggers independent augmentation pipelines in __getitem__
                view_1 = dataset[idx]
                view_2 = dataset[idx]
                
                if not view_1 or not view_2: 
                    continue
                
                emb_1 = self.probe.get_embedding(view_1)
                emb_2 = self.probe.get_embedding(view_2)

                sim = self.probe.compute_similarity(emb_1, emb_2)
                results.append({'id': view_1['meta']['hdf5_key'], 'similarity': sim})
                
                # Cleanup to prevent VRAM fragmentation
                if i % 20 == 0: 
                    torch.cuda.empty_cache()
                    
        finally:
            # Restore canonical configuration
            self.probe.pt_config = original_config
            dataset.dataset.cfg = original_config

        self._save_results(results, "exp1_geometric_invariance.json")

    @torch.no_grad()
    def run_inter_patient_separability(self, dataset: Dataset, num_pairs: int = 5000):
        """
        Experiment 2: Inter-Patient Baseline.
        Measures cosine similarity between random pairs of different patients.
        """
        logging.info("--- Experiment 2: Inter-Patient Separability ---")
        
        # Pre-cache embeddings to avoid repeated forward passes
        cache = []
        subset_size = min(2000, len(dataset))
        indices = np.random.choice(len(dataset), subset_size, replace=False)
        
        for i, idx in enumerate(tqdm(indices, desc="Caching Embeddings")):
            item = dataset[idx]
            if item:
                cache.append({
                    'pid': item['meta']['patient_id'], 
                    'emb': self.probe.get_embedding(item)
                })
            if i % 50 == 0: 
                gc.collect()
                torch.cuda.empty_cache()
        
        results = []
        for _ in tqdm(range(num_pairs), desc="Comparing Random Pairs"):
            if len(cache) < 2: break
            
            sample_a, sample_b = random.sample(cache, 2)
            if sample_a['pid'] == sample_b['pid']: 
                continue
            
            sim = self.probe.compute_similarity(sample_a['emb'], sample_b['emb'])
            results.append({'similarity': sim})
            
        self._save_results(results, "exp2_inter_patient_separability.json")

    @torch.no_grad()
    def run_bilateral_symmetry(self, dataset: Dataset, paired_map: Dict):
        """
        Experiment 3: Bilateral Symmetry.
        Compares Intra-Patient (Left vs Right) similarity against Anatomical Symmetry (Left vs Flipped Right).
        """
        logging.info("--- Experiment 3: Bilateral Symmetry ---")
        
        lookup = { (idt[0], idt[1]): i for i, idt in enumerate(dataset.identifiers) }
        results = []
        
        for i, (scan_id, _) in enumerate(tqdm(paired_map.items(), desc="Analyzing Pairs")):
            idx_l = lookup.get((scan_id, 'left_joint'))
            idx_r = lookup.get((scan_id, 'right_joint'))
            
            if idx_l is None or idx_r is None: 
                continue
            
            emb_l = self.probe.get_embedding(dataset[idx_l])
            emb_r = self.probe.get_embedding(dataset[idx_r])
            
            # Compute symmetry by flipping the right joint to match left anatomical orientation
            emb_r_flipped = self.probe.get_embedding(dataset[idx_r], force_side_override='left_joint')
            
            results.append({
                'scan_id': scan_id, 
                'sim_intra': self.probe.compute_similarity(emb_l, emb_r), 
                'sim_symmetry': self.probe.compute_similarity(emb_l, emb_r_flipped)
            })
            
            if i % 50 == 0: 
                torch.cuda.empty_cache()

        # Statistical Significance Test
        sim_intra = [r['sim_intra'] for r in results]
        sim_symm = [r['sim_symmetry'] for r in results]
        
        if len(sim_intra) > 5:
            try:
                stat, p_val = wilcoxon(sim_intra, sim_symm)
                logging.info(f"Wilcoxon Test (Intra vs Symmetry): p={p_val:.4e}")
                with open(self.out_dir / "stats_bilateral.json", 'w') as f:
                    json.dump({'p_value': p_val, 'mean_intra': np.mean(sim_intra), 
                               'mean_symmetry': np.mean(sim_symm)}, f, indent=4)
            except Exception as e:
                logging.warning(f"Statistical test failed: {e}")
                
        self._save_results(results, "exp3_bilateral_symmetry.json")

    @torch.no_grad()
    def run_longitudinal_stability(self, dataset: Dataset, longitudinal_map: Dict):
        """
        Experiment 4: Longitudinal Stability.
        Compares embeddings of the same patient across different timepoints.
        """
        logging.info("--- Experiment 4: Longitudinal Stability ---")
        
        lookup = { (idt[0], idt[1]): i for i, idt in enumerate(dataset.identifiers) }
        results = []
        
        for i, (pid, scans) in enumerate(tqdm(longitudinal_map.items(), desc="Analyzing Timepoints")):
            # Pairwise comparison of all timepoints for a single patient
            for a in range(len(scans)):
                for b in range(a+1, len(scans)):
                    scan_1, scan_2 = scans[a], scans[b]
                    
                    for side in ['left_joint', 'right_joint']:
                        key_1, key_2 = (scan_1['hdf5_key'], side), (scan_2['hdf5_key'], side)
                        
                        if key_1 in lookup and key_2 in lookup:
                            emb_1 = self.probe.get_embedding(dataset[lookup[key_1]])
                            emb_2 = self.probe.get_embedding(dataset[lookup[key_2]])
                            
                            try:
                                date_diff = abs(int(scan_1['date']) - int(scan_2['date']))
                            except ValueError:
                                date_diff = -1
                                
                            results.append({
                                'patient_id': pid, 
                                'date_diff': date_diff,
                                'similarity': self.probe.compute_similarity(emb_1, emb_2)
                            })
            if i % 50 == 0: 
                torch.cuda.empty_cache()
            
        self._save_results(results, "exp4_longitudinal_stability.json")

    def _save_results(self, data: List[Dict], filename: str):
        """Serializes results and statistics to JSON."""
        values = []
        if data:
            # Determine primary metric key dynamically
            key = 'similarity' if 'similarity' in data[0] else 'sim_intra'
            values = [d[key] for d in data]
            
        stats = compute_descriptive_statistics(values)
        
        with open(self.out_dir / filename, 'w') as f:
            json.dump({'statistics': stats, 'raw_data': data}, f, indent=4)
            
        logging.info(f"Saved {filename} | Mean Similarity: {stats.get('mean', 0.0):.4f}")


# ==============================================================================
# 5. Main Entry Point
# ==============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TMJ Feature Space Analysis")
    parser.add_argument('--model', required=True, help="Model architecture key from registry")
    parser.add_argument('--output_dir', default='./results_featprobes', help="Root output directory")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--samples_invariance', type=int, default=500, help="Samples for Exp 1")
    parser.add_argument('--samples_whitening', type=int, default=2000, help="Samples for Whitening fit")
    
    args = parser.parse_args()
    
    # Setup
    output_path = Path(args.output_dir) / args.model
    device = setup_reproducibility(args.seed, output_path)
    
    # Load Configuration & Modules
    cfg = EvalConfig()
    pt_modules, pt_cfg = load_pretrain_modules(cfg.PRETRAIN_SCRIPT_PATH)
    
    # Create Configurations
    # 1. Canonical (Evaluation) Config
    cfg_canonical = copy.deepcopy(pt_cfg)
    cfg_canonical.AUG_ROTATION_DEGREES = [0.0, 0.0]
    cfg_canonical.AUG_SCALE = [1.0, 1.0]
    cfg_canonical.AUG_TRANSLATE_PERCENT_RANGE = [0.0, 0.0]
    
    # 2. Augmented (Invariance Test) Config
    cfg_augmented = copy.deepcopy(pt_cfg)
    cfg_augmented.AUG_ROTATION_DEGREES = [-15, 15]
    cfg_augmented.AUG_SCALE = [0.9, 1.1]

    # Initialize Model
    try:
        model_info = EvalConfig.MODEL_REGISTRY[args.model]
    except KeyError:
        raise ValueError(f"Model '{args.model}' not found in registry.")

    backbone, feature_dim = ModelFactory.load_backbone(
        model_info['arch_type'], model_info['path'], pt_cfg, pt_modules
    )
    
    # Wrap in Analysis Model (Identity Head for direct feature extraction)
    model = TMJAnalysisModel(
        backbone, 
        head=torch.nn.Identity(), 
        feature_dim=feature_dim, 
        arch_type=model_info['arch_type'], 
        inference_mode='full_image'
    )

    # Initialize Managers
    cohort = CohortManager.build(cfg, cfg_canonical, pt_modules['TMJDataset'])
    
    probe = FeatureSpaceProbe(
        model=model, 
        device=device, 
        gpu_process_fn=pt_modules['process_batch_on_gpu'], 
        collate_fn=pt_modules['collate_fn'], 
        pt_config=cfg_canonical,
        cache_dir=output_path
    )
    
    suite = ExperimentSuite(probe, output_path)

    # Execution Pipeline
    logging.info("Starting Analysis Pipeline...")
    
    probe.fit_whitening(cohort['dataset'], num_samples=args.samples_whitening)
    
    suite.run_geometric_invariance(cohort['dataset'], cfg_augmented, num_samples=args.samples_invariance)
    suite.run_inter_patient_separability(cohort['dataset'])
    suite.run_bilateral_symmetry(cohort['dataset'], cohort['paired'])
    suite.run_longitudinal_stability(cohort['dataset'], cohort['longitudinal'])
    
    logging.info("Analysis Suite Completed Successfully.")