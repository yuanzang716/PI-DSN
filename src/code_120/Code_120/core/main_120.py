# -*- coding: utf-8 -*-
import os
import glob
import time
import re
import shutil
import hashlib
import json
import multiprocessing
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import math
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
try:
    from torchvision import models, transforms
except Exception:
    models = None
    transforms = None
from scipy.optimize import differential_evolution
from joblib import Parallel, delayed
import tqdm
from PIL import Image
import random
# ============================================================
# Data_Engine (Leakage-Safe)
# - Physical isolation: train_real/ vs val_real/
# - Generation isolation:
#   * Simulations are generated ONLY from TRAIN real images (sources).
#   * Training diffs are generated ONLY against TRAIN real images (targets).
#   * Optional validation diffs can be generated against VAL real images
#     using the same TRAIN simulations, without ever using VAL as source.
# ============================================================

# ============================================================
# 0. Global Configuration
# ============================================================
class DataConfig:
    # --------------------------
    # Mode
    # --------------------------
    # FAST verification mode (replaces old VERIFICATION_MODE)
    # - designed to quickly check that the whole pipeline can run end-to-end.
    FAST_VERIFY = False
    FAST_VERIFY_MAX_OPT_IMAGES = 2          # optimize only first K train real images
    FAST_VERIFY_MAX_SIMS_PER_SRC = 2        # generate only K sims per source
    FAST_VERIFY_MAX_DIFF_TARGETS = 1        # for each sim, diff against only K targets
    FAST_VERIFY_SKIP_VAL = False             # skip VAL generation by default
    FAST_VERIFY_SKIP_PART3 = False          # if True, skip Part-3 (only DE+rp+csv)


    # --------------------------
    # Data root / physical isolation
    # --------------------------
    # Recommended folder layout:
    #   DATA_ROOT/
    #     train_real/   <-- only training real images
    #     val_real/     <-- only validation real images
    #     simulation_dataset/
    #
    # If you still keep all images in one folder, you can set
    # AUTO_SPLIT_FROM_SINGLE_FOLDER=True and point DATA_FOLDER
    # to that folder. The script will create train_real/ and val_real/
    # by copying files (non-destructive).
    # Data root can be configured via environment variable for reproducibility across machines.
    # Default assumes this repository layout: <repo_root>/data/Diameter_100.2_120mm
    _DEFAULT_DATA_ROOT = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", "data", "Diameter_100.2_120mm")
    )
    DATA_ROOT = os.environ.get("DATA_ROOT", _DEFAULT_DATA_ROOT)
    TRAIN_REAL_FOLDER = os.path.join(DATA_ROOT, 'train_real')
    VAL_REAL_FOLDER   = os.path.join(DATA_ROOT, 'val_real')

    AUTO_SPLIT_FROM_SINGLE_FOLDER = True
    DATA_FOLDER = os.environ.get("DATA_FOLDER", DATA_ROOT)  # used only if AUTO_SPLIT...=True

    # Split control (used only if AUTO_SPLIT_FROM_SINGLE_FOLDER=True)
    # Option A: provide explicit validation IDs extracted from filename
    VAL_REAL_IDS: Optional[List[int]] = None  # e.g., [9, 10]
    # Option B: random split by ratio (deterministic with seed)
    VAL_RATE = 0.3
    SPLIT_SEED = 42

    # --------------------------
    # Part outputs
    # --------------------------
    # Coarse diameter estimation + optimized params are computed ONLY on TRAIN reals.
    INIT_DIAMETER_FILE = os.path.join(TRAIN_REAL_FOLDER, 'initial_diameter.txt')
    RESULTS_FILE = os.path.join(TRAIN_REAL_FOLDER, 'optimized_params.csv')
    # Coarse real-parameter tables (for conditioning rp in main.py)
    # These are computed from REAL images (train and val separately) and are leakage-safe.
    TRAIN_REAL_RP_FILE = os.path.join(TRAIN_REAL_FOLDER, 'train_real_rp.csv')
    VAL_REAL_RP_FILE   = os.path.join(VAL_REAL_FOLDER,   'val_real_rp.csv')

    # Quick optimization settings to estimate VAL real rp (kept lightweight)
    VAL_RP_OPT_MAXITER = 25
    VAL_RP_OPT_POPSIZE = 25

    # Dataset outputs (kept separate)
    DATASET_ROOT = os.path.join(DATA_ROOT, 'simulation_dataset')
    TRAIN_DATASET_FOLDER = os.path.join(DATASET_ROOT, 'train')
    VAL_DATASET_FOLDER   = os.path.join(DATASET_ROOT, 'val')

    TRAIN_SIM_FOLDER  = os.path.join(TRAIN_DATASET_FOLDER, 'sim')
    TRAIN_DIFF_FOLDER = os.path.join(TRAIN_DATASET_FOLDER, 'diff')
    VAL_DIFF_FOLDER   = os.path.join(VAL_DATASET_FOLDER, 'diff')
    VAL_SIM_FOLDER    = os.path.join(VAL_DATASET_FOLDER, 'sim')

    TRAIN_LABELS = os.path.join(TRAIN_DATASET_FOLDER, 'dataset_labels_train.csv')
    VAL_LABELS   = os.path.join(VAL_DATASET_FOLDER,   'dataset_labels_val.csv')

    # Whether to pre-generate validation diffs to disk.
    # If False, you can compute val diff on-the-fly in main.py.
    GENERATE_VAL_DIFF = True

    # --------------------------
    # Physical constants (um)
    # --------------------------
    WAVELENGTH_UM = 0.78024
    FOCAL_LENGTH_UM = 120000.0
    PIXEL_SIZE_UM = 4.8


    # --------------------------
    # Real/Sim alignment + center-band mask
    # --------------------------
    APPLY_REAL_ALIGN = True   # align (derotate+recenter) REAL before mask & network input
    APPLY_CENTER_MASK = True
    CENTER_MASK_BAND_PX = 50  # keep [cx-50, cx+50] columns
    CENTER_MASK_MODE = 'keep' # 'keep' => keep center band only; 'drop' => drop center band
    # Bounds (Part 2): [phi, r, b, x, a(dynamic), gamma]
    BOUNDS_BASE = [
        (-5.0, 0.0),        # phi (deg) allow small tilt (estimated on raw REAL)
        (-15*1.05, -15*0.95),     # r (mm)
        (900.0, 1600.0),    # b (um)
        (-300.0, 300.0),    # x (um) allow small horizontal shift (estimated on raw REAL)
        # a inserted dynamically
        (0.1, 1.0)          # gamma
    ]

    # --------------------------
    # Regularization for diameter 'a' during optimization
    # --------------------------
    # When optimizing parameters on REAL images, 'a' can be strongly coupled with other parameters.
    # We therefore add a soft prior pulling 'a' towards the coarse estimate from Part 1.
    # - A_PRIOR_RATIO: allowed search range around coarse a (e.g. 0.05 => ±5%)
    # - A_REG_WEIGHT : strength of the quadratic prior term (dimensionless)
    A_PRIOR_RATIO = 0.05
    A_REG_WEIGHT  = 0.5

    # --------------------------
    # use coarse (FFT-peak) during DE optimization 
    # --------------------------
    A_TO_COARSE = False  # allow 'a' to be estimated within ±A_PRIOR_RATIO during DE
    # Small relative epsilon used to build legal bounds for SciPy DE (a is still clamped exactly to a_prior).
    A_EPS_REL = 1e-6

    # --------------------------
    # Generation (Part 3)
    # --------------------------
    SAMPLES_PER_ROW = 1000
    PERTURB_RATE = 0.05
    # Per-parameter absolute perturbation floor (very small) to avoid zero-variance when center≈0.
    # Order: [Phi, R_mm, B_um, X_um, A_um, Gamma]
    # Keep these small; adjust if a parameter is frequently near zero.

    # Generate VAL sims/diffs from VAL sources/targets (deployment-consistent), with strict no leakage.
    VAL_SELF_SIM = True
    # If True, VAL generation will NOT mix TRAIN and VAL in any sim/diff computation.
    STRICT_NO_LEAKAGE = True

    N_JOBS = 8

    # Safety: overwrite existing generated folders/csv?
    OVERWRITE_DATASET = False

    # --- RP table protection ---
    OVERWRITE_REAL_RP = True
    SIGN_RP_TABLES = False  # (disabled) no CSV protection/signature

import os
import re
import gc
import math
import random
import glob
import time
import json
import socket
import getpass
import hashlib
from collections import defaultdict

import pandas as pd
import numpy as np
from tqdm import tqdm
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, Sampler
from torch.optim.lr_scheduler import SequentialLR, LinearLR, CosineAnnealingLR
from torchvision.utils import save_image
from torchvision import transforms, models
from PIL import Image

import matplotlib
matplotlib.use('Agg')
warnings.filterwarnings("ignore", message=".*TF32.*")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


    # =========================================================
    # 0. Configuration
    # =========================================================
class TrainConfig:
    # --- Working / outputs ---
    WORK_DIR = r"/root/autodl-tmp/Diameter_100.2_120mm/cache"
    # BASE_MODEL_SAVE_DIR
    BASE_MODEL_SAVE_DIR = os.environ.get("BASE_MODEL_SAVE_DIR", r"/root/autodl-tmp/Diameter_100.2_120mm/BaseLine_Epoch50")

    # --- Leakage-safe data root ---
    DATA_ROOT = r"/root/autodl-tmp/Diameter_100.2_120mm"
    TRAIN_REAL_FOLDER = os.path.join(DATA_ROOT, "train_real")
    VAL_REAL_FOLDER   = os.path.join(DATA_ROOT, "val_real")

    DATASET_ROOT = os.path.join(DATA_ROOT, "simulation_dataset")
    TRAIN_LABELS_CSV = os.path.join(DATASET_ROOT, "train", "dataset_labels_train.csv")
    VAL_LABELS_CSV   = os.path.join(DATASET_ROOT, "val",   "dataset_labels_val.csv")

    INIT_DIAMETER_TXT = os.path.join(TRAIN_REAL_FOLDER, "initial_diameter.txt")
    TRAIN_REAL_RP_CSV = os.path.join(TRAIN_REAL_FOLDER, 'train_real_rp.csv')
    VAL_REAL_RP_CSV   = os.path.join(VAL_REAL_FOLDER,   'val_real_rp.csv')

    FFT_CACHE_DIR = r"/root/autodl-tmp/Diameter_100.2_120mm/cache/fft_tensors"

    # --- FFT cache subdir names (single source of truth for Dataset + Precomputer) ---
    FFT_REAL_SUBDIR = "real"
    FFT_SIMDIFF_SUBDIR = "sim_diff"
    # PNG prefixes that should be cached into FFT_SIMDIFF_SUBDIR
    FFT_CACHE_PNG_PREFIXES = ("sim_", "diff_", "simv_", "diffv_")

    # --- Backbone (torchvision pretrained, efficient) ---
    # Using ResNet34 as an efficient alternative to ConvNeXt-Tiny, with official torchvision weights.
    BACKBONE_NAME = "convnext_t"
    INPUT_CHANNELS_FULL = 6      # [real(2), sim(2), diff(2)] for caching; not directly fed into stems
    REAL_BRANCH_CH = 4           # [real(2), diff(2)]
    SIM_BRANCH_CH  = 2           # [sim(2)] = spatial+fft
    PARAM_DIM = 10               # rp(5)+sp(5)

    # --- Training ---
    # --- Progressive unfreeze (trunk) ---
    ENABLE_PROGRESSIVE_UNFREEZE = True
    FREEZE_RATIO_HIGH = 0.85      # freeze ratio for early epochs
    FREEZE_RATIO_LOW  = 0.65      # freeze ratio after ramp
    FREEZE_EPOCHS_HIGH = 5        # keep HIGH ratio for first N epochs
    UNFREEZE_RAMP_EPOCHS = 5      # linearly ramp from HIGH->LOW over these epochs

    # --- Physics debug print ---
    ENABLE_PHYS_DEBUG_PRINT = False
    PHYS_DEBUG_PRINT_FIRST_N_BATCHES = 1  # per epoch in validation

    GROUND_TRUTH_DIAMETER = 100.2
    NUM_EPOCHS = int(os.environ.get("NUM_EPOCHS", "60")) 
    BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-3
    NUM_WORKERS = 8
    SEED = int(os.environ.get("SEED", "42")) 

    # --- Sensitivity Analysis Mode ---
    SENSITIVITY_MODE = os.environ.get("SENSITIVITY_MODE", "0").strip() in {"1", "true", "True"}
    BASELINE_DIR = os.environ.get("BASELINE_DIR", "")  
    MAX_BATCHES = int(os.environ.get("MAX_BATCHES", "-1"))

    # --- Sensitivity Perturbation Parameters ---
    SENSITIVITY_COARSE_PERTURB = float(os.environ.get("SENSITIVITY_COARSE_PERTURB", "0.0"))  
    SENSITIVITY_IMAGE_NOISE = float(os.environ.get("SENSITIVITY_IMAGE_NOISE", "0.0")) 
    SENSITIVITY_PARAMS_ZERO = os.environ.get("SENSITIVITY_PARAMS_ZERO", "0").strip() in {"1", "true", "True"}  

    # --- Ablation: Data Quantity ---
    SIM_SAMPLE_COUNT = int(os.environ.get("SIM_SAMPLE_COUNT", "-1"))

    # --- Image ---
    RESIZE = 224
    CROP_PIXELS = 3
    COARSE_NOISE_STD = 0.003
    ORIG_H, ORIG_W = 960, 1280


    # --- Real preprocessing (already centered/un-tilted) ---
    APPLY_REAL_ALIGN = True
    APPLY_CENTER_MASK = True
    CENTER_MASK_BAND_PX = 50
    CENTER_MASK_MODE = 'keep'

    # --- Physics Constants (Microns) ---
    PHYSICS_SUBSAMPLE = 1
    PINN_WAVELENGTH_UM = 0.78024
    PINN_FOCLENGTH_UM = 120000.0
    PINN_PIXEL_SIZE_UM = 4.8

    # --- Switches ---
    ENABLE_PHYS  = True
    ENABLE_WEAK  = True
    ENABLE_PAIRCONS = True
    ENABLE_VICREG = True
    ENABLE_TTA_EMB = True
    ENABLE_GROUP_EMB = True

    USE_DIFF_INPUT = False
    USE_FFT_INPUT = True

    # --- Residual diameter parameterization ---
    LOG_SPACE = True
    DELTA_CLIP = 0.08   # ~ ±8.3% in exp space

    # --- Loss Weights (base) ---
    W_PHYS  = 3.0
    W_WEAK  = 3.0        # weak guidance (relative)
    W_PAIR  = 1.5        # pair consistency (semi) for paired only

    # --- Stabilization / Debug switches (paper-friendly) ---
    # NOTE: These can be toggled via environment variables to make runner/probe comparisons easy
    # without editing code, e.g. LOSS_WARMUP_ENABLE=1 LOSS_WARMUP_EPOCHS=3.
    ENABLE_AMP = os.environ.get("ENABLE_AMP", "1").strip() not in {"0", "false", "False"}

    # Loss warmup: ramp selected loss weights from 0 -> target over first K epochs.
    LOSS_WARMUP_ENABLE = os.environ.get("LOSS_WARMUP_ENABLE", "0").strip() in {"1", "true", "True"}
    LOSS_WARMUP_EPOCHS = int(os.environ.get("LOSS_WARMUP_EPOCHS", "3"))
    LOSS_WARMUP_MODE = os.environ.get("LOSS_WARMUP_MODE", "linear").strip().lower()  # linear|cosine
    # Comma-separated: phys,pair,weak,vic,tta,group (default phys only)
    LOSS_WARMUP_TARGETS = [s.strip().lower() for s in os.environ.get("LOSS_WARMUP_TARGETS", "phys").split(",") if s.strip()]
    
    # Reverse warmup: ramp selected loss weights from target -> reduced over first K epochs (for bad seeds)
    # This allows pair loss to dominate early learning, then gradually introduce other losses
    LOSS_REVERSE_WARMUP_ENABLE = os.environ.get("LOSS_REVERSE_WARMUP_ENABLE", "0").strip() in {"1", "true", "True"}
    LOSS_REVERSE_WARMUP_EPOCHS = int(os.environ.get("LOSS_REVERSE_WARMUP_EPOCHS", "5"))
    LOSS_REVERSE_WARMUP_MODE = os.environ.get("LOSS_REVERSE_WARMUP_MODE", "linear").strip().lower()  # linear|cosine
    LOSS_REVERSE_WARMUP_TARGETS = [s.strip().lower() for s in os.environ.get("LOSS_REVERSE_WARMUP_TARGETS", "phys,weak").split(",") if s.strip()]
    LOSS_REVERSE_WARMUP_MIN_SCALE = float(os.environ.get("LOSS_REVERSE_WARMUP_MIN_SCALE", "0.1"))  # minimum scale (0.1 = 10% of original weight)
    
    # Simple adaptive loss weighting (lightweight, no extra memory)
    # Automatically boost pair loss weight when pairloss is high relative to other losses
    LOSS_ADAPTIVE_PAIR_BOOST = os.environ.get("LOSS_ADAPTIVE_PAIR_BOOST", "0").strip() in {"1", "true", "True"}
    LOSS_ADAPTIVE_PAIR_THRESHOLD = float(os.environ.get("LOSS_ADAPTIVE_PAIR_THRESHOLD", "0.03"))  # boost when pairloss > this
    LOSS_ADAPTIVE_PAIR_MAX_BOOST = float(os.environ.get("LOSS_ADAPTIVE_PAIR_MAX_BOOST", "2.0"))  # max multiplier
    
    # Gradual introduction: gradually ramp up other losses after pair-only period
    # This prevents sudden jumps when other losses are enabled after pair-only mode
    LOSS_GRADUAL_INTRO_ENABLE = os.environ.get("LOSS_GRADUAL_INTRO_ENABLE", "0").strip() in {"1", "true", "True"}
    LOSS_GRADUAL_INTRO_EPOCHS = int(os.environ.get("LOSS_GRADUAL_INTRO_EPOCHS", "5"))  # ramp-up period after pair-only
    LOSS_GRADUAL_INTRO_MODE = os.environ.get("LOSS_GRADUAL_INTRO_MODE", "cosine").strip().lower()  # linear|cosine
    LOSS_GRADUAL_INTRO_TARGETS = [s.strip().lower() for s in os.environ.get("LOSS_GRADUAL_INTRO_TARGETS", "phys,weak,vic").split(",") if s.strip()]
    
    # Conditional enable: only enable other losses when pair loss is stable (below threshold)
    # This prevents enabling other losses when pair loss is still high/fluctuating
    LOSS_CONDITIONAL_ENABLE = os.environ.get("LOSS_CONDITIONAL_ENABLE", "0").strip() in {"1", "true", "True"}
    LOSS_CONDITIONAL_PAIR_THRESHOLD = float(os.environ.get("LOSS_CONDITIONAL_PAIR_THRESHOLD", "0.030"))  # enable others when pair < this
    LOSS_CONDITIONAL_STABLE_EPOCHS = int(os.environ.get("LOSS_CONDITIONAL_STABLE_EPOCHS", "2"))  # pair must be stable for N epochs
    
    # Reduced phys weight for bad seeds (to reduce interference)
    W_PHYS_REDUCED = float(os.environ.get("W_PHYS_REDUCED", str(W_PHYS)))  # default same as W_PHYS, can be set lower (e.g., 1.0)

    # Gradient conflict / balancing (debuggable, mutually exclusive)
    USE_PCGRAD = os.environ.get("USE_PCGRAD", "0").strip() in {"1", "true", "True"}
    USE_GRADNORM = os.environ.get("USE_GRADNORM", "0").strip() in {"1", "true", "True"}

    # PCGrad tasks: optional subset to reduce compute/memory.
    # Comma-separated: pair,weak,phys,vic,tta,group. Empty -> include all enabled tasks.
    _PCGRAD_TASKS_RAW = os.environ.get("PCGRAD_TASKS", "").strip()
    PCGRAD_TASKS = [s.strip().lower() for s in _PCGRAD_TASKS_RAW.split(",") if s.strip()] if _PCGRAD_TASKS_RAW else []
    GRADNORM_ALPHA = float(os.environ.get("GRADNORM_ALPHA", "0.5"))
    GRADNORM_LR = float(os.environ.get("GRADNORM_LR", "0.025"))
    GRADNORM_TASKS = [s.strip().lower() for s in os.environ.get(
        "GRADNORM_TASKS",
        "pair,weak,phys,vic,tta,group"
    ).split(",") if s.strip()]

    # VICReg
    W_VICREG_INV = 1.0
    W_VICREG_VAR = 1.0
    W_VICREG_COV = 1.0
    W_VICREG_START = 1.0
    W_VICREG_END   = 0.1

    # TTA / Group on embedding with decay
    W_TTA_START   = 1.0
    W_TTA_END     = 0.5
    W_GROUP_START = 1.0
    W_GROUP_END   = 0.5

    # TTA params
    TTA_NUM_TRAIN = 2
    TTA_NUM_VAL   = 5
    TTA_NOISE_TRAIN = 0.1
    TTA_NOISE_VAL   = 0.1

    # Weak tolerance (relative)
    WEAK_TOLERANCE = 0.1

    # --- Advanced Physics weights ---
    W_PHYS_CORR = 1.0
    W_PHYS_FFT  = 2.0
    W_PHYS_GRAD = 1.0

    # --- FFT band selection (low-frequency band-pass) ---
    # radial frequency normalized to [0.005, 0.5]; low-pass uses [0, FFT_HIGH]
    FFT_LOW  = 0.005
    FFT_HIGH = 0.5

    # --- Selection score weights (no hard jumps) ---
    SCORE_W_PHYS = 3.0
    SCORE_W_WEAK = 3.0
    SCORE_W_PAIR = 1.5
    SCORE_W_VICVAR = 1.5
    
    # --- Improved score: pair emphasis for bad seeds (no GT dependency) ---
    # Increase pair weight for bad seeds (bad seeds have high pairloss)
    # This helps select models with better pair consistency without using GT
    SCORE_W_PAIR_BOOST = float(os.environ.get("SCORE_W_PAIR_BOOST", "1.0"))  # multiplier for pair weight

    # --- Misc ---
    USE_EMA_TEACHER = True
    EMA_DECAY = 0.999

# ============================================================
# 1. Helper Functions
# ============================================================
def _um_to_px(x_um: float, pixel_size_um: float) -> float:
    return float(x_um) / float(pixel_size_um)

def apply_center_band_mask(img: np.ndarray, band_px: int, mode: str = 'keep') -> np.ndarray:
    """Apply a vertical-band mask around the image center.

    mode:
      - 'keep': keep only [cx-band, cx+band], zero elsewhere
      - 'drop': zero-out [cx-band, cx+band], keep elsewhere
    """
    if img is None:
        return img
    band_px = int(band_px) if band_px is not None else 0
    if band_px <= 0:
        return img
    H, W = img.shape[:2]
    cx = W // 2
    l = max(0, cx - band_px)
    r = min(W, cx + band_px + 1)
    out = img.copy()
    if str(mode).lower() == 'drop':
        out[:, l:r] = 0
    else:
        out[:, :l] = 0
        out[:, r:] = 0
    return out

def apply_derotate_and_shift(img: np.ndarray, phi_deg: float, x_shift_um: float,
                             pixel_size_um: float, *,
                             apply_mask: bool = True,
                             mask_band_px: int = 50,
                             mask_mode: str = 'keep') -> np.ndarray:
    """Inverse-transform real image by (-phi, -x_shift)."""
    if img is None:
        return img
    H, W = img.shape[:2]
    phi_deg = float(phi_deg or 0.0)
    x_shift_um = float(x_shift_um or 0.0)

    # inverse rotation
    M = cv2.getRotationMatrix2D((W / 2.0, H / 2.0), -phi_deg, 1.0)
    img_r = cv2.warpAffine(img, M, (W, H), flags=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    # inverse horizontal shift
    dx_px = -_um_to_px(x_shift_um, pixel_size_um)
    M2 = np.array([[1.0, 0.0, dx_px], [0.0, 1.0, 0.0]], dtype=np.float32)
    img_rs = cv2.warpAffine(img_r, M2, (W, H), flags=cv2.INTER_LINEAR,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0)

    if apply_mask:
        img_rs = apply_center_band_mask(img_rs, band_px=int(mask_band_px), mode=str(mask_mode))
    return img_rs

def preprocess_image_basic(img_path: str) -> np.ndarray:
    """Read image, to gray, flip, crop borders. (legacy behavior)"""
    if not os.path.exists(img_path):
        raise FileNotFoundError(f"Image path does not exist: {img_path}")
    img = cv2.imread(img_path)
    if img is None:
        raise ValueError(f"Failed to decode image: {img_path}")
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.flip(img, 0)
    crop = 3
    if img.shape[0] <= 2*crop or img.shape[1] <= 2*crop:
        raise ValueError(f"Image too small after crop: {img_path} shape={img.shape}")
    img = img[crop:-crop, crop:-crop]
    return img.astype(np.float64)

def preprocess_image(img_path: str,
                     *,
                     align_phi_deg: Optional[float] = None,
                     align_x_shift_um: Optional[float] = None,
                     apply_align: bool = False,
                     apply_mask: bool = False,
                     mask_band_px: int = 50,
                     mask_mode: str = 'keep') -> np.ndarray:
    """Unified loader with optional alignment/mask."""
    img = preprocess_image_basic(img_path)
    if apply_align:
        img = apply_derotate_and_shift(
            img,
            phi_deg=float(align_phi_deg or 0.0),
            x_shift_um=float(align_x_shift_um or 0.0),
            pixel_size_um=DataConfig.PIXEL_SIZE_UM,
            apply_mask=bool(apply_mask),
            mask_band_px=int(mask_band_px),
            mask_mode=str(mask_mode),
        )
    elif apply_mask:
        img = apply_center_band_mask(img, band_px=int(mask_band_px), mode=str(mask_mode))
    return img.astype(np.float64)

def extract_real_id(fname: str) -> int:
    """Extract ID from filename: 'foc_120_(4).BMP' -> 4"""
    m = re.search(r"\((\d+)\)", fname)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", fname)
    if m:
        return int(m.group(1))
    return abs(hash(fname)) % 10000

def list_bmp_files(folder: str) -> List[str]:
    files = glob.glob(os.path.join(folder, '*.BMP'))
    # keep deterministic ordering
    return sorted(files)

def count_real_bmp(folder: str) -> int:
    """Count 'real' images by extension (.bmp only, case-insensitive)."""
    return len(list_bmp_files(folder))

def validate_real_folders():
    """Validate that we are counting only .bmp as real images (not .png sims, etc)."""
    # If user keeps everything in one folder, validate DATA_FOLDER (used for splitting).
    if DataConfig.AUTO_SPLIT_FROM_SINGLE_FOLDER:
        n_all = len(os.listdir(DataConfig.DATA_FOLDER)) if os.path.isdir(DataConfig.DATA_FOLDER) else 0
        n_bmp = count_real_bmp(DataConfig.DATA_FOLDER) if os.path.isdir(DataConfig.DATA_FOLDER) else 0
        print(f"[CHECK] DATA_FOLDER total files={n_all}, real_bmp={n_bmp} (only .bmp counted as real).")
        if n_bmp == 0:
            raise FileNotFoundError(f"No real .bmp images found in DATA_FOLDER: {DataConfig.DATA_FOLDER}")

    # Validate train/val folders if they exist
    if os.path.isdir(DataConfig.TRAIN_REAL_FOLDER):
        n_all = len(os.listdir(DataConfig.TRAIN_REAL_FOLDER))
        n_bmp = count_real_bmp(DataConfig.TRAIN_REAL_FOLDER)
        print(f"[CHECK] TRAIN_REAL_FOLDER total files={n_all}, real_bmp={n_bmp}.")
        if n_bmp == 0:
            raise FileNotFoundError(f"No real .bmp images found in TRAIN_REAL_FOLDER: {DataConfig.TRAIN_REAL_FOLDER}")

    if os.path.isdir(DataConfig.VAL_REAL_FOLDER):
        n_all = len(os.listdir(DataConfig.VAL_REAL_FOLDER))
        n_bmp = count_real_bmp(DataConfig.VAL_REAL_FOLDER)
        print(f"[CHECK] VAL_REAL_FOLDER total files={n_all}, real_bmp={n_bmp}.")

def _fast_verify_limit_list(items, k: int):
    try:
        k = int(k)
    except Exception:
        k = 0
    if k <= 0:
        return list(items)
    return list(items)[:k]

def ensure_empty_dir(path: str, overwrite: bool):
    if os.path.exists(path):
        if overwrite:
            shutil.rmtree(path)
        else:
            return
    os.makedirs(path, exist_ok=True)

def maybe_auto_split_to_folders():
    """Optional: split from a single folder into train_real/ and val_real/ (copy, non-destructive)."""
    if not DataConfig.AUTO_SPLIT_FROM_SINGLE_FOLDER:
        return

    src_folder = DataConfig.DATA_FOLDER
    all_files = list_bmp_files(src_folder)
    if not all_files:
        raise FileNotFoundError(f"No .bmp files found in {src_folder}")

    os.makedirs(DataConfig.TRAIN_REAL_FOLDER, exist_ok=True)
    os.makedirs(DataConfig.VAL_REAL_FOLDER, exist_ok=True)

    if DataConfig.VAL_REAL_IDS is not None:
        val_ids = set(DataConfig.VAL_REAL_IDS)
        val_files = []
        train_files = []
        for f in all_files:
            rid = extract_real_id(os.path.basename(f))
            (val_files if rid in val_ids else train_files).append(f)
    else:
        rng = np.random.RandomState(DataConfig.SPLIT_SEED)
        idx = np.arange(len(all_files))
        rng.shuffle(idx)
        split = int(len(all_files) * DataConfig.VAL_RATE)
        val_idx = set(idx[:split].tolist())
        train_files, val_files = [], []
        for i, f in enumerate(all_files):
            (val_files if i in val_idx else train_files).append(f)

    # copy files (do not delete original)
    for f in train_files:
        dst = os.path.join(DataConfig.TRAIN_REAL_FOLDER, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)

    for f in val_files:
        dst = os.path.join(DataConfig.VAL_REAL_FOLDER, os.path.basename(f))
        if not os.path.exists(dst):
            shutil.copy2(f, dst)

    print(f"[AUTO_SPLIT] train={len(train_files)} val={len(val_files)} copied to folders.")


# ============================================================
# PART 1: Initial Diameter Estimation (FFT)  [TRAIN ONLY]
# ============================================================
def estimate_initial_diameter() -> float:
    print("\n" + "="*60)
    print(" [PART 1] INITIAL DIAMETER ESTIMATION (TRAIN ONLY) ")
    print("="*60)

    if os.path.exists(DataConfig.INIT_DIAMETER_FILE):
        try:
            with open(DataConfig.INIT_DIAMETER_FILE, 'r') as f:
                val = float(f.read().strip())
            print(f" [SKIP] Found existing file. Diameter: {val:.3f} um")
            return val
        except Exception:
            pass

    bmp_files = list_bmp_files(DataConfig.TRAIN_REAL_FOLDER)
    if not bmp_files:
        raise FileNotFoundError(f"No .bmp files found in {DataConfig.TRAIN_REAL_FOLDER}")

    target_files = bmp_files[:1] if DataConfig.FAST_VERIFY else bmp_files[:5]

    estimates = []
    for filepath in target_files:
        img = preprocess_image_basic(filepath)
        profile = np.max(img, axis=1)
        window = max(3, int(len(profile) / 20))
        profile = np.convolve(profile, np.ones(window) / window, mode='same')
        profile = (profile - profile.min()) / (profile.max() - profile.min() + 1e-8)

        N = len(profile)
        Y = np.fft.fft(profile)
        P1 = np.abs(Y / N)[:N // 2 + 1]
        Fs = 1.0 / DataConfig.PIXEL_SIZE_UM
        freqs = np.linspace(0, Fs / 2, len(P1))

        if len(P1) > 5:
            peak_idx = np.argmax(P1[5:]) + 5
            d_est = freqs[peak_idx] * DataConfig.WAVELENGTH_UM * DataConfig.FOCAL_LENGTH_UM
            estimates.append(d_est)
            print(f" -> {os.path.basename(filepath)}: {d_est:.3f} um")

    final_a = float(np.median(estimates)) if estimates else 102.555
    os.makedirs(os.path.dirname(DataConfig.INIT_DIAMETER_FILE), exist_ok=True)
    with open(DataConfig.INIT_DIAMETER_FILE, 'w') as f:
        f.write(str(final_a))

    print(f" [RESULT] Final Coarse a = {final_a:.3f} um")
    return final_a


# ============================================================
# PART 2: Parameter Optimization (DE)  [TRAIN ONLY]
# ============================================================
def generate_field_raw(params, shape, consts):
    phi, r_mm, b_um, x_shift_um, a_um, _ = params
    H, W = shape

    r_um = r_mm * 1000.0
    angle_rad = np.arctan(r_um / consts.FOCAL_LENGTH_UM)

    x_vec = np.linspace(-W * consts.PIXEL_SIZE_UM / 2, W * consts.PIXEL_SIZE_UM / 2, W)
    y_vec = np.linspace(np.tan(angle_rad) * consts.FOCAL_LENGTH_UM,
                        np.tan(angle_rad) * consts.FOCAL_LENGTH_UM + H * consts.PIXEL_SIZE_UM, H)
    X, Y = np.meshgrid(x_vec, y_vec)

    y_c = y_vec[0] + (y_vec[-1] - y_vec[0]) / 2
    Xs = X
    Ys = Y - y_c

    th = np.deg2rad(phi)
    Xr = np.cos(th) * Xs - np.sin(th) * Ys + x_shift_um
    Yr = np.sin(th) * Xs + np.cos(th) * Ys + y_c

    dist = np.sqrt(Xr ** 2 + Yr ** 2 + consts.FOCAL_LENGTH_UM ** 2)
    sin_tx = Xr / dist
    sin_ty = Yr / dist

    alpha = (b_um * sin_tx) / consts.WAVELENGTH_UM
    beta = (a_um * sin_ty) / consts.WAVELENGTH_UM

    field = (np.sinc(alpha) ** 2) * (np.sinc(beta) ** 2)
    return field


def calculate_analytic_gain(I_real, field_nonlinear):
    mask = I_real < 254
    if np.sum(mask) < 100:
        mask = np.ones_like(I_real, dtype=bool)

    vec_real = I_real[mask]
    vec_sim = field_nonlinear[mask]

    num = np.dot(vec_real, vec_sim)
    den = np.dot(vec_sim, vec_sim)
    gain_ls = num / (den + 1e-10)

    max_r = np.max(I_real)
    max_s = np.max(field_nonlinear) + 1e-9
    gain_peak = max_r / max_s
    return max(0.0, min(gain_ls, gain_peak * 1.05))


def objective_function(params, I_real, consts, a_prior: Optional[float] = None):
    gamma = params[5]
    field = generate_field_raw(params, I_real.shape, consts)
    field_non = np.power(np.abs(field), gamma)
    gain = calculate_analytic_gain(I_real, field_non)
    I_sim = np.clip(field_non * gain, 0, 255)

    mae = np.mean(np.abs(I_real - I_sim)) / 255.0

    f_r = np.log(np.abs(np.fft.fftshift(np.fft.fft2(I_real))) + 1)
    f_s = np.log(np.abs(np.fft.fftshift(np.fft.fft2(I_sim))) + 1)
    f_r = (f_r - f_r.mean()) / (f_r.std() + 1e-6)
    f_s = (f_s - f_s.mean()) / (f_s.std() + 1e-6)
    fft_loss = np.mean(np.abs(f_r - f_s))

    flat_r = I_real.flatten()
    flat_s = I_sim.flatten()
    corr = np.corrcoef(flat_r, flat_s)[0, 1] if np.std(flat_r) > 1e-5 else 0.0

    # --- soft prior on diameter a (optional) ---
    reg_a = 0.0
    if a_prior is not None and getattr(consts, "A_REG_WEIGHT", 0.0) > 0:
        a_um = float(params[4])
        ap = float(a_prior)
        reg_a = float(consts.A_REG_WEIGHT) * ((a_um - ap) / (ap + 1e-9))**2

    return 2.5 * (1 - corr) + 2.0 * fft_loss + 0.5 * mae + reg_a



def _optimize_single_image(I_real: np.ndarray, bounds: List[Tuple[float, float]], consts, a_prior: Optional[float],
                           maxiter: int, popsize: int, seed: int) -> Tuple[np.ndarray, float, float]:
    """Run DE for a single image; returns (params, loss, gain)."""
    import time

    t0 = time.time()
    gen = [0]
    last_print_t = [t0]

    def _cb(xk, convergence):
        # SciPy: callback is called once per generation
        gen[0] += 1
        now = time.time()

        if (now - last_print_t[0]) >= 2.0 or gen[0] == 1:
            print(f"    [DE] gen={gen[0]:04d}/{maxiter}  conv={convergence:.3e}  elapsed={now - t0:.1f}s",
                  flush=True)
            last_print_t[0] = now
        return False  # return True would stop early

    res = differential_evolution(
        objective_function, bounds, args=(I_real, consts, a_prior),
        strategy='best1bin',
        maxiter=maxiter,
        popsize=popsize,
        tol=0.01,
        workers=DataConfig.N_JOBS,
        updating='deferred',  
        seed=seed,
        polish=True,
        init='latinhypercube',
        disp=True,             
        callback=_cb          
    )

    p = res.x
    field = generate_field_raw(p, I_real.shape, consts)
    field_non = np.power(np.abs(field), p[5])
    gain = calculate_analytic_gain(I_real, field_non)
    return p, float(res.fun), float(gain)



def optimize_real_folder(
    folder: str,
    init_a: float,
    bounds: List[Tuple[float, float]],
    a_prior: Optional[float],
    maxiter: int,
    popsize: int,
    mode: str,
    file_list: Optional[List[str]] = None
    ) -> pd.DataFrame:
    """Unified optimizer for both TRAIN and VAL real images.

    mode:
      - 'part2': return columns compatible with optimized_params.csv (Filename, Phi, R_mm, B_um, X_um, A_um, Gamma, Gain_Calc, Loss)
      - 'rp'   : return columns for rp tables (fname, phi, r, b, x, a, gamma, gain_calc, loss)
    """
    bmp_files = list_bmp_files(folder) if (file_list is None) else list(file_list)
    if not bmp_files:
        raise FileNotFoundError(f"No .bmp files found in {folder}")

    records = []
    for i, f in enumerate(bmp_files):
        fname = os.path.basename(f)
        try:
            # For optimization we use RAW real image (may be tilted/shifted). Alignment+mask happen later before network.
            I_real = preprocess_image_basic(f)
        except Exception as e:
            print(f"[WARN] skip {fname}: {e}")
            continue

        p, loss, gain = _optimize_single_image(
            I_real=I_real, bounds=bounds, consts=DataConfig, a_prior=a_prior,
            maxiter=maxiter, popsize=popsize, seed=42
        )

        if mode == 'part2':
            records.append({
                'Filename': fname,
                'Phi': float(p[0]), 'R_mm': float(p[1]), 'B_um': float(p[2]), 'X_um': float(p[3]),
                'A_um': float(p[4]), 'Gamma': float(p[5]), 'Gain_Calc': float(gain),
                'Loss': float(loss)
            })
        elif mode == 'rp':
            records.append({
                'fname': fname,
                'phi': float(p[0]), 'r': float(p[1]), 'b': float(p[2]), 'x': float(p[3]),
                'a': float(p[4]), 'gamma': float(p[5]),
                'gain_calc': float(gain), 'loss': float(loss)
            })
        else:
            raise ValueError(f"Unknown mode: {mode}")

        print(f"[{i+1}/{len(bmp_files)}] {fname} -> Loss: {loss:.4f}")

    return pd.DataFrame(records)

def run_optimization(init_a: float) -> pd.DataFrame:
    print("\n" + "="*60)
    print(" [PART 2] PARAMETER OPTIMIZATION (TRAIN ONLY) ")
    print("="*60)

    if os.path.exists(DataConfig.RESULTS_FILE):
        print(f" [SKIP] Found existing results: {DataConfig.RESULTS_FILE}")
        df_exist = pd.read_csv(DataConfig.RESULTS_FILE)
        # ---- minimal post-process (idempotent) ----
        # Keep DE-estimated Phi/X_um for alignment; do NOT overwrite them here.
        # a weighting: a_est = 0.3*a_DE + 0.7*a_coarse
        if 'A_um' in df_exist.columns:
            if 'A_um_DE' not in df_exist.columns:
                df_exist['A_um_DE'] = df_exist['A_um']
                df_exist['A_um'] = 0.3 * df_exist['A_um'].astype(float) + 0.7 * float(init_a)

        df_exist.to_csv(DataConfig.RESULTS_FILE, index=False)
        return df_exist

    # bounds: [phi, r, b, x, a, gamma]
    bounds = list(DataConfig.BOUNDS_BASE)
    if bool(getattr(DataConfig, 'A_TO_COARSE', False)):
        eps = float(getattr(DataConfig, 'A_EPS_REL', 1e-6))
        bounds.insert(4, (init_a * (1 - eps), init_a * (1 + eps)))
    else:
        bounds.insert(4, (init_a * (1 - DataConfig.A_PRIOR_RATIO), init_a * (1 + DataConfig.A_PRIOR_RATIO)))
    bmp_files = list_bmp_files(DataConfig.TRAIN_REAL_FOLDER)
    if DataConfig.FAST_VERIFY:
        maxiter = 1
        popsize = 1
        bmp_files = _fast_verify_limit_list(bmp_files, DataConfig.FAST_VERIFY_MAX_OPT_IMAGES)
    else:
        maxiter = 25
        popsize = 25

    # Unified optimizer (TRAIN)
    df = optimize_real_folder(
        folder=DataConfig.TRAIN_REAL_FOLDER,
        init_a=init_a,
        bounds=bounds,
        a_prior=init_a,
        maxiter=maxiter,
        popsize=popsize,
        mode='part2',
        file_list=bmp_files
    )
    # ---- post-process optimized parameters (minimal patch) ----
    # Keep DE-estimated Phi/X_um for later REAL alignment (derotate+recenter) before mask/network.
    # (2) Diameter weighting: a_est = 0.3*a_DE + 0.7*a_coarse (FFT-peak is more robust)
    if 'A_um' in df.columns:
        df['A_um_DE'] = df['A_um']
        df['A_um'] = 0.3 * df['A_um'].astype(float) + 0.7 * float(init_a)

    cols = ['Filename', 'Phi', 'R_mm', 'B_um', 'X_um', 'A_um', 'A_um_DE', 'Gamma', 'Gain_Calc', 'Loss']
    df = df[cols]
    os.makedirs(os.path.dirname(DataConfig.RESULTS_FILE), exist_ok=True)
    df.to_csv(DataConfig.RESULTS_FILE, index=False)
    return df


# ============================================================
# PART 2.5: Export REAL coarse parameters (rp) for conditioning  [TRAIN & VAL]
# ============================================================

def export_real_rp_tables(df_opt_train: pd.DataFrame, init_a: float):
    """Create train_real_rp.csv (PHYSICAL UNITS) and val_real_rp.csv (leakage-safe)."""
    os.makedirs(os.path.dirname(DataConfig.TRAIN_REAL_RP_FILE), exist_ok=True)
    os.makedirs(os.path.dirname(DataConfig.VAL_REAL_RP_FILE), exist_ok=True)

    # -------- Train rp (from optimized_params.csv) --------
    overwrite = bool(getattr(DataConfig, "OVERWRITE_REAL_RP", True))
    if overwrite or (not os.path.exists(DataConfig.TRAIN_REAL_RP_FILE)):
        train_rp = df_opt_train.copy()
        cols_map = {'Filename':'fname','Phi':'phi','R_mm':'r','B_um':'b','X_um':'x','A_um':'a','Gamma':'gamma'}
        missing = [k for k in cols_map.keys() if k not in train_rp.columns]
        if missing:
            raise KeyError(f"Missing column(s) in optimized params: {missing}")
        train_rp = train_rp[list(cols_map.keys())].rename(columns=cols_map)

        def _assert_range(col, lo, hi):
            s = train_rp[col].astype(float)
            ok = (s >= lo) & (s <= hi)
            if not ok.all():
                bad = train_rp.loc[~ok, ['fname', col]].head(8)
                raise ValueError(f"[SANITY FAIL] {col} out of [{lo},{hi}]. Examples:\n{bad}")

        _assert_range('phi', -6.0, 6.0)
        _assert_range('r', -20.0, 0.0)
        _assert_range('b', 700.0, 2000.0)
        _assert_range('x', -500.0, 500.0)
        _assert_range('a', 50.0, 200.0)
        _assert_range('gamma', 0.05, 2.0)

        train_rp.to_csv(DataConfig.TRAIN_REAL_RP_FILE, index=False)
        print(f" [DONE] Train real rp saved (PHYSICAL): {DataConfig.TRAIN_REAL_RP_FILE}")
    else:
        print(f" [SKIP] Train real rp exists: {DataConfig.TRAIN_REAL_RP_FILE}")

    # -------- Val rp (unified DE on val real images only) --------

    if DataConfig.FAST_VERIFY and bool(DataConfig.FAST_VERIFY_SKIP_VAL):
        print(" [FAST_VERIFY] FAST_VERIFY_SKIP_VAL=True -> skip val_real_rp.csv generation.")
        return

    if os.path.exists(DataConfig.VAL_REAL_RP_FILE):
        print(f" [SKIP] Val real rp exists: {DataConfig.VAL_REAL_RP_FILE}")
        return

    val_files = list_bmp_files(DataConfig.VAL_REAL_FOLDER)
    if not val_files:
        print(" [INFO] No val real images found; skip val_real_rp.csv generation.")
        return

    # bounds: [phi, r, b, x, a, gamma]
    bounds = list(DataConfig.BOUNDS_BASE)
    if bool(getattr(DataConfig, 'A_TO_COARSE', False)):
        eps = float(getattr(DataConfig, 'A_EPS_REL', 1e-6))
        bounds.insert(4, (init_a * (1 - eps), init_a * (1 + eps)))
    else:
        bounds.insert(4, (init_a * (1 - DataConfig.A_PRIOR_RATIO), init_a * (1 + DataConfig.A_PRIOR_RATIO)))
    # lightweight settings for VAL rp (configurable)
    maxiter = 1 if DataConfig.FAST_VERIFY else int(DataConfig.VAL_RP_OPT_MAXITER)
    popsize = 1 if DataConfig.FAST_VERIFY else int(DataConfig.VAL_RP_OPT_POPSIZE)

    print(f"Estimating VAL real rp for {len(val_files)} images (unified DE with a-prior)...")
    val_bmps = list_bmp_files(DataConfig.VAL_REAL_FOLDER)
    if DataConfig.FAST_VERIFY:
        val_bmps = _fast_verify_limit_list(val_bmps, DataConfig.FAST_VERIFY_MAX_OPT_IMAGES)
    dfv = optimize_real_folder(
        folder=DataConfig.VAL_REAL_FOLDER,
        init_a=init_a,
        bounds=bounds,
        a_prior=init_a,
        maxiter=maxiter,
        popsize=popsize,
        mode='rp',
        file_list=val_bmps
    )

    if len(dfv) == 0:
        print(" [WARN] No val rp records produced; skip writing val_real_rp.csv")
        return
    dfv.to_csv(DataConfig.VAL_REAL_RP_FILE, index=False)
    print(f" [DONE] Val real rp saved: {DataConfig.VAL_REAL_RP_FILE}")

# ============================================================
# PART 3: Data Generation (Leakage-Safe)
# ============================================================
def latin_hypercube(n: int, d: int, seed: int) -> np.ndarray:
    """
    Pure-numpy Latin Hypercube Sampling in [0,1]^d.
    Ensures each dimension is stratified into n bins with one sample per bin.
    """
    rng = np.random.default_rng(seed)
    # Stratified positions within each bin
    cut = (np.arange(n) + rng.random(n)) / n  # (n,)
    u = np.empty((n, d), dtype=np.float64)
    for j in range(d):
        perm = rng.permutation(n)
        u[:, j] = cut[perm]
    return u


def lhs_perturb_params(p_center: np.ndarray, n: int, rate: float, seed: int) -> np.ndarray:
    """
    LHS perturbation around p_center with relative +/-rate and per-dimension absolute floor.
    Returns (n, d) array in physical units.
    """
    p_center = np.asarray(p_center, dtype=np.float64)
    d = p_center.shape[0]

    delta = np.abs(p_center) * rate
    p_min = p_center - delta
    p_max = p_center + delta

    u = latin_hypercube(n=n, d=d, seed=seed)
    return p_min + u * (p_max - p_min)

def generate_worker_train_sim_and_train_diffs(args):
    """
    Worker:
    - Generate one sim from TRAIN source (LHS-perturbed params)
    - Generate diffs against TRAIN targets ONLY (no leakage)
    - Save sim + diffs to TRAIN folders
    - Return metadata rows for TRAIN csv
    """
    row_data, src_rid, s_idx, p_sample, shape, consts, train_real_imgs = args
    # p_sample is provided by LHS table in run_generator

    phi, r, b, x, a, gamma = p_sample
    # Inputs are already centered/un-tilted -> enforce phi=0 and x_shift=0 for sims.
    phi = 0.0
    x = 0.0
    p_sample = np.array([phi, r, b, x, a, gamma], dtype=np.float64)

    # generate sim using source real for analytic gain
    src_fname = row_data['Filename']
    src_real_img = train_real_imgs[src_fname]

    field = generate_field_raw(p_sample, shape, consts)
    field_non = np.power(np.abs(field), gamma)
    gain = calculate_analytic_gain(src_real_img, field_non)
    I_sim = np.clip(field_non * gain, 0, 255).astype(np.uint8)
    # Apply the same center-band mask to SIM as REAL (direct masked, no extra rotate/shift).
    if bool(getattr(DataConfig, 'APPLY_CENTER_MASK', False)):
        I_sim = apply_center_band_mask(I_sim, band_px=int(getattr(DataConfig,'CENTER_MASK_BAND_PX',50)), mode=str(getattr(DataConfig,'CENTER_MASK_MODE','keep')))

    sim_name = f"sim_src{src_rid}_sam{s_idx}.png"
    sim_path_full = os.path.join(consts.TRAIN_SIM_FOLDER, sim_name)

    # Resume-friendly: if sim already exists and is readable, reuse it; otherwise write atomically.
    if os.path.exists(sim_path_full):
        I_sim_exist = cv2.imread(sim_path_full, cv2.IMREAD_GRAYSCALE)
        if I_sim_exist is not None and I_sim_exist.size > 0:
            I_sim = I_sim_exist
        else:
            cv2.imwrite(sim_path_full, I_sim)
    else:
        cv2.imwrite(sim_path_full, I_sim)

    metadata_records = []
    for _ti, (tgt_fname, tgt_img) in enumerate(train_real_imgs.items()):
        if DataConfig.FAST_VERIFY and _ti >= int(DataConfig.FAST_VERIFY_MAX_DIFF_TARGETS):
            break
        tgt_rid = extract_real_id(tgt_fname)

        tgt_img_u8 = np.clip(tgt_img, 0, 255).astype(np.uint8)
        I_diff = cv2.absdiff(tgt_img_u8, I_sim)

        diff_name = f"diff_tgt{tgt_rid}_src{src_rid}_sam{s_idx}.png"
        diff_path_full = os.path.join(consts.TRAIN_DIFF_FOLDER, diff_name)
        if os.path.exists(diff_path_full):
            # If existing diff is unreadable (corrupted), rewrite.
            d_exist = cv2.imread(diff_path_full, cv2.IMREAD_GRAYSCALE)
            if d_exist is None or d_exist.size == 0:
                cv2.imwrite(diff_path_full, I_diff)
        else:
            cv2.imwrite(diff_path_full, I_diff)

        metadata_records.append({
            # identity fields (make leakage auditing trivial)
            'tgt_fname': tgt_fname,
            'tgt_rid': int(tgt_rid),
            'src_fname': src_fname,
            'src_rid': int(src_rid),

            # numeric labels (sim params)
            'phi': float(phi), 'r': float(r), 'b': float(b), 'x': float(x),
            'a': float(a), 'gamma': float(gamma), 'gain': float(gain),

            # relative paths (under dataset root)
            'sim_path': os.path.join('train', 'sim', sim_name).replace('\\', '/'),
            'diff_path': os.path.join('train', 'diff', diff_name).replace('\\', '/'),

            'is_paired': bool(tgt_fname == src_fname)
        })

    return metadata_records


def generate_worker_val_sim_and_val_diffs(args):
    """
    Worker:
    - Generate one sim from VAL source (LHS-perturbed params)
    - Generate diffs against VAL targets ONLY (strict no leakage)
    - Save sim + diffs to VAL folders
    - Return metadata rows for VAL csv
    """
    row_data, src_rid, s_idx, p_sample, shape, consts, val_real_imgs = args

    phi, r, b, x, a, gamma = p_sample
    # Inputs are already centered/un-tilted -> enforce phi=0 and x_shift=0 for sims.
    phi = 0.0
    x = 0.0
    p_sample = np.array([phi, r, b, x, a, gamma], dtype=np.float64)

    src_fname = row_data['Filename']
    src_real_img = val_real_imgs[src_fname]

    # generate sim using source real for analytic gain
    field = generate_field_raw(p_sample, shape, consts)
    field_non = np.power(np.abs(field), gamma)
    gain = calculate_analytic_gain(src_real_img, field_non)
    I_sim = np.clip(gain * field_non, 0, 255).astype(np.uint8)
    # Apply the same center-band mask to SIM as REAL (direct masked, no extra rotate/shift).
    if bool(getattr(DataConfig, 'APPLY_CENTER_MASK', False)):
        I_sim = apply_center_band_mask(I_sim, band_px=int(getattr(DataConfig,'CENTER_MASK_BAND_PX',50)), mode=str(getattr(DataConfig,'CENTER_MASK_MODE','keep')))

    sim_name = f"simv_src{int(src_rid):04d}_sam{int(s_idx):04d}.png"
    sim_path = os.path.join(DataConfig.VAL_SIM_FOLDER, sim_name)

    # Resume-friendly: reuse existing sim if readable
    if os.path.exists(sim_path):
        I_sim_exist = cv2.imread(sim_path, cv2.IMREAD_GRAYSCALE)
        if I_sim_exist is not None and I_sim_exist.size > 0:
            I_sim = I_sim_exist
        else:
            cv2.imwrite(sim_path, I_sim)
    else:
        cv2.imwrite(sim_path, I_sim)

    metadata_records = []
    for _ti, (tgt_fname, tgt_real_img) in enumerate(val_real_imgs.items()):
        if DataConfig.FAST_VERIFY and _ti >= int(DataConfig.FAST_VERIFY_MAX_DIFF_TARGETS):
            break
        tgt_img_u8 = np.clip(tgt_real_img, 0, 255).astype(np.uint8)
        diff_img = cv2.absdiff(tgt_img_u8, I_sim)
        diff_name = f"diffv_tgt{extract_real_id(tgt_fname):04d}_src{int(src_rid):04d}_sam{int(s_idx):04d}.png"
        diff_path = os.path.join(DataConfig.VAL_DIFF_FOLDER, diff_name)
        if os.path.exists(diff_path):
            d_exist = cv2.imread(diff_path, cv2.IMREAD_GRAYSCALE)
            if d_exist is None or d_exist.size == 0:
                cv2.imwrite(diff_path, diff_img)
        else:
            cv2.imwrite(diff_path, diff_img)

        metadata_records.append({
            'tgt_fname': tgt_fname,
            'tgt_rid': int(extract_real_id(tgt_fname)),
            'src_fname': src_fname,
            'src_rid': int(src_rid),

            'phi': float(phi), 'r': float(r), 'b': float(b), 'x': float(x),
            'a': float(a), 'gamma': float(gamma), 'gain': float(gain),

            'sim_path': os.path.join('val', 'sim', sim_name).replace('\\', '/'),
            'diff_path': os.path.join('val', 'diff', diff_name).replace('\\', '/'),

            'is_paired': bool(tgt_fname == src_fname)
        })

    return metadata_records

def generate_val_diffs_from_train_sims(
    train_csv: pd.DataFrame,
    val_real_imgs: Dict[str, np.ndarray],
    dataset_root: str,
    n_jobs: int
) -> pd.DataFrame:
    """
    Generate validation diffs:
    - For each row in train_csv (each has a sim_path and src_rid/tgt_rid),
      pair its sim with each VAL target real and compute absdiff.
    - IMPORTANT: sim stays from TRAIN; VAL is only target.
    """
    # load all sims referenced in train_csv only once if possible
    # but to reduce memory, do per-task loading.
    train_sim_rel = train_csv['sim_path'].unique().tolist()

    # prepare val targets list
    val_targets = list(val_real_imgs.items())

    def _worker(sim_rel_path: str):
        sim_full = os.path.join(dataset_root, sim_rel_path)
        I_sim = cv2.imread(sim_full, cv2.IMREAD_GRAYSCALE)
        if I_sim is None:
            raise FileNotFoundError(f"Missing sim file: {sim_full}")

        # parse src_rid + sam from filename to keep consistent naming
        base = os.path.basename(sim_rel_path)
        m = re.search(r"sim_src(\d+)_sam(\d+)\.png$", base)
        if not m:
            raise ValueError(f"Unexpected sim filename: {base}")
        src_rid = int(m.group(1))
        s_idx = int(m.group(2))

        rows = []
        for tgt_fname, tgt_img in val_targets:
            tgt_rid = extract_real_id(tgt_fname)
            tgt_img_u8 = np.clip(tgt_img, 0, 255).astype(np.uint8)
            I_diff = cv2.absdiff(tgt_img_u8, I_sim)

            diff_name = f"diff_tgt{tgt_rid}_src{src_rid}_sam{s_idx}.png"
            diff_full = os.path.join(DataConfig.VAL_DIFF_FOLDER, diff_name)
            cv2.imwrite(diff_full, I_diff)

            rows.append({
                'tgt_fname': tgt_fname,
                'tgt_rid': int(tgt_rid),
                # src is still the sim source
                'src_fname': None,
                'src_rid': int(src_rid),

                # keep sim params by joining later
                'sim_path': sim_rel_path.replace('\\', '/'),
                'diff_path': os.path.join('val', 'diff', diff_name).replace('\\', '/'),
                'is_paired': False,  # cross-domain by design
            })
        return rows

    results = Parallel(n_jobs=n_jobs, verbose=5)(delayed(_worker)(p) for p in train_sim_rel)
    flat = []
    for sub in results:
        flat.extend(sub)
    val_df = pd.DataFrame(flat)

    # join sim parameter columns from train_csv (unique per sim_path)
    sim_params_cols = ['sim_path', 'phi', 'r', 'b', 'x', 'a', 'gamma', 'gain', 'src_rid', 'src_fname']
    sim_params = train_csv[sim_params_cols].drop_duplicates(subset=['sim_path']).copy()
    sim_params['sim_path'] = sim_params['sim_path'].astype(str)

    val_df = val_df.merge(sim_params, on=['sim_path', 'src_rid'], how='left', suffixes=('', '_train'))
    # reorder columns
    cols = [
        'tgt_fname', 'tgt_rid', 'src_fname', 'src_rid',
        'phi', 'r', 'b', 'x', 'a', 'gamma', 'gain',
        'sim_path', 'diff_path', 'is_paired'
    ]
    # fill missing src_fname for cleanliness (optional)
    if 'src_fname' not in val_df.columns:
        val_df['src_fname'] = None
    val_df = val_df[cols]
    return val_df


def run_generator(df_opt_train: pd.DataFrame):
    """
    Generate leakage-safe dataset artifacts:
      - Train: sims from train reals; diffs only against train reals
      - Val  : sims from val reals; diffs only against val reals (no train/val mixing)
    """
    print("\n" + "="*60)
    print(" [PART 3] DATASET GENERATION (LEAKAGE-SAFE) ")
    print("="*60)
    # If dataset is already complete and overwrite is disabled, skip heavy generation.
    if (not bool(getattr(DataConfig, "OVERWRITE_DATASET", False))) and dataset_png_is_complete(DataConfig.DATASET_ROOT, DataConfig.TRAIN_LABELS, min_coverage=0.999):
        if (not bool(getattr(DataConfig, "GENERATE_VAL_DIFF", True))) or dataset_png_is_complete(DataConfig.DATASET_ROOT, DataConfig.VAL_LABELS, min_coverage=0.999):
            print(" [SKIP] Dataset CSV+PNG already complete. Set OVERWRITE_DATASET=True to force regeneration.")
            return


    # prepare folders
    if DataConfig.OVERWRITE_DATASET:
        ensure_empty_dir(DataConfig.DATASET_ROOT, overwrite=True)
    os.makedirs(DataConfig.DATASET_ROOT, exist_ok=True)

    ensure_empty_dir(DataConfig.TRAIN_SIM_FOLDER, overwrite=DataConfig.OVERWRITE_DATASET)
    ensure_empty_dir(DataConfig.TRAIN_DIFF_FOLDER, overwrite=DataConfig.OVERWRITE_DATASET)
    # Prepare VAL folders (self-sim/diff) if enabled
    if DataConfig.VAL_SELF_SIM:
        ensure_empty_dir(DataConfig.VAL_SIM_FOLDER, overwrite=DataConfig.OVERWRITE_DATASET)
        ensure_empty_dir(DataConfig.VAL_DIFF_FOLDER, overwrite=DataConfig.OVERWRITE_DATASET)
    ensure_empty_dir(DataConfig.VAL_DIFF_FOLDER, overwrite=DataConfig.OVERWRITE_DATASET) if DataConfig.GENERATE_VAL_DIFF else None

    # load TRAIN real images only
    print(" Loading TRAIN real images...")
    train_img_map: Dict[str, np.ndarray] = {}
    valid_fnames = []
    
    for _, row in df_opt_train.iterrows():
        fname = str(row['Filename'])
        path = os.path.join(DataConfig.TRAIN_REAL_FOLDER, fname)
        if not os.path.exists(path):
            continue

        row_phi = float(row.get('Phi', 0.0))
        row_x   = float(row.get('X_um', 0.0))
        train_img_map[fname] = preprocess_image(
            path,
            align_phi_deg=row_phi,
            align_x_shift_um=row_x,
            apply_align=bool(DataConfig.APPLY_REAL_ALIGN),
            apply_mask=bool(DataConfig.APPLY_CENTER_MASK),
            mask_band_px=int(DataConfig.CENTER_MASK_BAND_PX),
            mask_mode=str(DataConfig.CENTER_MASK_MODE),
        )
        valid_fnames.append(fname)


    if not train_img_map:
        raise ValueError("No valid TRAIN images found for generation (check TRAIN_REAL_FOLDER).")
    first_shape = list(train_img_map.values())[0].shape
    print(f" Loaded {len(train_img_map)} TRAIN real images. shape={first_shape}")

    # sample count
    if DataConfig.FAST_VERIFY:
        samples_per_row = int(DataConfig.FAST_VERIFY_MAX_SIMS_PER_SRC)
        print(" [FAST_VERIFY] Generating minimal samples.")
    else:
        samples_per_row = int(DataConfig.SAMPLES_PER_ROW)

    # build tasks: only TRAIN sources
    tasks = []
    df_valid = df_opt_train[df_opt_train['Filename'].isin(valid_fnames)].copy()
    for _, row in df_valid.iterrows():
        src_fname = row['Filename']
        src_rid = extract_real_id(src_fname)
        # LHS table for this source (deterministic per src_rid)
        p_center = np.array([0.0, row['R_mm'], row['B_um'], 0.0, row['A_um'], row['Gamma']], dtype=np.float64)
        lhs = lhs_perturb_params(p_center, n=samples_per_row, rate=DataConfig.PERTURB_RATE, seed=int(src_rid))
        for s in range(samples_per_row):
            tasks.append((row, src_rid, s, lhs[s], first_shape, DataConfig, train_img_map))

    print(f" Generating TRAIN sims & TRAIN diffs...")
    print(f"  - Sim tasks: {len(tasks)}")
    print(f"  - Train diffs per sim: {len(train_img_map)}")
    print(f"  - Expected train diff files: {len(tasks) * len(train_img_map)}")

    results_lists = Parallel(n_jobs=DataConfig.N_JOBS, verbose=5)(
        delayed(generate_worker_train_sim_and_train_diffs)(t) for t in tasks
    )

    # flatten and save train csv
    train_records = []
    for sublist in results_lists:
        train_records.extend(sublist)
    train_df = pd.DataFrame(train_records)

    train_df.sort_values(by=['tgt_fname', 'is_paired'], ascending=[True, False], inplace=True)

    train_cols = [
        'tgt_fname', 'tgt_rid', 'src_fname', 'src_rid',
        'phi', 'r', 'b', 'x', 'a', 'gamma', 'gain',
        'sim_path', 'diff_path', 'is_paired'
    ]
    train_df = train_df[train_cols]
    os.makedirs(os.path.dirname(DataConfig.TRAIN_LABELS), exist_ok=True)
    train_df.to_csv(DataConfig.TRAIN_LABELS, index=False)
    print(f" [DONE] Train CSV rows: {len(train_df)}")
    print(f"        Train labels saved to: {DataConfig.TRAIN_LABELS}")


    # --------------------------
    # Validation dataset generation (STRICT NO LEAKAGE)
    # --------------------------
    if DataConfig.FAST_VERIFY and bool(DataConfig.FAST_VERIFY_SKIP_VAL):
        print("\n [FAST_VERIFY] FAST_VERIFY_SKIP_VAL=True -> skip VAL sims/diffs generation.")
    else:
        val_files = list_bmp_files(DataConfig.VAL_REAL_FOLDER) if os.path.isdir(DataConfig.VAL_REAL_FOLDER) else []
        if DataConfig.VAL_SELF_SIM and val_files:
            print("\n Generating VAL sims & VAL diffs (sources=VAL, targets=VAL only)...")
        
            val_img_map: Dict[str, np.ndarray] = {}
            for f in val_files:
                fname = os.path.basename(f)
                # Load raw (legacy) first; then optionally align after reading rp table
                val_img_map[fname] = preprocess_image_basic(f)

            # Need val rp table (estimated by DE in Part 2 export)
            val_rp_csv = DataConfig.VAL_REAL_RP_FILE
            if not os.path.isfile(val_rp_csv):
                raise FileNotFoundError(f"Missing val rp table: {val_rp_csv}. Run Part 2 export first.")
            df_val_rp = pd.read_csv(val_rp_csv)

            # Support both schemas: (Filename/Phi/...) or (fname/phi/...)
            if 'Filename' in df_val_rp.columns:
                fn_col = 'Filename'
            elif 'fname' in df_val_rp.columns:
                fn_col = 'fname'
            else:
                raise KeyError(
                    f"val_real_rp.csv has no valid filename column. "
                    f"Expected 'Filename' or 'fname', but got: {list(df_val_rp.columns)}"
                )

            # Filter to actually-present files
            df_val_rp = df_val_rp[df_val_rp[fn_col].isin(val_img_map.keys())].copy()
            if df_val_rp.empty:
                raise RuntimeError("val_real_rp.csv has no rows matching VAL_REAL_FOLDER filenames.")

            # Apply alignment + center mask (in-place) on VAL real images using estimated Phi/X
            if bool(DataConfig.APPLY_REAL_ALIGN) or bool(DataConfig.APPLY_CENTER_MASK):
                if fn_col == 'Filename':
                    rp_map = {str(r['Filename']): (float(r.get('Phi', 0.0)), float(r.get('X_um', 0.0))) for _, r in df_val_rp.iterrows()}
                else:
                    rp_map = {str(r['fname']): (float(r.get('phi', 0.0)), float(r.get('x', 0.0))) for _, r in df_val_rp.iterrows()}

                for fname, img_np in list(val_img_map.items()):
                    phi_v, x_v = rp_map.get(fname, (0.0, 0.0))
                    if bool(DataConfig.APPLY_REAL_ALIGN):
                        img_np = apply_derotate_and_shift(
                            img_np,
                            phi_deg=phi_v,
                            x_shift_um=x_v,
                            pixel_size_um=DataConfig.PIXEL_SIZE_UM,
                            apply_mask=bool(DataConfig.APPLY_CENTER_MASK),
                            mask_band_px=int(DataConfig.CENTER_MASK_BAND_PX),
                            mask_mode=str(DataConfig.CENTER_MASK_MODE),
                        )
                    elif bool(DataConfig.APPLY_CENTER_MASK):
                        img_np = apply_center_band_mask(img_np, band_px=int(DataConfig.CENTER_MASK_BAND_PX), mode=str(DataConfig.CENTER_MASK_MODE))
                    val_img_map[fname] = img_np.astype(np.float64) 

            first_shape_val = next(iter(val_img_map.values())).shape

            val_tasks = []
            # In FAST_VERIFY, also reduce number of sources and sims per source
            max_sources = int(DataConfig.FAST_VERIFY_MAX_OPT_IMAGES) if DataConfig.FAST_VERIFY else None
            sources_iter = df_val_rp.iterrows()
            if max_sources is not None:
                sources_iter = list(df_val_rp.iterrows())[:max_sources]

            sims_per_src = int(DataConfig.FAST_VERIFY_MAX_SIMS_PER_SRC) if DataConfig.FAST_VERIFY else int(DataConfig.SAMPLES_PER_ROW)
            for _, row in sources_iter:
                src_fname = row[fn_col]
                src_rid = extract_real_id(src_fname)
                if fn_col == 'Filename':
                    p_center = np.array([
                        row['Phi'],
                        row['R_mm'],
                        row['B_um'],
                        row['X_um'],
                        row['A_um'],
                        row['Gamma'],
                    ], dtype=np.float64)
                else:
                    p_center = np.array([
                        row['phi'],
                        row['r'],
                        row['b'],
                        row['x'],
                        row['a'],
                        row['gamma'],
                    ], dtype=np.float64)
                lhs = lhs_perturb_params(p_center, n=sims_per_src, rate=DataConfig.PERTURB_RATE, seed=int(src_rid) + 777777)
                for s in range(sims_per_src):
                    # adapt row to expected keys in worker
                    row2 = {'Filename': src_fname}
                    val_tasks.append((row2, src_rid, s, lhs[s], first_shape_val, DataConfig, val_img_map))

            print(f"  - Val sim tasks: {len(val_tasks)}")
            print(f"  - Val diffs per sim (capped when FAST_VERIFY): {min(len(val_img_map), int(DataConfig.FAST_VERIFY_MAX_DIFF_TARGETS)) if DataConfig.FAST_VERIFY else len(val_img_map)}")

            val_results = Parallel(n_jobs=DataConfig.N_JOBS, backend='loky', verbose=5)(
                delayed(generate_worker_val_sim_and_val_diffs)(t) for t in val_tasks
            )

            val_records = []
            for sublist in val_results:
                val_records.extend(sublist)
            val_df = pd.DataFrame(val_records)
            val_df.sort_values(by=['tgt_fname', 'is_paired'], ascending=[True, False], inplace=True)

            val_cols = [
                'tgt_fname', 'tgt_rid', 'src_fname', 'src_rid',
                'phi', 'r', 'b', 'x', 'a', 'gamma', 'gain',
                'sim_path', 'diff_path', 'is_paired'
            ]
            val_df = val_df[val_cols]
            os.makedirs(os.path.dirname(DataConfig.VAL_LABELS), exist_ok=True)
            val_df.to_csv(DataConfig.VAL_LABELS, index=False)
            print(f" [DONE] Val CSV rows: {len(val_df)}")
            print(f"        Val labels saved to: {DataConfig.VAL_LABELS}")

        else:
            print("\n [SKIP] VAL_SELF_SIM is disabled or VAL folder empty; val sims/diffs not generated.")
            if not DataConfig.VAL_SELF_SIM:
                print("        Reason: VAL_SELF_SIM is False.")
            elif not (os.path.isdir(DataConfig.VAL_REAL_FOLDER) and val_files):
                print("        Reason: No val real images found in VAL_REAL_FOLDER.")

        # quick sanity checks for leakage
        if DataConfig.STRICT_NO_LEAKAGE:
            print("\n [SANITY] Leakage-safety checks:")
            assert train_df['sim_path'].str.startswith('train/sim').all()
            assert train_df['diff_path'].str.startswith('train/diff').all()
            if os.path.exists(DataConfig.VAL_LABELS):
                vdf = pd.read_csv(DataConfig.VAL_LABELS)
                # val labels may either be val self-sim (val/sim) or train-sim-based (train/sim)
                # We keep both valid depending on which path you enabled.
                assert vdf['diff_path'].str.startswith('val/diff').all()
            print("  - OK (paths are isolated as expected).")



    # ============================================================
    # Main Execution
    # ============================================================



    # -*- coding: utf-8 -*-
    """
    Revised Main Training Script (per reviewer)
    Key updates vs previous main.py:
    1) No jump hard-penalty in selection; weak term is continuous + relative normalized.
    2) Replace MMD with VICReg (invariance + variance/covariance anti-collapse). Selection uses VarPenalty.
    3) Dual-stem / adapter design:
       - sim branch: adapter_sim consumes sim spatial+fft (2ch)
       - real branch: adapter_real consumes real+diff spatial+fft (4ch)
       Shared trunk after the stems (not fully shared backbone).
    4) Pair consistency (semi) computed ONLY on is_paired samples.
    5) Group & TTA computed on embeddings, with cosine decay to keep gradients stable.
    6) Diameter is residual-param in log-space around coarse: log(d)=log(coarse)+Δ, Δ clipped.
    7) rp_refine enabled with strong regularization + small relative range (±5%).
    8) Physics loss computed after strong intensity normalization (instance norm) for both real and sim.
    9) Frequency-domain physics term uses low-frequency band-pass energy (low-pass) instead of full spectrum.

    Note: This file is intended to replace your existing main.py.

    vNext additions:
    - Physics forward runs in FP32 during validation + prints min/max/mean debug stats.
    - Progressive trunk unfreezing schedule (config switch).
    - VICReg computed on pooled *stem* features (adapter outputs), not trunk embeddings.
    - Validation logs pred_std to metrics.csv.
    """

    # =========================================================
    # Freeze / Unfreeze Helpers
    # =========================================================
def _set_requires_grad(params, flag: bool):
    for p in params:
        p.requires_grad = flag

def apply_progressive_freeze(model, epoch: int, config):
    """
    Freeze a portion of the shared trunk parameters.
    Early epochs: higher freeze ratio; later epochs: gradually unfreeze.
    Controlled by TrainConfig.ENABLE_PROGRESSIVE_UNFREEZE.

    We freeze trunk blocks in order: layer1 -> layer2 -> layer3 -> layer4.
    """
    if not getattr(config, "ENABLE_PROGRESSIVE_UNFREEZE", False):
        return None

    high = float(config.FREEZE_RATIO_HIGH)
    low  = float(config.FREEZE_RATIO_LOW)
    n_high = int(config.FREEZE_EPOCHS_HIGH)
    ramp = int(config.UNFREEZE_RAMP_EPOCHS)

    if epoch <= n_high:
        ratio = high
    elif ramp <= 0:
        ratio = low
    else:
        t = min(1.0, max(0.0, (epoch - n_high) / float(ramp)))
        ratio = high + (low - high) * t

    trunk_params = []
    for blk in [model.layer1, model.layer2, model.layer3, model.layer4]:
        trunk_params += list(blk.parameters())

    k = int(len(trunk_params) * ratio)
    _set_requires_grad(trunk_params[:k], False)
    _set_requires_grad(trunk_params[k:], True)

    # Keep stems + heads trainable
    _set_requires_grad(model.stem_real.parameters(), True)
    _set_requires_grad(model.stem_sim.parameters(), True)
    _set_requires_grad(model.param_mlp.parameters(), True)
    _set_requires_grad(model.fuse.parameters(), True)
    _set_requires_grad(model.delta_head.parameters(), True)
    _set_requires_grad(model.emb_head.parameters(), True)

    return ratio

# =========================================================
# 1. Utils
# =========================================================
def seed_everything(seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False

def clean_runtime_artifacts(config):
    force_clean = os.environ.get("FORCE_CLEAN_ARTIFACTS", "0").strip() == "1"
    if not force_clean:
        print(f"[Clean] Skip cleaning (if cleaning is needed, please set FORCE_CLEAN_ARTIFACTS=1)")
        return
    
    print(f"[Clean] Warning: Cleaning up old files in the experiment directory...")
    if not os.path.exists(config.BASE_MODEL_SAVE_DIR):
        return
    cache_abs = os.path.abspath(config.FFT_CACHE_DIR)
    removed_count = 0
    for root, dirs, files in os.walk(config.BASE_MODEL_SAVE_DIR):
        if cache_abs in os.path.abspath(root):
            continue
        for file in files:
            fp = os.path.join(root, file)
            # Only delete temporary files, not key result files
            if file.endswith(".csv") or file.endswith(".pth") or file.endswith(".png"):
                # Additional protection: Do not delete key files such as best_model.pth, best_ema_model.pth, metrics.csv
                if file in ["best_model.pth", "best_ema_model.pth", "metrics.csv", "config_snapshot.json", "DONE.txt"]:
                    continue
                try:
                    os.remove(fp)
                    removed_count += 1
                except:
                    pass
    print(f"[Clean] Completed, {removed_count} temporary files have been deleted")

def _extract_real_id(fname: str):
    m = re.search(r"\((\d+)\)", fname)
    if m:
        return int(m.group(1))
    m = re.search(r"(\d+)", fname)
    if m:
        return int(m.group(1))
    return -1

def cosine_decay(epoch, total_epochs):
    return 0.5 * (1 + math.cos(math.pi * epoch / total_epochs))

def get_dynamic_weight(epoch, start_val, end_val, total_epochs):
    cd = cosine_decay(epoch, total_epochs)
    return end_val + (start_val - end_val) * cd


def _warmup_scale(epoch: int, warmup_epochs: int, mode: str) -> float:
    if warmup_epochs is None or int(warmup_epochs) <= 0:
        return 1.0
    e = int(epoch)
    k = int(warmup_epochs)
    if e >= k:
        return 1.0
    if str(mode).lower() == "cosine":
        # cosine ramp 0->1 over k epochs
        return float(0.5 * (1 - math.cos(math.pi * e / max(1, k))))
    # default: linear ramp
    return float(e / max(1, k))


def get_loss_weight_scales(config, epoch: int, pair_only_epochs: int = 0, pair_loss_history: list = None, conditional_enable: bool = False) -> Dict[str, float]:
    """Return per-loss multiplicative scales (for warmup, reverse warmup, and gradual introduction).
    
    Args:
        epoch: Current epoch
        pair_only_epochs: Number of epochs for pair-only mode
        pair_loss_history: List of recent pair loss values (for conditional enable)
        conditional_enable: Whether to use conditional enable based on pair loss stability
    """
    scales = {"pair": 1.0, "weak": 1.0, "phys": 1.0, "vic": 1.0, "tta": 1.0, "group": 1.0}
    
    # Early pair-only mode: completely disable other losses
    if pair_only_epochs > 0 and epoch <= pair_only_epochs:
        # Keep pair at 1.0, set others to 0
        for t in ["weak", "phys", "vic", "tta", "group"]:
            scales[t] = 0.0
        return scales  # Early return: no other scaling applied in pair-only mode
    
    # Conditional enable: only enable other losses when pair loss is stable and below threshold
    if conditional_enable and pair_loss_history is not None and len(pair_loss_history) > 0:
        threshold = float(getattr(config, "LOSS_CONDITIONAL_PAIR_THRESHOLD", 0.030))
        stable_epochs = int(getattr(config, "LOSS_CONDITIONAL_STABLE_EPOCHS", 2))
        
        # Check if pair loss is stable and below threshold
        recent_pairs = pair_loss_history[-stable_epochs:] if len(pair_loss_history) >= stable_epochs else pair_loss_history
        avg_pair = sum(recent_pairs) / len(recent_pairs) if recent_pairs else float('inf')
        max_pair = max(recent_pairs) if recent_pairs else float('inf')
        min_pair = min(recent_pairs) if recent_pairs else float('inf')
        stable = (max_pair - min_pair) < 0.005  # variation < 0.005
        
        # Only enable if pair is stable and below threshold
        if not (avg_pair < threshold and stable and len(recent_pairs) >= stable_epochs):
            # Keep pair at 1.0, set others to 0
            for t in ["weak", "phys", "vic", "tta", "group"]:
                scales[t] = 0.0
            return scales
    
    # Gradual introduction: gradually ramp up other losses after pair-only period
    # This prevents sudden jumps when other losses are enabled
    if getattr(config, "LOSS_GRADUAL_INTRO_ENABLE", False):
        intro_epochs = int(getattr(config, "LOSS_GRADUAL_INTRO_EPOCHS", 5))
        intro_start = pair_only_epochs + 1  # Start after pair-only period
        intro_end = intro_start + intro_epochs - 1
        
        if intro_start <= epoch <= intro_end:
            targets = set([s.lower() for s in (getattr(config, "LOSS_GRADUAL_INTRO_TARGETS", []) or [])])
            if targets:
                # Calculate progress: 0 at intro_start, 1 at intro_end
                progress = (epoch - intro_start) / max(intro_epochs, 1)
                mode = getattr(config, "LOSS_GRADUAL_INTRO_MODE", "cosine").lower()
                
                if mode == "cosine":
                    # Cosine ramp: smooth start and end
                    s = 0.5 * (1 - math.cos(math.pi * progress))
                else:  # linear
                    s = progress
                
                for t in targets:
                    if t in scales:
                        scales[t] = float(s)
    
    # Forward warmup: ramp from 0 -> 1
    if getattr(config, "LOSS_WARMUP_ENABLE", False):
        k = int(getattr(config, "LOSS_WARMUP_EPOCHS", 0) or 0)
        if k > 0:
            targets = set([s.lower() for s in (getattr(config, "LOSS_WARMUP_TARGETS", []) or [])])
            if targets:
                s = _warmup_scale(epoch, k, getattr(config, "LOSS_WARMUP_MODE", "linear"))
                for t in targets:
                    if t in scales:
                        scales[t] = float(s)
    
    # Reverse warmup: ramp from 1 -> min_scale (for bad seeds: reduce phys/weak early, let pair dominate)
    if getattr(config, "LOSS_REVERSE_WARMUP_ENABLE", False):
        k = int(getattr(config, "LOSS_REVERSE_WARMUP_EPOCHS", 0) or 0)
        if k > 0:
            targets = set([s.lower() for s in (getattr(config, "LOSS_REVERSE_WARMUP_TARGETS", []) or [])])
            min_scale = float(getattr(config, "LOSS_REVERSE_WARMUP_MIN_SCALE", 0.1))
            if targets:
                # Reverse: start at 1.0, end at min_scale over k epochs
                # s_reverse = 1.0 - (1.0 - min_scale) * warmup_progress
                s_forward = _warmup_scale(epoch, k, getattr(config, "LOSS_REVERSE_WARMUP_MODE", "linear"))
                s_reverse = 1.0 - (1.0 - min_scale) * s_forward
                for t in targets:
                    if t in scales:
                        # Apply reverse warmup (but don't override forward warmup if both apply)
                        # If forward warmup is active, it takes precedence
                        if not (getattr(config, "LOSS_WARMUP_ENABLE", False) and t in set([s.lower() for s in (getattr(config, "LOSS_WARMUP_TARGETS", []) or [])])):
                            # Also don't override gradual introduction if active
                            if not (getattr(config, "LOSS_GRADUAL_INTRO_ENABLE", False) and t in set([s.lower() for s in (getattr(config, "LOSS_GRADUAL_INTRO_TARGETS", []) or [])])):
                                scales[t] = float(s_reverse)
    
    return scales


def _dot_grads(gi: List[Optional[torch.Tensor]], gj: List[Optional[torch.Tensor]]) -> torch.Tensor:
    dot = None
    for a, b in zip(gi, gj):
        if a is None or b is None:
            continue
        v = (a * b).sum()
        dot = v if dot is None else (dot + v)
    if dot is None:
        return torch.tensor(0.0, device='cuda')
    return dot


def _norm2_grads(g: List[Optional[torch.Tensor]]) -> torch.Tensor:
    n2 = None
    for a in g:
        if a is None:
            continue
        v = (a * a).sum()
        n2 = v if n2 is None else (n2 + v)
    if n2 is None:
        return torch.tensor(0.0, device='cuda')
    return n2


def pcgrad_backward(task_losses: List[torch.Tensor], params: List[torch.nn.Parameter]) -> None:
    """PCGrad (Yu et al.) for a list of task losses.

    This runs a manual backward and writes .grad into params.
    """
    # Skip non-differentiable losses (can happen when a term is effectively constant
    # for a given batch, e.g. no paired samples -> L_pair is a constant 0 tensor).
    task_losses = [L for L in task_losses if isinstance(L, torch.Tensor) and bool(getattr(L, "requires_grad", False))]
    if len(task_losses) == 0:
        return

    grads: List[List[Optional[torch.Tensor]]] = []
    for L in task_losses:
        g = torch.autograd.grad(L, params, retain_graph=True, allow_unused=True)
        grads.append([gi if gi is None else gi.detach() for gi in g])

    # project conflicting gradients
    for i in range(len(grads)):
        gi = grads[i]
        for j in range(len(grads)):
            if i == j:
                continue
            gj = grads[j]
            dot = _dot_grads(gi, gj)
            if float(dot.item()) < 0.0:
                n2 = _norm2_grads(gj)
                if float(n2.item()) <= 1e-12:
                    continue
                coeff = dot / (n2 + 1e-12)
                # gi = gi - coeff * gj
                for k in range(len(gi)):
                    if gi[k] is None or gj[k] is None:
                        continue
                    gi[k] = gi[k] - coeff * gj[k]
        grads[i] = gi

    # accumulate into params.grad
    for p in params:
        p.grad = None
    for k, p in enumerate(params):
        acc = None
        for i in range(len(grads)):
            gk = grads[i][k]
            if gk is None:
                continue
            acc = gk if acc is None else (acc + gk)
        if acc is not None:
            p.grad = acc


class GradNormBalancer:
    """GradNorm (Chen et al.) task-weight balancer for a fixed set of task losses."""

    def __init__(self, task_names: List[str], *, alpha: float, lr: float, device: str = 'cuda'):
        self.task_names = list(task_names)
        self.alpha = float(alpha)
        self.weights = torch.nn.Parameter(torch.ones(len(self.task_names), device=device))
        self.opt_w = torch.optim.Adam([self.weights], lr=float(lr))
        self._L0 = None  # type: Optional[torch.Tensor]

    def step(self, task_losses: Dict[str, torch.Tensor], shared_params: List[torch.nn.Parameter]) -> Dict[str, float]:
        names = self.task_names
        L = torch.stack([task_losses[n].float() for n in names])
        if self._L0 is None:
            self._L0 = L.detach().clamp(min=1e-12)

        # Compute gradient norms G_i = || d (w_i * L_i) / dW ||
        G = []
        self.opt_w.zero_grad(set_to_none=True)
        for i, n in enumerate(names):
            Li = task_losses[n].float()
            wi = self.weights[i].clamp(min=0.0)
            if not bool(getattr(Li, "requires_grad", False)):
                G.append(torch.tensor(0.0, device=self.weights.device))
                continue
            g = torch.autograd.grad(wi * Li, shared_params, retain_graph=True, allow_unused=True, create_graph=True)
            n2 = None
            for gg in g:
                if gg is None:
                    continue
                v = (gg * gg).sum()
                n2 = v if n2 is None else (n2 + v)
            Gi = torch.sqrt((n2 if n2 is not None else torch.tensor(0.0, device=self.weights.device)) + 1e-12)
            G.append(Gi)
        G = torch.stack(G)

        # Relative inverse training rates
        Li_ratio = (L.detach() / self._L0).clamp(min=1e-12)
        r = Li_ratio / Li_ratio.mean()
        G_avg = G.detach().mean()
        target = (G_avg * (r ** self.alpha)).detach()

        gradnorm_loss = (G - target).abs().sum()
        gradnorm_loss.backward(retain_graph=True)
        self.opt_w.step()

        # Renormalize weights to keep sum = n_tasks, and keep non-negative
        with torch.no_grad():
            w = self.weights.data.clamp(min=0.0)
            s = float(w.sum().item())
            if s > 1e-12:
                w = w * (len(names) / s)
            self.weights.data.copy_(w)

        return {f"gradnorm_w_{n}": float(self.weights[i].item()) for i, n in enumerate(names)}



# =========================================================
# RP table integrity + parameter normalization
# =========================================================
# Physical range sanity (keep aligned with data.py export_real_rp_tables)
RP_RANGE = {
    "phi": (-6.0, 1.0),
    "r": (-20.0, 0.0),
    "b": (700.0, 2000.0),
    "x": (-500.0, 500.0),
    "a": (50.0, 200.0),
    "gamma": (0.05, 2.0),
    "gain": (0.0, 1e9),
}

def _assert_in_range(name: str, v: float, lo: float, hi: float, ctx: str = ""):
    if not (lo <= float(v) <= hi):
        raise ValueError(f"[RP_RANGE] {name}={v} out of [{lo},{hi}] {ctx}".strip())

def validate_params5_phys(p5: torch.Tensor, ctx: str = ""):
    """p5: (B,5) in physical units => [phi,r,b,x,gamma]."""
    if p5.ndim != 2 or p5.size(1) != 5:
        raise ValueError(f"[RP_RANGE] expect (B,5) but got {tuple(p5.shape)} {ctx}".strip())
    phi = p5[:, 0]; r = p5[:, 1]; b = p5[:, 2]; x = p5[:, 3]; gamma = p5[:, 4]
    _assert_in_range("phi", float(phi.min().item()), *RP_RANGE["phi"], ctx + " (min)")
    _assert_in_range("phi", float(phi.max().item()), *RP_RANGE["phi"], ctx + " (max)")
    _assert_in_range("r", float(r.min().item()), *RP_RANGE["r"], ctx + " (min)")
    _assert_in_range("r", float(r.max().item()), *RP_RANGE["r"], ctx + " (max)")
    _assert_in_range("b", float(b.min().item()), *RP_RANGE["b"], ctx + " (min)")
    _assert_in_range("b", float(b.max().item()), *RP_RANGE["b"], ctx + " (max)")
    _assert_in_range("x", float(x.min().item()), *RP_RANGE["x"], ctx + " (min)")
    _assert_in_range("x", float(x.max().item()), *RP_RANGE["x"], ctx + " (max)")
    _assert_in_range("gamma", float(gamma.min().item()), *RP_RANGE["gamma"], ctx + " (min)")
    _assert_in_range("gamma", float(gamma.max().item()), *RP_RANGE["gamma"], ctx + " (max)")

def _is_params5_norm_pm1(p5: torch.Tensor, eps: float = 1e-3) -> bool:
    """Heuristic: treat as normalized if all entries are within [-1,1] (with small tolerance)."""
    try:
        if p5.ndim != 2 or p5.size(1) != 5:
            return False
        mn = float(p5.min().item())
        mx = float(p5.max().item())
        return (mn >= -1.0 - eps) and (mx <= 1.0 + eps)
    except Exception:
        return False


def normalize_params5(p5_maybe_phys_or_norm: torch.Tensor) -> torch.Tensor:
    """
    NN-side parameter normalization for [phi,r,b,x,gamma].

    Requirement:
      - If already normalized to [-1,1], return as-is.
      - Else treat as PHYSICAL units and normalize to [-1,1] using RP_RANGE bounds:
            norm = 2*(x - lo)/(hi - lo) - 1
    NOTE:
      - This function is for NN consumption only (conditioning input).
      - Physics layer must always receive PHYSICAL units.
    """
    if p5_maybe_phys_or_norm.ndim != 2 or p5_maybe_phys_or_norm.size(1) != 5:
        raise ValueError(f"[normalize_params5] expect (B,5) but got {tuple(p5_maybe_phys_or_norm.shape)}")
    if _is_params5_norm_pm1(p5_maybe_phys_or_norm):
        return p5_maybe_phys_or_norm

    # treat as physical
    p5_phys = p5_maybe_phys_or_norm
    # validate as physical for early failure visibility (decision is based on normalization check above)
    validate_params5_phys(p5_phys, ctx="normalize_params5(input_phys)")

    bounds = torch.tensor([
        RP_RANGE["phi"],
        RP_RANGE["r"],
        RP_RANGE["b"],
        RP_RANGE["x"],
        RP_RANGE["gamma"],
    ], device=p5_phys.device, dtype=p5_phys.dtype)  # (5,2)
    lo = bounds[:, 0].view(1, 5)
    hi = bounds[:, 1].view(1, 5)
    den = (hi - lo).clamp_min(1e-12)
    # map to [-1,1]
    return 2.0 * (p5_phys - lo) / den - 1.0


def denormalize_params5(norm_p5_pm1: torch.Tensor) -> torch.Tensor:
    """Inverse of normalize_params5 for [-1,1] normalized params -> physical units."""
    if norm_p5_pm1.ndim != 2 or norm_p5_pm1.size(1) != 5:
        raise ValueError(f"[denormalize_params5] expect (B,5) but got {tuple(norm_p5_pm1.shape)}")
    bounds = torch.tensor([
        RP_RANGE["phi"],
        RP_RANGE["r"],
        RP_RANGE["b"],
        RP_RANGE["x"],
        RP_RANGE["gamma"],
    ], device=norm_p5_pm1.device, dtype=norm_p5_pm1.dtype)
    lo = bounds[:, 0].view(1, 5)
    hi = bounds[:, 1].view(1, 5)
    den = (hi - lo).clamp_min(1e-12)
    # invert: x = lo + (norm+1)/2 * (hi-lo)
    return lo + (norm_p5_pm1 + 1.0) * 0.5 * den


def ensure_params5_phys_for_physics(p5_maybe_phys_or_norm: torch.Tensor, ctx: str = "") -> torch.Tensor:
    """
    Physics-side guard:
      - If already within physical RP_RANGE => OK.
      - Else if normalized ([-1,1]) => denormalize to physical and re-validate.
      - Else => error (unknown scale).
    This is applied uniformly for BOTH rp/sp before any Fraunhofer-related usage.
    """
    if p5_maybe_phys_or_norm.ndim != 2 or p5_maybe_phys_or_norm.size(1) != 5:
        raise ValueError(f"[ensure_params5_phys_for_physics] expect (B,5) but got {tuple(p5_maybe_phys_or_norm.shape)} {ctx}".strip())

    try:
        validate_params5_phys(p5_maybe_phys_or_norm, ctx=(ctx + " (phys)").strip())
        return p5_maybe_phys_or_norm
    except Exception:
        pass

    if _is_params5_norm_pm1(p5_maybe_phys_or_norm):
        p_phys = denormalize_params5(p5_maybe_phys_or_norm)
        validate_params5_phys(p_phys, ctx=(ctx + " (denorm_pm1)").strip())
        return p_phys

    raise ValueError(
        f"[ensure_params5_phys_for_physics] Params are neither within physical RP_RANGE nor normalized to [-1,1]. {ctx}".strip()
    )

ensure_params5_phys_for_physics_compat = ensure_params5_phys_for_physics

def tta_noise_only(tensor, n, noise_level=0.02):
    outs = [tensor]
    for _ in range(n - 1):
        noise = torch.randn_like(tensor) * noise_level
        outs.append(tensor + noise)
    return outs[:n]

def ema_init(model):
    ema_state = {}
    with torch.no_grad():
        for k, v in model.state_dict().items():
            ema_state[k] = v.detach().clone()
    return ema_state

def ema_update(model, ema_state, decay):
    with torch.no_grad():
        for k, v in model.state_dict().items():
            if k not in ema_state:
                ema_state[k] = v.detach().clone()
            else:
                if v.dtype in [torch.float16, torch.float32, torch.float64]:
                    ema_state[k].mul_(decay).add_(v.detach(), alpha=1.0-decay)
                else:
                    ema_state[k].copy_(v.detach())


# =========================================================
# 2. Advanced Physics Loss (normalized + low-freq band)
# =========================================================
class AdvancedPhysicsLoss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.register_buffer('sobel_x', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1,1,3,3))
        self.register_buffer('sobel_y', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32).view(1,1,3,3))
        self._fft_mask_cache = {}

    @staticmethod
    def instance_norm(x):
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        return (x - mean) / std

    def _radial_mask(self, H, W, low, high, device):
        key = (H, W, float(low), float(high), str(device))
        if key in self._fft_mask_cache:
            return self._fft_mask_cache[key]
        yy = torch.linspace(-0.5, 0.5, H, device=device)
        xx = torch.linspace(-0.5, 0.5, W, device=device)
        Y, X = torch.meshgrid(yy, xx, indexing='ij')
        R = torch.sqrt(X**2 + Y**2)
        mask = (R >= low) & (R <= high)
        mask = mask.float()[None, None, :, :]  # (1,1,H,W)
        self._fft_mask_cache[key] = mask
        return mask

    def gradient_loss(self, pred, target):
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        pred_g = torch.sqrt(pred_gx**2 + pred_gy**2 + 1e-8)
        target_g = torch.sqrt(target_gx**2 + target_gy**2 + 1e-8)
        return F.l1_loss(self.instance_norm(pred_g), self.instance_norm(target_g))

    def fft_band_loss(self, pred, target):
        # pred/target are already intensity-normalized spatial images (B,1,H,W)
        B, _, H, W = pred.shape
        fft_pred = torch.fft.fft2(pred)
        fft_target = torch.fft.fft2(target)
        fft_pred = torch.fft.fftshift(fft_pred, dim=(-2, -1))
        fft_target = torch.fft.fftshift(fft_target, dim=(-2, -1))
        mag_pred = torch.log1p(torch.abs(fft_pred))
        mag_target = torch.log1p(torch.abs(fft_target))

        mask = self._radial_mask(H, W, self.config.FFT_LOW, self.config.FFT_HIGH, pred.device)
        mag_pred = mag_pred * mask
        mag_target = mag_target * mask
        return F.mse_loss(self.instance_norm(mag_pred), self.instance_norm(mag_target))

    def correlation_loss(self, pred, target):
        # correlation on normalized images to reduce gain/gamma coupling
        x = pred.view(pred.size(0), -1)
        y = target.view(target.size(0), -1)
        x = x - x.mean(dim=1, keepdim=True)
        y = y - y.mean(dim=1, keepdim=True)
        r_num = torch.sum(x * y, dim=1)
        r_den = torch.sqrt(torch.sum(x**2, dim=1) * torch.sum(y**2, dim=1) + 1e-8)
        r = r_num / r_den
        return 1.0 - r.mean()

    def forward(self, pred_img, target_img):
        # Strong intensity normalization BEFORE all physics terms
        pred_n = self.instance_norm(pred_img)
        tgt_n  = self.instance_norm(target_img)

        l_grad = self.gradient_loss(pred_n, tgt_n)
        l_fft  = self.fft_band_loss(pred_n, tgt_n)
        l_corr = self.correlation_loss(pred_n, tgt_n)

        total = (self.config.W_PHYS_CORR * l_corr +
                 self.config.W_PHYS_FFT  * l_fft +
                 self.config.W_PHYS_GRAD * l_grad)
        return total, l_corr, l_fft, l_grad


# =========================================================
# =========================================================
# 3. FFT Precomputer (unchanged)
# =========================================================

class FFTPrecomputer:
    def __init__(self, config):
        self.config = config

        # (Optional) for REAL BMPs, apply inverse (derotate, recenter) using estimated Phi/X_um
        self.real_rp_map: Dict[str, Tuple[float, float]] = {}
        try:
            rp_files = []
            if hasattr(config, 'TRAIN_REAL_RP_FILE'):
                rp_files.append(getattr(config, 'TRAIN_REAL_RP_FILE'))
            if hasattr(config, 'VAL_REAL_RP_FILE'):
                rp_files.append(getattr(config, 'VAL_REAL_RP_FILE'))
            for rp in rp_files:
                if rp and os.path.isfile(rp):
                    df = pd.read_csv(rp)
                    key = 'Filename' if 'Filename' in df.columns else ('fname' if 'fname' in df.columns else None)
                    if key is None:
                        continue
                    col_phi = 'Phi' if 'Phi' in df.columns else ('phi' if 'phi' in df.columns else None)
                    col_x = 'X_um' if 'X_um' in df.columns else ('x_um' if 'x_um' in df.columns else ('x' if 'x' in df.columns else None))
                    if col_phi is None:
                        continue
                    for _, r in df.iterrows():
                        fn = str(r[key])
                        ph = float(r[col_phi])
                        xv = float(r[col_x]) if col_x else 0.0
                        self.real_rp_map[fn] = (ph, xv)
        except Exception:
            self.real_rp_map = {}

        self.transform = transforms.Compose([
            transforms.Lambda(lambda img: self.dynamic_crop(img, config.CROP_PIXELS)),
            transforms.Resize((config.RESIZE, config.RESIZE)),
            transforms.ToTensor()
        ])

    @staticmethod
    def _is_valid_pt(path: str) -> bool:
        """Best-effort corruption check for cached .pt tensors."""
        try:
            if (not os.path.exists(path)) or (os.path.getsize(path) < 64):
                return False
            _ = torch.load(path, map_location="cpu")
            return True
        except Exception:
            return False

    @staticmethod
    def _atomic_torch_save(tensor, dst: str):
        """Atomic save to avoid half-written cache files on interruptions."""
        tmp = dst + ".tmp"
        torch.save(tensor, tmp)
        os.replace(tmp, dst)


    def dynamic_crop(self, im, pixels):
        if pixels <= 0:
            return im
        return im.crop((pixels, pixels, im.width - pixels, im.height - pixels))


    def process(self, img_path):
        try:
            ext = os.path.splitext(img_path)[1].lower()

            # Align REAL BMPs after Part2 estimation (optional)
            if ext == ".bmp" and bool(getattr(self.config, "APPLY_REAL_ALIGN", False)):
                fname = os.path.basename(img_path)
                phi, x_um = self.real_rp_map.get(fname, (0.0, 0.0))
                img_np = preprocess_image(
                    img_path,
                    align_phi_deg=phi,
                    align_x_shift_um=x_um,
                    apply_align=True,
                    apply_mask=bool(getattr(self.config, "APPLY_CENTER_MASK", False)),
                    mask_band_px=int(getattr(self.config, "CENTER_MASK_BAND_PX", 50)),
                    mask_mode=str(getattr(self.config, "CENTER_MASK_MODE", "keep")),
                )
                img_u8 = np.clip(img_np, 0, 255).astype(np.uint8)
                img = Image.fromarray(img_u8, mode="L")
            else:
                img = Image.open(img_path).convert("L")

            sp = self.transform(img)
            fft = torch.fft.fft2(sp)
            mag = torch.log1p(torch.abs(fft))
            mag = (mag - mag.min()) / (mag.max() - mag.min() + 1e-8)
            return torch.cat([sp, mag], dim=0).half()
        except Exception:
            return None

    def run(self):
        res_dir = os.path.join(self.config.FFT_CACHE_DIR, f"size_{self.config.RESIZE}")
        real_dir = os.path.join(res_dir, getattr(self.config, 'FFT_REAL_SUBDIR', 'real'))
        sim_dir = os.path.join(res_dir, getattr(self.config, 'FFT_SIMDIFF_SUBDIR', 'sim_diff'))

        # Resume-friendly: do NOT early-return just because some cache exists.
        # Scan and (re)generate missing/corrupted tensors.
        if os.path.exists(real_dir) and os.path.exists(sim_dir):
            print(f"[Precompute] Cache dir exists at {res_dir} (resume mode: will fill missing items).")

        os.makedirs(real_dir, exist_ok=True)
        os.makedirs(sim_dir, exist_ok=True)

        print(f"[Precompute] Generating to: {res_dir}")

        real_files = []
        for folder in [self.config.TRAIN_REAL_FOLDER, self.config.VAL_REAL_FOLDER]:
            real_files += glob.glob(os.path.join(folder, "*.BMP"))

        for f in tqdm(sorted(real_files), desc="Caching Real(BMP)", ncols=80):
            name = os.path.splitext(os.path.basename(f))[0] + ".pt"
            dst = os.path.join(real_dir, name)
            if (not os.path.exists(dst)) or (not self._is_valid_pt(dst)):
                t = self.process(f)
                if t is not None:
                    self._atomic_torch_save(t, dst)

        png_files = glob.glob(os.path.join(self.config.DATASET_ROOT, "**", "*.png"), recursive=True)
        for f in tqdm(sorted(png_files), desc="Caching Sim/Diff(PNG)", ncols=80):
            fname = os.path.basename(f)
            prefixes = getattr(self.config, 'FFT_CACHE_PNG_PREFIXES', ("sim_", "diff_", "simv_", "diffv_"))
            if not fname.startswith(prefixes):
                continue
            name = os.path.splitext(fname)[0] + ".pt"
            dst = os.path.join(sim_dir, name)
            if (not os.path.exists(dst)) or (not self._is_valid_pt(dst)):
                t = self.process(f)
                if t is not None:
                    self._atomic_torch_save(t, dst)


# =========================================================
# 5. Dataset (add is_paired)
# =========================================================
class DiffSimRealDataset(Dataset):
    def __init__(self, dataset_root, real_root, labels_csv, init_diameter_txt,
                 real_rp_csv=None,
                 crop_pixels=3, resize=224, coarse_noise_std=0.003, is_train=True, config=None):
        self.dataset_root = dataset_root.replace("\\", "/")
        self.real_root = real_root.replace("\\", "/")
        self.is_train = is_train
        self.config = config

        self.cache_root = os.path.join(config.FFT_CACHE_DIR, f"size_{config.RESIZE}")
        self.real_cache = os.path.join(self.cache_root, getattr(config, 'FFT_REAL_SUBDIR', 'real'))
        self.sim_cache = os.path.join(self.cache_root, getattr(config, 'FFT_SIMDIFF_SUBDIR', 'sim_diff'))

        self.df = pd.read_csv(labels_csv)
        self.df['tgt_fname'] = self.df['tgt_fname'].astype(str)

        self.real_rp_csv = real_rp_csv
        self.real_rp_map = self._load_real_rp_map(real_rp_csv) if real_rp_csv is not None else {}

        self.coarse_noise_std = coarse_noise_std
        self._coarse_val = self._load_coarse(init_diameter_txt)

        self.real_id_to_global_idx = {}
        self.samples, self.grouped_by_real_indices, self.unique_tgt_fnames = self._build_table()

    def _load_coarse(self, path):
        if os.path.exists(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    v = float(f.read().strip())
                return v
            except:
                pass
        return 102.555

    def _build_table(self):
        df = self.df

        # Ablation: Data Quantity - sample a subset of sim data for each real image in the training set
        if self.is_train and self.config.SIM_SAMPLE_COUNT > 0:
            print(f"[Dataset] Training mode: Sampling {self.config.SIM_SAMPLE_COUNT} sim samples per real image.")
            
            # Use a reproducible random state based on the global seed
            # Handle groups smaller than the sample count gracefully
            df = df.groupby('tgt_fname', group_keys=False).apply(
                lambda x: x.sample(
                    n=min(len(x), self.config.SIM_SAMPLE_COUNT),
                    random_state=self.config.SEED
                )
            )

        records = []
        tmp_group = defaultdict(list)
        unique_tgts = sorted(df['tgt_fname'].unique().tolist())

        for idx, row in df.iterrows():
            tgt_fname = str(row['tgt_fname'])
            sim_rel = str(row['sim_path'])
            diff_rel = str(row['diff_path'])

            real_pt_name = os.path.splitext(os.path.basename(tgt_fname))[0] + ".pt"
            sim_pt_name  = os.path.splitext(os.path.basename(sim_rel))[0] + ".pt"
            diff_pt_name = os.path.splitext(os.path.basename(diff_rel))[0] + ".pt"

            real_pt_path = os.path.join(self.real_cache, real_pt_name)
            sim_pt_path  = os.path.join(self.sim_cache, sim_pt_name)
            diff_pt_path = os.path.join(self.sim_cache, diff_pt_name)

            if os.path.exists(real_pt_path) and os.path.exists(sim_pt_path) and os.path.exists(diff_pt_path):
                rid = _extract_real_id(tgt_fname)
                r = {
                    "real_pt": real_pt_path,
                    "sim_pt": sim_pt_path,
                    "diff_pt": diff_pt_path,
                    "real_fname": tgt_fname,
                    "row_idx": idx,
                    "real_id": rid
                }
                tmp_group[rid].append(r)

        grouped = defaultdict(list)
        curr = 0
        for rid in sorted(tmp_group.keys()):
            for r in tmp_group[rid]:
                records.append(r)
                grouped[rid].append(curr)
                curr += 1

        return records, grouped, unique_tgts

    def _get_real_coarse(self):
        v = float(self._coarse_val)
        if self.is_train:
            v = v + np.random.randn() * self.coarse_noise_std * v
        return torch.tensor(v, dtype=torch.float32)

    def _load_real_rp_map(self, csv_path: str):
        if (csv_path is None) or (not os.path.exists(csv_path)):
            raise FileNotFoundError(f"Missing real rp csv: {csv_path}")
        dfp = pd.read_csv(csv_path)

        col_map = {}
        for c in dfp.columns:
            lc = c.lower()
            if lc in ["fname", "filename"]:
                col_map[c] = "fname"
            elif lc in ["phi"]:
                col_map[c] = "phi"
            elif lc in ["r", "r_mm"]:
                col_map[c] = "r"
            elif lc in ["b", "b_um"]:
                col_map[c] = "b"
            elif lc in ["x", "x_um"]:
                col_map[c] = "x"
            elif lc in ["gamma"]:
                col_map[c] = "gamma"
        dfp = dfp.rename(columns=col_map)

        required = ["fname", "phi", "r", "b", "x", "gamma"]
        for rcol in required:
            if rcol not in dfp.columns:
                raise KeyError(f"real rp csv missing column '{rcol}': {csv_path}")

        mp = {}
        for _, row in dfp.iterrows():
            fn = str(row["fname"])
            mp[fn] = row
        return mp

    def _normalize_real_rp(self, rp_row):
        """Return PHYSICAL units [phi,r,b,x,gamma]. Normalization is done in main() before model/physics."""
        return torch.tensor([
            0.0,
            float(rp_row["r"]),
            float(rp_row["b"]),
            0.0,
            float(rp_row["gamma"]),
        ], dtype=torch.float32)

    def _get_real_rp(self, tgt_fname: str):
        if not self.real_rp_map:
            raise RuntimeError("real rp map not loaded; please pass real_rp_csv to dataset.")
        key = str(tgt_fname)
        if key not in self.real_rp_map:
            base = os.path.basename(key)
            if base in self.real_rp_map:
                key = base
            else:
                raise KeyError(f"Real rp not found for '{tgt_fname}'. Check rp csv: {self.real_rp_csv}")
        return self._normalize_real_rp(self.real_rp_map[key])

    def _get_sim_params(self, idx):
        """Return SIM params in PHYSICAL units [phi,r,b,x,gamma] (no legacy hard-coded normalization)."""
        row = self.df.iloc[idx]
        p_phys = torch.tensor([
            0.0,
            float(row['r']),
            float(row['b']),
            0.0,
            float(row['gamma']),
        ], dtype=torch.float32)
        a = torch.tensor(float(row['a']), dtype=torch.float32)
        is_paired = torch.tensor(float(row.get('is_paired', 0.0)), dtype=torch.float32)
        return p_phys, a, is_paired


    def _load_pt(self, path):
        return torch.load(path).float()

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        r = self.samples[idx]
        real_pt = self._load_pt(r["real_pt"])  # (2,H,W) [spatial, fftmag]
        sim_pt  = self._load_pt(r["sim_pt"])   # (2,H,W)
        diff_pt = self._load_pt(r["diff_pt"])  # (2,H,W)

        # -------------------------------
        # input channel switches
        # -------------------------------
        if not self.config.USE_FFT_INPUT:
            # Channel 1 is FFT ([spatial, fft])
            real_pt = real_pt.clone()
            sim_pt  = sim_pt.clone()
            diff_pt = diff_pt.clone()

            real_pt[1:2].zero_()
            sim_pt[1:2].zero_()
            diff_pt[1:2].zero_()

        if not self.config.USE_DIFF_INPUT:
            diff_pt = diff_pt.clone()
            diff_pt.zero_()

        # Keep compatibility: full_input = [real(2), sim(2), diff(2)] for debug/physics targeting
        full_input = torch.cat([real_pt, sim_pt, diff_pt], dim=0)

        sim_p, target_a, is_paired = self._get_sim_params(r["row_idx"])
        real_p_initial = self._get_real_rp(r["real_fname"])
        coarse = self._get_real_coarse()

        global_idx = self.real_id_to_global_idx.get(r["real_id"], 0)
        return r["real_fname"], full_input, sim_pt, real_p_initial, sim_p, coarse, target_a, is_paired, global_idx


class GroupedBatchSampler(Sampler):
    def __init__(self, groups, batch_size, shuffle=True):
        self.groups = groups
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.group_keys = list(groups.keys())

    def __iter__(self):
        keys = self.group_keys[:]
        if self.shuffle:
            random.shuffle(keys)
        batch = []
        for k in keys:
            idxs = self.groups[k][:]
            if self.shuffle:
                random.shuffle(idxs)
            for i in idxs:
                batch.append(i)
                if len(batch) == self.batch_size:
                    yield batch
                    batch = []
        if len(batch) > 0:
            yield batch

    def __len__(self):
        total = sum(len(v) for v in self.groups.values())
        return (total + self.batch_size - 1) // self.batch_size


# =========================================================
# 6. Physics Layer
# =========================================================
class FraunhoferPhysicsLayer(nn.Module):
    def __init__(self, orig_h, orig_w, crop_pixels, resize_target, wavelength_um, foclength_um, p_size_um):
        super().__init__()
        self.orig_h, self.orig_w = orig_h, orig_w
        self.crop_pixels = crop_pixels
        self.resize_target = (resize_target, resize_target)
        self.wavelength = wavelength_um
        self.foclength = foclength_um
        self.p_size = p_size_um

        x_len = self.orig_w * self.p_size
        y_len = self.orig_h * self.p_size
        x = torch.linspace(-x_len/2, x_len/2, self.orig_w)
        y = torch.linspace(-y_len/2, y_len/2, self.orig_h)
        Y_grid, X_grid = torch.meshgrid(y, x, indexing='ij')
        self.register_buffer('X_orig', X_grid)
        self.register_buffer('Y_orig', Y_grid)

    def forward(self, pred_diameter_um, norm_params, target_real_img=None):
        # NOTE: FraunhoferPhysicsLayer expects PHYSICAL units here (no normalization inside).
        # norm_params naming is kept for backward compatibility with older call sites.
        phi_deg = norm_params[:, 0:1]
        r_offset_mm = norm_params[:, 1:2]
        b_um = norm_params[:, 2:3]
        x_shift_um = norm_params[:, 3:4]
        gamma = norm_params[:, 4:5]

        d_um = pred_diameter_um
        r_offset_um = r_offset_mm * 1000.0
        B = pred_diameter_um.shape[0]

        angle_rad = torch.atan(r_offset_um / self.foclength)
        max_dist = torch.tan(angle_rad) * self.foclength
        y_center_um = max_dist + (self.orig_h * self.p_size / 2.0)

        theta_rad = torch.deg2rad(phi_deg).view(B, 1, 1)
        cos_t, sin_t = torch.cos(theta_rad), torch.sin(theta_rad)

        Xs = self.X_orig
        Ys = self.Y_orig

        Xr = cos_t * Xs - sin_t * Ys + x_shift_um.view(B, 1, 1)
        Yr = sin_t * Xs + cos_t * Ys + y_center_um.view(B, 1, 1)

        f_sq = self.foclength**2
        dist_x = torch.sqrt(Xr**2 + f_sq)
        dist_y = torch.sqrt(Yr**2 + f_sq)
        sin_tx, sin_ty = Xr / dist_x, Yr / dist_y

        arg_x = (b_um.view(B, 1, 1) * sin_tx) / self.wavelength
        arg_y = (d_um.view(B, 1, 1) * sin_ty) / self.wavelength

        field = (torch.sinc(arg_x))**2 * (torch.sinc(arg_y))**2
        field_nonlin = torch.pow(field + 1e-8, gamma.view(B, 1, 1))

        img_flipped = torch.flip(field_nonlin.unsqueeze(1), dims=[2])
        c = self.crop_pixels
        cropped = img_flipped[:, :, c:-c, c:-c] if c > 0 else img_flipped
        sim_final_size = F.interpolate(cropped, size=self.resize_target, mode='bilinear', align_corners=False)

        if target_real_img is not None:
            sim_flat = sim_final_size.view(B, -1)
            real_flat = target_real_img.view(B, -1)
            mask = real_flat < 0.99

            gains = []
            for i in range(B):
                m = mask[i]
                if m.sum() < 100:
                    m = torch.ones_like(m)
                v_real = real_flat[i][m]
                v_sim = sim_flat[i][m]
                num = torch.dot(v_real, v_sim)
                den = torch.dot(v_sim, v_sim)
                g_ls = num / (den + 1e-8)
                max_r = real_flat[i].max()
                max_s = sim_flat[i].max()
                g_peak = max_r / (max_s + 1e-8)
                g = torch.min(g_ls, g_peak * 1.05)
                gains.append(g)

            gain_tensor = torch.stack(gains).view(B, 1, 1, 1)
            sim_out = torch.clamp(sim_final_size * gain_tensor, 0, 1)
            return sim_out

        return sim_final_size


# =========================================================
# 7. Dual-stem shared backbone + residual log-d head
# =========================================================
class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for (N,C,H,W) tensors."""
    def __init__(self, n_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (N,C,H,W)
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight.view(1, -1, 1, 1) * x + self.bias.view(1, -1, 1, 1)


class DualStemNet(nn.Module):
    """
    Two stems (adapters), shared trunk (selectable backbone):
      - real stem: 4ch -> trunk
      - sim  stem: 2ch -> trunk
    Heads:
      - delta head: predicts Δ in log-space (clipped)
      - embedding head: projects to z for VICReg / group / TTA

    Backbones supported (via TrainConfig.BACKBONE_NAME):
      - 'resnet34' (default fallback), 'resnet18', 'resnet50' (torchvision)
      - 'convnext_t', 'convnext_tiny' (torchvision convnext_tiny)
    Notes:
      - For ConvNeXt, we expose pseudo layer1..layer4 to keep the existing
        progressive-freeze code unchanged.
    """
    def __init__(self, config: TrainConfig, emb_dim: int = 256):
        super().__init__()
        self.config = config

        if models is None:
            raise ImportError("torchvision is required for selectable backbones, but torchvision import failed.")

        name = str(getattr(config, 'BACKBONE_NAME', 'resnet34')).lower().strip()
        # normalize a few aliases
        if name in ['convnext_t', 'convnext-t', 'convnext_tiny', 'convnext-tiny', 'convnext']:
            name = 'convnext_tiny'
        if name in ['', 'default', 'auto']:
            name = 'resnet34'

        self.backbone_name = name
        self.backbone_type = None

        if name.startswith('convnext'):
            self._init_convnext_tiny(emb_dim=emb_dim)
        else:
            self._init_resnet(name=name, emb_dim=emb_dim)

    # --------------------------
    # Backbone builders
    # --------------------------
    def _init_resnet(self, name: str, emb_dim: int):
        # Choose resnet variant
        name = str(name).lower().strip()
        if name not in ['resnet18', 'resnet34', 'resnet50']:
            name = 'resnet34'

        # Try to use official pretrained weights (if available), otherwise fall back to random init.
        base = None
        if name == 'resnet18':
            try:
                weights = getattr(models, 'ResNet18_Weights', None)
                base = models.resnet18(weights=weights.DEFAULT if weights else 'DEFAULT')
            except Exception:
                base = models.resnet18(weights=None)
            feat_dim = 512
        elif name == 'resnet50':
            try:
                weights = getattr(models, 'ResNet50_Weights', None)
                base = models.resnet50(weights=weights.DEFAULT if weights else 'DEFAULT')
            except Exception:
                base = models.resnet50(weights=None)
            feat_dim = 2048
        else:
            try:
                weights = getattr(models, 'ResNet34_Weights', None)
                base = models.resnet34(weights=weights.DEFAULT if weights else 'DEFAULT')
            except Exception:
                base = models.resnet34(weights=None)
            feat_dim = 512

        self.backbone_type = name

        # Build stems mirroring ResNet stem (conv/bn/relu/maxpool)
        def make_stem(in_ch: int):
            conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
            bn = nn.BatchNorm2d(64)
            relu = nn.ReLU(inplace=True)
            mp = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
            return nn.Sequential(conv, bn, relu, mp)

        self.stem_real = make_stem(self.config.REAL_BRANCH_CH)
        self.stem_sim = make_stem(self.config.SIM_BRANCH_CH)

        # Initialize stem weights from pretrained RGB conv1 by channel-tiling
        try:
            with torch.no_grad():
                w = base.conv1.weight  # (64,3,7,7)
                # real stem (4ch)
                self.stem_real[0].weight.copy_(w.repeat(1, 2, 1, 1)[:, :self.config.REAL_BRANCH_CH])
                # sim stem (2ch)
                self.stem_sim[0].weight.copy_(w.repeat(1, 1, 1, 1)[:, :self.config.SIM_BRANCH_CH])
        except Exception:
            # if pretrained weights not available, keep random init
            pass

        # Shared trunk (keep same attribute names used elsewhere)
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self._feat_dim = int(feat_dim)
        self._post_norm = None  # for API parity
        self._trunk_is_convnext = False

        self._init_heads(feat_dim=self._feat_dim, emb_dim=emb_dim)

    def _init_convnext_tiny(self, emb_dim: int):
        # Build ConvNeXt-Tiny (torchvision)
        try:
            weights = getattr(models, 'ConvNeXt_Tiny_Weights', None)
            base = models.convnext_tiny(weights=weights.DEFAULT if weights else None)
        except Exception:
            base = models.convnext_tiny(weights=None)

        self.backbone_type = 'convnext_tiny'

        # Determine stem output channels if possible; default to 96 for convnext_tiny
        stem_out = 96
        try:
            # base.features[0] is usually Conv2dNormActivation; try to access its conv out_channels
            conv0 = None
            try:
                conv0 = base.features[0][0]
            except Exception:
                conv0 = getattr(base.features[0], '0', None)
            if conv0 is not None and hasattr(conv0, 'out_channels'):
                stem_out = int(conv0.out_channels)
        except Exception:
            stem_out = 96

        def make_stem(in_ch: int):
            # ConvNeXt patchify stem: k=4, s=4
            return nn.Sequential(
                nn.Conv2d(in_ch, stem_out, kernel_size=4, stride=4, padding=0, bias=True),
                LayerNorm2d(stem_out, eps=1e-6),
                nn.GELU(),
            )

        self.stem_real = make_stem(self.config.REAL_BRANCH_CH)
        self.stem_sim = make_stem(self.config.SIM_BRANCH_CH)

        # Initialize from pretrained stem conv if accessible
        try:
            with torch.no_grad():
                w = None
                try:
                    w = base.features[0][0].weight
                except Exception:
                    try:
                        w = base.features[0].weight
                    except Exception:
                        w = None
                if w is not None and w.ndim == 4:
                    # Average RGB weights -> 1ch, then repeat to needed channels
                    w1 = w.mean(dim=1, keepdim=True)
                    self.stem_real[0].weight.copy_(w1.repeat(1, self.config.REAL_BRANCH_CH, 1, 1))
                    self.stem_sim[0].weight.copy_(w1.repeat(1, self.config.SIM_BRANCH_CH, 1, 1))
        except Exception:
            pass

        # Expose pseudo layer1..4 for compatibility with progressive freeze code.
        feats = list(getattr(base, 'features', nn.Sequential()).children())
        # Typical layout: [stem, stage1, stage2, stage3, stage4, norm]
        stages = feats[1:] if len(feats) >= 2 else []
        self.layer1 = stages[0] if len(stages) > 0 else nn.Identity()
        self.layer2 = stages[1] if len(stages) > 1 else nn.Identity()
        self.layer3 = stages[2] if len(stages) > 2 else nn.Identity()
        self.layer4 = stages[3] if len(stages) > 3 else nn.Identity()

        # Post norm (if exists)
        self._post_norm = stages[4] if len(stages) > 4 else nn.Identity()

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                # Infer actual feature dim by running the trunk once on a CPU dummy batch.
        # This avoids torchvision-version-dependent mismatches (e.g., 384 vs 768).
        feat_dim = int(self._infer_feat_dim())
        self._feat_dim = feat_dim

        self._trunk_is_convnext = True

        self._init_heads(feat_dim=self._feat_dim, emb_dim=emb_dim)

    def _infer_feat_dim(self) -> int:
        """Infer trunk feature dimension with a CPU dummy forward to avoid version-dependent mismatches."""
        try:
            # Make sure convnext-related flags won't crash _trunk before they are set.
            if not hasattr(self, "_trunk_is_convnext"):
                self._trunk_is_convnext = True
            if not hasattr(self, "_post_norm"):
                self._post_norm = None

            h = int(getattr(self.config, 'RESIZE', 224))
            w = int(getattr(self.config, 'RESIZE', 224))
            dummy = torch.zeros(1, int(self.config.REAL_BRANCH_CH), h, w, dtype=torch.float32)
            with torch.no_grad():
                feat = self._trunk(dummy, self.stem_real)
            return int(feat.shape[1])
        except Exception:
            # Conservative fallback for convnext_tiny
            return 768


    def _init_heads(self, feat_dim: int, emb_dim: int):
        # Condition on params + coarse
        self.param_mlp = nn.Sequential(
            nn.Linear(self.config.PARAM_DIM + 1, 256),
            nn.SiLU(),
            nn.Linear(256, 256),
            nn.SiLU()
        )

        self.fuse = nn.Sequential(
            nn.Linear(int(feat_dim) + 256, 512),
            nn.SiLU(),
            nn.Dropout(0.1)
        )

        self.delta_head = nn.Linear(512, 1)   # Δ in log-space
        self.emb_head = nn.Sequential(
            nn.Linear(512, emb_dim),
            nn.LayerNorm(emb_dim)
        )

    # --------------------------
    # Internal forward pieces
    # --------------------------
    def _stem_pool(self, x: torch.Tensor, stem: nn.Module) -> torch.Tensor:
        x = stem(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)

    def _trunk(self, x: torch.Tensor, stem: nn.Module) -> torch.Tensor:
        x = stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        if self._trunk_is_convnext and self._post_norm is not None:
            x = self._post_norm(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    # ---------------------------------------------------------
    # Backbone accessors (for debugging / feature inspection)
    # ---------------------------------------------------------
    def getbackbone(self):
        # Keep backward compatibility: return a nn.Sequential of "trunk blocks"
        return nn.Sequential(self.layer1, self.layer2, self.layer3, self.layer4)

    def get_backbone(self):
        return self.getbackbone()

    # --------------------------
    # Public forward APIs (UNCHANGED)
    # --------------------------
    def forward_real(self, x_real4: torch.Tensor, params10: torch.Tensor, coarse1: torch.Tensor, return_stem_pool: bool = False):
        stem_pool = self._stem_pool(x_real4, self.stem_real) if return_stem_pool else None
        feat = self._trunk(x_real4, self.stem_real)
        c = torch.log(coarse1.clamp_min(1e-6))
        cond = self.param_mlp(torch.cat([params10, c], dim=1))
        h = self.fuse(torch.cat([feat, cond], dim=1))
        z = self.emb_head(h)

        # residual log-space diameter
        delta_raw = self.delta_head(h)
        delta = self.config.DELTA_CLIP * torch.tanh(delta_raw)  # smooth clip
        log_d = c + delta
        d_um = torch.exp(log_d).clamp(min=1e-4)
        if return_stem_pool:
            return d_um, log_d, z, stem_pool
        return d_um, log_d, z

    def forward_sim(self, x_sim2: torch.Tensor, params10: torch.Tensor, coarse1: torch.Tensor, return_stem_pool: bool = False):
        # Single-branch ablation: Make the sim branch also go through stem_real (by padding to 4 channels)
        # Disabled by default, only enabled when the environment variable FORCE_SINGLE_BRANCH=1 is set
        force_single = os.environ.get("FORCE_SINGLE_BRANCH", "0").strip() in {"1", "true", "True"}
        if force_single:
            x_sim4 = torch.cat([x_sim2, torch.zeros_like(x_sim2)], dim=1)  # 2ch -> 4ch
            stem = self.stem_real
            stem_pool = self._stem_pool(x_sim4, stem) if return_stem_pool else None
            feat = self._trunk(x_sim4, stem)
        else:
            stem = self.stem_sim
            stem_pool = self._stem_pool(x_sim2, stem) if return_stem_pool else None
            feat = self._trunk(x_sim2, stem)

        c = torch.log(coarse1.clamp_min(1e-6))
        cond = self.param_mlp(torch.cat([params10, c], dim=1))
        h = self.fuse(torch.cat([feat, cond], dim=1))
        z = self.emb_head(h)
        if return_stem_pool:
            return z, stem_pool
        return z

# =========================================================
# 8. Weak + Pair losses
# =========================================================
def weak_guidance_rel(pred, coarse, tolerance=0.1):
    # Continuous, relative-normalized deviation outside band
    ratio = pred / coarse.clamp_min(1e-6)
    lower = 1.0 - tolerance
    upper = 1.0 + tolerance
    loss = F.relu(lower - ratio) + F.relu(ratio - upper)
    return loss.mean()

def pair_consistency_loss_logd(log_d, target_a, is_paired_mask):
    # log-space L1 for stability
    if is_paired_mask.sum() < 1:
        # Return a tensor with requires_grad=True to avoid gradient computation errors
        return torch.tensor(0.0, device=log_d.device, requires_grad=True)
    log_a = torch.log(target_a.clamp_min(1e-6))
    diff = (log_d.squeeze(1) - log_a).abs()
    return (diff * is_paired_mask).sum() / (is_paired_mask.sum() + 1e-6)


# =========================================================
# 9. VICReg (anti-collapse)
# =========================================================
def vicreg_loss(z1, z2, sim_coeff=25.0, var_coeff=25.0, cov_coeff=1.0, eps=1e-4, return_components: bool = False):
    """
    z1, z2: (B, D)
    Returns: total, var_penalty (for selection monitoring)
    """
    # invariance
    inv = F.mse_loss(z1, z2)

    # variance
    def var_term(z):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))
    v1 = var_term(z1)
    v2 = var_term(z2)
    var = v1 + v2

    # covariance
    def cov_term(z):
        z = z - z.mean(dim=0)
        B, D = z.shape
        cov = (z.T @ z) / (B - 1)
        off = cov - torch.diag(torch.diag(cov))
        return (off**2).sum() / D
    cov = cov_term(z1) + cov_term(z2)

    total = sim_coeff * inv + var_coeff * var + cov_coeff * cov
    if return_components:
        return total, inv.detach(), var.detach(), cov.detach()
    return total, var.detach()


# =========================================================
# 10. Validation (updated selection & VICReg var penalty)
# =========================================================
def validate_unified(model, physics, val_loader, config, adv_phys_loss):
    model.eval()
    val_l_phys, val_l_weak, val_l_pair = [], [], []
    val_l_corr, val_l_fft, val_l_grad = [], [], []
    val_l_vicvar = []
    val_l_vicinv = []
    val_l_viccov = []
    preds_by_realid = defaultdict(list)
    all_preds_err = []
    all_preds = []

    
    phys_debug_batches = 0
    with torch.no_grad():
        for fnames, full_input, sim_pt, rp, sp, coarse, target_a, is_paired, real_idx in val_loader:
            full_input = full_input.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            sim_pt = sim_pt.cuda(non_blocking=True)
            rp = rp.cuda(non_blocking=True)
            sp = sp.cuda(non_blocking=True)
            coarse = coarse.cuda().view(-1, 1)
            target_a = target_a.cuda().view(-1, 1)
            is_paired = is_paired.cuda().view(-1)
            real_idx = real_idx.cuda()

            rp_refined = rp

            # ---- Parameter pipeline ----
            # 1) Physics-side: before any Fraunhofer usage, ensure PHYSICAL units (denorm if needed)
            rp_phys = ensure_params5_phys_for_physics(rp_refined, ctx="val/rp_for_phys")
            _ = ensure_params5_phys_for_physics(sp,        ctx="val/sp_for_phys")  # sp checked symmetrically

            # 2) NN-side: condition inputs must be normalized to [-1,1] (detect by range, not by bounds)
            rp_n = normalize_params5(rp_refined)
            sp_n = normalize_params5(sp).to(rp_n.device)
            params = torch.cat([rp_n, sp_n], dim=1)

            # Build branch inputs
            x_real4 = torch.cat([full_input[:, 0:2], full_input[:, 4:6]], dim=1)  # real+diff
            x_sim2 = sim_pt

            # TTA on embeddings + diameter (using mean)
            d_preds = []
            z_real0 = None
            stem_real_pool0 = None
            with torch.amp.autocast('cuda'):
                for i, aug in enumerate(tta_noise_only(x_real4, config.TTA_NUM_VAL, noise_level=config.TTA_NOISE_VAL)):
                    d_um, log_d, z_real, stem_real_pool = model.forward_real(aug, params, coarse, return_stem_pool=True)
                    d_preds.append(d_um)
                    if i == 0:
                        z_real0 = z_real
                        stem_real_pool0 = stem_real_pool

            d_stack = torch.stack(d_preds)
            d_final = d_stack.mean(0)

            # weak (relative)
            val_l_weak.append(float(weak_guidance_rel(d_final, coarse, config.WEAK_TOLERANCE).item()))

            # pair consistency ONLY on is_paired
            if config.ENABLE_PAIRCONS:
                # reuse log_d from first aug for stability
                with torch.amp.autocast('cuda'):
                    _, log_d0, _ = model.forward_real(x_real4, params, coarse)
                val_l_pair.append(float(pair_consistency_loss_logd(log_d0, target_a.squeeze(1), is_paired).item()))

            # record err (for observation only; not used in selection)
            errs = (d_final - config.GROUND_TRUTH_DIAMETER).abs() / config.GROUND_TRUTH_DIAMETER
            all_preds_err.extend(errs.cpu().tolist())
            all_preds.extend(d_final.detach().view(-1).cpu().tolist())

            for i, fn in enumerate(fnames):
                rid = _extract_real_id(fn)
                val = float(d_final[i].item())
                if not math.isnan(val):
                    preds_by_realid[rid].append(val)

            if config.ENABLE_PHYS:
                # physics compares normalized intensities inside AdvancedPhysicsLoss
                with torch.cuda.amp.autocast(enabled=False):
                    img_pred = physics(d_final.float(), rp_phys.float(), target_real_img=full_input[:, 0:1].float())
                if getattr(config, 'ENABLE_PHYS_DEBUG_PRINT', False) and phys_debug_batches < getattr(config, 'PHYS_DEBUG_PRINT_FIRST_N_BATCHES', 1):
                    print('[PhysDebug] img_pred.min =', float(img_pred.min().item()))
                    print('[PhysDebug] img_pred.max =', float(img_pred.max().item()))
                    print('[PhysDebug] img_pred.mean=', float(img_pred.mean().item()))
                    phys_debug_batches += 1

                l_adv_total, l_corr, l_fft, l_grad = adv_phys_loss(img_pred, full_input[:, 0:1])
                if not math.isnan(l_adv_total.item()):
                    val_l_phys.append(float(l_adv_total.item()))
                    val_l_corr.append(float(l_corr.item()))
                    val_l_fft.append(float(l_fft.item()))
                    val_l_grad.append(float(l_grad.item()))

            if config.ENABLE_VICREG:
                with torch.amp.autocast('cuda'):
                    z_sim, stem_sim_pool = model.forward_sim(x_sim2, params, coarse, return_stem_pool=True)
                    # VICReg on pooled stems (adapters)
                    vtot, vinv, vvar, vcov = vicreg_loss(
                        stem_real_pool0.float(), stem_sim_pool.float(),
                        sim_coeff=config.W_VICREG_INV,
                        var_coeff=config.W_VICREG_VAR,
                        cov_coeff=config.W_VICREG_COV,
                        return_components=True
                    )
                if not math.isnan(float(vvar.item())):
                    val_l_vicvar.append(float(vvar.item()))
                    val_l_vicinv.append(float(vinv.item()))
                    val_l_viccov.append(float(vcov.item()))

    # group var is still reported (but NOT used for selection)
    group_vars = []
    for rid, arr in preds_by_realid.items():
        if len(arr) > 1:
            group_vars.append(np.var(arr))
    avg_group = float(np.mean(group_vars)) if group_vars else 0.0

    avg_phys = float(np.mean(val_l_phys)) if val_l_phys else 10.0
    avg_weak = float(np.mean(val_l_weak)) if val_l_weak else 10.0
    avg_pair = float(np.mean(val_l_pair)) if val_l_pair else 0.0
    avg_vicv = float(np.mean(val_l_vicvar)) if val_l_vicvar else 10.0

    # Enhanced pair weight (for bad seeds with high pairloss)
    # Bad seeds typically have high pairloss (0.066 vs 0.027), so boosting pair weight
    # helps select models with better pair consistency (no GT dependency)
    w_pair_eff = float(config.SCORE_W_PAIR) * float(getattr(config, "SCORE_W_PAIR_BOOST", 1.0))

    # Continuous, relative-normalized selection score
    # NOTE: No GT information used - this is an unsupervised measurement system
    selection_score = (
        config.SCORE_W_PHYS * avg_phys
        + config.SCORE_W_WEAK * avg_weak
        + w_pair_eff * avg_pair
        + config.SCORE_W_VICVAR * avg_vicv
    )

    return selection_score, {
        "val_score": selection_score,
        "val_err": float(np.mean(all_preds_err)) if all_preds_err else 0.0,
        "val_pred_std": float(np.std(all_preds)) if all_preds else 0.0,
        "val_pred_mean": float(np.mean(all_preds)) if all_preds else 0.0,
        "val_phys": avg_phys,
        "val_sub_corr": float(np.mean(val_l_corr)) if val_l_corr else 0.0,
        "val_sub_fft": float(np.mean(val_l_fft)) if val_l_fft else 0.0,
        "val_sub_grad": float(np.mean(val_l_grad)) if val_l_grad else 0.0,
        "val_weak": avg_weak,
        "val_pair": avg_pair,
        "val_vicvar": avg_vicv,
        "val_vicinv": float(np.mean(val_l_vicinv)) if val_l_vicinv else 0.0,
        "val_viccov": float(np.mean(val_l_viccov)) if val_l_viccov else 0.0,
        "val_group": avg_group,
    }


# =========================================================
# 11. Scheduler
# =========================================================
def build_warmup_cosine_scheduler(optimizer, num_epochs, warmup_ratio=0.1, min_lr=1e-6):
    warm = max(1, int(num_epochs * warmup_ratio))
    main_iter = num_epochs - warm
    s1 = LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=warm)
    s2 = CosineAnnealingLR(optimizer, T_max=main_iter, eta_min=min_lr)
    return SequentialLR(optimizer, schedulers=[s1, s2], milestones=[warm])


# =========================================================
# 12. Dryrun check (updated batch indices)
# =========================================================
def dryrun_check(config):
    print("\n[Dry Run] Verifying leakage-safe paths + dataset integrity...")
    if not os.path.isdir(config.TRAIN_REAL_FOLDER):
        raise FileNotFoundError(f"Missing TRAIN_REAL_FOLDER: {config.TRAIN_REAL_FOLDER}")
    if not os.path.isdir(config.VAL_REAL_FOLDER):
        print(f"[Warn] Missing VAL_REAL_FOLDER: {config.VAL_REAL_FOLDER} (val may fail)")
    if not os.path.exists(config.TRAIN_LABELS_CSV):
        raise FileNotFoundError(f"Missing TRAIN_LABELS_CSV: {config.TRAIN_LABELS_CSV}")

    try:
        ds_dry = DiffSimRealDataset(
            dataset_root=config.DATASET_ROOT,
            real_root=config.TRAIN_REAL_FOLDER,
            labels_csv=config.TRAIN_LABELS_CSV,
            init_diameter_txt=config.INIT_DIAMETER_TXT,
            real_rp_csv=config.TRAIN_REAL_RP_CSV,
            is_train=True,
            config=config
        )
        if len(ds_dry) == 0:
            print("[AutoRepair] Dryrun found empty train dataset. Trying to repair artifacts/cache...")
            ensure_main_required_artifacts(config, attempt=2)
            ds_dry = DiffSimRealDataset(
                dataset_root=config.DATASET_ROOT,
                real_root=config.TRAIN_REAL_FOLDER,
                labels_csv=config.TRAIN_LABELS_CSV,
                init_diameter_txt=config.INIT_DIAMETER_TXT,
                real_rp_csv=config.TRAIN_REAL_RP_CSV,
                is_train=True,
                config=config
            )
            if len(ds_dry) == 0:
                raise ValueError("Train dataset empty after auto-repair (check FFT cache + csv paths).")

        dl_dry = DataLoader(ds_dry, batch_size=2, shuffle=True, num_workers=0)

        model_dry = DualStemNet(config).cuda()
        phys_dry = FraunhoferPhysicsLayer(
            config.ORIG_H, config.ORIG_W, config.CROP_PIXELS, config.RESIZE,
            config.PINN_WAVELENGTH_UM, config.PINN_FOCLENGTH_UM, config.PINN_PIXEL_SIZE_UM
        ).cuda()
        adv_loss_dry = AdvancedPhysicsLoss(config).cuda()

        batch = next(iter(dl_dry))
        full_input = batch[1].cuda()
        sim_pt = batch[2].cuda()
        rp = batch[3].cuda()
        sp = batch[4].cuda()
        coarse = batch[5].cuda().view(-1, 1)
        idx = batch[8].cuda()

        with torch.amp.autocast('cuda'):
            params = torch.cat([normalize_params5(rp), normalize_params5(sp)], dim=1)
            x_real4 = torch.cat([full_input[:, 0:2], full_input[:, 4:6]], dim=1)
            d_um, _, z = model_dry.forward_real(x_real4, params, coarse)
            rp_phys = ensure_params5_phys_for_physics(rp, ctx="dry/rp_for_phys")
            phys_out = phys_dry(d_um, rp_phys, target_real_img=full_input[:, 0:1])
            loss, _, _, _ = adv_loss_dry(phys_out, full_input[:, 0:1])
            loss.backward()

        print("[Dry Run] Train pipeline forward/backward OK!")
        del model_dry, phys_dry, adv_loss_dry, dl_dry, ds_dry, batch
        torch.cuda.empty_cache()
    except Exception as e:
        raise RuntimeError(f"[Dry Run] Failed: {e}")
    print("[Dry Run] All checks passed!\n")


# =========================================================
# 13. Debug visualization
# =========================================================
def save_debug_visualization(model, physics, loader, config, epoch, out_dir):
    model.eval()
    os.makedirs(out_dir, exist_ok=True)
    try:
        batch = next(iter(loader))
        fnames, full_input, sim_pt, rp, sp, coarse, target_a, is_paired, real_idx = batch
        full_input = full_input.cuda()
        rp = rp.cuda()
        sp = sp.cuda()
        coarse = coarse.cuda().view(-1, 1)
        real_idx = real_idx.cuda()

        with torch.no_grad():
            params = torch.cat([normalize_params5(rp), normalize_params5(sp)], dim=1)
            x_real4 = torch.cat([full_input[:, 0:2], full_input[:, 4:6]], dim=1)
            d_um, _, _ = model.forward_real(x_real4, params, coarse)
            rp_phys = ensure_params5_phys_for_physics(rp, ctx="debug/rp_for_phys")
            img_pred = physics(d_um, rp_phys, target_real_img=full_input[:, 0:1])
            real_img = full_input[0, 0:1, :, :]
            pred_img = img_pred[0, :, :, :]
            # When outputting debug_vis, flip the sim image once to match the display direction of the real image
            # Note: The flipped image is used in loss calculation (it has been flipped internally by physics), which is correct
            # This is just to make the directions consistent during debug_vis display
            pred_img_flipped_for_display = torch.flip(pred_img, dims=[1])
            combined = torch.cat([real_img, pred_img_flipped_for_display], dim=2)
            save_path = os.path.join(out_dir, f"epoch_{epoch}_pred_{d_um[0].item():.2f}um.png")
            save_image(combined, save_path)
    except Exception:
        pass




# =========================================================
# 13.2 Dataset PNG integrity + incremental repair (NO unnecessary regeneration)
# =========================================================
def _png_readable(path: str) -> bool:
    """Lightweight check: exists + decodable by cv2."""
    try:
        if (not os.path.isfile(path)) or (os.path.getsize(path) < 128):
            return False
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        return (img is not None) and (img.size > 0)
    except Exception:
        return False

def dataset_png_report(dataset_root: str, labels_csv: str, *, sample_rows: int = 0) -> dict:
    """Check that CSV exists and referenced sim/diff PNG files exist (and optionally readable)."""
    rep = {
        "labels_csv": labels_csv,
        "csv_exists": os.path.isfile(labels_csv),
        "rows": 0,
        "unique_sims": 0,
        "unique_diffs": 0,
        "missing_sims": 0,
        "missing_diffs": 0,
        "bad_decode": 0,
        "coverage": 0.0,
    }
    if not rep["csv_exists"]:
        return rep

    df = pd.read_csv(labels_csv)
    rep["rows"] = int(len(df))
    if rep["rows"] == 0:
        return rep

    sims = df["sim_path"].astype(str).unique().tolist() if "sim_path" in df.columns else []
    diffs = df["diff_path"].astype(str).unique().tolist() if "diff_path" in df.columns else []
    rep["unique_sims"] = int(len(sims))
    rep["unique_diffs"] = int(len(diffs))

    miss_s = sum(1 for p in sims if not os.path.isfile(os.path.join(dataset_root, p)))
    miss_d = sum(1 for p in diffs if not os.path.isfile(os.path.join(dataset_root, p)))
    rep["missing_sims"] = int(miss_s)
    rep["missing_diffs"] = int(miss_d)

    total = rep["unique_sims"] + rep["unique_diffs"]
    ok = total - (rep["missing_sims"] + rep["missing_diffs"])
    rep["coverage"] = float(ok / max(1, total))

    if sample_rows and sample_rows > 0:
        try:
            n = min(int(sample_rows), rep["rows"])
            sub = df.sample(n=n, random_state=42) if rep["rows"] > n else df
            bad = 0
            for _, r in sub.iterrows():
                sp = os.path.join(dataset_root, str(r["sim_path"]))
                dp = os.path.join(dataset_root, str(r["diff_path"]))
                if os.path.isfile(sp) and (not _png_readable(sp)):
                    bad += 1
                if os.path.isfile(dp) and (not _png_readable(dp)):
                    bad += 1
            rep["bad_decode"] = int(bad)
        except Exception:
            rep["bad_decode"] = -1

    return rep

def dataset_png_is_complete(dataset_root: str, labels_csv: str, *, min_coverage: float = 0.999) -> bool:
    rep = dataset_png_report(dataset_root, labels_csv, sample_rows=0)
    return bool(rep["csv_exists"]) and rep["rows"] > 0 and rep["coverage"] >= float(min_coverage) and rep["missing_sims"] == 0 and rep["missing_diffs"] == 0

def repair_dataset_from_labels(
    *,
    dataset_root: str,
    labels_csv: str,
    real_folder: str,
    real_rp_csv: Optional[str],
    apply_align: bool,
    apply_mask: bool,
    mask_band_px: int,
    mask_mode: str,
    n_jobs: int,
    fast_limit_targets: int = 0
) -> dict:
    """Incrementally repair missing sim/diff PNGs referenced by an existing labels CSV.
    This DOES NOT regenerate new samples; it only fills missing files.
    """
    if not os.path.isfile(labels_csv):
        return {"ok": False, "reason": "labels_csv_missing", "labels_csv": labels_csv}

    df = pd.read_csv(labels_csv)
    if len(df) == 0:
        return {"ok": False, "reason": "labels_csv_empty", "labels_csv": labels_csv}

    # Load alignment parameters (optional)
    rp_map = {}
    if real_rp_csv and os.path.isfile(real_rp_csv):
        try:
            dfrp = pd.read_csv(real_rp_csv)
            key = "Filename" if "Filename" in dfrp.columns else ("fname" if "fname" in dfrp.columns else None)
            if key is not None:
                if "Phi" in dfrp.columns:
                    col_phi, col_x = "Phi", ("X_um" if "X_um" in dfrp.columns else "x")
                else:
                    col_phi, col_x = "phi", ("x" if "x" in dfrp.columns else "x_um")
                for _, rr in dfrp.iterrows():
                    fn = str(rr[key])
                    rp_map[fn] = (float(rr.get(col_phi, 0.0)), float(rr.get(col_x, 0.0)))
        except Exception:
            rp_map = {}

    needed = set(df["tgt_fname"].astype(str).tolist())
    if "src_fname" in df.columns:
        needed |= set([str(x) for x in df["src_fname"].dropna().astype(str).tolist() if str(x).lower() != "none"])

    real_imgs: Dict[str, np.ndarray] = {}
    for fn in sorted(needed):
        fp = os.path.join(real_folder, os.path.basename(fn))
        if not os.path.isfile(fp):
            fp = os.path.join(real_folder, fn)
        if not os.path.isfile(fp):
            continue
        try:
            phi, x_um = rp_map.get(os.path.basename(fn), (0.0, 0.0))
            real_imgs[os.path.basename(fn)] = preprocess_image(
                fp,
                align_phi_deg=phi,
                align_x_shift_um=x_um,
                apply_align=bool(apply_align),
                apply_mask=bool(apply_mask),
                mask_band_px=int(mask_band_px),
                mask_mode=str(mask_mode),
            )
        except Exception:
            continue

    if not real_imgs:
        return {"ok": False, "reason": "no_real_images_loaded", "real_folder": real_folder}

    first_shape = next(iter(real_imgs.values())).shape

    sim_rows = df.drop_duplicates(subset=["sim_path"]).copy()
    if "src_fname" not in sim_rows.columns:
        sim_rows["src_fname"] = sim_rows["tgt_fname"]

    missing_sims = []
    for _, r in sim_rows.iterrows():
        sim_full = os.path.join(dataset_root, str(r["sim_path"]))
        if not os.path.isfile(sim_full):
            missing_sims.append(r)

    missing_diffs = []
    for _, r in df.iterrows():
        diff_full = os.path.join(dataset_root, str(r["diff_path"]))
        if not os.path.isfile(diff_full):
            missing_diffs.append(r)

    if fast_limit_targets and fast_limit_targets > 0 and len(missing_diffs) > fast_limit_targets:
        missing_diffs = missing_diffs[:fast_limit_targets]

    def _make_sim(row):
        sim_full = os.path.join(dataset_root, str(row["sim_path"]))
        os.makedirs(os.path.dirname(sim_full), exist_ok=True)

        src_fn = os.path.basename(str(row.get("src_fname", "")) or "")
        src_img = real_imgs.get(src_fn, None)
        if src_img is None:
            tgt_fn = os.path.basename(str(row.get("tgt_fname", "")) or "")
            src_img = real_imgs.get(tgt_fn, None)
        if src_img is None:
            return False

        p = np.array([float(row["phi"]), float(row["r"]), float(row["b"]), float(row["x"]), float(row["a"]), float(row["gamma"])], dtype=np.float64)
        field = generate_field_raw(p, first_shape, DataConfig)
        field_non = np.power(np.abs(field), float(row["gamma"]))
        gain = calculate_analytic_gain(src_img, field_non)
        I_sim = np.clip(field_non * gain, 0, 255).astype(np.uint8)
        cv2.imwrite(sim_full, I_sim)
        return True

    sim_full_map = {sim_rel: os.path.join(dataset_root, sim_rel) for sim_rel in df["sim_path"].astype(str).unique().tolist()}

    def _make_diff(row):
        diff_full = os.path.join(dataset_root, str(row["diff_path"]))
        os.makedirs(os.path.dirname(diff_full), exist_ok=True)

        tgt_fn = os.path.basename(str(row["tgt_fname"]))
        tgt_img = real_imgs.get(tgt_fn, None)
        if tgt_img is None:
            return False

        sim_rel = str(row["sim_path"])
        sim_full = sim_full_map.get(sim_rel, os.path.join(dataset_root, sim_rel))
        I_sim = cv2.imread(sim_full, cv2.IMREAD_GRAYSCALE)
        if I_sim is None:
            return False

        tgt_u8 = np.clip(tgt_img, 0, 255).astype(np.uint8)
        I_diff = cv2.absdiff(tgt_u8, I_sim)
        cv2.imwrite(diff_full, I_diff)
        return True

    ok_s = 0
    if missing_sims:
        print(f"[Repair] Missing sims: {len(missing_sims)} -> regenerating ONLY missing sims...")
        res = Parallel(n_jobs=int(n_jobs), verbose=5)(delayed(_make_sim)(r) for r in missing_sims)
        ok_s = int(sum(bool(x) for x in res))

    ok_d = 0
    if missing_diffs:
        print(f"[Repair] Missing diffs: {len(missing_diffs)} -> regenerating ONLY missing diffs...")
        res = Parallel(n_jobs=int(n_jobs), verbose=5)(delayed(_make_diff)(r) for r in missing_diffs)
        ok_d = int(sum(bool(x) for x in res))

    rep2 = dataset_png_report(dataset_root, labels_csv, sample_rows=0)
    return {
        "ok": bool(rep2["csv_exists"]) and rep2["missing_sims"] == 0 and rep2["missing_diffs"] == 0,
        "repaired_sims": ok_s,
        "repaired_diffs": ok_d,
        "after": rep2,
    }

# =========================================================
# 13.5 Data-artifact self-healing for main()
# =========================================================
def _file_nonempty(path: str) -> bool:
    try:
        return os.path.isfile(path) and os.path.getsize(path) > 0
    except OSError:
        return False


def _csv_readable(path: str) -> bool:
    if not _file_nonempty(path):
        return False
    try:
        # small read to confirm it's a valid CSV
        _ = pd.read_csv(path, nrows=5)
        return True
    except Exception:
        return False


def ensure_main_required_artifacts(config: 'TrainConfig', *, attempt: int = 1) -> None:
    """Best-effort: if main-required artifacts are missing/corrupt, regenerate via data-pipeline,
    then (re)build FFT cache incrementally.

    This function makes *minimal* changes to behavior: it only runs when required files are missing
    or unreadable, and relies on the data pipeline's own skip/resume behavior.
    """
    # What main expects (paths come from TrainConfig)
    required_csvs = [
        getattr(config, "TRAIN_LABELS_CSV", None),
        getattr(config, "VAL_LABELS_CSV", None),
        getattr(config, "TRAIN_REAL_RP_CSV", None),
        getattr(config, "VAL_REAL_RP_CSV", None),
    ]
    required_csvs = [p for p in required_csvs if isinstance(p, str) and p.strip()]

    required_other = [
        getattr(config, "INIT_DIAMETER_TXT", None),
    ]
    required_other = [p for p in required_other if isinstance(p, str) and p.strip()]

    missing = []
    for p in required_csvs:
        if not _csv_readable(p):
            missing.append(p)
    for p in required_other:
        if not _file_nonempty(p):
            missing.append(p)

    if not missing:
        return

    print(f"[AutoRepair] Detected missing/corrupt artifacts (attempt {attempt}):")
    for p in missing:
        print(f"  - {p}")

    # Run the data pipeline (resumable) to regenerate whatever is missing.
    # NOTE: This is a best-effort repair. It does NOT delete existing artifacts.
    try:
        validate_real_folders()
        maybe_auto_split_to_folders()
        validate_real_folders()

        # Part 1/2/2.5
        init_a = estimate_initial_diameter()
        df_train = run_optimization(init_a)
        export_real_rp_tables(df_train, init_a)

        # Part 3 (labels csv & sim/diff png)  -- DO NOT regenerate if complete
        if DataConfig.FAST_VERIFY and bool(DataConfig.FAST_VERIFY_SKIP_PART3):
            print("[AutoRepair] FAST_VERIFY_SKIP_PART3=True -> Part 3 dataset generation skipped by config.")
        else:
            tr_ok = dataset_png_is_complete(DataConfig.DATASET_ROOT, DataConfig.TRAIN_LABELS, min_coverage=0.999)
            va_ok = (not bool(getattr(DataConfig, "GENERATE_VAL_DIFF", True))) or dataset_png_is_complete(DataConfig.DATASET_ROOT, DataConfig.VAL_LABELS, min_coverage=0.999)
            if tr_ok and va_ok and (not bool(getattr(DataConfig, "OVERWRITE_DATASET", False))):
                print("[AutoRepair] Dataset already complete (CSV+PNG). Skip Part 3 regeneration.")
            else:
                if os.path.isfile(DataConfig.TRAIN_LABELS):
                    rep = dataset_png_report(DataConfig.DATASET_ROOT, DataConfig.TRAIN_LABELS, sample_rows=0)
                    if rep["csv_exists"] and rep["rows"] > 0 and (rep["missing_sims"] + rep["missing_diffs"]) > 0:
                        _ = repair_dataset_from_labels(
                            dataset_root=DataConfig.DATASET_ROOT,
                            labels_csv=DataConfig.TRAIN_LABELS,
                            real_folder=DataConfig.TRAIN_REAL_FOLDER,
                            real_rp_csv=DataConfig.TRAIN_REAL_RP_FILE,
                            apply_align=bool(DataConfig.APPLY_REAL_ALIGN),
                            apply_mask=bool(DataConfig.APPLY_CENTER_MASK),
                            mask_band_px=int(DataConfig.CENTER_MASK_BAND_PX),
                            mask_mode=str(DataConfig.CENTER_MASK_MODE),
                            n_jobs=int(DataConfig.N_JOBS),
                        )
                    else:
                        run_generator(df_train)
                else:
                    run_generator(df_train)
    except Exception as e:
        print(f"[AutoRepair] Data pipeline repair failed: {e!r}")

    # Ensure FFT cache is present/healthy
    try:
        FFTPrecomputer(config).run()
    except Exception as e:
        print(f"[AutoRepair] FFT cache repair failed: {e!r}")
# =========================================================
# 14. Main Loop
# =========================================================
def main(config: TrainConfig):
    # Best-effort auto-repair of required artifacts before any dataset checks.
    ensure_main_required_artifacts(config, attempt=1)
    dryrun_check(config)
    clean_runtime_artifacts(config)
    seed_everything(config.SEED)
    # Ensure that the output directory exists (passed by run_all_experiments.py through environment variables)
    os.makedirs(config.BASE_MODEL_SAVE_DIR, exist_ok=True)
    print(f"[Main] Model save directory: {config.BASE_MODEL_SAVE_DIR}")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    debug_root = os.path.join(config.BASE_MODEL_SAVE_DIR, "debug_vis")
    debug_run_dir = os.path.join(debug_root, f"run_{run_stamp}")
    os.makedirs(debug_run_dir, exist_ok=True)

    print("[Main] Initializing leakage-safe datasets...")

    train_ds = DiffSimRealDataset(
        dataset_root=config.DATASET_ROOT,
        real_root=config.TRAIN_REAL_FOLDER,
        labels_csv=config.TRAIN_LABELS_CSV,
        init_diameter_txt=config.INIT_DIAMETER_TXT,
        real_rp_csv=config.TRAIN_REAL_RP_CSV,
        is_train=True,
        config=config
    )
    if len(train_ds) == 0:
        raise RuntimeError("Train dataset empty.")

    if not os.path.exists(config.VAL_LABELS_CSV):
        raise FileNotFoundError(
            f"Missing VAL_LABELS_CSV: {config.VAL_LABELS_CSV}\n"
            f"Please generate val diffs/csv or implement on-the-fly val."
        )

    val_ds = DiffSimRealDataset(
        dataset_root=config.DATASET_ROOT,
        real_root=config.VAL_REAL_FOLDER,
        labels_csv=config.VAL_LABELS_CSV,
        init_diameter_txt=config.INIT_DIAMETER_TXT,
        real_rp_csv=config.VAL_REAL_RP_CSV,
        is_train=False,
        config=config
    )
    if len(val_ds) == 0:
        print("[AutoRepair] Val dataset empty. Trying to repair missing artifacts and rebuild cache...")
        ensure_main_required_artifacts(config, attempt=2)
        val_ds = DiffSimRealDataset(
            dataset_root=config.DATASET_ROOT,
            real_root=config.VAL_REAL_FOLDER,
            labels_csv=config.VAL_LABELS_CSV,
            init_diameter_txt=config.INIT_DIAMETER_TXT,
            real_rp_csv=config.VAL_REAL_RP_CSV,
            is_train=False,
            config=config
        )
        if len(val_ds) == 0:
            raise RuntimeError("Val dataset empty after auto-repair (check VAL_LABELS_CSV + FFT cache + VAL_REAL_FOLDER).")

    # global id space for refiner
    union_fnames = sorted(set(train_ds.unique_tgt_fnames) | set(val_ds.unique_tgt_fnames))
    rid_list = [_extract_real_id(fn) for fn in union_fnames]
    rid_to_gidx = {rid: i for i, rid in enumerate(sorted(set(rid_list)))}
    train_ds.real_id_to_global_idx = rid_to_gidx
    val_ds.real_id_to_global_idx = rid_to_gidx
    num_real_samples = len(rid_to_gidx)

    print(f"[Main] Train samples: {len(train_ds)} | Val samples: {len(val_ds)}")
    print(f"[Main] Unique real IDs (train+val union): {num_real_samples}")

    tr_sampler = GroupedBatchSampler(train_ds.grouped_by_real_indices, config.BATCH_SIZE, shuffle=True)
    train_loader = DataLoader(train_ds, batch_sampler=tr_sampler, num_workers=config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    model = DualStemNet(config).cuda().to(memory_format=torch.channels_last)
    physics = FraunhoferPhysicsLayer(
        config.ORIG_H, config.ORIG_W, config.CROP_PIXELS, config.RESIZE,
        config.PINN_WAVELENGTH_UM, config.PINN_FOCLENGTH_UM, config.PINN_PIXEL_SIZE_UM
    ).cuda()
    adv_phys_loss = AdvancedPhysicsLoss(config).cuda()

    opt = torch.optim.AdamW(
        model.parameters(),
        lr=config.LEARNING_RATE, weight_decay=config.WEIGHT_DECAY
    )

    if getattr(config, "USE_PCGRAD", False) and getattr(config, "USE_GRADNORM", False):
        raise ValueError("USE_PCGRAD and USE_GRADNORM are mutually exclusive")

    # NOTE: PCGrad/GradNorm do manual grad handling. By default we keep them in fp32
    # to avoid subtle AMP + custom-grad interactions.
    # Optional: allow forward-only autocast (reduces activation memory) while keeping
    # manual grad steps in fp32.
    use_custom_grad = bool(getattr(config, "USE_PCGRAD", False) or getattr(config, "USE_GRADNORM", False))
    force_amp_fwd = os.environ.get("FORCE_AMP_FWD", "0").strip() in {"1", "true", "True"}
    use_amp = bool(getattr(config, "ENABLE_AMP", True)) and (not use_custom_grad)
    use_amp_fwd_only = bool(getattr(config, "ENABLE_AMP", True)) and use_custom_grad and force_amp_fwd
    autocast_enabled = bool(use_amp or use_amp_fwd_only)

    run_variant = os.environ.get("RUN_VARIANT", "").strip()
    if not run_variant:
        if bool(getattr(config, "USE_GRADNORM", False)):
            run_variant = "gradnorm"
        elif bool(getattr(config, "USE_PCGRAD", False)):
            run_variant = "pcgrad"
        elif bool(getattr(config, "LOSS_WARMUP_ENABLE", False)):
            run_variant = "warmup"
        else:
            run_variant = "baseline"

    # Print policy configuration information
    strategy_info = []
    strategy_info.append(f"variant={run_variant}")
    if getattr(config, 'EARLY_PAIR_ONLY_EPOCHS', 0) > 0:
        strategy_info.append(f"early_pair_only={getattr(config, 'EARLY_PAIR_ONLY_EPOCHS', 0)}ep")
    if getattr(config, 'LOSS_GRADUAL_INTRO_ENABLE', False):
        intro_epochs = getattr(config, 'LOSS_GRADUAL_INTRO_EPOCHS', 0)
        intro_mode = getattr(config, 'LOSS_GRADUAL_INTRO_MODE', 'cosine')
        strategy_info.append(f"gradual_intro={intro_epochs}ep({intro_mode})")
    if getattr(config, 'LOSS_CONDITIONAL_ENABLE', False):
        threshold = getattr(config, 'LOSS_CONDITIONAL_PAIR_THRESHOLD', 0.03)
        strategy_info.append(f"conditional(threshold={threshold})")
    w_phys_reduced = getattr(config, 'W_PHYS_REDUCED', None)
    if w_phys_reduced is not None and float(w_phys_reduced) != 1.0:
        strategy_info.append(f"w_phys_reduced={w_phys_reduced}x")
    if getattr(config, 'LOSS_ADAPTIVE_PAIR_BOOST', False):
        strategy_info.append(f"adaptive_pair_boost")
    strategy_info.append(f"amp={'on' if use_amp else ('fwd' if use_amp_fwd_only else 'off')}")
    
    print("Strategy configuration | " + " | ".join(strategy_info))

    # Sensitivity mode: Load model weights from the baseline
    if config.SENSITIVITY_MODE:
        if not config.BASELINE_DIR or not os.path.exists(config.BASELINE_DIR):
            raise ValueError(f"The Sensitivity mode requires BASELINE_DIR, but the directory does not exist: {config.BASELINE_DIR}")
        
        baseline_model_path = os.path.join(config.BASELINE_DIR, "best_ema_model.pth")
        if not os.path.exists(baseline_model_path):
            baseline_model_path = os.path.join(config.BASELINE_DIR, "best_model.pth")
        
        if os.path.exists(baseline_model_path):
            print(f"[Sensitivity] Load the model from baseline: {baseline_model_path}")
            checkpoint = torch.load(baseline_model_path, map_location='cuda')
            model.load_state_dict(checkpoint, strict=False)
            print(f"[Sensitivity] Model loading completed")
            
            # Print the disturbance parameter
            if config.SENSITIVITY_COARSE_PERTURB != 0.0:
                print(f"[Sensitivity] Coarse diameter perturbation: {config.SENSITIVITY_COARSE_PERTURB*100:.1f}%")
            if config.SENSITIVITY_IMAGE_NOISE > 0.0:
                print(f"[Sensitivity] Image noise: std={config.SENSITIVITY_IMAGE_NOISE}")
            if config.SENSITIVITY_PARAMS_ZERO:
                print(f"[Sensitivity] parameter zeroing mode")
        else:
            raise FileNotFoundError(f"Baseline model file does not exist: {baseline_model_path}")

    sched = build_warmup_cosine_scheduler(opt, config.NUM_EPOCHS)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None
    ema_state = ema_init(model) if config.USE_EMA_TEACHER else None

    gradnorm = None
    if getattr(config, "USE_GRADNORM", False):
        tasks = [t for t in getattr(config, "GRADNORM_TASKS", []) if t]
        if not tasks:
            tasks = ["pair", "weak", "phys", "vic", "tta", "group"]
        gradnorm = GradNormBalancer(tasks, alpha=getattr(config, "GRADNORM_ALPHA", 0.5), lr=getattr(config, "GRADNORM_LR", 0.025))

    best_score = float('inf')
    best_ema_score = float('inf')
    metrics_history = []
    
    # Track pair loss history for conditional enable (based on stability)
    pair_loss_history = []  # List of recent pair loss values (for conditional enable)
    conditional_enable = getattr(config, "LOSS_CONDITIONAL_ENABLE", False)

    print(f"Start Training: {config.NUM_EPOCHS} Epochs")
    global_step = 0
    
    # Get pair-only epochs and other settings
    EARLY_PAIR_ONLY_EPOCHS = int(os.environ.get("EARLY_PAIR_ONLY_EPOCHS", "0"))

    for epoch in range(1, config.NUM_EPOCHS + 1):
        model.train()
        train_stats = defaultdict(float)
        ct = 0

        # smooth decay weights for stability
        w_vic = get_dynamic_weight(epoch, config.W_VICREG_START, config.W_VICREG_END, config.NUM_EPOCHS)
        w_tta = get_dynamic_weight(epoch, config.W_TTA_START, config.W_TTA_END, config.NUM_EPOCHS)
        w_grp = get_dynamic_weight(epoch, config.W_GROUP_START, config.W_GROUP_END, config.NUM_EPOCHS)

        epoch_start_time = time.time()
        if epoch == 1:
            if config.SENSITIVITY_MODE:
                print(f"[Sensitivity] Evaluation starts, with a total of {config.NUM_EPOCHS} epochs")
                if config.MAX_BATCHES > 0:
                    print(f"[Sensitivity] Limit each epoch to a maximum of {config.MAX_BATCHES} steps (batches), with batch_size={config.BATCH_SIZE}")
            else:
                print(f"Start training, with a total of {config.NUM_EPOCHS} epochs, training in progress...")
        
        batch_count = 0
        for fnames, full_input, sim_pt, rp, sp, coarse, target_a, is_paired, real_idx in train_loader:
            # Sensitivity mode: limit the number of batches
            if config.SENSITIVITY_MODE and config.MAX_BATCHES > 0:
                if batch_count >= config.MAX_BATCHES:
                    break
                batch_count += 1
            full_input = full_input.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            sim_pt = sim_pt.cuda(non_blocking=True)
            rp = rp.cuda(non_blocking=True)
            sp = sp.cuda(non_blocking=True)
            coarse = coarse.cuda().view(-1, 1)
            target_a = target_a.cuda().view(-1, 1)
            is_paired = is_paired.cuda().view(-1)
            real_idx = real_idx.cuda()
            
            # Sensitivity mode: Apply perturbation
            if config.SENSITIVITY_MODE:
                if config.SENSITIVITY_COARSE_PERTURB != 0.0:
                    coarse = coarse * (1.0 + config.SENSITIVITY_COARSE_PERTURB)
                if config.SENSITIVITY_IMAGE_NOISE > 0.0:
                    noise = torch.randn_like(full_input) * config.SENSITIVITY_IMAGE_NOISE
                    full_input = full_input + noise
                if config.SENSITIVITY_PARAMS_ZERO:
                    rp = torch.zeros_like(rp)
                    sp = torch.zeros_like(sp)

            with torch.amp.autocast('cuda', enabled=autocast_enabled):
                # ---- Parameter pipeline ----
                # 1) Physics-side: before any Fraunhofer usage, ensure PHYSICAL units (denorm if needed)
                rp_phys = ensure_params5_phys_for_physics(rp, ctx="train/rp_for_phys")
                _ = ensure_params5_phys_for_physics(sp,ctx="train/sp_for_phys")  # sp checked symmetrically

                # 2) NN-side: condition inputs must be normalized to [-1,1] (detect by range, not by bounds)
                rp_n = normalize_params5(rp)
                sp_n = normalize_params5(sp).to(rp_n.device)
                params = torch.cat([rp_n, sp_n], dim=1)

                # branch inputs
                x_real4 = torch.cat([full_input[:, 0:2], full_input[:, 4:6]], dim=1)  # real+diff (4ch)
                x_sim2 = sim_pt  # sim (2ch)

                d_um, log_d, z_real, stem_real_pool = model.forward_real(x_real4, params, coarse, return_stem_pool=True)

                # Determine if we should enable other losses based on pair-only mode and conditional enable
                # Early epoch pair-only mode: completely disable other losses in first few epochs
                # This prevents phys/weak from interfering with pair learning when pair loss is already low
                early_pair_only = EARLY_PAIR_ONLY_EPOCHS > 0 and epoch <= EARLY_PAIR_ONLY_EPOCHS
                
                # Get loss weight scales (with pair loss history for conditional enable)
                scales = get_loss_weight_scales(config, epoch, pair_only_epochs=EARLY_PAIR_ONLY_EPOCHS, 
                                                 pair_loss_history=pair_loss_history, conditional_enable=conditional_enable)
                
                # Determine if other losses should be enabled based on scales
                # If scales['phys']/['weak']/['vic'] are 0, then disable those losses
                enable_other_losses = (scales.get('phys', 1.0) > 1e-6 or scales.get('weak', 1.0) > 1e-6 or 
                                      scales.get('vic', 1.0) > 1e-6)
                
                # PairCons only for paired samples
                if config.ENABLE_PAIRCONS:
                    L_pair = pair_consistency_loss_logd(log_d, target_a.squeeze(1), is_paired)
                else:
                    L_pair = torch.tensor(0.0, device='cuda', requires_grad=True)

                # Weak guidance (continuous + relative)
                L_weak = torch.tensor(0.0, device='cuda')
                if config.ENABLE_WEAK and enable_other_losses and scales.get('weak', 0.0) > 1e-6:
                    L_weak = weak_guidance_rel(d_um, coarse, config.WEAK_TOLERANCE)

                # Physics (normalized + low-freq band)
                # Use reduced phys weight for bad seeds if configured
                w_phys_base = float(getattr(config, "W_PHYS_REDUCED", config.W_PHYS))
                L_phys = torch.tensor(0.0, device='cuda')
                l_corr = l_fft = l_grad = torch.tensor(0.0, device='cuda')
                if config.ENABLE_PHYS and enable_other_losses and scales.get('phys', 0.0) > 1e-6:
                    img_pred = physics(d_um, rp_phys, target_real_img=full_input[:, 0:1])
                    L_phys, l_corr, l_fft, l_grad = adv_phys_loss(img_pred, full_input[:, 0:1])

                # VICReg alignment (anti-collapse)
                L_vic = torch.tensor(0.0, device='cuda')
                vic_inv = torch.tensor(0.0, device='cuda')
                vic_var = torch.tensor(0.0, device='cuda')
                vic_cov = torch.tensor(0.0, device='cuda')
                if config.ENABLE_VICREG and w_vic > 1e-6 and enable_other_losses and scales.get('vic', 0.0) > 1e-6:
                    z_sim, stem_sim_pool = model.forward_sim(x_sim2, params, coarse, return_stem_pool=True)
                    # VICReg on pooled stem (adapter) features
                    L_vic, vic_inv, vic_var, vic_cov = vicreg_loss(
                    stem_real_pool.float(), stem_sim_pool.float(),
                    sim_coeff=config.W_VICREG_INV,
                    var_coeff=config.W_VICREG_VAR,
                    cov_coeff=config.W_VICREG_COV,
                    return_components=True
                )

                # TTA / Group on embedding only, with decay
                L_tta = torch.tensor(0.0, device='cuda')
                if config.ENABLE_TTA_EMB and config.TTA_NUM_TRAIN > 1 and w_tta > 1e-6 and enable_other_losses and scales.get('tta', 0.0) > 1e-6:
                    zs = []
                    for aug in tta_noise_only(x_real4, config.TTA_NUM_TRAIN, noise_level=config.TTA_NOISE_TRAIN):
                        _, _, z_aug = model.forward_real(aug, params, coarse)
                        zs.append(z_aug)
                    z_stack = torch.stack(zs)  # (T,B,D)
                    L_tta = z_stack.var(dim=0, unbiased=False).mean()

                L_group = torch.tensor(0.0, device='cuda')
                if config.ENABLE_GROUP_EMB and w_grp > 1e-6 and enable_other_losses and scales.get('group', 0.0) > 1e-6:
                    rids = torch.tensor([_extract_real_id(n) for n in fnames], device='cuda')
                    unique_ids, counts = rids.unique(return_counts=True)
                    denom = 0.0
                    for uid in unique_ids[counts > 1]:
                        zz = z_real[rids == uid]
                        L_group = L_group + zz.var(unbiased=False)
                        denom += 1.0
                    L_group = L_group / (denom + 1e-6)

                # Calculate effective weights (with reduced phys weight for bad seeds)
                w_pair_base = float(config.W_PAIR) * float(scales.get('pair', 1.0))
                w_weak_eff = float(config.W_WEAK) * float(scales.get('weak', 1.0))
                w_phys_eff = float(w_phys_base) * float(scales.get('phys', 1.0))  # Use reduced phys weight
                w_vic_eff = float(w_vic) * float(scales.get('vic', 1.0))
                w_tta_eff = float(w_tta) * float(scales.get('tta', 1.0))
                w_grp_eff = float(w_grp) * float(scales.get('group', 1.0))
                
                # Adaptive pair boost: automatically increase pair weight when pairloss is high
                # This helps bad seeds without using GT information
                w_pair_eff = w_pair_base
                if getattr(config, "LOSS_ADAPTIVE_PAIR_BOOST", False):
                    pair_val = float(L_pair.item()) if isinstance(L_pair, torch.Tensor) else 0.0
                    threshold = float(getattr(config, "LOSS_ADAPTIVE_PAIR_THRESHOLD", 0.03))
                    max_boost = float(getattr(config, "LOSS_ADAPTIVE_PAIR_MAX_BOOST", 2.0))
                    if pair_val > threshold:
                        # Boost factor: 1.0 + (pair_val - threshold) / threshold * (max_boost - 1.0)
                        # Clamped to [1.0, max_boost]
                        boost = 1.0 + min((pair_val - threshold) / max(threshold, 1e-6) * (max_boost - 1.0), max_boost - 1.0)
                        w_pair_eff = w_pair_base * float(boost)

                # total loss
                # If early_pair_only or scales disable other losses, only use pair loss
                if early_pair_only or not enable_other_losses:
                    # Pair-only mode: only pair loss
                    # Skip batch if no paired samples (pair-only mode needs paired samples)
                    if is_paired.sum() == 0:
                        # No paired samples: skip this batch by using zero loss
                        loss = torch.tensor(0.0, device='cuda', requires_grad=False)
                    else:
                        # Ensure L_pair has gradient
                        if not L_pair.requires_grad:
                            # Recalculate using log_d to ensure gradient
                            log_a = torch.log(target_a[is_paired.bool()].clamp_min(1e-6))
                            diff = (log_d.squeeze(1)[is_paired.bool()] - log_a).abs()
                            L_pair = diff.mean() if diff.numel() > 0 else torch.tensor(0.0, device='cuda', requires_grad=True)
                        loss = w_pair_eff * L_pair
                else:
                    # Normal mode: all losses
                    loss = (
                        w_pair_eff * L_pair
                        + w_weak_eff * L_weak
                        + w_phys_eff * L_phys
                        + w_vic_eff * L_vic
                        + w_tta_eff * L_tta
                        + w_grp_eff * L_group
                    )

            opt.zero_grad(set_to_none=True)
            # Skip backward if loss has no gradient (e.g., no paired samples in pair-only mode)
            # Sensitivity mode: Skip training and only perform evaluation
            skip_backward = config.SENSITIVITY_MODE or ((early_pair_only or not enable_other_losses) and (is_paired.sum() == 0 or not loss.requires_grad or (isinstance(loss, torch.Tensor) and loss.item() == 0.0 and not loss.requires_grad)))
            
            if not skip_backward:
                if use_amp:
                    assert scaler is not None
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    params = [p for p in model.parameters() if p.requires_grad]
                    if getattr(config, "USE_PCGRAD", False):
                        allowed = set(getattr(config, "PCGRAD_TASKS", []) or [])
                        use_subset = bool(allowed)
                        task_losses = []
                        if (not use_subset or 'pair' in allowed) and w_pair_eff > 0 and config.ENABLE_PAIRCONS:
                            task_losses.append((w_pair_eff * L_pair).float())
                        if (not use_subset or 'weak' in allowed) and w_weak_eff > 0 and config.ENABLE_WEAK:
                            task_losses.append((w_weak_eff * L_weak).float())
                        if (not use_subset or 'phys' in allowed) and w_phys_eff > 0 and config.ENABLE_PHYS:
                            task_losses.append((w_phys_eff * L_phys).float())
                        if (not use_subset or 'vic' in allowed) and w_vic_eff > 0 and config.ENABLE_VICREG:
                            task_losses.append((w_vic_eff * L_vic).float())
                        if (not use_subset or 'tta' in allowed) and w_tta_eff > 0 and config.ENABLE_TTA_EMB:
                            task_losses.append((w_tta_eff * L_tta).float())
                        if (not use_subset or 'group' in allowed) and w_grp_eff > 0 and config.ENABLE_GROUP_EMB:
                            task_losses.append((w_grp_eff * L_group).float())
                        if len(task_losses) == 0:
                            loss.backward()
                            opt.step()
                        else:
                            pcgrad_backward(task_losses, params)
                            opt.step()
                    elif getattr(config, "USE_GRADNORM", False) and gradnorm is not None:
                        # Collect enabled task losses
                        task_dict: Dict[str, torch.Tensor] = {}
                        if 'pair' in gradnorm.task_names and config.ENABLE_PAIRCONS:
                            task_dict['pair'] = (float(scales.get('pair', 1.0)) * L_pair).float()
                        if 'weak' in gradnorm.task_names and config.ENABLE_WEAK:
                            task_dict['weak'] = (float(scales.get('weak', 1.0)) * L_weak).float()
                        if 'phys' in gradnorm.task_names and config.ENABLE_PHYS:
                            task_dict['phys'] = (float(scales.get('phys', 1.0)) * L_phys).float()
                        if 'vic' in gradnorm.task_names and config.ENABLE_VICREG:
                            task_dict['vic'] = (float(scales.get('vic', 1.0)) * L_vic).float()
                        if 'tta' in gradnorm.task_names and config.ENABLE_TTA_EMB:
                            task_dict['tta'] = (float(scales.get('tta', 1.0)) * L_tta).float()
                        if 'group' in gradnorm.task_names and config.ENABLE_GROUP_EMB:
                            task_dict['group'] = (float(scales.get('group', 1.0)) * L_group).float()

                        # Ensure all configured tasks exist (fill zeros)
                        for name in gradnorm.task_names:
                            if name not in task_dict:
                                task_dict[name] = torch.tensor(0.0, device='cuda')

                        w_info = gradnorm.step(task_dict, params)

                        # Model update with detached weights (no grad to weights)
                        loss_gn = 0.0
                        for i, name in enumerate(gradnorm.task_names):
                            wi = float(gradnorm.weights[i].detach().item())
                            loss_gn = loss_gn + wi * task_dict[name]
                        loss_gn.backward()
                        opt.step()

                        # store a tiny bit of info for debugging
                        train_stats['gn_w_mean'] += float(np.mean([v for v in w_info.values()]))
                    else:
                        loss.backward()
                        opt.step()

            if config.USE_EMA_TEACHER and ema_state is not None:
                ema_update(model, ema_state, config.EMA_DECAY)

            train_stats['loss'] += float(loss.item())
            train_stats['pair'] += float(L_pair.item())
            train_stats['weak'] += float(L_weak.item())
            train_stats['phys'] += float(L_phys.item())
            train_stats['corr'] += float(l_corr.item())
            train_stats['fft']  += float(l_fft.item())
            train_stats['grad'] += float(l_grad.item())
            train_stats['vic']  += float(L_vic.item())
            train_stats['vic_inv'] += float(vic_inv.item())
            train_stats['vic_var'] += float(vic_var.item())
            train_stats['vic_cov'] += float(vic_cov.item())
            train_stats['tta']  += float(L_tta.item())
            train_stats['group'] += float(L_group.item())

            global_step += 1
            ct += 1
        
        sched.step()

        # validation (student)
        current_weights = model.state_dict()
        val_score, val_metrics = validate_unified(
            model, physics, val_loader, config, adv_phys_loss
        )

        # validation (EMA teacher)
        val_ema_score = float('nan')
        val_ema_metrics = {}
        if config.USE_EMA_TEACHER and ema_state is not None:
            model.load_state_dict(ema_state, strict=False)
            val_ema_score, val_ema_metrics = validate_unified(
                model, physics, val_loader, config, adv_phys_loss
            )
            model.load_state_dict(current_weights)

        avg_train = {k: v / max(1, ct) for k, v in train_stats.items()}
        epoch_time = time.time() - epoch_start_time
        
        # Update pair loss history for conditional enable (use validation pair loss)
        if conditional_enable:
            val_pair_loss = float(val_metrics.get('val_pair', float('nan')))
            if not math.isnan(val_pair_loss):
                pair_loss_history.append(val_pair_loss)
                # Keep only recent history (last N epochs for stability check)
                stable_epochs = int(getattr(config, "LOSS_CONDITIONAL_STABLE_EPOCHS", 2))
                max_history = stable_epochs + 5  # Keep a bit more for smoothing
                if len(pair_loss_history) > max_history:
                    pair_loss_history = pair_loss_history[-max_history:]

        gt_d = float(getattr(config, 'GROUND_TRUTH_DIAMETER', float('nan')))
        pred_mean = float(val_metrics.get('val_pred_mean', float('nan')))
        pred_bias = (pred_mean - gt_d) if (not math.isnan(pred_mean) and not math.isnan(gt_d)) else float('nan')
        ema_pred_mean = float(val_ema_metrics.get('val_pred_mean', float('nan')))
        ema_pred_bias = (ema_pred_mean - gt_d) if (not math.isnan(ema_pred_mean) and not math.isnan(gt_d)) else float('nan')
        
        print(
            f"Ep {epoch:3d}/{config.NUM_EPOCHS} | "
            f"Loss={avg_train.get('loss',0.0):.4f} | "
            f"Val_err={val_metrics.get('val_err',0.0):.4f} | "
            f"PredMean={pred_mean:.2f} Bias={pred_bias:+.2f} | "
            f"Score={val_score:.4f} | "
            f"EMA_err={val_ema_metrics.get('val_err',float('nan')):.4f} | "
            f"EMA_PredMean={ema_pred_mean:.2f} EMA_Bias={ema_pred_bias:+.2f} | "
            f"EMA_score={val_ema_score:.4f} | "
            f"Time={epoch_time:.1f}s"
        )
        is_best_model = 0
        if not math.isnan(val_score) and val_score < best_score:
            best_score = val_score
            is_best_model = 1
            print(f"  >> New Best Model! (Score: {best_score:.4f})")
            torch.save(model.state_dict(), os.path.join(config.BASE_MODEL_SAVE_DIR, "best_model.pth"))

        is_best_ema = 0
        if not math.isnan(val_ema_score) and val_ema_score < best_ema_score:
            best_ema_score = val_ema_score
            is_best_ema = 1
            print(f"  >> New Best EMA! (Score: {best_ema_score:.4f})")
            torch.save(ema_state, os.path.join(config.BASE_MODEL_SAVE_DIR, "best_ema_model.pth"))

        metrics = {
            "epoch": epoch,

            "train_loss_total": avg_train['loss'],
            "train_pair": avg_train['pair'],
            "train_weak": avg_train['weak'],
            "train_phys": avg_train['phys'],
            "train_sub_corr": avg_train['corr'],
            "train_sub_fft": avg_train['fft'],
            "train_sub_grad": avg_train['grad'],
            "train_vic": avg_train['vic'],
            "train_vic_var": avg_train['vic_var'],
            "train_vic_inv": avg_train.get('vic_inv', 0.0),
            "train_vic_cov": avg_train.get('vic_cov', 0.0),
            "train_tta": avg_train['tta'],
            "train_group": avg_train['group'],

            "val_err_obs": val_metrics['val_err'],
            "val_pred_mean": val_metrics.get('val_pred_mean', 0.0),
            "val_pred_std": val_metrics.get('val_pred_std', 0.0),
            "val_score": val_score,
            "val_phys": val_metrics['val_phys'],
            "val_sub_corr": val_metrics.get('val_sub_corr', 0.0),
            "val_sub_fft": val_metrics.get('val_sub_fft', 0.0),
            "val_sub_grad": val_metrics.get('val_sub_grad', 0.0),
            "val_weak": val_metrics['val_weak'],
            "val_pair": val_metrics['val_pair'],
            "val_vicvar": val_metrics['val_vicvar'],
            "val_vicinv": val_metrics.get('val_vicinv', 0.0),
            "val_viccov": val_metrics.get('val_viccov', 0.0),
            "val_group_var": val_metrics['val_group'],

            "val_ema_err_obs": val_ema_metrics.get('val_err', float('nan')),
            "val_ema_score": val_ema_score,
            "is_best_model": is_best_model,
            "is_best_ema": is_best_ema
        }
        metrics_history.append(metrics)
        pd.DataFrame(metrics_history).to_csv(os.path.join(config.BASE_MODEL_SAVE_DIR, "metrics.csv"), index=False)

        save_debug_visualization(model, physics, val_loader, config, epoch, debug_run_dir)
        
    print("Training Finished.")


# ============================================================
# Entry: run data pipeline first, then training pipeline
# ============================================================
def run_data_then_train():
    """
    Run order:
      1) Data pipeline (prepare train/val split, estimate coarse diameter, optimize rp tables,
         generate simulation dataset + labels csv)  [from data.py]
      2) Training pipeline (FFT cache -> train loop) [from main.py]
    """
    # ---------- Part A: data pipeline ----------
    multiprocessing.freeze_support()
    try:
        validate_real_folders()
        maybe_auto_split_to_folders()
        validate_real_folders()

        if not os.path.isdir(DataConfig.TRAIN_REAL_FOLDER):
            raise FileNotFoundError(f"Missing TRAIN_REAL_FOLDER: {DataConfig.TRAIN_REAL_FOLDER}")
        if not os.path.isdir(DataConfig.VAL_REAL_FOLDER):
            print(f"[WARN] VAL_REAL_FOLDER not found: {DataConfig.VAL_REAL_FOLDER} (val diffs will be skipped)")

        init_a = estimate_initial_diameter()
        df_train = run_optimization(init_a)
        export_real_rp_tables(df_train, init_a)
        # Part 3: generate dataset ONLY if missing/incomplete or overwrite requested
        if DataConfig.FAST_VERIFY and bool(DataConfig.FAST_VERIFY_SKIP_PART3):
            print("[FAST_VERIFY] FAST_VERIFY_SKIP_PART3=True -> skip Part 3 dataset generation.")
        else:
            tr_ok = dataset_png_is_complete(DataConfig.DATASET_ROOT, DataConfig.TRAIN_LABELS, min_coverage=0.999)
            va_ok = (not bool(getattr(DataConfig, "GENERATE_VAL_DIFF", True))) or dataset_png_is_complete(DataConfig.DATASET_ROOT, DataConfig.VAL_LABELS, min_coverage=0.999)
            if tr_ok and va_ok and (not bool(getattr(DataConfig, "OVERWRITE_DATASET", False))):
                print("[Data] Dataset already complete (CSV+PNG). Skip Part 3 generation.")
            else:
                if os.path.isfile(DataConfig.TRAIN_LABELS):
                    rep = dataset_png_report(DataConfig.DATASET_ROOT, DataConfig.TRAIN_LABELS, sample_rows=0)
                    if rep["csv_exists"] and rep["rows"] > 0 and (rep["missing_sims"] + rep["missing_diffs"]) > 0:
                        _ = repair_dataset_from_labels(
                            dataset_root=DataConfig.DATASET_ROOT,
                            labels_csv=DataConfig.TRAIN_LABELS,
                            real_folder=DataConfig.TRAIN_REAL_FOLDER,
                            real_rp_csv=DataConfig.TRAIN_REAL_RP_FILE,
                            apply_align=bool(DataConfig.APPLY_REAL_ALIGN),
                            apply_mask=bool(DataConfig.APPLY_CENTER_MASK),
                            mask_band_px=int(DataConfig.CENTER_MASK_BAND_PX),
                            mask_mode=str(DataConfig.CENTER_MASK_MODE),
                            n_jobs=int(DataConfig.N_JOBS),
                        )
                    else:
                        run_generator(df_train)
                else:
                    run_generator(df_train)

        print("\n[Data] Pipeline Complete (Leakage-Safe).")
    except Exception as e:
        print(f"[Data] ERROR: {e}")
        import traceback
        traceback.print_exc()
        raise

    # ---------- Part B: training pipeline ----------
    if os.name == 'nt':
        torch.multiprocessing.freeze_support()

    torch.set_float32_matmul_precision('medium')
    torch.backends.cudnn.benchmark = True

    cfg = TrainConfig()
    FFTPrecomputer(cfg).run()
    main(cfg)

def export_predictions(config, ckpt_path: str, split: str, out_csv: str, use_ema_state: bool = False):
    model = DualStemNet(config).cuda()
    model.eval()

    if use_ema_state:
        state = torch.load(ckpt_path, map_location='cpu') 
        if isinstance(state, dict) and 'model_ema' in state:
            state = state['model_ema']
        elif isinstance(state, dict) and 'ema_model' in state:
            state = state['ema_model']
        elif isinstance(state, dict) and 'model' in state:
            state = state['model']
        model.load_state_dict(state, strict=True)
    else:
        state = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(state, strict=True)

    if split == 'train':
        labels_csv = config.TRAIN_LABELS_CSV
        real_root = config.TRAIN_REAL_FOLDER
        real_rp_csv = config.TRAIN_REAL_RP_CSV
    elif split == 'val':
        labels_csv = config.VAL_LABELS_CSV
        real_root = config.VAL_REAL_FOLDER
        real_rp_csv = config.VAL_REAL_RP_CSV
    else:
        raise ValueError("split must be 'train' or 'val'")

    ds = DiffSimRealDataset(
        dataset_root=config.DATASET_ROOT,
        real_root=real_root,
        labels_csv=labels_csv,
        init_diameter_txt=config.INIT_DIAMETER_TXT,
        real_rp_csv=real_rp_csv,
        crop_pixels=config.CROP_PIXELS,
        resize=config.RESIZE,
        coarse_noise_std=0.0,
        is_train=False,
        config=config,
    )

    loader = DataLoader(ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS, pin_memory=True)

    rows = []
    from tqdm import tqdm
    with torch.no_grad():
        for fnames, full_input, sim_pt, rp, sp, coarse, target_a, is_paired, real_idx in tqdm(loader, desc=f"Predict {split}", ncols=80):
            full_input = full_input.cuda(non_blocking=True).to(memory_format=torch.channels_last)
            rp = rp.cuda(non_blocking=True)
            sp = sp.cuda(non_blocking=True)
            coarse = coarse.cuda().view(-1, 1)

            rp_n = normalize_params5(rp)
            sp_n = normalize_params5(sp).to(rp_n.device)
            params = torch.cat([rp_n, sp_n], dim=1)
            x_real4 = torch.cat([full_input[:, 0:2], full_input[:, 4:6]], dim=1)

            d_preds = []
            with torch.amp.autocast('cuda'):
                for aug in tta_noise_only(x_real4, config.TTA_NUM_VAL, noise_level=config.TTA_NOISE_VAL):
                    d_um, _, _, _ = model.forward_real(aug, params, coarse, return_stem_pool=True)
                    d_preds.append(d_um)
            d_final = torch.stack(d_preds).mean(0).detach().view(-1).cpu().numpy().tolist()

            for f, d in zip(fnames, d_final):
                rows.append({"split": split, "fname": str(f), "real_id": int(_extract_real_id(str(f))), "pred_um": float(d)})

    pd.DataFrame(rows).to_csv(out_csv, index=False)

def run_export_predictions():
    cfg = TrainConfig()
    FFTPrecomputer(cfg).run()

    base_dir = cfg.BASE_MODEL_SAVE_DIR
    ckpt_type = os.environ.get('EXPORT_CKPT', 'best').lower()
    if ckpt_type == 'best':
        ckpt_path = os.path.join(base_dir, 'best_model.pth')
        use_ema = False
        ckpt_tag = 'best'
    elif ckpt_type in ['ema', 'best_ema']:
        ckpt_path = os.path.join(base_dir, 'best_ema_model.pth')
        use_ema = True
        ckpt_tag = 'ema'
    else:
        raise ValueError("EXPORT_CKPT must be 'best' or 'ema'")

    out_root = os.environ.get('EXPORT_OUT_DIR', os.path.join(base_dir, 'pred_exports'))
    os.makedirs(out_root, exist_ok=True)

    seed = os.environ.get('SEED', 'unknown')

    export_predictions(cfg, ckpt_path, 'train', os.path.join(out_root, f'pred_seed{seed}_{ckpt_tag}_train.csv'), use_ema_state=use_ema)
    export_predictions(cfg, ckpt_path, 'val', os.path.join(out_root, f'pred_seed{seed}_{ckpt_tag}_val.csv'), use_ema_state=use_ema)

if __name__ == "__main__":
    mode = os.environ.get('MODE', 'train').lower()
    if mode == 'export_preds':
        print("[INFO] Running in EXPORT_PREDS mode. Training will be skipped.")
        run_export_predictions()
        import sys
        sys.exit(0)

    # Hard safety guard for training mode
    allow_train = os.environ.get('ALLOW_TRAIN', '0').strip().lower() in ('1', 'true', 'yes', 'y')
    if not allow_train:
        raise RuntimeError(
            "Training is disabled by default to prevent accidental GPU cost. "
            "Set ALLOW_TRAIN=1 to enable training, or set MODE=export_preds to export predictions."
        )
    print("[INFO] Running in TRAIN mode (ALLOW_TRAIN=1).")

    # --- Run environment snapshot (for audit / anti-silent-reuse) ---
    try:
        base_dir = TrainConfig.BASE_MODEL_SAVE_DIR
        os.makedirs(base_dir, exist_ok=True)

        def _sha256_file(path: str) -> str:
            h = hashlib.sha256()
            with open(path, 'rb') as f:
                for b in iter(lambda: f.read(1024 * 1024), b''):
                    h.update(b)
            return h.hexdigest()

        snapshot_env_keys = [
            'SEED', 'MODE', 'ALLOW_TRAIN', 'BASE_MODEL_SAVE_DIR', 'NUM_EPOCHS',
            'USE_DIFF_INPUT', 'USE_FFT_INPUT',
            'EARLY_PAIR_ONLY_EPOCHS',
            'LOSS_GRADUAL_INTRO_ENABLE', 'LOSS_GRADUAL_INTRO_EPOCHS', 'LOSS_GRADUAL_INTRO_MODE', 'LOSS_GRADUAL_INTRO_TARGETS',
            'LOSS_CONDITIONAL_ENABLE', 'LOSS_CONDITIONAL_PAIR_THRESHOLD', 'LOSS_CONDITIONAL_STABLE_EPOCHS',
            'W_PHYS_REDUCED', 'LOSS_ADAPTIVE_PAIR_BOOST', 'LOSS_ADAPTIVE_PAIR_THRESHOLD', 'LOSS_ADAPTIVE_PAIR_MAX_BOOST',
            'SIM_SAMPLE_COUNT',
        ]

        snapshot = {
            'created_at': time.strftime('%Y-%m-%d %H:%M:%S'),
            'host': socket.gethostname(),
            'user': getpass.getuser(),
            'cwd': os.getcwd(),
            'python': sys.executable if 'sys' in globals() else None,
            'env': {k: os.environ.get(k) for k in snapshot_env_keys if k in os.environ},
            'train_config': {
                'BASE_MODEL_SAVE_DIR': getattr(TrainConfig, 'BASE_MODEL_SAVE_DIR', None),
                'INPUT_CHANNELS_FULL': getattr(TrainConfig, 'INPUT_CHANNELS_FULL', None),
                'REAL_BRANCH_CH': getattr(TrainConfig, 'REAL_BRANCH_CH', None),
                'SIM_BRANCH_CH': getattr(TrainConfig, 'SIM_BRANCH_CH', None),
                'BACKBONE_NAME': getattr(TrainConfig, 'BACKBONE_NAME', None),
            },
        }

        # If a previous best_model exists, record its hash to detect accidental carry-over.
        best_model_path = os.path.join(base_dir, 'best_model.pth')
        best_ema_path = os.path.join(base_dir, 'best_ema_model.pth')
        if os.path.exists(best_model_path):
            snapshot['preexisting_best_model_sha256'] = _sha256_file(best_model_path)
        if os.path.exists(best_ema_path):
            snapshot['preexisting_best_ema_model_sha256'] = _sha256_file(best_ema_path)

        with open(os.path.join(base_dir, 'run_env_snapshot.json'), 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2)

        # For compatibility with experiment scripts that expect a config snapshot.
        with open(os.path.join(base_dir, 'config_snapshot.json'), 'w', encoding='utf-8') as f:
            json.dump(snapshot, f, indent=2)
    except Exception as e:
        print(f"[WARN] Failed to write run_env_snapshot.json: {e}")
    run_data_then_train()

