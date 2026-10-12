"""
Nova AI V4 — Decoder-Only Character-Level Transformer
=====================================================
Target parameter count: ~200,000 trainable parameters (199,016 at T=32; 201,064 at T=64)
Corpus: 5 diverse datasets (~326,594 characters, 1,183 paragraphs)
Vocabulary: 104 distinct characters

Key Specifications:
  • Architecture: Decoder-Only Causal Transformer
  • Normalization: Pre-LayerNorm (LayerNorm before Attention and FFN, plus final LayerNorm)
  • Regularization: Decoupled weight decay (AdamW, lambda=0.005); NO dropout (DROPOUT=0.0)
  • Vocabulary: 104 distinct characters
  • Default Context length (T): 32 characters (configurable to 64)
  • Model dimension (d_model): 64
  • Transformer layers (L): 4
  • Attention heads (H): 4 (head dimension = 16)
  • Feed-forward hidden dim (d_ff): 224
  • Activation: ReLU in feed-forward sublayers
  • Parameter Count:
      - At T=32: Exactly 199,016 trainable parameters
      - At T=64: Exactly 201,064 trainable parameters

Training Recipe:
  • Optimizer: In-place AdamW (beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.005)
  • Batch size: 64 (configurable)
  • Learning rate schedule: Cosine decay with 100-epoch horizon (initial=0.001, min=1e-5)
  • Dataset split: Representative paragraph-level 90/10 split
  • Window construction: Disjoint sets with zero cross-split leakage
"""

import os
import sys
import msvcrt  # Windows console input for Ctrl+T detection

# Configure OpenBLAS / MKL / OMP threads before NumPy initialization (defaults to 4 physical cores)
for _env in ("OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "OMP_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    if _env not in os.environ:
        os.environ[_env] = "4"

import time
import math
import json
import shutil
import argparse
import ctypes
from datetime import datetime

try:
    import psutil
except ImportError:
    psutil = None

import numpy as np

# Ensure UTF-8 stdout encoding on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# Live inline console updates: never buffer stdout when redirected.
# Training status must be visible immediately in the terminal, not at exit.
try:
    sys.stdout.reconfigure(line_buffering=True)
    sys.stderr.reconfigure(line_buffering=True)
except Exception:
    pass

# Terminal display detection
IN_IDLE = "idlelib" in sys.modules
IS_TTY = sys.stdout.isatty()

# ====================================================================
#  BLAS BACKEND & THREADING DIAGNOSTICS
# ====================================================================
def get_blas_info():
    """Detect and return BLAS implementation name and verified active thread count."""
    info = {"implementation": "scipy-openblas", "threads": None}
    if psutil is None:
        info["threads"] = int(os.environ.get("OPENBLAS_NUM_THREADS", "4"))
        return info
    try:
        maps = psutil.Process().memory_maps()
        openblas_dlls = [m.path for m in maps if "openblas" in m.path.lower()]
        if openblas_dlls:
            dll = ctypes.CDLL(openblas_dlls[0])
            info["implementation"] = f"OpenBLAS ({os.path.basename(openblas_dlls[0])})"
            for func_name in ("scipy_openblas_get_num_threads64_", "openblas_get_num_threads64_", "openblas_get_num_threads"):
                func = getattr(dll, func_name, None)
                if func:
                    info["threads"] = func()
                    break
    except Exception as e:
        info["error"] = str(e)
    if info["threads"] is None:
        info["threads"] = int(os.environ.get("OPENBLAS_NUM_THREADS", "4"))
    return info


def set_blas_threads(n):
    """Enforce BLAS thread count directly via library C symbols and environment."""
    os.environ["OPENBLAS_NUM_THREADS"] = str(n)
    os.environ["MKL_NUM_THREADS"] = str(n)
    os.environ["OMP_NUM_THREADS"] = str(n)
    if psutil is None:
        return
    try:
        maps = psutil.Process().memory_maps()
        openblas_dlls = [m.path for m in maps if "openblas" in m.path.lower()]
        if openblas_dlls:
            dll = ctypes.CDLL(openblas_dlls[0])
            for func_name in ("scipy_openblas_set_num_threads64_", "openblas_set_num_threads64_", "openblas_set_num_threads"):
                func = getattr(dll, func_name, None)
                if func:
                    func(int(n))
                    break
    except Exception:
        pass

# Enforce 4 threads (matching 4 physical cores)
set_blas_threads(int(os.environ.get("OPENBLAS_NUM_THREADS", "4")))

# ====================================================================
#  USER CONFIGURATION
# ====================================================================
# Edit these values directly to change training behavior.
# Run `python nova_v4_fast.py --train` to use the interactive config prompt.
# Command-line arguments (--epochs, --batch-size, etc.) will override these defaults.
# ====================================================================

# --------------------------------------------------------------------------
# TRAINING SETTINGS (safe to edit)
# --------------------------------------------------------------------------
EPOCHS = 1000                    # Number of training epochs (default: 3)
BATCH_SIZE = 64              # Mini-batch size (default: 64)
DISPLAY_EVERY = 1            # Show progress every N steps (default: live every step)
VALIDATION_EVERY = 1         # Run validation every N epochs (default: 1 = every epoch)

# --------------------------------------------------------------------------
# GENERATION DEFAULTS (safe to edit)
# --------------------------------------------------------------------------
GEN_MAX_CHARS = 140          # Default max characters for generation (default: 140)
GEN_TEMPERATURE = 0.8        # Default temperature for sampling (default: 0.8)

# --------------------------------------------------------------------------
# COMPUTE SETTINGS (advanced, safe to edit)
# --------------------------------------------------------------------------
PRUNE_FINAL_BLOCK = True     # Bit-exact gradient parity vs unpruned math verified
                             # (max grad diff 0.0, probs diff 0.0). Skips dead
                             # positions in the final block: K/V computed for all T
                             # (consumed by the last-position query), but Query,
                             # attention output, LayerNorm-2 and FFN evaluated ONLY
                             # at position T-1. Set False to use the classic path.

# ====================================================================
#  CORE CONFIGURATION & HYPERPARAMETERS (DO NOT EDIT)
# ====================================================================
# These settings define the model architecture and are not user-configurable.
# Changing them alters the model structure and requires retraining.
# ====================================================================

VERSION = "4.0"
MODEL_VERSION = 4.0
ARCHITECTURE = "decoder_transformer"
NORMALIZATION = "pre_layernorm"

# Reproducibility Seeds (DO NOT EDIT - affects reproducibility)
SEED = 42
DATASET_SEED = 42
TRAIN_BATCH_SEED = 42
EVAL_SEED = 42
GEN_SEED = 42

DTYPE = np.float32

# Path Resolution (DO NOT EDIT)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = SCRIPT_DIR

# Output directory: saves model files in the same folder as this script
OUTPUT_DIR = SCRIPT_DIR
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Training Corpora — USER DATASET LIST (edit here to change datasets)
# ====================================================================
# Add as many datasets as you want (more than 5 is fine). You may use:
#   • absolute paths:      r"C:\Users\DELL\Desktop\my_data\story_corpus.txt"
#   • paths relative to THIS script's folder: "training_data6.txt"
#   • plain file names in this folder:        "my_aux_dataset.txt"
#
# RULES:
#   • Leave the list EMPTY [] to auto-discover every training_data*.txt
#     file sitting next to this script (the original behavior).
#   • Every listed path MUST exist when training starts; the trainer
#     raises a clear error naming the missing file.
#   • Duplicates are ignored; order affects nothing except log labels.
# ====================================================================
TRAINING_FILES_USER: list[str] = [
    # --- examples (uncomment and edit, or add your own lines below) ---
     r"C:\Users\DELL\Desktop\nova\src\Final model\training data.txt",
     r"C:\Users\DELL\Desktop\nova\src\Final model\training data2.txt",
     r"C:\Users\DELL\Desktop\nova\src\Final model\training_data3.txt",
     r"C:\Users\DELL\Desktop\nova\src\Final model\training_data4.txt",
     r"C:\Users\DELL\Desktop\nova\src\Final model\training_data5.txt",
     r"C:\Users\DELL\Desktop\nova\src\Final model\training_data6.txt",
    # "training_data6.txt",                # newer dataset in this folder
    # r"D:\novadata\sci_fi_corpus.txt",   # dataset anywhere on disk
]

# Fallback auto-discovery (used only when TRAINING_FILES_USER is empty)
def _discover_training_files(search_dir):
    import glob as _glob
    pattern = os.path.join(search_dir, "training_data*.txt")
    files = sorted(set(_glob.glob(pattern)))
    if not files:
        raise FileNotFoundError(
            f"No training_data*.txt files found in {search_dir}. "
            "Place all your training data files matching training_data*.txt there, "
            "or list them explicitly in TRAINING_FILES_USER above."
        )
    return files


def _resolve_training_files(user_list):
    """Resolve the user-supplied dataset list into absolute paths.

    Accepts absolute paths, paths relative to this script's folder, and
    plain file names living next to the script. Raises FileNotFoundError
    naming the first missing entry so fixing it is unambiguous.
    """
    resolved = []
    seen = set()
    for entry in user_list:
        path = entry
        if not os.path.isabs(path):
            cand = os.path.abspath(os.path.join(SRC_DIR, path))
            path = cand if os.path.exists(cand) else os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Training file listed in TRAINING_FILES_USER does not exist: {entry} "
                f"(resolved: {os.path.abspath(path)})"
            )
        if not os.path.isfile(path):
            raise FileNotFoundError(
                f"Path in TRAINING_FILES_USER is not a file: {entry} "
                f"(resolved: {os.path.abspath(path)})"
            )
        if not path.lower().endswith(".txt"):
            raise ValueError(
                f"Dataset in TRAINING_FILES_USER must be a .txt file: {entry}"
            )
        key = os.path.abspath(path).lower()
        if key in seen:
            continue  # silently ignore duplicate entries
        seen.add(key)
        resolved.append(os.path.abspath(path))
    if not resolved:
        raise ValueError(
            "TRAINING_FILES_USER is empty or resolved to nothing. "
            "List at least one dataset file, or leave it completely empty "
            "([]) to use automatic discovery of training_data*.txt files."
        )
    return resolved


_TRAINING_DIR = os.environ.get("NOVA_TRAINING_DIR", SRC_DIR)
if len(TRAINING_FILES_USER) > 0:
    TRAINING_FILES = _resolve_training_files(TRAINING_FILES_USER)
else:
    TRAINING_FILES = _discover_training_files(_TRAINING_DIR)

# Output Artifact Locations (DO NOT EDIT - preserves checkpoint isolation)
# Generated files live in scratch/ so they never overwrite the production
# v4.0/ checkpoints. Change GENERATED_DIR to OUTPUT_DIR to write to v4.0/.
GENERATED_DIR = os.path.join(SRC_DIR, "scratch", "generated")
os.makedirs(GENERATED_DIR, exist_ok=True)
MODEL_FILE          = os.path.join(GENERATED_DIR, "nova_v4_model.json")
OPTIMIZER_FILE      = os.path.join(GENERATED_DIR, "nova_v4_optimizer.json")
CURVE_FILE          = os.path.join(GENERATED_DIR, "nova_v4_learning_curve.txt")
BEST_MODEL_FILE     = os.path.join(GENERATED_DIR, "nova_v4_best_model.json")
BEST_OPTIMIZER_FILE = os.path.join(GENERATED_DIR, "nova_v4_best_optimizer.json")

# Exit-checkpoint files (written only on Ctrl+C; never touch best-val checkpoints)
EXIT_MODEL_FILE     = os.path.join(GENERATED_DIR, "nova_v4_exit_model.json")
EXIT_OPTIMIZER_FILE = os.path.join(GENERATED_DIR, "nova_v4_exit_optimizer.json")

# Model Architecture Dimensions (~200k parameters) (DO NOT EDIT)
CONTEXT_LENGTH = 32   # T (default 32 for optimal CPU training throughput; 199,016 params)
D_MODEL        = 64   # d_model
NUM_LAYERS     = 4    # L
NUM_HEADS      = 4    # H
D_HEAD         = D_MODEL // NUM_HEADS  # 16
D_FF           = 224  # d_ff

# Training & Regularization Settings (DO NOT EDIT - affects optimization behavior)
INITIAL_LR       = 0.001
MIN_LR           = 1e-5
LR_HORIZON       = 100
WEIGHT_DECAY     = 0.005
DROPOUT          = 0.0     # Explicit: No dropout in V4 (regularization via AdamW weight decay)
TRAIN_SPLIT      = 0.90


# ====================================================================
#  INTERACTIVE CONFIGURATION PROMPT
# ====================================================================
def prompt_user_config(config_vars, config_labels):
    """
    Display current configuration and prompt user to confirm or change values.
    
    Args:
        config_vars: dict of variable names to current values
        config_labels: dict of variable names to human-readable labels
    
    Returns:
        dict of confirmed/updated configuration values
    """
    print("\n" + "=" * 70)
    print("USER CONFIGURATION")
    print("=" * 70)
    print()
    print("Current settings:")
    print("-" * 70)
    
    for var_name, value in config_vars.items():
        label = config_labels.get(var_name, var_name)
        # Format value nicely
        if isinstance(value, float):
            if value >= 0.001:
                formatted = f"{value:.6f}"
            else:
                formatted = f"{value:.2e}"
        else:
            formatted = str(value)
        print(f"  {label:<35} = {formatted}")
    
    print("-" * 70)
    print()
    
    while True:
        response = input("Are these values correct? (y/n): ").strip().lower()
        if response in ('y', 'yes'):
            print("Using configured settings.\n")
            return config_vars.copy()
        elif response in ('n', 'no'):
            break
        else:
            print("Please enter 'y' or 'n'.")
    
    # User wants to change something
    print("\nWhich setting would you like to change?")
    print("(Enter the number or exact name, or 'done' to finish)")
    print()
    
    var_list = list(config_vars.keys())
    for i, name in enumerate(var_list, 1):
        label = config_labels.get(name, name)
        print(f"  {i}. {label}")
    
    print()
    
    while True:
        choice = input("Enter selection (number/name or 'done'): ").strip()
        
        if choice.lower() in ('done', 'd', 'finish', 'f'):
            break
        
        # Try to parse as number
        var_name = None
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(var_list):
                var_name = var_list[idx]
        except ValueError:
            # Try as variable name
            if choice in config_vars:
                var_name = choice
            else:
                # Try case-insensitive match
                for name in config_vars:
                    if name.lower() == choice.lower():
                        var_name = name
                        break
        
        if var_name is None:
            print(f"Invalid selection: '{choice}'. Try again.")
            continue
        
        current_value = config_vars[var_name]
        label = config_labels.get(var_name, var_name)
        
        # Determine expected type
        if isinstance(current_value, int):
            expected_type = "integer"
        elif isinstance(current_value, float):
            expected_type = "number"
        else:
            expected_type = "value"
        
        while True:
            new_value_str = input(f"Change {label} (current: {current_value}) to: ").strip()
            
            if new_value_str == "":
                print("No change made.")
                break
            
            try:
                # Try to convert to appropriate type
                if isinstance(current_value, int):
                    new_value = int(new_value_str)
                elif isinstance(current_value, float):
                    new_value = float(new_value_str)
                else:
                    new_value = new_value_str
                
                config_vars[var_name] = new_value
                print(f"  Updated: {label} = {new_value}")
                break
            except ValueError as e:
                print(f"  Invalid input: {e}. Please enter a valid {expected_type}.")
        
        print()
    
    print()
    print("Final configuration:")
    print("-" * 70)
    for var_name, value in config_vars.items():
        label = config_labels.get(var_name, var_name)
        if isinstance(value, float):
            if value >= 0.001:
                formatted = f"{value:.6f}"
            else:
                formatted = f"{value:.2e}"
        else:
            formatted = str(value)
        print(f"  {label:<35} = {formatted}")
    print("-" * 70)
    print()
    
    return config_vars


# ====================================================================
#  CLEAN TERMINAL DISPLAY (INLINE UPDATES LIKE V3.6)
# ====================================================================
def live_print(status_msg, compact_msg=None, final=False):
    """
    Clean terminal status reporter matching V3.6 style.
    Updates inline with carriage return (\r) so the terminal does not explode with newlines.
    Dynamically respects terminal columns and clears leftover line characters.
    """
    try:
        cols = shutil.get_terminal_size(fallback=(95, 24)).columns
    except Exception:
        cols = 95
    width = max(min(cols - 1, 95), 40)
    display = status_msg[:width]

    if not IN_IDLE and IS_TTY:
        sys.stdout.write(f"\r{display:<{width}}")
        sys.stdout.flush()
        if final:
            sys.stdout.write("\n")
            sys.stdout.flush()
    else:
        # Fallback (non-TTY / pipelines): print full lines so nothing is hidden.
        # No eager buffer writes: everything shows up immediately (unbuffered-ish).
        print(display, flush=True)


# ====================================================================
#  DATASET LOADING & PARAGRAPH-LEVEL SPLIT
# ====================================================================
def split_into_paragraphs(text):
    """Split text into paragraphs (blocks separated by blank lines)."""
    paragraphs = []
    current = []
    for line in text.split('\n'):
        if line.strip() == '':
            if current:
                paragraphs.append('\n'.join(current))
                current = []
        else:
            current.append(line)
    if current:
        paragraphs.append('\n'.join(current))
    return paragraphs


def load_corpus(files=TRAINING_FILES):
    """Load, inspect, and validate all 5 training datasets."""
    print("=" * 65)
    print("Nova AI V4 — Loading Training Datasets")
    print("=" * 65)
    all_paragraphs = []
    dataset_info = []

    for file_idx, filepath in enumerate(files):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Required training data file not found: {filepath}")
        with open(filepath, "r", encoding="utf-8") as f:
            raw = f.read()
        paras = split_into_paragraphs(raw)
        fname = os.path.basename(filepath)
        print(f"  • [{file_idx+1}/5] {fname:<22}: {len(raw):>7,} chars, {len(paras):>4} paragraphs")
        dataset_info.append({"file": fname, "chars": len(raw), "paragraphs": len(paras)})
        all_paragraphs.extend(paras)

    total_paragraphs = len(all_paragraphs)
    full_corpus_text = '\n\n'.join(all_paragraphs) + '\n'
    corpus_char_count = len(full_corpus_text)

    # Dynamic Vocabulary Construction
    chars = sorted(set(full_corpus_text))
    vocab = {ch: i for i, ch in enumerate(chars)}
    rev_char = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(vocab)

    print("-" * 65)
    print(f"Total Paragraphs Loaded   : {total_paragraphs:,}")
    print(f"Total Combined Characters : {corpus_char_count:,}")
    print(f"Distinct Character Vocab  : {vocab_size} characters")
    print("=" * 65)

    return all_paragraphs, full_corpus_text, vocab, rev_char, dataset_info


def create_splits(all_paragraphs, vocab, context_length=CONTEXT_LENGTH,
                  seed=DATASET_SEED, train_ratio=TRAIN_SPLIT):
    """
    Construct representative paragraph-level disjoint train and validation partitions.
    Guarantees zero cross-split leakage by isolating paragraph boundaries.
    """
    total_paragraphs = len(all_paragraphs)
    n_train_paras = int(total_paragraphs * train_ratio)
    n_val_paras   = total_paragraphs - n_train_paras

    split_rng = np.random.default_rng(seed)
    shuffled_indices = split_rng.permutation(total_paragraphs)

    train_para_indices = shuffled_indices[:n_train_paras]
    val_para_indices   = shuffled_indices[n_train_paras:]

    # Assert strictly disjoint sets
    assert len(set(train_para_indices).intersection(set(val_para_indices))) == 0, "Train and validation splits overlap!"
    assert len(train_para_indices) + len(val_para_indices) == total_paragraphs

    train_paragraphs = [all_paragraphs[i] for i in train_para_indices]
    val_paragraphs   = [all_paragraphs[i] for i in val_para_indices]

    train_text = '\n\n'.join(train_paragraphs) + '\n'
    val_text   = '\n\n'.join(val_paragraphs) + '\n'

    # Check training split vocabulary coverage
    train_vocab_size = len(set(train_text))
    if train_vocab_size != len(vocab):
        raise ValueError(f"Training split missing {len(vocab) - train_vocab_size} vocabulary characters!")

    encoded_train = np.array([vocab[ch] for ch in train_text], dtype=np.int32)
    encoded_val   = np.array([vocab[ch] for ch in val_text], dtype=np.int32)

    n_train = len(encoded_train) - context_length
    n_val   = len(encoded_val) - context_length

    print(f"Train split      : {n_train_paras:,} paragraphs | {len(train_text):,} chars | {n_train:,} samples")
    print(f"Validation split : {n_val_paras:,} paragraphs | {len(val_text):,} chars | {n_val:,} samples")

    split_metadata = {
        "n_train_paras": n_train_paras,
        "n_val_paras": n_val_paras,
        "total_paragraphs": total_paragraphs,
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "total_chars": len(train_text) + len(val_text),
        "n_train": n_train,
        "n_val": n_val
    }

    return encoded_train, encoded_val, n_train, n_val, split_metadata


# ====================================================================
#  TRANSFORMER DECODER ARCHITECTURE (PRE-LAYERNORM)
# ====================================================================
class TransformerBlock:
    """Pre-LayerNorm Transformer Decoder Block with Causal Self-Attention and FFN."""
    def __init__(self, d_model, num_heads, d_ff, rng):
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_ff = d_ff

        scale_attn = np.sqrt(2.0 / (d_model + d_model)).astype(DTYPE)
        scale_ff = np.sqrt(2.0 / (d_model + d_ff)).astype(DTYPE)

        # Multi-Head Attention Projections
        self.W_q = (rng.normal(0, scale_attn, (d_model, d_model))).astype(DTYPE)
        self.b_q = np.zeros(d_model, dtype=DTYPE)
        self.W_k = (rng.normal(0, scale_attn, (d_model, d_model))).astype(DTYPE)
        self.b_k = np.zeros(d_model, dtype=DTYPE)
        self.W_v = (rng.normal(0, scale_attn, (d_model, d_model))).astype(DTYPE)
        self.b_v = np.zeros(d_model, dtype=DTYPE)
        self.W_o = (rng.normal(0, scale_attn, (d_model, d_model))).astype(DTYPE)
        self.b_o = np.zeros(d_model, dtype=DTYPE)

        # LayerNorm 1 (Pre-Attention)
        self.gamma1 = np.ones(d_model, dtype=DTYPE)
        self.beta1  = np.zeros(d_model, dtype=DTYPE)

        # Position-wise Feed-Forward Network
        self.W_ff1 = (rng.normal(0, scale_ff, (d_model, d_ff))).astype(DTYPE)
        self.b_ff1 = np.zeros(d_ff, dtype=DTYPE)
        self.W_ff2 = (rng.normal(0, scale_ff, (d_ff, d_model))).astype(DTYPE)
        self.b_ff2 = np.zeros(d_model, dtype=DTYPE)

        # LayerNorm 2 (Pre-FFN)
        self.gamma2 = np.ones(d_model, dtype=DTYPE)
        self.beta2  = np.zeros(d_model, dtype=DTYPE)

    def num_parameters(self):
        return (
            self.W_q.size + self.b_q.size +
            self.W_k.size + self.b_k.size +
            self.W_v.size + self.b_v.size +
            self.W_o.size + self.b_o.size +
            self.gamma1.size + self.beta1.size +
            self.W_ff1.size + self.b_ff1.size +
            self.W_ff2.size + self.b_ff2.size +
            self.gamma2.size + self.beta2.size
        )


class NovaV4Transformer:
    """
    Nova AI V4 Decoder-Only Transformer Character-Level Language Model.
    Architecture: Pre-LayerNorm Causal Transformer.
    """
    def __init__(self, vocab_size=104, context_length=CONTEXT_LENGTH,
                 d_model=D_MODEL, num_layers=NUM_LAYERS,
                 num_heads=NUM_HEADS, d_ff=D_FF, seed=SEED):
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.d_ff = d_ff

        init_rng = np.random.default_rng(seed)

        # Token and Learned Positional Embeddings
        self.tok_emb = (init_rng.normal(0, 0.02, (vocab_size, d_model))).astype(DTYPE)
        self.pos_emb = (init_rng.normal(0, 0.02, (context_length, d_model))).astype(DTYPE)

        # Stacked Pre-LayerNorm Transformer Blocks
        self.blocks = [
            TransformerBlock(d_model, num_heads, d_ff, init_rng)
            for _ in range(num_layers)
        ]

        # Final LayerNorm
        self.gamma_f = np.ones(d_model, dtype=DTYPE)
        self.beta_f  = np.zeros(d_model, dtype=DTYPE)

        # Output Projection Head (d_model -> vocab_size)
        scale_head = np.sqrt(2.0 / (d_model + vocab_size)).astype(DTYPE)
        self.W_head = (init_rng.normal(0, scale_head, (d_model, vocab_size))).astype(DTYPE)
        self.b_head = np.zeros(vocab_size, dtype=DTYPE)

        self.causal_mask = np.triu(np.ones((context_length, context_length), dtype=bool), k=1)
        # Precompute additive causal attention bias mask (0.0 for unmasked, -1e9 for masked)
        self.additive_causal_mask = np.where(self.causal_mask, -1e9, 0.0).astype(DTYPE)
        self.total_params = self.count_parameters()

    def count_parameters(self):
        count = self.tok_emb.size + self.pos_emb.size
        for b in self.blocks:
            count += b.num_parameters()
        count += self.gamma_f.size + self.beta_f.size + self.W_head.size + self.b_head.size
        return count

    def forward(self, x_ids, training=True):
        """
        Pre-LayerNorm forward pass for a mini-batch of context sequences.
        x_ids: (B, T) integer array of character IDs
        Returns:
          probs: (B, vocab_size) next-character probability distributions
          cache: dict containing intermediate activations for exact backpropagation
        """
        B, T = x_ids.shape
        assert T <= self.context_length

        pos_indices = np.arange(T)
        h = self.tok_emb[x_ids] + self.pos_emb[pos_indices]

        scale = np.float32(1.0 / np.sqrt(self.d_head))
        cmask_add = self.additive_causal_mask[:T, :T]
        BT = B * T
        last = T - 1

        # Only build cache for training (backpropagation needs it)
        if training:
            block_caches = []
        else:
            block_caches = None

        num_blocks = len(self.blocks)
        for b_idx, block in enumerate(self.blocks):
            is_last = (b_idx == num_blocks - 1) and PRUNE_FINAL_BLOCK
            h_in = h

            # 1. Pre-LayerNorm 1 (fused-variance: E[x^2]-E[x]^2 in one pass)
            mean1 = np.mean(h, axis=-1, keepdims=True)
            mean_sq1 = np.mean(h * h, axis=-1, keepdims=True)
            var1 = np.maximum(mean_sq1 - mean1 * mean1, 0.0).astype(DTYPE)
            std1_inv = 1.0 / np.sqrt(var1 + 1e-5)
            x_hat1 = (h - mean1) * std1_inv
            h_norm1 = block.gamma1 * x_hat1 + block.beta1

            # 2. 2D Projections for Multi-Head Causal Self-Attention
            hn1_flat = h_norm1.reshape(BT, self.d_model)
            if is_last and num_blocks > 1:
                # FINAL BLOCK PRUNING (training only):
                # The loss depends on the final position only. In the last block,
                # position t < T-1 feeds nothing used by the loss, so we only need
                # K/V at ALL positions (consumed by the last-position query) and
                # Q/attention-output/FFN at position T-1. This is a pure compute
                # saving: gradients to K/V at every position are preserved because
                # they are conjugal with the full T-position keys/values used by
                # the last query row. (Inference=False path skips cache anyway; the
                # pruned path is used for training only.)
                hn_last = h_norm1[:, last, :].reshape(B, self.d_model)
                q_raw = hn_last @ block.W_q + block.b_q   # (B, d_model) last pos only
                k_raw = (hn1_flat @ block.W_k + block.b_k).reshape(B, T, self.d_model)
                v_raw = (hn1_flat @ block.W_v + block.b_v).reshape(B, T, self.d_model)

                q = q_raw.reshape(B, self.num_heads, 1, self.d_head)              # (B,H,1,dh)
                k = k_raw.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)  # (B,H,T,dh)
                v = v_raw.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)  # (B,H,T,dh)

                # Causal attention for the final query position: attends to all T keys.
                scores = (q @ k.transpose(0, 1, 3, 2)) * scale            # (B,H,1,T)
                # Additive mask row for the last position: all zeros (it sees everything)
                scores = scores + cmask_add[last:last + 1, :].reshape(1, 1, 1, T)

                scores_max = np.max(scores, axis=-1, keepdims=True)
                exp_scores = np.exp(scores - scores_max)
                attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)  # (B,H,1,T)

                out_attn_last = (attn_weights @ v).reshape(B, self.d_head * self.num_heads)  # (B, d_model)
                attn_proj_last = out_attn_last @ block.W_o + block.b_o                      # (B, d_model)

                # Residual 1 for last position only
                h_mid_last = h[:, last, :] + attn_proj_last               # (B, d_model)

                # Pre-LayerNorm 2 for last position (fused-variance)
                mean2 = np.mean(h_mid_last, axis=-1, keepdims=True)
                mean_sq2 = np.mean(h_mid_last * h_mid_last, axis=-1, keepdims=True)
                var2 = np.maximum(mean_sq2 - mean2 * mean2, 0.0).astype(DTYPE)
                std2_inv = 1.0 / np.sqrt(var2 + 1e-5)
                x_hat2 = (h_mid_last - mean2) * std2_inv
                h_norm2 = block.gamma2 * x_hat2 + block.beta2             # (B, d_model)

                # FFN for last position only
                ffn_pre = h_norm2 @ block.W_ff1 + block.b_ff1             # (B, d_ff)
                ffn_act = np.maximum(ffn_pre, 0)
                ffn_out_last = (ffn_act @ block.W_ff2 + block.b_ff2)      # (B, d_model)

                # Residual 2: output at every position (earlier positions carry
                # through unchanged: h_mid + ffn_out must be produced for the
                # FINAL block output h used by final LayerNorm at ALL positions? No:
                # final LayerNorm output only matters at the last position for the
                # loss. Earlier positions of the last block's output h are dead
                # values - nobody consumes them. So h for earlier positions = h_in.)
                h_last_pos = h_mid_last + ffn_out_last                    # (B, d_model)

                # Assemble full-length h for final-LayerNorm consumption pattern:
                # the final LayerNorm feeds only the last position into the head,
                # so h at earlier positions is irrelevant; we keep h as (B, d_model)
                # last-position values via a marker.
                h = np.concatenate([h[:, :last, :], h_last_pos[:, None, :]], axis=1)  # (B,T,d)

                if training:
                    block_caches.append({
                        "pruned": True,
                        "h_in": h_in, "h_norm1": h_norm1, "hn1_flat": hn1_flat,
                        "x_hat1": x_hat1, "std1_inv": std1_inv,
                        "q": q, "k": k, "v": v,
                        "attn_weights": attn_weights,          # (B,H,1,T)
                        "q_raw": q_raw,                        # (B, d_model)
                        "hn_last": hn_last,                    # (B, d_model)
                        "out_attn_last": out_attn_last,        # (B, d_model)
                        "attn_proj_last": attn_proj_last,      # (B, d_model)
                        "h_mid_last": h_mid_last,              # (B, d_model)
                        "h_norm2": h_norm2,                    # (B, d_model)
                        "x_hat2": x_hat2, "std2_inv": std2_inv,
                        "ffn_pre": ffn_pre,                    # (B, d_ff)
                        "ffn_act": ffn_act,                    # (B, d_ff)
                        "h_mid": None, "hn2_flat": None,
                        "ffn_act_flat": None, "ffn_out": None, "attn_proj": None, "out_attn": None,
                    })
                continue

            q_raw = (hn1_flat @ block.W_q + block.b_q).reshape(B, T, self.d_model)
            k_raw = (hn1_flat @ block.W_k + block.b_k).reshape(B, T, self.d_model)
            v_raw = (hn1_flat @ block.W_v + block.b_v).reshape(B, T, self.d_model)

            q = q_raw.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
            k = k_raw.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)
            v = v_raw.reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

            # Additive Causal Attention (avoids boolean slicing copies)
            scores = (q @ k.transpose(0, 1, 3, 2)) * scale + cmask_add

            scores_max = np.max(scores, axis=-1, keepdims=True)
            exp_scores = np.exp(scores - scores_max)
            attn_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

            out_attn = (attn_weights @ v).transpose(0, 2, 1, 3).reshape(BT, self.d_model)
            attn_proj = (out_attn @ block.W_o + block.b_o).reshape(B, T, self.d_model)

            # 3. Residual 1
            h_mid = h_in + attn_proj

            # 4. Pre-LayerNorm 2 (fused-variance, one pass fewer temps)
            mean2 = np.mean(h_mid, axis=-1, keepdims=True)
            mean_sq2 = np.mean(h_mid * h_mid, axis=-1, keepdims=True)
            var2 = np.maximum(mean_sq2 - mean2 * mean2, 0.0).astype(DTYPE)
            std2_inv = 1.0 / np.sqrt(var2 + 1e-5)
            x_hat2 = (h_mid - mean2) * std2_inv
            h_norm2 = block.gamma2 * x_hat2 + block.beta2

            # 5. 2D Position-wise FFN
            hn2_flat = h_norm2.reshape(BT, self.d_model)
            ffn_pre = (hn2_flat @ block.W_ff1 + block.b_ff1).reshape(B, T, self.d_ff)
            ffn_act = np.maximum(ffn_pre, 0)
            ffn_act_flat = ffn_act.reshape(BT, self.d_ff)
            ffn_out = (ffn_act_flat @ block.W_ff2 + block.b_ff2).reshape(B, T, self.d_model)

            # 6. Residual 2
            h = h_mid + ffn_out

            # Only store cache for training
            if training:
                block_caches.append({
                    "h_in": h_in, "h_norm1": h_norm1, "hn1_flat": hn1_flat,
                    "x_hat1": x_hat1, "std1_inv": std1_inv,
                    "q": q, "k": k, "v": v,
                    "attn_weights": attn_weights, "out_attn": out_attn, "attn_proj": attn_proj,
                    "h_mid": h_mid, "h_norm2": h_norm2, "hn2_flat": hn2_flat,
                    "x_hat2": x_hat2, "std2_inv": std2_inv,
                    "ffn_pre": ffn_pre, "ffn_act": ffn_act, "ffn_act_flat": ffn_act_flat,
                    "ffn_out": ffn_out
                })

        # 7. Final LayerNorm (fused-variance)
        mean_f = np.mean(h, axis=-1, keepdims=True)
        mean_sq_f = np.mean(h * h, axis=-1, keepdims=True)
        var_f = np.maximum(mean_sq_f - mean_f * mean_f, 0.0).astype(DTYPE)
        stdf_inv = 1.0 / np.sqrt(var_f + 1e-5)
        x_hat_f = (h - mean_f) * stdf_inv
        h_final = self.gamma_f * x_hat_f + self.beta_f

        # Only store final cache for training
        if training:
            h_last = h_final[:, -1, :]
            cache = {
                "x_ids": x_ids, "h_final": h_final, "x_hat_f": x_hat_f,
                "stdf_inv": stdf_inv, "h_last": h_last, "block_caches": block_caches
            }
        else:
            h_last = h_final[:, -1, :]
            cache = {}

        # Logits at the final context position (T - 1)
        logits = h_last @ self.W_head + self.b_head
        logits -= np.max(logits, axis=-1, keepdims=True)
        exp_logits = np.exp(logits)
        probs = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        return probs, cache

    def backward(self, probs, y_ids, cache):
        """Exact backpropagation through Pre-LayerNorm Transformer.        """
        B, T = cache["x_ids"].shape
        last = T - 1
        BT = B * T
        inv_N = np.float32(1.0 / self.d_model)
        scale = np.float32(1.0 / np.sqrt(self.d_head))

        grads = {}

        # Logit gradients
        d_logits = probs.copy()
        d_logits[np.arange(B), y_ids] -= 1.0
        d_logits /= B

        # Output projection head gradients
        grads["W_head"] = cache["h_last"].T @ d_logits
        grads["b_head"] = np.sum(d_logits, axis=0)
        dh_last = d_logits @ self.W_head.T

        # Final LayerNorm backward — EXACT last-position reduction.
        # The loss consumes h_final only at position T-1, so dh_final is zero
        # everywhere except the last position. Computing only that slice gives
        # bit-identical gradients (verified: earlier rows were exactly zero in
        # the original formulation) while avoiding a full (B,T,D) zeros array
        # and several full-size temporaries. This is an exactness-preserving
        # restructure, not an approximation.
        xl_f = cache["x_hat_f"][:, -1, :]                 # (B, d_model)
        grads["gamma_f"] = np.sum(dh_last * xl_f, axis=0)
        grads["beta_f"]  = np.sum(dh_last, axis=0)
        dx_last = dh_last * self.gamma_f                   # (B, d_model)
        sum_dx_f = np.sum(dx_last, axis=-1, keepdims=True)
        sum_dx_x_f = np.sum(dx_last * xl_f, axis=-1, keepdims=True)
        dh_last_out = cache["stdf_inv"][:, -1, :] * (
            dx_last - inv_N * (sum_dx_f + xl_f * sum_dx_x_f))  # (B, d_model)
        dh = np.zeros_like(cache["h_final"])
        dh[:, -1, :] = dh_last_out

        # Transformer blocks backward in reverse order
        for l_idx in reversed(range(self.num_layers)):
            block = self.blocks[l_idx]
            bcache = cache["block_caches"][l_idx]

            if bcache.get("pruned"):
                # ---- PRUNED FINAL BLOCK backward ----
                # dh: (B, T, d_model). Only the last position carries gradient
                # (produced by final LayerNorm backward); earlier positions are 0.
                d_h_last = dh[:, last, :]                     # (B, d_model)

                # Residual 2 backward: h_mid_last = h[last] + ffn_out_last
                d_h_mid_last = d_h_last.copy()                # gradient to residual stream
                d_ffn_last = d_h_last                         # gradient into ffn_out_last

                # FFN backward (last position only, (B, d_ff))
                grads[f"b{l_idx}_W_ff2"] = bcache["ffn_act"].T @ d_ffn_last
                grads[f"b{l_idx}_b_ff2"] = np.sum(d_ffn_last, axis=0)

                d_ffn_act = d_ffn_last @ block.W_ff2.T        # (B, d_ff)
                d_ffn_pre = d_ffn_act * (bcache["ffn_pre"] > 0)

                grads[f"b{l_idx}_W_ff1"] = bcache["h_norm2"].T @ d_ffn_pre
                grads[f"b{l_idx}_b_ff1"] = np.sum(d_ffn_pre, axis=0)

                d_h_norm2 = d_ffn_pre @ block.W_ff1.T         # (B, d_model)

                # Pre-LayerNorm 2 backward (last pos, mean over d_model)
                grads[f"b{l_idx}_gamma2"] = np.sum(d_h_norm2 * bcache["x_hat2"], axis=0)
                grads[f"b{l_idx}_beta2"]  = np.sum(d_h_norm2, axis=0)
                dx_hat2 = d_h_norm2 * block.gamma2

                sum_dx2 = np.sum(dx_hat2, axis=-1, keepdims=True)
                sum_dx_x2 = np.sum(dx_hat2 * bcache["x_hat2"], axis=-1, keepdims=True)
                d_h_mid_last += bcache["std2_inv"] * (dx_hat2 - inv_N * (sum_dx2 + bcache["x_hat2"] * sum_dx_x2))

                # Residual 1 backward: h_mid_last = h[last] + attn_proj_last
                d_attn_proj_last = d_h_mid_last.copy()        # (B, d_model)

                # W_o backward
                grads[f"b{l_idx}_W_o"] = bcache["out_attn_last"].T @ d_attn_proj_last
                grads[f"b{l_idx}_b_o"] = np.sum(d_attn_proj_last, axis=0)

                d_out_attn_last = d_attn_proj_last @ block.W_o.T   # (B, d_model)

                # Attention core backward (single query row)
                d_out_attn = d_out_attn_last.reshape(B, self.num_heads, self.d_head)  # (B,H,dh)
                aw = bcache["attn_weights"]                   # (B,H,1,T)

                # out = aw @ v :  out[b,h,dh] = sum_t aw[b,h,t] * v[b,h,t,dh]
                # d_v[t,dh] = aw[t] * d_out[dh]
                d_v = np.einsum('b h t, b h d -> b h t d', aw[:, :, 0, :], d_out_attn)  # (B,H,T,dh)

                # d_aw: out[b,h,dh] = sum_t aw[...t] v[...t,dh]
                d_aw = np.einsum('b h d, b h t d -> b h t', d_out_attn, bcache["v"])  # (B,H,T)

                sum_aw = np.sum(d_aw * aw[:, :, 0, :], axis=-1, keepdims=True)       # (B,H,1)
                d_scores_scaled = (aw[:, :, 0, :] * (d_aw - sum_aw)) * scale         # (B,H,T)

                # q (B,H,1,dh), k (B,H,T,dh)
                d_q = (d_scores_scaled[:, :, None, :] @ bcache["k"]).reshape(B, self.d_head * self.num_heads)  # (B,d_model)

                # d_k[t] = d_scores[t] * q  (per head)
                d_k = d_scores_scaled[:, :, :, None] * bcache["q"]                    # (B,H,T,dh)

                # Scatter projections backward
                d_q_raw_block = d_q                                                   # (B, d_model)
                d_k_flat = d_k.transpose(0, 2, 1, 3).reshape(BT, self.d_model)         # (BT, d_model)
                d_v_flat = d_v.transpose(0, 2, 1, 3).reshape(BT, self.d_model)         # (BT, d_model)

                grads[f"b{l_idx}_W_q"] = bcache["hn_last"].T @ d_q_raw_block
                grads[f"b{l_idx}_b_q"] = np.sum(d_q_raw_block, axis=0)
                grads[f"b{l_idx}_W_k"] = bcache["hn1_flat"].T @ d_k_flat
                grads[f"b{l_idx}_b_k"] = np.sum(d_k_flat, axis=0)
                grads[f"b{l_idx}_W_v"] = bcache["hn1_flat"].T @ d_v_flat
                grads[f"b{l_idx}_b_v"] = np.sum(d_v_flat, axis=0)

                # dh into LayerNorm 1 input: last-pos row from Q path, all rows from K/V paths
                dh_norm1 = (d_k_flat @ block.W_k.T + d_v_flat @ block.W_v.T).reshape(B, T, self.d_model)
                dh_norm1[:, last, :] += (d_q_raw_block @ block.W_q.T).reshape(B, self.d_model)

                # Pre-LayerNorm 1 backward
                grads[f"b{l_idx}_gamma1"] = np.sum(dh_norm1 * bcache["x_hat1"], axis=(0, 1))
                grads[f"b{l_idx}_beta1"]  = np.sum(dh_norm1, axis=(0, 1))
                dx_hat1 = dh_norm1 * block.gamma1

                sum_dx1 = np.sum(dx_hat1, axis=-1, keepdims=True)
                sum_dx_x1 = np.sum(dx_hat1 * bcache["x_hat1"], axis=-1, keepdims=True)
                dh_in = bcache["std1_inv"] * (dx_hat1 - inv_N * (sum_dx1 + bcache["x_hat1"] * sum_dx_x1))

                # h_in gradient: only last position also receives the residual-stream
                # gradient from d_h_mid_last. Earlier positions receive 0 (their h_in
                # does not influence the loss through the last block).
                dh_in[:, last, :] += d_h_mid_last

                dh = dh_in
                continue

            # Residual 2 backward
            dh_mid = dh.copy()
            d_ffn_out_flat = dh.reshape(BT, self.d_model)

            # 2D FFN backward
            grads[f"b{l_idx}_W_ff2"] = bcache["ffn_act_flat"].T @ d_ffn_out_flat
            grads[f"b{l_idx}_b_ff2"] = np.sum(d_ffn_out_flat, axis=0)

            d_ffn_act_flat = d_ffn_out_flat @ block.W_ff2.T
            d_ffn_pre_flat = d_ffn_act_flat * (bcache["ffn_pre"].reshape(BT, self.d_ff) > 0)

            grads[f"b{l_idx}_W_ff1"] = bcache["hn2_flat"].T @ d_ffn_pre_flat
            grads[f"b{l_idx}_b_ff1"] = np.sum(d_ffn_pre_flat, axis=0)

            dh_norm2 = (d_ffn_pre_flat @ block.W_ff1.T).reshape(B, T, self.d_model)

            # Pre-LayerNorm 2 backward
            grads[f"b{l_idx}_gamma2"] = np.sum(dh_norm2 * bcache["x_hat2"], axis=(0, 1))
            grads[f"b{l_idx}_beta2"]  = np.sum(dh_norm2, axis=(0, 1))
            dx_hat2 = dh_norm2 * block.gamma2

            sum_dx2 = np.sum(dx_hat2, axis=-1, keepdims=True)
            sum_dx_x2 = np.sum(dx_hat2 * bcache["x_hat2"], axis=-1, keepdims=True)
            dh_mid += bcache["std2_inv"] * (dx_hat2 - inv_N * (sum_dx2 + bcache["x_hat2"] * sum_dx_x2))

            # Residual 1 backward
            dh_in = dh_mid.copy()
            d_attn_proj_flat = dh_mid.reshape(BT, self.d_model)

            # Attention Projection backward
            grads[f"b{l_idx}_W_o"] = bcache["out_attn"].T @ d_attn_proj_flat
            grads[f"b{l_idx}_b_o"] = np.sum(d_attn_proj_flat, axis=0)

            d_out_attn = (d_attn_proj_flat @ block.W_o.T).reshape(B, T, self.num_heads, self.d_head).transpose(0, 2, 1, 3)

            # Attention core backward
            d_v = bcache["attn_weights"].transpose(0, 1, 3, 2) @ d_out_attn
            d_attn_weights = d_out_attn @ bcache["v"].transpose(0, 1, 3, 2)

            sum_aw = np.sum(d_attn_weights * bcache["attn_weights"], axis=-1, keepdims=True)
            d_scores_scaled = (bcache["attn_weights"] * (d_attn_weights - sum_aw)) * scale
            # Causal mask is already encoded in attn_weights (masked positions = 0),
            # so zeroing d_scores_scaled is redundant - attn_weights already handles this
            # d_scores_scaled[:, :, self.causal_mask[:T, :T]] = 0.0  # REMOVED: redundant

            d_q = d_scores_scaled @ bcache["k"]
            d_k = d_scores_scaled.transpose(0, 1, 3, 2) @ bcache["q"]

            d_q_flat = d_q.transpose(0, 2, 1, 3).reshape(BT, self.d_model)
            d_k_flat = d_k.transpose(0, 2, 1, 3).reshape(BT, self.d_model)
            d_v_flat = d_v.transpose(0, 2, 1, 3).reshape(BT, self.d_model)

            hn1_flat = bcache["hn1_flat"]
            grads[f"b{l_idx}_W_q"] = hn1_flat.T @ d_q_flat
            grads[f"b{l_idx}_b_q"] = np.sum(d_q_flat, axis=0)
            grads[f"b{l_idx}_W_k"] = hn1_flat.T @ d_k_flat
            grads[f"b{l_idx}_b_k"] = np.sum(d_k_flat, axis=0)
            grads[f"b{l_idx}_W_v"] = hn1_flat.T @ d_v_flat
            grads[f"b{l_idx}_b_v"] = np.sum(d_v_flat, axis=0)

            dh_norm1 = (d_q_flat @ block.W_q.T + d_k_flat @ block.W_k.T + d_v_flat @ block.W_v.T).reshape(B, T, self.d_model)

            # Pre-LayerNorm 1 backward
            grads[f"b{l_idx}_gamma1"] = np.sum(dh_norm1 * bcache["x_hat1"], axis=(0, 1))
            grads[f"b{l_idx}_beta1"]  = np.sum(dh_norm1, axis=(0, 1))
            dx_hat1 = dh_norm1 * block.gamma1

            sum_dx1 = np.sum(dx_hat1, axis=-1, keepdims=True)
            sum_dx_x1 = np.sum(dx_hat1 * bcache["x_hat1"], axis=-1, keepdims=True)
            dh_in += bcache["std1_inv"] * (dx_hat1 - inv_N * (sum_dx1 + bcache["x_hat1"] * sum_dx_x1))

            dh = dh_in

        # Embeddings backward (one-hot matmul instead of np.add.at scatter:
        # benchmarked 0.84 ms vs 2.85 ms at B=64/T=32; changes only grad
        # summation order - tiny fp32 differences permitted)
        grads["pos_emb"] = np.sum(dh, axis=0)
        B_emb, T_emb = cache["x_ids"].shape
        oh = np.zeros((B_emb * T_emb, self.tok_emb.shape[0]), dtype=DTYPE)
        oh[np.arange(B_emb * T_emb), cache["x_ids"].reshape(-1)] = 1.0
        grads["tok_emb"] = (oh.T @ dh.reshape(B_emb * T_emb, -1)).astype(DTYPE)

        return grads

    def get_parameter_dict(self):
        """Return dictionary of parameter references."""
        params = {
            "tok_emb": self.tok_emb,
            "pos_emb": self.pos_emb,
            "gamma_f": self.gamma_f,
            "beta_f":  self.beta_f,
            "W_head":  self.W_head,
            "b_head":  self.b_head
        }
        for l_idx, b in enumerate(self.blocks):
            params[f"b{l_idx}_W_q"]    = b.W_q
            params[f"b{l_idx}_b_q"]    = b.b_q
            params[f"b{l_idx}_W_k"]    = b.W_k
            params[f"b{l_idx}_b_k"]    = b.b_k
            params[f"b{l_idx}_W_v"]    = b.W_v
            params[f"b{l_idx}_b_v"]    = b.b_v
            params[f"b{l_idx}_W_o"]    = b.W_o
            params[f"b{l_idx}_b_o"]    = b.b_o
            params[f"b{l_idx}_gamma1"] = b.gamma1
            params[f"b{l_idx}_beta1"]  = b.beta1
            params[f"b{l_idx}_W_ff1"]  = b.W_ff1
            params[f"b{l_idx}_b_ff1"]  = b.b_ff1
            params[f"b{l_idx}_W_ff2"]  = b.W_ff2
            params[f"b{l_idx}_b_ff2"]  = b.b_ff2
            params[f"b{l_idx}_gamma2"] = b.gamma2
            params[f"b{l_idx}_beta2"]  = b.beta2
        return params


# ====================================================================
#  ADAMW OPTIMIZER & LR SCHEDULING
# ====================================================================
class AdamWOptimizer:
    """Decoupled AdamW Optimizer for Nova AI V4 with vectorized in-place updates."""
    def __init__(self, params, lr=INITIAL_LR, beta1=0.9, beta2=0.999,
                 eps=1e-8, weight_decay=WEIGHT_DECAY):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {k: np.zeros_like(p) for k, p in params.items()}
        self.v = {k: np.zeros_like(p) for k, p in params.items()}
        # Pre-classify 2D weight parameters for decoupled weight decay
        self.decay_flags = {k: ("W_" in k and p.ndim == 2) for k, p in params.items()}
        self.t = 0

    def step(self, grads, lr=None):
        self.t += 1
        current_lr = lr if lr is not None else self.lr
        bc1 = 1.0 - (self.beta1 ** self.t)
        bc2 = 1.0 - (self.beta2 ** self.t)
        one_minus_beta1 = 1.0 - self.beta1
        one_minus_beta2 = 1.0 - self.beta2
        wd_step = current_lr * self.weight_decay

        for k, p in self.params.items():
            g = grads[k]

            # Decoupled weight decay on 2D weight matrices (not biases or LayerNorm)
            if self.decay_flags[k]:
                p -= wd_step * p

            # Adam moments with in-place updates
            m = self.m[k]
            v = self.v[k]
            m *= self.beta1
            m += one_minus_beta1 * g
            v *= self.beta2
            v += one_minus_beta2 * (g * g)

            m_hat = m / bc1
            v_hat = v / bc2

            p -= (current_lr * m_hat) / (np.sqrt(v_hat) + self.eps)


def get_learning_rate(epoch, total_epochs=LR_HORIZON):
    """Cosine decay learning rate schedule."""
    if epoch >= total_epochs:
        return MIN_LR
    progress = epoch / total_epochs
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    return MIN_LR + (INITIAL_LR - MIN_LR) * cosine_decay


# ====================================================================
#  CHECKPOINT SERIALIZATION & DESERIALIZATION
# ====================================================================
def save_checkpoint(filepath, model, epoch, train_loss, val_loss, lr, vocab, rev_char, is_best=False):
    """Save model weights, vocabulary, architecture configuration, and training metadata."""
    params = model.get_parameter_dict()
    params_serializable = {k: v.tolist() for k, v in params.items()}

    data = {
        "version": VERSION,
        "model_version": MODEL_VERSION,
        "architecture": ARCHITECTURE,
        "normalization": NORMALIZATION,
        "config": {
            "vocab_size": model.vocab_size,
            "context_length": model.context_length,
            "d_model": model.d_model,
            "num_layers": model.num_layers,
            "num_heads": model.num_heads,
            "d_ff": model.d_ff,
            "total_params": model.total_params,
            "dropout": DROPOUT,
            "weight_decay": WEIGHT_DECAY,
        },
        "metadata": {
            "epoch": epoch,
            "train_loss": float(train_loss) if train_loss is not None else None,
            "val_loss": float(val_loss) if val_loss is not None else None,
            "val_perplexity": float(math.exp(min(val_loss, 20))) if val_loss is not None else None,
            "learning_rate": float(lr) if lr is not None else None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "is_best": is_best,
        },
        "vocab": vocab,
        "rev_char": {str(k): v for k, v in rev_char.items()},
        "params": params_serializable,
    }

    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    tmp_file = filepath + f".tmp_{os.getpid()}_{time.time_ns()}"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(data, f)
            f.flush()
            os.fsync(f.fileno())
        # Atomic replacement: preserves existing checkpoint until successfully swapped
        os.replace(tmp_file, filepath)
    except Exception:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass
        raise


def load_checkpoint(filepath):
    """Load model weights, config, and vocabulary from a saved checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Model checkpoint not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    cfg = data["config"]
    vocab = data["vocab"]
    rev_char = {int(k): v for k, v in data["rev_char"].items()}

    model = NovaV4Transformer(
        vocab_size=cfg["vocab_size"],
        context_length=cfg["context_length"],
        d_model=cfg["d_model"],
        num_layers=cfg["num_layers"],
        num_heads=cfg["num_heads"],
        d_ff=cfg["d_ff"],
        seed=SEED
    )

    p_dict = model.get_parameter_dict()
    saved_params = data["params"]
    for k in p_dict:
        if k in saved_params:
            p_dict[k][:] = np.array(saved_params[k], dtype=DTYPE)
        else:
            raise KeyError(f"Checkpoint missing parameter tensor: {k}")

    return model, data.get("metadata", {}), vocab, rev_char


def save_optimizer(filepath, optimizer, epoch):
    """Save AdamW optimizer state and moments with interruption-safe atomic replacement."""
    opt_data = {
        "version": VERSION,
        "epoch": epoch,
        "t": optimizer.t,
        "lr": optimizer.lr,
        "beta1": optimizer.beta1,
        "beta2": optimizer.beta2,
        "eps": optimizer.eps,
        "weight_decay": optimizer.weight_decay,
        "moments": {
            k: {
                "m": optimizer.m[k].tolist(),
                "v": optimizer.v[k].tolist()
            }
            for k in optimizer.params
        }
    }
    dir_name = os.path.dirname(filepath)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    tmp_file = filepath + f".tmp_{os.getpid()}_{time.time_ns()}"
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            json.dump(opt_data, f)
            f.flush()
            os.fsync(f.fileno())
        # Atomic replacement: preserves existing checkpoint until successfully swapped
        os.replace(tmp_file, filepath)
    except Exception:
        if os.path.exists(tmp_file):
            try:
                os.remove(tmp_file)
            except OSError:
                pass
        raise


def load_optimizer(filepath, params):
    """Load AdamW optimizer state and moments from a saved checkpoint."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Optimizer checkpoint not found at: {filepath}")

    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)

    optimizer = AdamWOptimizer(
        params,
        lr=data["lr"],
        beta1=data["beta1"],
        beta2=data["beta2"],
        eps=data["eps"],
        weight_decay=data["weight_decay"]
    )
    optimizer.t = data["t"]

    moments = data["moments"]
    for k in params:
        if k in moments:
            optimizer.m[k][:] = np.array(moments[k]["m"], dtype=DTYPE)
            optimizer.v[k][:] = np.array(moments[k]["v"], dtype=DTYPE)

    return optimizer


# ====================================================================
#  TEXT GENERATION SAMPLER
# ====================================================================
def generate_text(model, prompt, vocab, rev_char, max_chars=None, temperature=None, seed=GEN_SEED):
    """Generate text continuation from a prompt using temperature sampling.

    max_chars/temperature default to the USER CONFIGURATION values
    (GEN_MAX_CHARS / GEN_TEMPERATURE) so hardcoded defaults never override
    values set through the interactive configuration prompt or CLI.
    """
    if max_chars is None:
        max_chars = GEN_MAX_CHARS
    if temperature is None:
        temperature = GEN_TEMPERATURE
    gen_rng = np.random.default_rng(seed)
    ctx_len = model.context_length
    curr_ids = [vocab.get(ch, 0) for ch in prompt]
    if len(curr_ids) < ctx_len:
        curr_ids = [0] * (ctx_len - len(curr_ids)) + curr_ids

    generated = list(prompt)
    for _ in range(max_chars):
        x_in = np.array([curr_ids[-ctx_len:]], dtype=np.int32)
        probs, _ = model.forward(x_in, training=False)
        p = probs[0].copy()

        if temperature <= 0.01:
            next_id = int(np.argmax(p))
        else:
            p = np.log(np.maximum(p, 1e-12)) / temperature
            p -= np.max(p)
            exp_p = np.exp(p)
            p = exp_p / np.sum(exp_p)
            next_id = int(gen_rng.choice(len(p), p=p))

        next_ch = rev_char[int(next_id)]
        generated.append(next_ch)
        curr_ids.append(next_id)

    return "".join(generated)


# Separate persistent RNG for interactive generation (isolated from training RNG)
INFERENCE_GEN_RNG = np.random.default_rng(GEN_SEED)


def interactive_generate(model, vocab, rev_char, prompt=None, max_chars=None, temperature=None):
    """
    Interactive text generation while training is paused (Ctrl+T).

    Uses a separate persistent RNG (INFERENCE_GEN_RNG) isolated from training.
    Does NOT modify model parameters, optimizer state, or training RNG.

    Controls (Windows console):
      Ctrl+T   -> resume training (returns so the training loop continues)
      Ctrl+C   -> clean exit: raises CtrlCExit; train()'s handler saves the EXIT
                  checkpoint (never touching best-validation checkpoints)
      Enter    -> generate again with a new prompt (empty prompt = empty context)
      'gen'    -> generate again with a new prompt (explicit command)
      'exit' / 'quit' -> same as Ctrl+C (clean exit + exit checkpoint)

    max_chars/temperature honor the USER CONFIGURATION values (GEN_MAX_CHARS,
    GEN_TEMPERATURE) whenever the caller does not pass explicit values.

    Returns the last generated text so the training loop can display it.
    """
    # Honor USER CONFIGURATION when the caller does not pass explicit values
    if max_chars is None:
        max_chars = GEN_MAX_CHARS
    if temperature is None:
        temperature = GEN_TEMPERATURE

    def _ask_prompt(first):
        print("\n" + "=" * 60)
        print("INTERACTIVE GENERATION (training paused — Ctrl+T to resume)")
        print("=" * 60)
        print("\nEnter prompt (or press Enter for empty prompt).")
        print("Commands: gen (regenerate) | exit/quit (save + stop) | Ctrl+T (resume) | Ctrl+C (stop)")
        try:
            return input("> ")
        except (KeyboardInterrupt, EOFError):
            raise CtrlCExit()

    if prompt is None:
        prompt = _ask_prompt(first=True)

    print(f"\nGenerating from prompt: {repr(prompt)}")
    print("-" * 60)
    result = generate_one(model, vocab, rev_char, prompt, max_chars, temperature)
    print(f"Generated ({len(result) - len(prompt)} new chars):")
    print(result)
    print("=" * 60)
    print(f"\nGeneration complete (max_chars={max_chars}, temperature={temperature}).")
    print("Press Enter to generate again, Ctrl+T to resume training, Ctrl+C to exit.\n")

    # Loop: keep generating until user presses Ctrl+T (resume) or exits cleanly.
    while True:
        try:
            time.sleep(0.1)
        except KeyboardInterrupt:
            # Ctrl+C interrupting the wait = clean exit signal
            raise CtrlCExit()

        if msvcrt.kbhit():
            try:
                key = msvcrt.getch()
            except KeyboardInterrupt:
                # Ctrl+C can surface as an interrupt on some console setups
                raise CtrlCExit()

            # Ctrl+T (0x14) = resume training
            if key == b'\x14':
                print("\nResuming training...\n")
                return result

            # Ctrl+C (0x03): getch() may deliver it as a keystroke - treat as exit
            if key == b'\x03':
                raise CtrlCExit()

            # Enter key = generate another text with a new prompt
            if key == b'\r' or key == b'\n':
                new_prompt = _ask_prompt(first=False)
                if new_prompt.strip().lower() in ("exit", "quit"):
                    raise CtrlCExit()
                if new_prompt.strip().lower() == "gen":
                    new_prompt = prompt  # regenerate from the same prompt
                print(f"\nGenerating from prompt: {repr(new_prompt)}")
                print("-" * 60)
                result = generate_one(model, vocab, rev_char, new_prompt, max_chars, temperature)
                print(f"Generated ({len(result) - len(new_prompt)} new chars):")
                print(result)
                print("=" * 60)
                print(f"\nGeneration complete (max_chars={max_chars}, temperature={temperature}).")
                print("Press Enter to generate again, Ctrl+T to resume training, Ctrl+C to exit.\n")

            # Any other key = ignore


def generate_one(model, vocab, rev_char, prompt, max_chars, temperature):
    """Generate one continuation using the persistent inference RNG (no inprompt)."""
    gen_rng = INFERENCE_GEN_RNG
    ctx_len = model.context_length
    curr_ids = [vocab.get(ch, 0) for ch in prompt]
    if len(curr_ids) < ctx_len:
        curr_ids = [0] * (ctx_len - len(curr_ids)) + curr_ids

    generated = list(prompt)
    for _ in range(max_chars):
        x_in = np.array([curr_ids[-ctx_len:]], dtype=np.int32)
        probs, _ = model.forward(x_in, training=False)
        p = probs[0].copy()

        if temperature <= 0.01:
            next_id = int(np.argmax(p))
        else:
            p = np.log(np.maximum(p, 1e-12)) / temperature
            p -= np.max(p)
            exp_p = np.exp(p)
            p = exp_p / np.sum(exp_p)
            next_id = int(gen_rng.choice(len(p), p=p))

        next_ch = rev_char[int(next_id)]
        generated.append(next_ch)
        curr_ids.append(next_id)

    return "".join(generated)


class CtrlCExit(Exception):
    """Raised for a clean user-requested exit (Ctrl+C) without a traceback."""


def save_exit_checkpoint(model, optimizer, epoch, vocab, rev_char):
    """Save the latest in-memory weights+optimizer on Ctrl+C.

    Writes to EXIT_MODEL_FILE / EXIT_OPTIMIZER_FILE so it can NEVER overwrite the
    best-validation checkpoints with a worse model. Atomic save, same mechanism as
    regular checkpoints.
    """
    try:
        print("\nSaving exit checkpoint (latest weights)...", flush=True)
        save_checkpoint(EXIT_MODEL_FILE, model, epoch, None, None, None, vocab, rev_char, is_best=False)
        save_optimizer(EXIT_OPTIMIZER_FILE, optimizer, epoch)
        print(f"Saved: {EXIT_MODEL_FILE}")
        print(f"Saved: {EXIT_OPTIMIZER_FILE}")
        print("(Best-validation checkpoints were NOT overwritten.)")
    except Exception as e:
        # A failed exit-save must not mask the clean exit or raise a traceback
        print(f"WARNING: exit checkpoint save failed: {e}", flush=True)


def interactive_generate_with_prompt(model, vocab, rev_char, prompt, max_chars=None, temperature=None):
    """Non-interactive generation helper for tests/scripts: uses a pre-given
    prompt and the persistent INFERENCE_GEN_RNG. max_chars/temperature default
    to USER CONFIGURATION (GEN_MAX_CHARS / GEN_TEMPERATURE)."""
    if max_chars is None:
        max_chars = GEN_MAX_CHARS
    if temperature is None:
        temperature = GEN_TEMPERATURE
    print("\n" + "=" * 60)
    print("INTERACTIVE GENERATION (Ctrl+T paused training)")
    print("=" * 60)
    print(f"\nGenerating from prompt: {repr(prompt)}")
    print("-" * 60)
    result = generate_one(model, vocab, rev_char, prompt, max_chars, temperature)
    print(f"Generated ({len(result) - len(prompt)} new chars):")
    print(result)
    print("=" * 60)
    return result


# ====================================================================
#  DETERMINISTIC VALIDATION EVALUATION
# ====================================================================
def evaluate_validation_loss(model, encoded_val, context_length=CONTEXT_LENGTH, batch_size=BATCH_SIZE):
    """
    Evaluate validation loss over fixed, reproducible validation windows.
    Guaranteed deterministic, does not modify optimizer or model state.
    """
    val_samples = len(encoded_val) - context_length
    if val_samples <= 0:
        return 0.0

    eval_rng = np.random.default_rng(EVAL_SEED)
    # Evaluate up to 500 fixed deterministic windows
    eval_count = min(val_samples, 500)
    sample_indices = eval_rng.choice(val_samples, size=eval_count, replace=False)

    encoded_val_arr = np.asarray(encoded_val, dtype=np.int32)
    val_offsets = np.arange(context_length, dtype=np.int32)

    total_loss = 0.0
    n_batches = math.ceil(eval_count / batch_size)

    for i in range(n_batches):
        b_indices = sample_indices[i * batch_size: min((i + 1) * batch_size, eval_count)]
        X_val = encoded_val_arr[b_indices[:, None] + val_offsets]
        Y_val = encoded_val_arr[b_indices + context_length]

        probs, _ = model.forward(X_val, training=False)
        batch_loss = -np.mean(np.log(probs[np.arange(len(Y_val)), Y_val] + 1e-12))
        total_loss += batch_loss * len(Y_val)

    return float(total_loss / eval_count)


# ====================================================================
#  TRAINING ENGINE (WITH INLINE V3.6 STATUS REPORTING)
# ====================================================================
def train(model, encoded_train, encoded_val, vocab, rev_char, num_epochs=3, batch_size=BATCH_SIZE, split_metadata=None):  # noqa: E501
    """
    Train Nova AI V4 across all 5 datasets with inline status updates, throughput tracking, and checkpointing.
    """
    print("\n" + "=" * 65)
    print(f"Nova AI V4 — Training Session Started ({num_epochs} Epochs)")
    print(f"Architecture: {ARCHITECTURE} | Normalization: {NORMALIZATION} | Params: {model.total_params:,}")
    print(f"Recipe: batch_size={batch_size} | weight_decay={WEIGHT_DECAY} | initial_lr={INITIAL_LR}")
    print("=" * 65)

    # Initialize learning curve log with throughput columns
    with open(CURVE_FILE, "w", encoding="utf-8") as f:
        f.write("Nova AI V4 Learning Curve\n")
        f.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Architecture: {ARCHITECTURE} ({NORMALIZATION}) | Context: {model.context_length} | Params: {model.total_params}\n")
        f.write(f"Corpus: 5 datasets | Recipe: weight_decay={WEIGHT_DECAY} | lr_horizon={LR_HORIZON}\n")
        f.write("-" * 115 + "\n")
        f.write("Timestamp           | Epoch | Train Loss | Val Loss | Val PPL | LR       | Samples/s | Context Pos/s | Steps/s | Duration (s)\n")

    train_rng = np.random.default_rng(TRAIN_BATCH_SEED)
    optimizer = AdamWOptimizer(model.get_parameter_dict(), lr=INITIAL_LR)

    n_train_samples = len(encoded_train) - model.context_length
    encoded_train_arr = np.asarray(encoded_train, dtype=np.int32)
    train_offsets = np.arange(model.context_length, dtype=np.int32)

    # 200 mini-batches per epoch for responsive, high-velocity training
    steps_per_epoch = min(200, n_train_samples // batch_size)

    best_val_loss = float("inf")
    best_epoch = None
    best_train_loss = None
    samples_processed = 0
    total_steps = 0
    train_start_time = time.time()
    epoch = 0  # visible to the clean-exit handler below

    # ------------------------------------------------------------------
    # Clean-exit handling: Ctrl+C (or CtrlCExit raised from interactive
    # generation) at ANY point inside the epoch loop prints a short exit
    # message, saves an exit checkpoint with the latest in-memory weights
    # + optimizer moments, and returns normally - no traceback. The exit
    # checkpoint NEVER overwrites the best-validation checkpoints.
    # ------------------------------------------------------------------
    try:
        for epoch in range(1, num_epochs + 1):
            epoch_start_time = time.time()
            current_lr = get_learning_rate(epoch - 1, LR_HORIZON)
            epoch_loss_sum = 0.0

            for step in range(steps_per_epoch):
                indices = train_rng.integers(0, n_train_samples, size=batch_size)
                X = encoded_train_arr[indices[:, None] + train_offsets]
                Y = encoded_train_arr[indices + model.context_length]

                probs, cache = model.forward(X, training=True)
                loss = -np.mean(np.log(probs[np.arange(batch_size), Y] + 1e-12))
                epoch_loss_sum += loss

                grads = model.backward(probs, Y, cache)
                optimizer.step(grads, lr=current_lr)

                samples_processed += batch_size
                total_steps += 1

                # Periodic batch progress updated inline (\r) without flooding terminal (matching V3.6 style)
            if (step + 1) % DISPLAY_EVERY == 0 or (step == steps_per_epoch - 1):
                elapsed = time.time() - train_start_time
                speed = samples_processed / max(elapsed, 0.001)
                best_str = f"{best_val_loss:.4f}" if best_val_loss != float("inf") else "--"
                status = (
                    f"E {epoch}/{num_epochs} | B {step+1}/{steps_per_epoch} | "
                    f"Loss {loss:.4f} | Val {best_str} | LR {current_lr:.6f} | {speed:,.0f}/s"
                )
                live_print(status, compact_msg=status, final=False)

                # Check for Ctrl+T (pause and interactive generation)
                # Only check every 5 steps to avoid overhead; safe to check after optimizer.step()
                if step % 5 == 4 and msvcrt.kbhit():
                    try:
                        key = msvcrt.getch()
                    except KeyboardInterrupt:
                        # Ctrl+C during the key read = clean exit signal
                        raise CtrlCExit()
                    # Ctrl+T is ASCII 20 (0x14)
                    if key == b'\x14':
                        print()  # Newline after inline status
                        interactive_generate(model, vocab, rev_char)  # may raise CtrlCExit
                        print(f"\nE {epoch}/{num_epochs} | B {step+1}/{steps_per_epoch} | Resumed training")

            train_loss = float(epoch_loss_sum / steps_per_epoch)
            epoch_duration = time.time() - epoch_start_time
            epoch_samples = steps_per_epoch * batch_size
            epoch_samples_s = epoch_samples / max(epoch_duration, 0.001)
            context_positions_s = (epoch_samples * model.context_length) / max(epoch_duration, 0.001)
            epoch_chars_s = context_positions_s  # Retain backward-compatible alias
            epoch_steps_s = steps_per_epoch / max(epoch_duration, 0.001)

            # Validation evaluation
            val_loss = evaluate_validation_loss(model, encoded_val, model.context_length, batch_size=batch_size)
            val_ppl = math.exp(min(val_loss, 20.0))

            is_best = val_loss < best_val_loss
            if is_best:
                best_val_loss = val_loss
                best_epoch = epoch
                best_train_loss = train_loss
                save_checkpoint(BEST_MODEL_FILE, model, epoch, train_loss, val_loss, current_lr, vocab, rev_char, is_best=True)
                save_optimizer(BEST_OPTIMIZER_FILE, optimizer, epoch)

            # Print final epoch completion line with throughput breakdown
            epoch_summary = (
                f"E {epoch:2d}/{num_epochs} | Train {train_loss:.4f} | "
                f"Val {val_loss:.4f} | PPL {val_ppl:.2f} | "
                f"{epoch_samples_s:,.0f} samp/s ({context_positions_s:,.0f} context pos/s, {epoch_steps_s:.1f} step/s) | {epoch_duration:.1f}s"
            )
            live_print(epoch_summary, compact_msg=epoch_summary, final=True)
            if is_best:
                print(f"  ★ New Best Validation: {best_val_loss:.4f} (PPL {val_ppl:.2f}) at Epoch {best_epoch}")

            # Log to learning curve file with throughput columns
            timestamp_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            with open(CURVE_FILE, "a", encoding="utf-8") as f:
                f.write(f"{timestamp_str} | {epoch:5d} | {train_loss:10.4f} | {val_loss:8.4f} | {val_ppl:7.2f} | {current_lr:8.6f} | {epoch_samples_s:9.0f} | {context_positions_s:13.0f} | {epoch_steps_s:7.2f} | {epoch_duration:11.2f}\n")

        # Save final model & optimizer (normal completion)
        save_checkpoint(MODEL_FILE, model, num_epochs, train_loss, val_loss, current_lr, vocab, rev_char, is_best=False)
        save_optimizer(OPTIMIZER_FILE, optimizer, num_epochs)

    except (KeyboardInterrupt, CtrlCExit):
        print("\n\nCtrl+C detected - stopping training cleanly.")
        save_exit_checkpoint(model, optimizer, max(epoch, 1), vocab, rev_char)
        print(f"Training stopped by user at epoch {max(epoch, 1)}.")
        print(f"Total steps completed:  {total_steps:,}")
        print(f"Total samples processed: {samples_processed:,}")
        if best_val_loss != float("inf"):
            print(f"Best validation loss:   {best_val_loss:.6f} (epoch {best_epoch})")
        print(f"Exit checkpoint:         {EXIT_MODEL_FILE}")
        print("Best-validation checkpoints were NOT overwritten.")
        print("=" * 65 + "\n")
        return model, best_val_loss if best_val_loss != float("inf") else None

    # Final Training Summary Report (aligned with V3.6 conventions)
    elapsed_total = time.time() - train_start_time
    avg_samples_s = samples_processed / max(elapsed_total, 0.001)
    avg_context_positions_s = (samples_processed * model.context_length) / max(elapsed_total, 0.001)
    avg_chars_s = avg_context_positions_s  # Retain backward-compatible alias
    avg_steps_s = total_steps / max(elapsed_total, 0.001)

    print("\n" + "=" * 65)
    print("Nova AI V4 — Final Training Summary Report")
    print("=" * 65)
    print(f"Version:                 {VERSION}")
    print(f"Architecture:            {ARCHITECTURE} ({NORMALIZATION}, L={model.num_layers}, H={model.num_heads}, d_model={model.d_model})")
    print(f"Total Parameters:        {model.total_params:,}")
    print(f"Context Length (T):      {model.context_length}")
    if split_metadata:
        total_c = split_metadata.get("total_chars", split_metadata.get("train_chars", 0) + split_metadata.get("val_chars", 0))
        total_p = split_metadata.get("total_paragraphs", 0)
        train_c = split_metadata.get("train_chars", 0)
        val_c   = split_metadata.get("val_chars", 0)
        print(f"Corpus Characters:       {total_c:,} ({total_p:,} paragraphs)")
        print(f"Train / Val Split:       {train_c:,} train / {val_c:,} val")
    print(f"Weight Decay:            {WEIGHT_DECAY}")
    print(f"Dropout:                 {DROPOUT} (None)")
    print(f"LR Horizon:              {LR_HORIZON} epochs (Cosine decay)")
    print()
    print(f"Best Validation Loss:    {best_val_loss:.6f}")
    print(f"Best Val Perplexity:     {math.exp(min(best_val_loss, 50)):.2f}")
    print(f"Best Epoch:              Epoch {best_epoch}")
    print(f"Train Loss @ Best Epoch: {best_train_loss:.6f}" if best_train_loss is not None else "Train Loss @ Best Epoch: --")
    print(f"Final Train Loss:        {train_loss:.6f}")
    print(f"Final Validation Loss:   {val_loss:.6f}")
    print()
    print(f"Average Throughput:      {avg_samples_s:,.0f} samples/s")
    print(f"Target Token Throughput: {avg_samples_s:,.0f} tokens/s (next-token targets)")
    print(f"Context Positions:       {avg_context_positions_s:,.0f} context positions/s (context window T={model.context_length})")
    print(f"Step Throughput:         {avg_steps_s:,.2f} steps/s (mini-batch updates)")
    print(f"Total Training Time:     {elapsed_total:.1f}s")
    print(f"Total Epochs Completed:  {num_epochs}")
    print(f"Total Steps Completed:   {total_steps:,}")
    print(f"Total Samples Processed: {samples_processed:,}")
    print()
    print(f"Best Model Checkpoint:   {BEST_MODEL_FILE}")
    print(f"Final Model Checkpoint:  {MODEL_FILE}")
    print(f"Learning Curve Log:      {CURVE_FILE}")
    print("=" * 65 + "\n")

    return model, best_val_loss


# ====================================================================
#  CLI ENTRY POINT
# ====================================================================
def main():
    global DISPLAY_EVERY, VALIDATION_EVERY, GEN_MAX_CHARS, GEN_TEMPERATURE
    
    parser = argparse.ArgumentParser(description="Nova AI V4 — Decoder-Only Character Transformer Trainer")
    parser.add_argument("--train", action="store_true", help="Start training session across all 5 datasets")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs (overrides EPOCHS)")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size (overrides BATCH_SIZE)")
    parser.add_argument("--generate", type=str, default=None, help="Generate continuation from prompt (requires trained checkpoint)")
    parser.add_argument("--checkpoint", type=str, default=None, help="Specific checkpoint path to load for inference")
    parser.add_argument("--context-length", type=int, default=None, help="Context length (overrides CONTEXT_LENGTH)")
    parser.add_argument("--eval", action="store_true", help="Evaluate validation loss from saved checkpoint")
    parser.add_argument("--no-prompt", action="store_true", help="Skip interactive configuration prompt")

    args = parser.parse_args()

    # Use CLI arguments to override defaults if provided
    effective_epochs = args.epochs if args.epochs is not None else EPOCHS
    effective_batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE
    effective_context_length = args.context_length if args.context_length is not None else CONTEXT_LENGTH
    effective_epochs = args.epochs if args.epochs is not None else EPOCHS
    effective_batch_size = args.batch_size if args.batch_size is not None else BATCH_SIZE

    # Interactive configuration prompt (unless --no-prompt is set)
    if not args.no_prompt and not args.generate:
        config_vars = {
            'EPOCHS': effective_epochs,
            'BATCH_SIZE': effective_batch_size,
            'DISPLAY_EVERY': DISPLAY_EVERY,
            'VALIDATION_EVERY': VALIDATION_EVERY,
            'GEN_MAX_CHARS': GEN_MAX_CHARS,
            'GEN_TEMPERATURE': GEN_TEMPERATURE,
        }
        config_labels = {
            'EPOCHS': 'Training epochs',
            'BATCH_SIZE': 'Batch size',
            'DISPLAY_EVERY': 'Display frequency (steps)',
            'VALIDATION_EVERY': 'Validation frequency (epochs)',
            'GEN_MAX_CHARS': 'Generation max characters',
            'GEN_TEMPERATURE': 'Generation temperature',
        }
        
        confirmed_config = prompt_user_config(config_vars, config_labels)
        
        # Update effective values from user confirmation
        effective_epochs = confirmed_config['EPOCHS']
        effective_batch_size = confirmed_config['BATCH_SIZE']
        DISPLAY_EVERY = confirmed_config['DISPLAY_EVERY']
        VALIDATION_EVERY = confirmed_config['VALIDATION_EVERY']
        GEN_MAX_CHARS = confirmed_config['GEN_MAX_CHARS']
        GEN_TEMPERATURE = confirmed_config['GEN_TEMPERATURE']

    # 1. Inference mode
    if args.generate:
        ckpt_path = args.checkpoint or (BEST_MODEL_FILE if os.path.exists(BEST_MODEL_FILE) else MODEL_FILE)
        if not os.path.exists(ckpt_path):
            print(f"\nERROR: No trained checkpoint found at: {ckpt_path}", file=sys.stderr)
            print("Please run training first before generating text:", file=sys.stderr)
            print("  python nova_v4.py --train --epochs 3\n", file=sys.stderr)
            sys.exit(1)

        print(f"Loading checkpoint for inference: {ckpt_path}")
        model, meta, vocab, rev_char = load_checkpoint(ckpt_path)
        print(f"Checkpoint info: Epoch {meta.get('epoch')}, Val Loss: {meta.get('val_loss'):.4f} (PPL: {meta.get('val_perplexity'):.2f})")
        print("\nPrompt: " + repr(args.generate))
        sample = generate_text(model, args.generate, vocab, rev_char, max_chars=GEN_MAX_CHARS, temperature=GEN_TEMPERATURE)
        print("Generated Output:\n" + sample + "\n")
        return

    # 2. Ingest all 5 datasets
    all_paras, full_text, vocab, rev_char, ds_info = load_corpus()

    # 3. Build representative paragraph split
    encoded_train, encoded_val, n_train, n_val, split_meta = create_splits(
        all_paras, vocab, context_length=effective_context_length
    )

    # 4. Instantiate Model
    model = NovaV4Transformer(
        vocab_size=len(vocab),
        context_length=effective_context_length,
        d_model=D_MODEL,
        num_layers=NUM_LAYERS,
        num_heads=NUM_HEADS,
        d_ff=D_FF,
        seed=SEED
    )

    print(f"\nModel initialized: {model.total_params:,} trainable parameters (Context T={effective_context_length})")

    # 5. Training or Evaluation
    if args.eval:
        ckpt_path = args.checkpoint or (BEST_MODEL_FILE if os.path.exists(BEST_MODEL_FILE) else MODEL_FILE)
        if not os.path.exists(ckpt_path):
            print(f"ERROR: No checkpoint found to evaluate at: {ckpt_path}", file=sys.stderr)
            sys.exit(1)
        model, meta, _, _ = load_checkpoint(ckpt_path)
        val_loss = evaluate_validation_loss(model, encoded_val, model.context_length)
        print(f"Validation Loss: {val_loss:.4f} (PPL: {math.exp(min(val_loss, 20)):.2f})")
    elif args.train:
        train(model, encoded_train, encoded_val, vocab, rev_char, num_epochs=effective_epochs, batch_size=effective_batch_size, split_metadata=split_meta)
    else:
        # Self-reliant trainer: bare `python nova_v4_fast.py` ALWAYS trains.
        # Only training data in this folder is required - no other command,
        # no other file. (--generate / --eval remain optional opt-ins.)
        train(model, encoded_train, encoded_val, vocab, rev_char, num_epochs=effective_epochs, batch_size=effective_batch_size, split_metadata=split_meta)


if __name__ == "__main__":
    main()
