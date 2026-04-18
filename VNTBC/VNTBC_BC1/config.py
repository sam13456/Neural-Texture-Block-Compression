"""
Centralized configuration for all NTBC BC1 pipeline scripts.

Edit the paths in this file once, and all scripts
(Dataset_Input_Extract, Train, Inference, Eval) will pick them up.
"""

from pathlib import Path


# ==================== EDIT THESE PATHS ====================

# Compressonator CLI executable
COMPRESSONATOR_CLI = r"D:\Compressonatorcli\bin\CLI\compressonatorcli.exe"

# Source texture images (all must share the same resolution)
SOURCE_IMAGES = [
    r"D:\BC1 extract\Data\MetalPlates013_4K-PNG\MetalPlates013_4K-PNG_Color.png",
    r"D:\BC1 extract\Data\MetalPlates013_4K-PNG\MetalPlates013_4K-PNG_NormalDX.png",
]

# Human-readable names matching SOURCE_IMAGES (used for output naming)
TEXTURE_NAMES = ["Color", "Normal"]

# Output directory (everything goes here: dataset, training, inference)
MODEL_DIR = r"D:\BC1 extract\VNTBC_Outputs\BC1\MetalPlates013_4K-PNG_model_bc1"

# ==================== VBQ CONFIGURATION ====================
# Number of bits per resolution level for Quantization-Aware Training.
# Length must match num_levels in the respective network (7 for endpoint, 8 for color).
QAT_BITS_ENDPOINT = [8, 8, 8, 8, 8, 8, 8]
QAT_BITS_COLOR = [8, 4, 4, 4, 4, 4, 4, 4]

# ==================== LOCAL POSITIONAL ENCODING (LPE) ====================
# NTBC future scope: append high-frequency sinusoidal position features to the
# bilinear grid features to help resolve fine details and block artifacts.
USE_LPE = True
LPE_N = 128
LPE_N_FREQ = 6
LPE_D0 = 8


# ==================== DERIVED PATHS (auto) ====================
# These are computed from MODEL_DIR above. You usually don't
# need to touch them unless your naming convention is different.

TRAIN_DATASET_JSON = str(Path(MODEL_DIR) / "Train_dataset.json")
INFERENCE_INPUT_JSON = str(Path(MODEL_DIR) / "Inference_input.json")
MERGED_CHECKPOINT = str(Path(MODEL_DIR) / "ntbc_bc1_merged_compressed.pt")
OUT_DIR_ENDPOINT = str(Path(MODEL_DIR) / "runs_endpoint")
OUT_DIR_COLOR = str(Path(MODEL_DIR) / "runs_color")

# Inference output subfolder
INFERENCE_OUTPUT_DIR = str(Path(MODEL_DIR) / "inference_output")
OUT_DDS = str(Path(INFERENCE_OUTPUT_DIR) / "ntbc_out.dds")
OUT_PREVIEW_PNG = str(Path(INFERENCE_OUTPUT_DIR) / "ntbc_out_preview.png")

# Eval: reference DDS files (auto-derived from source image stems)
REF_DDS = [
    str(Path(MODEL_DIR) / f"{Path(s).stem}_ref_bc1.dds")
    for s in SOURCE_IMAGES
]

# Eval: NTBC output DDS files (in inference_output, auto-derived from texture names)
TEST_DDS = [
    str(Path(INFERENCE_OUTPUT_DIR) / f"ntbc_out_{n}.dds")
    for n in TEXTURE_NAMES
]

