"""
Centralized configuration for all V-NTBC BC4 pipeline scripts.

V-NTBC BC4 = NTBC BC4 + Variable Bitrate Quantization (VBQ) + Residual Fusion.
(No luminance-chrominance loss — inapplicable to single-channel data.)

Edit the paths in this file once, and all scripts
(Dataset_Input_Extract, Train, Inference, Eval) will pick them up.
"""

from pathlib import Path


# ==================== EDIT THESE PATHS ====================

# Compressonator CLI executable
COMPRESSONATOR_CLI = r"D:\Compressonatorcli\bin\CLI\compressonatorcli.exe"

# Source texture images — single-channel (grayscale) textures for BC4
SOURCE_IMAGES = [
    r"D:\BC1 extract\Data\PavingStones070_4K-PNG\PavingStones070_4K-PNG_Displacement.png",
    r"D:\BC1 extract\Data\PavingStones070_4K-PNG\PavingStones070_4K-PNG_AmbientOcclusion.png",
    r"D:\BC1 extract\Data\PavingStones070_4K-PNG\PavingStones070_4K-PNG_Roughness.png"
   
]

# Human-readable names matching SOURCE_IMAGES (used for output naming)
TEXTURE_NAMES = ["Disp","AO","Roughness"]

# Output directory (everything goes here: dataset, training, inference)
MODEL_DIR = r"D:\BC1 extract\VNTBC_Outputs\BC4\PavingStones070_4K-PNG_model_vbc4"


# ==================== VBQ CONFIGURATION ====================
# Number of bits per resolution level for Quantization-Aware Training.
# Length must match num_levels in the respective network (7 for endpoint, 8 for color).
QAT_BITS_ENDPOINT = [8, 4, 4, 4, 4, 4, 8]
QAT_BITS_COLOR = [8, 4, 4, 4, 4, 4, 4, 4]


# ==================== LOCAL POSITIONAL ENCODING (LPE) ====================
# Append high-frequency sinusoidal position features to the bilinear grid
# features to help resolve fine details and block artifacts.
USE_LPE = True
LPE_N = 128
LPE_N_FREQ = 6
LPE_D0 = 8


# ==================== DERIVED PATHS (auto) ====================

TRAIN_DATASET_JSON = str(Path(MODEL_DIR) / "Train_dataset.json")
INFERENCE_INPUT_JSON = str(Path(MODEL_DIR) / "Inference_input.json")
MERGED_CHECKPOINT = str(Path(MODEL_DIR) / "v_ntbc_bc4_merged_compressed.pt")
OUT_DIR_ENDPOINT = str(Path(MODEL_DIR) / "runs_endpoint")
OUT_DIR_COLOR = str(Path(MODEL_DIR) / "runs_color")

# Inference output subfolder
INFERENCE_OUTPUT_DIR = str(Path(MODEL_DIR) / "inference_output")
OUT_DDS = str(Path(INFERENCE_OUTPUT_DIR) / "ntbc_out.dds")
OUT_PREVIEW_PNG = str(Path(INFERENCE_OUTPUT_DIR) / "ntbc_out_preview.png")

# Eval: reference DDS files
REF_DDS = [
    str(Path(MODEL_DIR) / f"{Path(s).stem}_ref_bc4.dds")
    for s in SOURCE_IMAGES
]

# Eval: NTBC output DDS files
TEST_DDS = [
    str(Path(INFERENCE_OUTPUT_DIR) / f"ntbc_out_{n}.dds")
    for n in TEXTURE_NAMES
]
