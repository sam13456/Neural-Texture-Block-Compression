"""
Centralized configuration for all NTBC BC4 pipeline scripts.

Edit the paths in this file once, and all scripts
(Dataset_Input_Extract, Train, Inference, Eval) will pick them up.
"""

from pathlib import Path


# ==================== EDIT THESE PATHS ====================

# Compressonator CLI executable
COMPRESSONATOR_CLI = r"D:\Compressonatorcli\bin\CLI\compressonatorcli.exe"

# Source texture images — single-channel (grayscale) textures for BC4
SOURCE_IMAGES = [
    r"D:\BC1 extract\Data\MetalPlates013_4K-PNG\MetalPlates013_4K-PNG_Displacement.png",
    r"D:\BC1 extract\Data\MetalPlates013_4K-PNG\MetalPlates013_4K-PNG_Roughness.png",
    r"D:\BC1 extract\Data\MetalPlates013_4K-PNG\MetalPlates013_4K-PNG_AmbientOcclusion.png",
    r"D:\BC1 extract\Data\MetalPlates013_4K-PNG\MetalPlates013_4K-PNG_Metalness.png"
   
]

# Human-readable names matching SOURCE_IMAGES (used for output naming)
TEXTURE_NAMES = ["Displacement","Roughness","AO","Metalness"]


# Output directory (everything goes here: dataset, training, inference)
MODEL_DIR = r"D:\BC1 extract\NTBC_Outputs\BC4\MetalPlates013_4K-PNG_model_vbc4"


# ==================== DERIVED PATHS (auto) ====================

TRAIN_DATASET_JSON = str(Path(MODEL_DIR) / "Train_dataset.json")
INFERENCE_INPUT_JSON = str(Path(MODEL_DIR) / "Inference_input.json")
MERGED_CHECKPOINT = str(Path(MODEL_DIR) / "ntbc_bc4_merged_compressed.pt")
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
