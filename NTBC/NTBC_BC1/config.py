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
    r"D:\BC1 extract\Data\roof_09_4k\roof_09_diff_4k.png",
    r"D:\BC1 extract\Data\roof_09_4k\roof_09_nor_dx_4k.png",
    r"D:\BC1 extract\Data\roof_09_4k\roof_09_arm_4k.png"
]

# Human-readable names matching SOURCE_IMAGES (used for output naming)
TEXTURE_NAMES = ["Color", "NormalDX","Arm"]  

# Output directory (everything goes here: dataset, training, inference)
MODEL_DIR = r"D:\BC1 extract\roof_09_4k_model"


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
    str(Path(INFERENCE_OUTPUT_DIR) / f"ntbc_out_Disp.dds")
    for n in TEXTURE_NAMES
]

