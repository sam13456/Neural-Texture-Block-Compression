"""Auto-generated config for VNTBC BC4 pipeline."""
from pathlib import Path

COMPRESSONATOR_CLI = r"D:/Compressonatorcli/bin/CLI/compressonatorcli.exe"

SOURCE_IMAGES = [
    r"D:\BC1 extract\Data\Bricks090_4K-PNG\Bricks090_4K-PNG_AmbientOcclusion.png",
    r"D:\BC1 extract\Data\Bricks090_4K-PNG\Bricks090_4K-PNG_Displacement.png",
    r"D:\BC1 extract\Data\Bricks090_4K-PNG\Bricks090_4K-PNG_Roughness.png",
]

TEXTURE_NAMES = ["AO", "Displacement", "Roughness"]

MODEL_DIR = r"D:\BC1 extract\VNTBC_UI_Test_Output\Bricks090_4K-PNG_VNTBC\BC4"

QAT_BITS_ENDPOINT = [8, 4, 4, 4, 4, 4, 8]
QAT_BITS_COLOR = [8, 4, 4, 4, 4, 4, 4, 4]

USE_LPE = True
LPE_N = 128
LPE_N_FREQ = 6
LPE_D0 = 8

TRAIN_DATASET_JSON = str(Path(MODEL_DIR) / "Train_dataset.json")
INFERENCE_INPUT_JSON = str(Path(MODEL_DIR) / "Inference_input.json")
MERGED_CHECKPOINT = str(Path(MODEL_DIR) / "v_ntbc_bc4_merged_compressed.pt")
OUT_DIR_ENDPOINT = str(Path(MODEL_DIR) / "runs_endpoint")
OUT_DIR_COLOR = str(Path(MODEL_DIR) / "runs_color")

INFERENCE_OUTPUT_DIR = str(Path(MODEL_DIR) / "inference_output")
OUT_DDS = str(Path(INFERENCE_OUTPUT_DIR) / "ntbc_out.dds")
OUT_PREVIEW_PNG = str(Path(INFERENCE_OUTPUT_DIR) / "ntbc_out_preview.png")

REF_DDS = [
    str(Path(MODEL_DIR) / f"{Path(s).stem}_ref_bc4.dds")
    for s in SOURCE_IMAGES
]

TEST_DDS = [
    str(Path(INFERENCE_OUTPUT_DIR) / f"ntbc_out_{n}.dds")
    for n in TEXTURE_NAMES
]

# Training overrides (from UI)
MAIN_STEPS = 20000
LR_GRID = 0.013
LR_MLP = 0.005
TEMPERATURE = 0.01
BATCH_SIZE_BLOCKS = 5096
BATCH_SIZE_TEXELS = 131072
