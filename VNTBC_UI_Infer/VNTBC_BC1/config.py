"""Auto-generated config for VNTBC BC1 inference-only."""
from pathlib import Path

COMPRESSONATOR_CLI = r"D:/Compressonatorcli/bin/CLI/compressonatorcli.exe"

SOURCE_IMAGES = [
    r"D:\VNTBC\Data\Bricks090_4K-PNG\Bricks090_4K-PNG_Color.png",
    r"D:\VNTBC\Data\Bricks090_4K-PNG\Bricks090_4K-PNG_NormalDX.png",
]

TEXTURE_NAMES = ["Color", "NormalDX"]

MODEL_DIR = r"D:\VNTBC\VNTBC_UI_Test_Output\Bricks090_4K-PNG_VNTBC\BC1"

INFERENCE_INPUT_JSON = r"D:\VNTBC\VNTBC_UI_Test_Output\Bricks090_4K-PNG_VNTBC\BC1\Inference_input.json"
MERGED_CHECKPOINT = r"D:/VNTBC/VNTBC_UI_Test_Output/Bricks090_4K-PNG_VNTBC/BC1/vntbc_bc1_merged_compressed.pt"

INFERENCE_OUTPUT_DIR = str(Path(MODEL_DIR) / "inference_output")
OUT_DDS = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out.dds")
OUT_PREVIEW_PNG = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out_preview.png")

REF_DDS = [
    str(Path(MODEL_DIR) / f"{Path(s).stem}_ref_bc1.dds")
    for s in SOURCE_IMAGES
]

TEST_DDS = [
    str(Path(INFERENCE_OUTPUT_DIR) / f"vntbc_out_{n}.dds")
    for n in TEXTURE_NAMES
]
