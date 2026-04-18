"""
VNTBC Pipeline Orchestrator

Generates temporary config files, runs the VNTBC pipeline stages as
subprocesses, generates diff maps, and reports progress via callbacks.
"""

import json
import os
import re
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image


# ---------------------------------------------------------------------------
# Default advanced settings
# ---------------------------------------------------------------------------

DEFAULT_ADVANCED = {
    "main_steps": 20000,
    "lr_grid": 0.013,
    "lr_mlp": 0.005,
    "temperature": 0.01,
    "batch_size_blocks": 5096,
    "batch_size_texels": 131072,
    "log_every_steps": 500,
    "save_every_steps": 5000,
    "bc1_qat_bits_endpoint": [8, 8, 8, 8, 8, 8, 8],
    "bc1_qat_bits_color": [8, 4, 4, 4, 4, 4, 4, 4],
    "bc1_use_lpe": True,
    "bc4_qat_bits_endpoint": [8, 4, 4, 4, 4, 4, 8],
    "bc4_qat_bits_color": [8, 4, 4, 4, 4, 4, 4, 4],
    "bc4_use_lpe": False,
}


# ---------------------------------------------------------------------------
# Settings persistence
# ---------------------------------------------------------------------------

SETTINGS_FILE = Path(__file__).parent / "vntbc_settings.json"


def load_settings() -> dict:
    defaults = {
        "compressonator_cli": "",
        "run_evaluation": True,
        "generate_diff_maps": True,
        "advanced": dict(DEFAULT_ADVANCED),
    }
    if SETTINGS_FILE.exists():
        try:
            saved = json.loads(SETTINGS_FILE.read_text())
            defaults.update(saved)
            # Ensure advanced has all keys
            for k, v in DEFAULT_ADVANCED.items():
                if k not in defaults.get("advanced", {}):
                    defaults.setdefault("advanced", {})[k] = v
        except Exception:
            pass
    return defaults


def save_settings(settings: dict):
    SETTINGS_FILE.write_text(json.dumps(settings, indent=2))


# ---------------------------------------------------------------------------
# Eval result dataclass
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    """One row of PSNR/SSIM evaluation results."""
    name: str
    fmt: str  # "BC1" or "BC4"
    psnr_ref: float = 0.0
    psnr_ntbc: float = 0.0
    psnr_delta: float = 0.0
    ssim_ref: Optional[float] = None
    ssim_ntbc: Optional[float] = None
    ssim_delta: Optional[float] = None


# Regex to parse eval output lines
_EVAL_RE = re.compile(
    r'^(.+?):\s+PSNR ref=([\d.]+)\s+ntbc=([\d.]+)\s+Delta=([+\-\d.]+)'
    r'(?:\s*\|\s*SSIM ref=([\d.]+)\s+ntbc=([\d.]+)\s+Delta=([+\-\d.]+))?'
)


def parse_eval_line(line: str, fmt: str) -> Optional[EvalResult]:
    """Try to parse an eval output line into an EvalResult."""
    m = _EVAL_RE.match(line.strip())
    if not m:
        return None
    r = EvalResult(
        name=m.group(1).strip(),
        fmt=fmt,
        psnr_ref=float(m.group(2)),
        psnr_ntbc=float(m.group(3)),
        psnr_delta=float(m.group(4)),
    )
    if m.group(5):
        r.ssim_ref = float(m.group(5))
        r.ssim_ntbc = float(m.group(6))
        r.ssim_delta = float(m.group(7))
    return r


# ---------------------------------------------------------------------------
# Config generation helpers
# ---------------------------------------------------------------------------

def _source_images_block(images: List[str]) -> str:
    lines = ["SOURCE_IMAGES = ["]
    for img in images:
        lines.append(f'    r"{img}",')
    lines.append("]")
    return "\n".join(lines)


def _texture_names_block(names: List[str]) -> str:
    inner = ", ".join(f'"{n}"' for n in names)
    return f"TEXTURE_NAMES = [{inner}]"


def _training_overrides(adv: dict) -> List[str]:
    """Generate optional training override lines for config."""
    lines = [
        "",
        "# Training overrides (from UI)",
        f"MAIN_STEPS = {adv.get('main_steps', 20000)}",
        f"LR_GRID = {adv.get('lr_grid', 0.013)}",
        f"LR_MLP = {adv.get('lr_mlp', 0.005)}",
        f"TEMPERATURE = {adv.get('temperature', 0.01)}",
        f"BATCH_SIZE_BLOCKS = {adv.get('batch_size_blocks', 5096)}",
        f"BATCH_SIZE_TEXELS = {adv.get('batch_size_texels', 131072)}",
        f"LOG_EVERY_STEPS = {adv.get('log_every_steps', 500)}",
        f"SAVE_EVERY_STEPS = {adv.get('save_every_steps', 5000)}",
    ]
    return lines


def generate_bc1_config(
    config_path: Path,
    source_images: List[str],
    texture_names: List[str],
    model_dir: str,
    compressonator_cli: str,
    advanced: Optional[dict] = None,
) -> None:
    adv = advanced or DEFAULT_ADVANCED
    qat_ep = adv.get("bc1_qat_bits_endpoint", [8, 8, 8, 8, 8, 8, 8])
    qat_co = adv.get("bc1_qat_bits_color", [8, 4, 4, 4, 4, 4, 4, 4])
    use_lpe = adv.get("bc1_use_lpe", True)

    lines = [
        '"""Auto-generated config for VNTBC BC1 pipeline."""',
        "from pathlib import Path",
        "",
        f'COMPRESSONATOR_CLI = r"{compressonator_cli}"',
        "",
        _source_images_block(source_images),
        "",
        _texture_names_block(texture_names),
        "",
        f'MODEL_DIR = r"{model_dir}"',
        "",
        f"QAT_BITS_ENDPOINT = {qat_ep}",
        f"QAT_BITS_COLOR = {qat_co}",
        "",
        f"USE_LPE = {use_lpe}",
        "LPE_N = 128",
        "LPE_N_FREQ = 6",
        "LPE_D0 = 8",
        "",
        'TRAIN_DATASET_JSON = str(Path(MODEL_DIR) / "Train_dataset.json")',
        'INFERENCE_INPUT_JSON = str(Path(MODEL_DIR) / "Inference_input.json")',
        'MERGED_CHECKPOINT = str(Path(MODEL_DIR) / "vntbc_bc1_merged_compressed.pt")',
        'OUT_DIR_ENDPOINT = str(Path(MODEL_DIR) / "runs_endpoint")',
        'OUT_DIR_COLOR = str(Path(MODEL_DIR) / "runs_color")',
        "",
        'INFERENCE_OUTPUT_DIR = str(Path(MODEL_DIR) / "inference_output")',
        'OUT_DDS = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out.dds")',
        'OUT_PREVIEW_PNG = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out_preview.png")',
        "",
        "REF_DDS = [",
        '    str(Path(MODEL_DIR) / f"{Path(s).stem}_ref_bc1.dds")',
        "    for s in SOURCE_IMAGES",
        "]",
        "",
        "TEST_DDS = [",
        '    str(Path(INFERENCE_OUTPUT_DIR) / f"vntbc_out_{n}.dds")',
        "    for n in TEXTURE_NAMES",
        "]",
    ]
    lines.extend(_training_overrides(adv))
    lines.append("")
    config_path.write_text("\n".join(lines))


def generate_bc4_config(
    config_path: Path,
    source_images: List[str],
    texture_names: List[str],
    model_dir: str,
    compressonator_cli: str,
    advanced: Optional[dict] = None,
) -> None:
    adv = advanced or DEFAULT_ADVANCED
    qat_ep = adv.get("bc4_qat_bits_endpoint", [8, 4, 4, 4, 4, 4, 8])
    qat_co = adv.get("bc4_qat_bits_color", [8, 4, 4, 4, 4, 4, 4, 4])
    use_lpe = adv.get("bc4_use_lpe", False)

    lines = [
        '"""Auto-generated config for VNTBC BC4 pipeline."""',
        "from pathlib import Path",
        "",
        f'COMPRESSONATOR_CLI = r"{compressonator_cli}"',
        "",
        _source_images_block(source_images),
        "",
        _texture_names_block(texture_names),
        "",
        f'MODEL_DIR = r"{model_dir}"',
        "",
        f"QAT_BITS_ENDPOINT = {qat_ep}",
        f"QAT_BITS_COLOR = {qat_co}",
        "",
        f"USE_LPE = {use_lpe}",
        "LPE_N = 128",
        "LPE_N_FREQ = 6",
        "LPE_D0 = 8",
        "",
        'TRAIN_DATASET_JSON = str(Path(MODEL_DIR) / "Train_dataset.json")',
        'INFERENCE_INPUT_JSON = str(Path(MODEL_DIR) / "Inference_input.json")',
        'MERGED_CHECKPOINT = str(Path(MODEL_DIR) / "v_ntbc_bc4_merged_compressed.pt")',
        'OUT_DIR_ENDPOINT = str(Path(MODEL_DIR) / "runs_endpoint")',
        'OUT_DIR_COLOR = str(Path(MODEL_DIR) / "runs_color")',
        "",
        'INFERENCE_OUTPUT_DIR = str(Path(MODEL_DIR) / "inference_output")',
        'OUT_DDS = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out.dds")',
        'OUT_PREVIEW_PNG = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out_preview.png")',
        "",
        "REF_DDS = [",
        '    str(Path(MODEL_DIR) / f"{Path(s).stem}_ref_bc4.dds")',
        "    for s in SOURCE_IMAGES",
        "]",
        "",
        "TEST_DDS = [",
        '    str(Path(INFERENCE_OUTPUT_DIR) / f"vntbc_out_{n}.dds")',
        "    for n in TEXTURE_NAMES",
        "]",
    ]
    lines.extend(_training_overrides(adv))
    lines.append("")
    config_path.write_text("\n".join(lines))


# ---------------------------------------------------------------------------
# Subprocess runner
# ---------------------------------------------------------------------------

def _clear_pycache(directory: Path):
    pc = directory / "__pycache__"
    if pc.exists():
        shutil.rmtree(pc, ignore_errors=True)


def run_subprocess(
    script_path: str,
    cwd: str,
    on_log: Callable[[str], None],
    cancel_event: threading.Event,
) -> int:
    proc = subprocess.Popen(
        [sys.executable, "-u", script_path],
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    try:
        for line in proc.stdout:
            if cancel_event.is_set():
                proc.terminate()
                proc.wait(timeout=5)
                raise RuntimeError("Cancelled by user")
            on_log(line.rstrip("\n"))
        proc.wait()
    except Exception:
        proc.kill()
        proc.wait()
        raise
    return proc.returncode


# ---------------------------------------------------------------------------
# Diff map generation
# ---------------------------------------------------------------------------

def generate_diff_map(ref_png: str, out_png: str, save_path: str, amplify: float = 5.0) -> Optional[str]:
    ref_p, out_p = Path(ref_png), Path(out_png)
    if not ref_p.exists() or not out_p.exists():
        return None
    ref_img = np.array(Image.open(ref_p).convert("RGB"))
    out_img = np.array(Image.open(out_p).convert("RGB"))
    if ref_img.shape != out_img.shape:
        h = min(ref_img.shape[0], out_img.shape[0])
        w = min(ref_img.shape[1], out_img.shape[1])
        ref_img, out_img = ref_img[:h, :w], out_img[:h, :w]
    diff = np.abs(ref_img.astype(np.float32) - out_img.astype(np.float32))
    diff_map = np.clip(diff * amplify, 0, 255).astype(np.uint8)
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(diff_map).save(save_path)
    return save_path


def _generate_all_diff_maps(model_dir, source_images, texture_names, fmt, on_log):
    inf_dir = Path(model_dir) / "inference_output"
    single = len(texture_names) == 1
    for src, name in zip(source_images, texture_names):
        safe_name = name.replace(" ", "_")
        ref_png = inf_dir / f"{Path(src).stem}_ref_{fmt}.png"
        out_png = inf_dir / ("vntbc_out_preview.png" if single else f"vntbc_out_preview_{safe_name}.png")
        diff_path = inf_dir / f"{safe_name}_diff_x5.png"
        if generate_diff_map(str(ref_png), str(out_png), str(diff_path)):
            on_log(f"[Diff Map] Generated: {diff_path.name}")
        else:
            on_log(f"[Diff Map] Skipped (missing files): {name}")


# ---------------------------------------------------------------------------
# Pipeline data classes
# ---------------------------------------------------------------------------

@dataclass
class StageInfo:
    name: str
    status: str = "pending"


@dataclass
class PipelineJob:
    bc1_images: List[str] = field(default_factory=list)
    bc1_names: List[str] = field(default_factory=list)
    bc4_images: List[str] = field(default_factory=list)
    bc4_names: List[str] = field(default_factory=list)
    output_dir: str = ""
    folder_name: str = ""
    compressonator_cli: str = ""
    run_evaluation: bool = True
    generate_diff_maps: bool = True
    advanced: dict = field(default_factory=lambda: dict(DEFAULT_ADVANCED))


def build_stages(job: PipelineJob) -> List[StageInfo]:
    stages = []
    if job.bc1_images:
        stages.append(StageInfo("Dataset Extraction (BC1)"))
        stages.append(StageInfo("Training (BC1)"))
        stages.append(StageInfo("Inference (BC1)"))
        if job.run_evaluation:
            stages.append(StageInfo("Evaluation (BC1)"))
    if job.bc4_images:
        stages.append(StageInfo("Dataset Extraction (BC4)"))
        stages.append(StageInfo("Training (BC4)"))
        stages.append(StageInfo("Inference (BC4)"))
        if job.run_evaluation:
            stages.append(StageInfo("Evaluation (BC4)"))
    return stages


# ---------------------------------------------------------------------------
# Pipeline runner
# ---------------------------------------------------------------------------

def run_pipeline(
    job: PipelineJob,
    on_stage_change: Callable[[int, str], None],
    on_log: Callable[[str], None],
    on_complete: Callable[[bool, str, List[EvalResult]], None],
    cancel_event: threading.Event,
):
    ui_dir = Path(__file__).parent
    bc1_dir = ui_dir / "VNTBC_BC1"
    bc4_dir = ui_dir / "VNTBC_BC4"

    stages = build_stages(job)
    stage_idx = 0
    eval_results: List[EvalResult] = []
    current_eval_fmt = ""

    def next_stage():
        nonlocal stage_idx
        stage_idx += 1

    def log_and_parse(line: str):
        on_log(line)
        if current_eval_fmt:
            r = parse_eval_line(line, current_eval_fmt)
            if r:
                eval_results.append(r)

    def run_script(script_name: str, cwd: Path, parse_eval: str = ""):
        nonlocal current_eval_fmt
        current_eval_fmt = parse_eval
        on_stage_change(stage_idx, "running")
        on_log(f"\n{'='*60}")
        on_log(f"  Running: {script_name}")
        on_log(f"{'='*60}\n")
        rc = run_subprocess(script_name, str(cwd), log_and_parse, cancel_event)
        current_eval_fmt = ""
        if rc != 0:
            on_stage_change(stage_idx, "error")
            raise RuntimeError(f"{script_name} exited with code {rc}")
        on_stage_change(stage_idx, "done")
        next_stage()

    try:
        # ---- BC1 pipeline ----
        if job.bc1_images:
            bc1_model = str(Path(job.output_dir) / f"{job.folder_name}_VNTBC" / "BC1")
            Path(bc1_model).mkdir(parents=True, exist_ok=True)

            on_log("[Config] Generating BC1 config...")
            generate_bc1_config(
                bc1_dir / "config.py",
                job.bc1_images, job.bc1_names,
                bc1_model, job.compressonator_cli, job.advanced,
            )
            _clear_pycache(bc1_dir)

            run_script("Dataset_Input_Extract.py", bc1_dir)
            run_script("Train_combined.py", bc1_dir)
            run_script("Inference_DDS.py", bc1_dir)
            if job.run_evaluation:
                _clear_pycache(bc1_dir)
                run_script("ntbc_eval.py", bc1_dir, parse_eval="BC1")

        # ---- BC4 pipeline ----
        if job.bc4_images:
            bc4_model = str(Path(job.output_dir) / f"{job.folder_name}_VNTBC"/ "BC4")
            Path(bc4_model).mkdir(parents=True, exist_ok=True)

            on_log("[Config] Generating BC4 config...")
            generate_bc4_config(
                bc4_dir / "config_BC4.py",
                job.bc4_images, job.bc4_names,
                bc4_model, job.compressonator_cli, job.advanced,
            )
            _clear_pycache(bc4_dir)

            run_script("Dataset_Input_Extract_BC4.py", bc4_dir)
            run_script("Train_combined_BC4.py", bc4_dir)
            run_script("Inference_DDS_BC4.py", bc4_dir)
            if job.run_evaluation:
                _clear_pycache(bc4_dir)
                run_script("ntbc_eval.py", bc4_dir, parse_eval="BC4")

        # ---- Diff maps (silent, no stage row) ----
        if job.generate_diff_maps:
            on_log(f"\n{'='*60}")
            on_log("  Generating Diff Maps")
            on_log(f"{'='*60}\n")
            if job.bc1_images:
                bc1_model = str(Path(job.output_dir) / f"{job.folder_name}_VNTBC" / "BC1")
                _generate_all_diff_maps(bc1_model, job.bc1_images, job.bc1_names, "bc1", on_log)
            if job.bc4_images:
                bc4_model = str(Path(job.output_dir) / f"{job.folder_name}_VNTBC" / "BC4")
                _generate_all_diff_maps(bc4_model, job.bc4_images, job.bc4_names, "bc4", on_log)

        on_complete(True, "Pipeline completed successfully!", eval_results)

    except RuntimeError as e:
        on_log(f"\n[ERROR] {e}")
        on_complete(False, str(e), eval_results)
    except Exception as e:
        on_log(f"\n[ERROR] Unexpected: {e}")
        on_complete(False, str(e), eval_results)


# ---------------------------------------------------------------------------
# Inference-only mode
# ---------------------------------------------------------------------------

@dataclass
class InferenceJob:
    """Job for running inference + eval from a pre-trained model."""
    model_checkpoint: str = ""          # path to .pt file
    inference_json: str = ""            # path to Inference_input.json
    source_images: List[str] = field(default_factory=list)  # original PNGs for eval
    texture_names: List[str] = field(default_factory=list)  # from JSON
    pipeline_type: str = "BC1"          # "BC1" or "BC4"
    output_dir: str = ""                # model dir (parent of inference_output)
    compressonator_cli: str = ""
    run_evaluation: bool = True
    generate_diff_maps: bool = True


def read_texture_names_from_json(json_path: str) -> List[str]:
    """Read texture_names from an Inference_input.json file."""
    try:
        data = json.loads(Path(json_path).read_text())
        names = data.get("texture_names", [])
        if not names:
            n = data.get("num_textures", 1)
            names = [f"tex{i:02d}" for i in range(n)]
        return names
    except Exception:
        return []


def generate_inference_bc1_config(
    config_path: Path,
    model_dir: str,
    checkpoint_path: str,
    inference_json: str,
    source_images: List[str],
    texture_names: List[str],
    compressonator_cli: str,
) -> None:
    """Generate a minimal config.py for inference-only BC1 runs."""
    lines = [
        '"""Auto-generated config for VNTBC BC1 inference-only."""',
        "from pathlib import Path",
        "",
        f'COMPRESSONATOR_CLI = r"{compressonator_cli}"',
        "",
        _source_images_block(source_images),
        "",
        _texture_names_block(texture_names),
        "",
        f'MODEL_DIR = r"{model_dir}"',
        "",
        f'INFERENCE_INPUT_JSON = r"{inference_json}"',
        f'MERGED_CHECKPOINT = r"{checkpoint_path}"',
        "",
        'INFERENCE_OUTPUT_DIR = str(Path(MODEL_DIR) / "inference_output")',
        'OUT_DDS = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out.dds")',
        'OUT_PREVIEW_PNG = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out_preview.png")',
        "",
        "REF_DDS = [",
        '    str(Path(MODEL_DIR) / f"{Path(s).stem}_ref_bc1.dds")',
        "    for s in SOURCE_IMAGES",
        "]",
        "",
        "TEST_DDS = [",
        '    str(Path(INFERENCE_OUTPUT_DIR) / f"vntbc_out_{n}.dds")',
        "    for n in TEXTURE_NAMES",
        "]",
        "",
    ]
    config_path.write_text("\n".join(lines))


def generate_inference_bc4_config(
    config_path: Path,
    model_dir: str,
    checkpoint_path: str,
    inference_json: str,
    source_images: List[str],
    texture_names: List[str],
    compressonator_cli: str,
) -> None:
    """Generate a minimal config_BC4.py for inference-only BC4 runs."""
    lines = [
        '"""Auto-generated config for VNTBC BC4 inference-only."""',
        "from pathlib import Path",
        "",
        f'COMPRESSONATOR_CLI = r"{compressonator_cli}"',
        "",
        _source_images_block(source_images),
        "",
        _texture_names_block(texture_names),
        "",
        f'MODEL_DIR = r"{model_dir}"',
        "",
        f'INFERENCE_INPUT_JSON = r"{inference_json}"',
        f'MERGED_CHECKPOINT = r"{checkpoint_path}"',
        "",
        'INFERENCE_OUTPUT_DIR = str(Path(MODEL_DIR) / "inference_output")',
        'OUT_DDS = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out.dds")',
        'OUT_PREVIEW_PNG = str(Path(INFERENCE_OUTPUT_DIR) / "vntbc_out_preview.png")',
        "",
        "REF_DDS = [",
        '    str(Path(MODEL_DIR) / f"{Path(s).stem}_ref_bc4.dds")',
        "    for s in SOURCE_IMAGES",
        "]",
        "",
        "TEST_DDS = [",
        '    str(Path(INFERENCE_OUTPUT_DIR) / f"vntbc_out_{n}.dds")',
        "    for n in TEXTURE_NAMES",
        "]",
        "",
    ]
    config_path.write_text("\n".join(lines))


def _generate_ref_dds(source_images: List[str], model_dir: str,
                      fmt: str, compressonator_cli: str,
                      on_log: Callable[[str], None]) -> None:
    """Generate reference DDS files using CompressonatorCLI if they don't exist."""
    bc_fmt = "BC1" if fmt == "bc1" else "BC4"
    for src in source_images:
        if not src:
            continue
        src_p = Path(src)
        ref_dds = Path(model_dir) / f"{src_p.stem}_ref_{fmt}.dds"
        if ref_dds.exists():
            on_log(f"[Ref DDS] Already exists: {ref_dds.name}")
            continue
        if not src_p.exists():
            on_log(f"[Ref DDS] Source image not found: {src}")
            continue
        on_log(f"[Ref DDS] Generating: {ref_dds.name}")
        try:
            subprocess.run(
                [compressonator_cli, "-fd", bc_fmt, str(src_p), str(ref_dds)],
                check=True, capture_output=True, text=True,
            )
        except Exception as e:
            on_log(f"[Ref DDS] Warning: {e}")


def build_inference_stages(job: InferenceJob) -> List[StageInfo]:
    """Build stage list for inference-only mode."""
    fmt = job.pipeline_type
    stages = [StageInfo(f"Inference ({fmt})")]
    if job.run_evaluation and job.source_images:
        stages.append(StageInfo(f"Evaluation ({fmt})"))
    return stages


def run_inference_pipeline(
    job: InferenceJob,
    on_stage_change: Callable[[int, str], None],
    on_log: Callable[[str], None],
    on_complete: Callable[[bool, str, List[EvalResult]], None],
    cancel_event: threading.Event,
):
    """Run inference + optional eval from a pre-trained model."""
    ui_dir = Path(__file__).parent
    bc1_dir = ui_dir / "VNTBC_BC1"
    bc4_dir = ui_dir / "VNTBC_BC4"

    stages = build_inference_stages(job)
    stage_idx = 0
    eval_results: List[EvalResult] = []
    current_eval_fmt = ""
    fmt = job.pipeline_type  # "BC1" or "BC4"

    def next_stage():
        nonlocal stage_idx
        stage_idx += 1

    def log_and_parse(line: str):
        on_log(line)
        if current_eval_fmt:
            r = parse_eval_line(line, current_eval_fmt)
            if r:
                eval_results.append(r)

    def run_script(script_name: str, cwd: Path, parse_eval: str = ""):
        nonlocal current_eval_fmt
        current_eval_fmt = parse_eval
        on_stage_change(stage_idx, "running")
        on_log(f"\n{'='*60}")
        on_log(f"  Running: {script_name}")
        on_log(f"{'='*60}\n")
        rc = run_subprocess(script_name, str(cwd), log_and_parse, cancel_event)
        current_eval_fmt = ""
        if rc != 0:
            on_stage_change(stage_idx, "error")
            raise RuntimeError(f"{script_name} exited with code {rc}")
        on_stage_change(stage_idx, "done")
        next_stage()

    try:
        model_dir = job.output_dir
        Path(model_dir).mkdir(parents=True, exist_ok=True)

        if fmt == "BC1":
            on_log("[Config] Generating BC1 inference config...")
            generate_inference_bc1_config(
                bc1_dir / "config.py",
                model_dir, job.model_checkpoint, job.inference_json,
                job.source_images, job.texture_names, job.compressonator_cli,
            )
            _clear_pycache(bc1_dir)

            # Generate ref DDS if we have source images + compressonator
            if job.source_images and job.compressonator_cli:
                _generate_ref_dds(job.source_images, model_dir, "bc1",
                                  job.compressonator_cli, on_log)

            run_script("Inference_DDS.py", bc1_dir)
            if job.run_evaluation and job.source_images:
                _clear_pycache(bc1_dir)
                run_script("ntbc_eval.py", bc1_dir, parse_eval="BC1")

        else:  # BC4
            on_log("[Config] Generating BC4 inference config...")
            generate_inference_bc4_config(
                bc4_dir / "config_BC4.py",
                model_dir, job.model_checkpoint, job.inference_json,
                job.source_images, job.texture_names, job.compressonator_cli,
            )
            _clear_pycache(bc4_dir)

            if job.source_images and job.compressonator_cli:
                _generate_ref_dds(job.source_images, model_dir, "bc4",
                                  job.compressonator_cli, on_log)

            run_script("Inference_DDS_BC4.py", bc4_dir)
            if job.run_evaluation and job.source_images:
                _clear_pycache(bc4_dir)
                run_script("ntbc_eval.py", bc4_dir, parse_eval="BC4")

        # Diff maps (silent)
        if job.generate_diff_maps and job.source_images:
            on_log(f"\n{'='*60}")
            on_log("  Generating Diff Maps")
            on_log(f"{'='*60}\n")
            _generate_all_diff_maps(model_dir, job.source_images,
                                    job.texture_names, fmt.lower(), on_log)

        on_complete(True, "Inference completed successfully!", eval_results)

    except RuntimeError as e:
        on_log(f"\n[ERROR] {e}")
        on_complete(False, str(e), eval_results)
    except Exception as e:
        on_log(f"\n[ERROR] Unexpected: {e}")
        on_complete(False, str(e), eval_results)
