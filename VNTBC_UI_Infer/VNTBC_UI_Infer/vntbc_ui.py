"""
VNTBC — Variable Bit Rate Neural Texture Block Compression UI

Native desktop application using CustomTkinter.
"""

import os
import sys
import threading
import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path
from typing import List, Optional

import customtkinter as ctk

from vntbc_classifier import TextureInfo, scan_texture_folder
from vntbc_orchestrator import (
    PipelineJob, InferenceJob, StageInfo, EvalResult, build_stages,
    build_inference_stages, load_settings, save_settings, run_pipeline,
    run_inference_pipeline, read_texture_names_from_json, DEFAULT_ADVANCED,
)

# ---------------------------------------------------------------------------
# Theme
# ---------------------------------------------------------------------------
ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

ACCENT = "#2277ba"
ACCENT_HOVER = "#1b5e94"
BC1_COLOR = "#2e7d32"
BC4_COLOR = "#2277ba"
ERROR_COLOR = "#EF5350"
SUCCESS_COLOR = "#2e7d32"
PENDING_COLOR = "#555555"
RUNNING_COLOR = "#2277ba"
BG_DARK = "#0D0D0D"
BG_CARD = "#1A1A1A"
FG_TEXT = "#E0E0E0"
FG_DIM = "#808080"

FONT_TITLE = ("Segoe UI", 24, "bold")
FONT_HEADING = ("Segoe UI", 14, "bold")
FONT_BODY = ("Segoe UI", 13)
FONT_SMALL = ("Segoe UI", 11)
FONT_MONO = ("Consolas", 11)


# ---------------------------------------------------------------------------
# Settings Dialog (Tabbed)
# ---------------------------------------------------------------------------
class SettingsDialog(ctk.CTkToplevel):
    def __init__(self, parent, settings: dict):
        super().__init__(parent)
        self.title("Settings")
        self.geometry("680x560")
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        self.result: Optional[dict] = None
        self.settings = dict(settings)
        self.adv = dict(self.settings.get("advanced", DEFAULT_ADVANCED))

        self.update_idletasks()
        px = parent.winfo_x() + (parent.winfo_width() - 680) // 2
        py = parent.winfo_y() + (parent.winfo_height() - 560) // 2
        self.geometry(f"+{px}+{py}")

        tabs = ctk.CTkTabview(self, width=640, height=460)
        tabs.pack(padx=20, pady=(10, 0), fill="both", expand=True)

        # ---- General tab ----
        gen_tab = tabs.add("General")
        ctk.CTkLabel(gen_tab, text="Compressonator CLI Executable", font=FONT_HEADING).pack(anchor="w", pady=(10, 2))
        ctk.CTkLabel(gen_tab, text="Required for reference DDS generation. Set once.", font=FONT_BODY, text_color=FG_DIM).pack(anchor="w")
        row = ctk.CTkFrame(gen_tab, fg_color="transparent")
        row.pack(fill="x", pady=(6, 0))
        self.comp_var = ctk.StringVar(value=self.settings.get("compressonator_cli", ""))
        ctk.CTkEntry(row, textvariable=self.comp_var, font=FONT_BODY).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row, text="Browse", width=80, fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._browse_comp).pack(side="right")

        # ---- Advanced tab ----
        adv_tab = tabs.add("Advanced")
        adv_scroll = ctk.CTkScrollableFrame(adv_tab, fg_color="transparent")
        adv_scroll.pack(fill="both", expand=True)

        ctk.CTkLabel(adv_scroll, text="Training Parameters", font=FONT_HEADING).pack(anchor="w", pady=(6, 4))

        self.adv_vars = {}
        train_params = [
            ("main_steps", "Training Steps", "int"),
            ("lr_grid", "Grid Learning Rate", "float"),
            ("lr_mlp", "MLP Learning Rate", "float"),
            ("temperature", "STE Temperature", "float"),
            ("batch_size_blocks", "Batch Size (Blocks)", "int"),
            ("batch_size_texels", "Batch Size (Texels)", "int"),
        ]
        for key, label, _ in train_params:
            f = ctk.CTkFrame(adv_scroll, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=label, font=FONT_BODY, width=200, anchor="w").pack(side="left")
            var = ctk.StringVar(value=str(self.adv.get(key, DEFAULT_ADVANCED[key])))
            ctk.CTkEntry(f, textvariable=var, font=FONT_MONO, width=160).pack(side="right")
            self.adv_vars[key] = (var, _)

        ctk.CTkLabel(adv_scroll, text="", font=("Segoe UI", 4)).pack()
        ctk.CTkLabel(adv_scroll, text="Logging & Checkpoints", font=FONT_HEADING).pack(anchor="w", pady=(6, 4))

        log_params = [
            ("log_every_steps", "Log Every N Steps", "int"),
            ("save_every_steps", "Save Checkpoint Every N Steps", "int"),
        ]
        for key, label, _ in log_params:
            f = ctk.CTkFrame(adv_scroll, fg_color="transparent")
            f.pack(fill="x", pady=2)
            ctk.CTkLabel(f, text=label, font=FONT_BODY, width=240, anchor="w").pack(side="left")
            var = ctk.StringVar(value=str(self.adv.get(key, DEFAULT_ADVANCED[key])))
            ctk.CTkEntry(f, textvariable=var, font=FONT_MONO, width=120).pack(side="right")
            self.adv_vars[key] = (var, _)

        ctk.CTkLabel(adv_scroll, text="", font=("Segoe UI", 4)).pack()  # spacer
        ctk.CTkLabel(adv_scroll, text="BC1 (RGB) Settings", font=FONT_HEADING).pack(anchor="w", pady=(6, 4))

        self.bc1_lpe_var = ctk.BooleanVar(value=self.adv.get("bc1_use_lpe", True))
        ctk.CTkCheckBox(adv_scroll, text="Enable LPE (Local Positional Encoding)", variable=self.bc1_lpe_var, font=FONT_BODY, fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(anchor="w", pady=2)

        f = ctk.CTkFrame(adv_scroll, fg_color="transparent")
        f.pack(fill="x", pady=2)
        ctk.CTkLabel(f, text="QAT Bits (Endpoint)", font=FONT_BODY, width=200, anchor="w").pack(side="left")
        self.bc1_qat_ep_var = ctk.StringVar(value=str(self.adv.get("bc1_qat_bits_endpoint", DEFAULT_ADVANCED["bc1_qat_bits_endpoint"])))
        ctk.CTkEntry(f, textvariable=self.bc1_qat_ep_var, font=FONT_MONO, width=220).pack(side="right")

        f = ctk.CTkFrame(adv_scroll, fg_color="transparent")
        f.pack(fill="x", pady=2)
        ctk.CTkLabel(f, text="QAT Bits (Color)", font=FONT_BODY, width=200, anchor="w").pack(side="left")
        self.bc1_qat_co_var = ctk.StringVar(value=str(self.adv.get("bc1_qat_bits_color", DEFAULT_ADVANCED["bc1_qat_bits_color"])))
        ctk.CTkEntry(f, textvariable=self.bc1_qat_co_var, font=FONT_MONO, width=220).pack(side="right")

        ctk.CTkLabel(adv_scroll, text="", font=("Segoe UI", 4)).pack()
        ctk.CTkLabel(adv_scroll, text="BC4 (Grayscale) Settings", font=FONT_HEADING).pack(anchor="w", pady=(6, 4))

        self.bc4_lpe_var = ctk.BooleanVar(value=self.adv.get("bc4_use_lpe", False))
        ctk.CTkCheckBox(adv_scroll, text="Enable LPE (Local Positional Encoding)", variable=self.bc4_lpe_var, font=FONT_BODY, fg_color=ACCENT, hover_color=ACCENT_HOVER).pack(anchor="w", pady=2)

        f = ctk.CTkFrame(adv_scroll, fg_color="transparent")
        f.pack(fill="x", pady=2)
        ctk.CTkLabel(f, text="QAT Bits (Endpoint)", font=FONT_BODY, width=200, anchor="w").pack(side="left")
        self.bc4_qat_ep_var = ctk.StringVar(value=str(self.adv.get("bc4_qat_bits_endpoint", DEFAULT_ADVANCED["bc4_qat_bits_endpoint"])))
        ctk.CTkEntry(f, textvariable=self.bc4_qat_ep_var, font=FONT_MONO, width=220).pack(side="right")

        f = ctk.CTkFrame(adv_scroll, fg_color="transparent")
        f.pack(fill="x", pady=2)
        ctk.CTkLabel(f, text="QAT Bits (Color)", font=FONT_BODY, width=200, anchor="w").pack(side="left")
        self.bc4_qat_co_var = ctk.StringVar(value=str(self.adv.get("bc4_qat_bits_color", DEFAULT_ADVANCED["bc4_qat_bits_color"])))
        ctk.CTkEntry(f, textvariable=self.bc4_qat_co_var, font=FONT_MONO, width=220).pack(side="right")

        ctk.CTkButton(adv_scroll, text="Reset to Defaults", width=140, fg_color="#18181A", hover_color="#2A2A2A",
                       command=self._reset_advanced).pack(anchor="w", pady=(10, 4))

        # ---- Evaluation tab ----
        eval_tab = tabs.add("Evaluation")
        ctk.CTkLabel(eval_tab, text="PSNR / SSIM Evaluation", font=FONT_HEADING).pack(anchor="w", pady=(10, 2))
        ctk.CTkLabel(eval_tab, text="Run ntbc_eval.py after inference to compare output quality\nagainst Compressonator reference. Results shown in the UI.",
                     font=FONT_BODY, text_color=FG_DIM, justify="left").pack(anchor="w", pady=(0, 10))
        self.eval_var = ctk.BooleanVar(value=self.settings.get("run_evaluation", True))
        ctk.CTkSwitch(eval_tab, text="  Enable Evaluation (PSNR + SSIM)", variable=self.eval_var, font=FONT_BODY, progress_color=ACCENT).pack(anchor="w", pady=6)

        ctk.CTkFrame(eval_tab, height=1, fg_color="#2A2A2A").pack(fill="x", pady=(12, 8))
        ctk.CTkLabel(eval_tab, text="Diff Map Generation", font=FONT_HEADING).pack(anchor="w", pady=(0, 2))
        ctk.CTkLabel(eval_tab, text="Generate amplified difference images (x5) after inference.",
                     font=FONT_BODY, text_color=FG_DIM).pack(anchor="w", pady=(0, 8))
        self.diffmap_var = ctk.BooleanVar(value=self.settings.get("generate_diff_maps", True))
        ctk.CTkSwitch(eval_tab, text="  Generate Diff Maps", variable=self.diffmap_var, font=FONT_BODY, progress_color=ACCENT).pack(anchor="w", pady=6)

        # ---- Save button ----
        ctk.CTkButton(self, text="Save", width=120, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                       command=self._save).pack(pady=(8, 15))

    def _browse_comp(self):
        p = filedialog.askopenfilename(title="Select Compressonator CLI", filetypes=[("Executable", "*.exe"), ("All", "*.*")])
        if p:
            self.comp_var.set(p)

    def _reset_advanced(self):
        for key, (var, _) in self.adv_vars.items():
            var.set(str(DEFAULT_ADVANCED[key]))
        self.bc1_lpe_var.set(DEFAULT_ADVANCED["bc1_use_lpe"])
        self.bc4_lpe_var.set(DEFAULT_ADVANCED["bc4_use_lpe"])
        self.bc1_qat_ep_var.set(str(DEFAULT_ADVANCED["bc1_qat_bits_endpoint"]))
        self.bc1_qat_co_var.set(str(DEFAULT_ADVANCED["bc1_qat_bits_color"]))
        self.bc4_qat_ep_var.set(str(DEFAULT_ADVANCED["bc4_qat_bits_endpoint"]))
        self.bc4_qat_co_var.set(str(DEFAULT_ADVANCED["bc4_qat_bits_color"]))

    def _parse_list(self, s: str) -> List[int]:
        import ast
        return list(ast.literal_eval(s.strip()))

    def _save(self):
        comp = self.comp_var.get().strip()
        if comp and not Path(comp).is_file():
            messagebox.showerror("Invalid Path", "Compressonator path is not a valid file.", parent=self)
            return
        try:
            adv = {}
            for key, (var, typ) in self.adv_vars.items():
                adv[key] = int(var.get()) if typ == "int" else float(var.get())
            adv["bc1_use_lpe"] = self.bc1_lpe_var.get()
            adv["bc4_use_lpe"] = self.bc4_lpe_var.get()
            adv["bc1_qat_bits_endpoint"] = self._parse_list(self.bc1_qat_ep_var.get())
            adv["bc1_qat_bits_color"] = self._parse_list(self.bc1_qat_co_var.get())
            adv["bc4_qat_bits_endpoint"] = self._parse_list(self.bc4_qat_ep_var.get())
            adv["bc4_qat_bits_color"] = self._parse_list(self.bc4_qat_co_var.get())
        except Exception as e:
            messagebox.showerror("Invalid Value", f"Error parsing settings:\n{e}", parent=self)
            return

        self.result = {
            "compressonator_cli": comp,
            "run_evaluation": self.eval_var.get(),
            "generate_diff_maps": self.diffmap_var.get(),
            "advanced": adv,
        }
        self.destroy()


# ---------------------------------------------------------------------------
# Texture Row Widget
# ---------------------------------------------------------------------------
class TextureRow(ctk.CTkFrame):
    def __init__(self, parent, texture: TextureInfo, on_change=None):
        super().__init__(parent, fg_color=BG_CARD, corner_radius=6, height=40)
        self.texture = texture
        self.on_change = on_change
        self.pack(fill="x", pady=2)
        self.grid_columnconfigure(1, weight=1)

        self.enabled_var = ctk.BooleanVar(value=texture.enabled)
        ctk.CTkCheckBox(self, text="", variable=self.enabled_var, width=24, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                        command=self._notify, checkbox_width=20, checkbox_height=20
                        ).grid(row=0, column=0, padx=(10, 6), pady=6)

        ctk.CTkLabel(self, text=texture.filename, font=FONT_BODY, anchor="w"
                     ).grid(row=0, column=1, sticky="w", padx=4)

        self.type_var = ctk.StringVar(value=texture.detected_type)
        ctk.CTkOptionMenu(self, variable=self.type_var, values=["BC1", "BC4"],
                          width=80, font=FONT_SMALL, command=self._on_type,
                          fg_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER
                          ).grid(row=0, column=2, padx=6, pady=6)

        self.badge = ctk.CTkLabel(self, text=texture.display_name, font=FONT_SMALL,
                                  width=100, corner_radius=4,
                                  fg_color=BC1_COLOR if texture.detected_type == "BC1" else BC4_COLOR)
        self.badge.grid(row=0, column=3, padx=(6, 10), pady=6)

    def _notify(self):
        self.texture.enabled = self.enabled_var.get()
        if self.on_change: self.on_change()

    def _on_type(self, v):
        self.texture.detected_type = v
        self.badge.configure(fg_color=BC1_COLOR if v == "BC1" else BC4_COLOR)
        if self.on_change: self.on_change()


# ---------------------------------------------------------------------------
# Setup Page
# ---------------------------------------------------------------------------
class SetupPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.texture_rows: List[TextureRow] = []

        ctk.CTkLabel(self, text="Input Texture Folder", font=FONT_HEADING).pack(anchor="w", padx=20, pady=(15, 4))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=20)
        self.input_var = ctk.StringVar()
        ctk.CTkEntry(row, textvariable=self.input_var, font=FONT_BODY).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row, text="Browse", width=90, fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._browse_input).pack(side="right")

        ctk.CTkLabel(self, text="Output Folder", font=FONT_HEADING).pack(anchor="w", padx=20, pady=(12, 4))
        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(fill="x", padx=20)
        self.output_var = ctk.StringVar()
        ctk.CTkEntry(row2, textvariable=self.output_var, font=FONT_BODY).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row2, text="Browse", width=90, fg_color=ACCENT, hover_color=ACCENT_HOVER, command=self._browse_output).pack(side="right")

        hdr = ctk.CTkFrame(self, fg_color="transparent")
        hdr.pack(fill="x", padx=20, pady=(15, 4))
        ctk.CTkLabel(hdr, text="Detected Textures", font=FONT_HEADING).pack(side="left")
        self.summary_label = ctk.CTkLabel(hdr, text="", font=FONT_SMALL, text_color=FG_DIM)
        self.summary_label.pack(side="right")

        self.tex_scroll = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=8, height=280)
        self.tex_scroll.pack(fill="both", expand=True, padx=20, pady=(0, 10))
        self.empty_label = ctk.CTkLabel(self.tex_scroll, text="Select an input folder to scan for textures",
                                        font=FONT_BODY, text_color=FG_DIM)
        self.empty_label.pack(pady=40)

        self.start_btn = ctk.CTkButton(self, text="Start Compression", font=FONT_HEADING,
                                        height=45, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                        command=self._on_start)
        self.start_btn.pack(pady=(4, 15))

    def _browse_input(self):
        f = filedialog.askdirectory(title="Select Texture Folder")
        if f:
            self.input_var.set(f)
            self._scan(f)
            name = Path(f).name
            self.output_var.set(str(Path(f).parent.parent / "VNTBC_Outputs" / name))

    def _browse_output(self):
        f = filedialog.askdirectory(title="Select Output Folder")
        if f: self.output_var.set(f)

    def _scan(self, folder):
        for r in self.texture_rows: r.destroy()
        self.texture_rows.clear()
        self.empty_label.pack_forget()
        try:
            textures = scan_texture_folder(folder)
        except ValueError as e:
            messagebox.showerror("Scan Error", str(e)); return
        for t in textures:
            self.texture_rows.append(TextureRow(self.tex_scroll, t, on_change=self._update_summary))
        self._update_summary()

    def _update_summary(self):
        bc1 = sum(1 for r in self.texture_rows if r.enabled_var.get() and r.type_var.get() == "BC1")
        bc4 = sum(1 for r in self.texture_rows if r.enabled_var.get() and r.type_var.get() == "BC4")
        self.summary_label.configure(text=f"BC1: {bc1}  |  BC4: {bc4}  |  Total: {bc1 + bc4}")

    def _on_start(self):
        inp = self.input_var.get().strip()
        out = self.output_var.get().strip()
        if not inp or not Path(inp).is_dir():
            messagebox.showerror("Error", "Select a valid input folder."); return
        if not out:
            messagebox.showerror("Error", "Select an output folder."); return
        enabled = [r for r in self.texture_rows if r.enabled_var.get()]
        if not enabled:
            messagebox.showerror("Error", "Enable at least one texture."); return

        bc1 = [r for r in enabled if r.type_var.get() == "BC1"]
        bc4 = [r for r in enabled if r.type_var.get() == "BC4"]
        job = PipelineJob(
            bc1_images=[r.texture.path for r in bc1],
            bc1_names=[r.texture.display_name for r in bc1],
            bc4_images=[r.texture.path for r in bc4],
            bc4_names=[r.texture.display_name for r in bc4],
            output_dir=out, folder_name=Path(inp).name,
            compressonator_cli=self.app.settings.get("compressonator_cli", ""),
            run_evaluation=self.app.settings.get("run_evaluation", True),
            generate_diff_maps=self.app.settings.get("generate_diff_maps", True),
            advanced=self.app.settings.get("advanced", dict(DEFAULT_ADVANCED)),
        )
        self.app.start_pipeline(job)


# ---------------------------------------------------------------------------
# Inference Setup Page
# ---------------------------------------------------------------------------
class InferenceSetupPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app

        # -- Model checkpoint --
        ctk.CTkLabel(self, text="Model Checkpoint (.pt)", font=FONT_HEADING).pack(anchor="w", padx=20, pady=(15, 4))
        row = ctk.CTkFrame(self, fg_color="transparent")
        row.pack(fill="x", padx=20)
        self.ckpt_var = ctk.StringVar()
        ctk.CTkEntry(row, textvariable=self.ckpt_var, font=FONT_BODY).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row, text="Browse", width=90, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                       command=self._browse_ckpt).pack(side="right")

        # -- Inference input JSON --
        ctk.CTkLabel(self, text="Inference Input JSON", font=FONT_HEADING).pack(anchor="w", padx=20, pady=(12, 4))
        row2 = ctk.CTkFrame(self, fg_color="transparent")
        row2.pack(fill="x", padx=20)
        self.json_var = ctk.StringVar()
        self.json_var.trace_add("write", self._on_json_changed)
        ctk.CTkEntry(row2, textvariable=self.json_var, font=FONT_BODY).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row2, text="Browse", width=90, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                       command=self._browse_json).pack(side="right")

        # -- Source images (optional, for eval) --
        ctk.CTkLabel(self, text="Source Images Folder (optional, for evaluation)", font=FONT_HEADING).pack(anchor="w", padx=20, pady=(12, 4))
        row3 = ctk.CTkFrame(self, fg_color="transparent")
        row3.pack(fill="x", padx=20)
        self.src_var = ctk.StringVar()
        ctk.CTkEntry(row3, textvariable=self.src_var, font=FONT_BODY).pack(side="left", fill="x", expand=True, padx=(0, 8))
        ctk.CTkButton(row3, text="Browse", width=90, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                       command=self._browse_src).pack(side="right")

        # -- Pipeline type --
        type_row = ctk.CTkFrame(self, fg_color="transparent")
        type_row.pack(fill="x", padx=20, pady=(12, 4))
        ctk.CTkLabel(type_row, text="Pipeline Type", font=FONT_HEADING).pack(side="left")
        self.type_var = ctk.StringVar(value="BC1")
        ctk.CTkOptionMenu(type_row, variable=self.type_var, values=["BC1", "BC4"],
                          width=100, font=FONT_BODY,
                          fg_color=ACCENT, button_color=ACCENT, button_hover_color=ACCENT_HOVER
                          ).pack(side="left", padx=(12, 0))

        # -- Detected texture names --
        self.names_label = ctk.CTkLabel(self, text="", font=FONT_SMALL, text_color=FG_DIM)
        self.names_label.pack(anchor="w", padx=20, pady=(8, 0))
        self._texture_names: List[str] = []

        # -- Start button --
        self.start_btn = ctk.CTkButton(self, text="Run Inference", font=FONT_HEADING,
                                        height=45, fg_color=ACCENT, hover_color=ACCENT_HOVER,
                                        command=self._on_start)
        self.start_btn.pack(pady=(20, 15))

    def _browse_ckpt(self):
        p = filedialog.askopenfilename(
            title="Select Model Checkpoint",
            filetypes=[("PyTorch checkpoint", "*.pt"), ("All", "*.*")]
        )
        if p:
            self.ckpt_var.set(p)
            # Auto-detect pipeline type from filename
            name = Path(p).name.lower()
            if "bc4" in name:
                self.type_var.set("BC4")
            elif "bc1" in name:
                self.type_var.set("BC1")
            # Auto-detect inference JSON in the same directory
            parent = Path(p).parent
            json_candidate = parent / "Inference_input.json"
            if json_candidate.exists() and not self.json_var.get():
                self.json_var.set(str(json_candidate))

    def _browse_json(self):
        p = filedialog.askopenfilename(
            title="Select Inference Input JSON",
            filetypes=[("JSON", "*.json"), ("All", "*.*")]
        )
        if p:
            self.json_var.set(p)

    def _on_json_changed(self, *_):
        """Read texture names from the JSON when path changes."""
        jp = self.json_var.get().strip()
        if jp and Path(jp).is_file():
            names = read_texture_names_from_json(jp)
            self._texture_names = names
            if names:
                self.names_label.configure(
                    text=f"Textures detected: {', '.join(names)} ({len(names)} total)"
                )
            else:
                self.names_label.configure(text="No texture names found in JSON")
        else:
            self._texture_names = []
            self.names_label.configure(text="")

    def _browse_src(self):
        f = filedialog.askdirectory(title="Select Source Images Folder")
        if f:
            self.src_var.set(f)

    def _on_start(self):
        ckpt = self.ckpt_var.get().strip()
        json_p = self.json_var.get().strip()
        if not ckpt or not Path(ckpt).is_file():
            messagebox.showerror("Error", "Select a valid model checkpoint (.pt)."); return
        if not json_p or not Path(json_p).is_file():
            messagebox.showerror("Error", "Select a valid Inference_input.json."); return

        texture_names = self._texture_names
        if not texture_names:
            texture_names = read_texture_names_from_json(json_p)
        if not texture_names:
            texture_names = ["tex00"]

        # Determine source images for eval — match by texture name
        source_images = []
        src_dir = self.src_var.get().strip()
        if src_dir and Path(src_dir).is_dir():
            all_imgs = sorted([
                f for f in Path(src_dir).iterdir()
                if f.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".exr")
            ])
            # Match each texture name to a source image
            for tname in texture_names:
                matched = None
                tname_lower = tname.lower()
                for img in all_imgs:
                    # Check if the texture name appears in the image filename
                    if tname_lower in img.stem.lower():
                        matched = str(img)
                        break
                if matched:
                    source_images.append(matched)
                else:
                    source_images.append("")  # placeholder — eval will skip
            # If no matches at all, clear the list so eval is skipped
            if all(s == "" for s in source_images):
                source_images = []

        # Output dir = directory containing the checkpoint
        output_dir = str(Path(ckpt).parent)

        job = InferenceJob(
            model_checkpoint=ckpt,
            inference_json=json_p,
            source_images=source_images,
            texture_names=texture_names,
            pipeline_type=self.type_var.get(),
            output_dir=output_dir,
            compressonator_cli=self.app.settings.get("compressonator_cli", ""),
            run_evaluation=self.app.settings.get("run_evaluation", True),
            generate_diff_maps=self.app.settings.get("generate_diff_maps", True),
        )
        self.app.start_inference_pipeline(job)


# ---------------------------------------------------------------------------
# Stage Row
# ---------------------------------------------------------------------------
class StageRow(ctk.CTkFrame):
    ICONS = {"pending": ("⏳", PENDING_COLOR), "running": ("🔄", RUNNING_COLOR),
             "done": ("✅", SUCCESS_COLOR), "error": ("❌", ERROR_COLOR)}

    def __init__(self, parent, stage):
        super().__init__(parent, fg_color=BG_CARD, corner_radius=6, height=36)
        self.pack(fill="x", pady=2)
        self.icon = ctk.CTkLabel(self, text="⏳", font=FONT_BODY, width=30)
        self.icon.pack(side="left", padx=(12, 6), pady=6)
        ctk.CTkLabel(self, text=stage.name, font=FONT_BODY, anchor="w").pack(side="left", fill="x", expand=True, pady=6)
        self.status = ctk.CTkLabel(self, text="Pending", font=FONT_SMALL, text_color=PENDING_COLOR, width=80)
        self.status.pack(side="right", padx=(6, 12), pady=6)

    def set_status(self, s):
        ic, col = self.ICONS.get(s, ("❓", FG_DIM))
        self.icon.configure(text=ic)
        self.status.configure(text=s.capitalize(), text_color=col)


# ---------------------------------------------------------------------------
# Progress Page
# ---------------------------------------------------------------------------
class ProgressPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self.stage_rows: List[StageRow] = []

        ctk.CTkLabel(self, text="Pipeline Progress", font=FONT_HEADING).pack(anchor="w", padx=20, pady=(15, 8))
        self.stage_frame = ctk.CTkFrame(self, fg_color=BG_DARK, corner_radius=8)
        self.stage_frame.pack(fill="x", padx=20, pady=(0, 10))

        ctk.CTkLabel(self, text="Live Log", font=FONT_HEADING).pack(anchor="w", padx=20)
        self.log_text = ctk.CTkTextbox(self, font=FONT_MONO, fg_color=BG_DARK, corner_radius=8, state="disabled")
        self.log_text.pack(fill="both", expand=True, padx=20, pady=(6, 10))

        ctk.CTkButton(self, text="Cancel", font=FONT_BODY, fg_color=ERROR_COLOR,
                       hover_color="#C62828", width=120, command=self._cancel).pack(pady=(0, 15))

    def setup_stages(self, stages):
        for r in self.stage_rows: r.destroy()
        self.stage_rows = [StageRow(self.stage_frame, s) for s in stages]
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def update_stage(self, idx, status):
        if 0 <= idx < len(self.stage_rows): self.stage_rows[idx].set_status(status)

    def append_log(self, line):
        self.log_text.configure(state="normal")
        self.log_text.insert("end", line + "\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _cancel(self):
        if messagebox.askyesno("Cancel", "Cancel the pipeline?"): self.app.cancel_pipeline()


# ---------------------------------------------------------------------------
# Results Page
# ---------------------------------------------------------------------------
class ResultsPage(ctk.CTkFrame):
    def __init__(self, parent, app):
        super().__init__(parent, fg_color="transparent")
        self.app = app
        self._output_dir = ""

        ctk.CTkFrame(self, fg_color="transparent", height=20).pack()
        self.icon_label = ctk.CTkLabel(self, text="", font=("Segoe UI", 48))
        self.icon_label.pack(pady=(10, 6))
        self.title_label = ctk.CTkLabel(self, text="", font=FONT_TITLE)
        self.title_label.pack(pady=(0, 4))
        self.msg_label = ctk.CTkLabel(self, text="", font=FONT_BODY, text_color=FG_DIM, wraplength=500)
        self.msg_label.pack(pady=(0, 10))

        # PSNR table area
        self.table_label = ctk.CTkLabel(self, text="", font=FONT_HEADING)
        self.table_label.pack(anchor="w", padx=30, pady=(4, 2))
        self.table_frame = ctk.CTkScrollableFrame(self, fg_color=BG_DARK, corner_radius=8, height=180)
        # Don't pack yet — only shown when eval results exist

        # Output path
        self.path_label = ctk.CTkLabel(self, text="", font=FONT_MONO, text_color=FG_DIM)
        self.path_label.pack(pady=(8, 12))

        btn_row = ctk.CTkFrame(self, fg_color="transparent")
        btn_row.pack()
        ctk.CTkButton(btn_row, text="📂  Open Output Folder", font=FONT_BODY, fg_color=ACCENT,
                       hover_color=ACCENT_HOVER, width=180, height=40, command=self._open).pack(side="left", padx=8)
        ctk.CTkButton(btn_row, text="🔄  New Job", font=FONT_BODY, fg_color="#2A2A2A",
                       hover_color="#3A3A3A", width=140, height=40,
                       command=lambda: app.show_page("setup")).pack(side="left", padx=8)

    def show_success(self, output_dir: str, eval_results: List[EvalResult]):
        self._output_dir = output_dir
        self.icon_label.configure(text="✅")
        self.title_label.configure(text="Compression Complete!", text_color=SUCCESS_COLOR)
        self.msg_label.configure(text="All textures processed. Diff maps saved to inference_output folders.")
        self.path_label.configure(text=output_dir)
        self._show_eval_table(eval_results)

    def show_error(self, message: str, output_dir: str = "", eval_results: List[EvalResult] = None):
        self._output_dir = output_dir
        self.icon_label.configure(text="❌")
        self.title_label.configure(text="Pipeline Failed", text_color=ERROR_COLOR)
        self.msg_label.configure(text=message)
        self.path_label.configure(text=output_dir)
        self._show_eval_table(eval_results or [])

    def _show_eval_table(self, results: List[EvalResult]):
        # Clear previous table contents
        for w in self.table_frame.winfo_children():
            w.destroy()

        if not results:
            self.table_label.configure(text="")
            self.table_frame.pack_forget()
            return

        self.table_label.configure(text="Evaluation Results")
        self.table_frame.pack(fill="x", padx=30, pady=(0, 6), before=self.path_label)

        has_ssim = any(r.ssim_ref is not None for r in results)

        # Column definitions: (header, width, anchor)
        col_defs = [
            ("Texture",    0, "w"),       # weight=1, expands to fill
            ("Format",    60, "center"),
            ("PSNR Ref",  85, "center"),
            ("PSNR VNTBC",90, "center"),
            ("Delta",     75, "center"),
        ]
        if has_ssim:
            col_defs += [
                ("SSIM Ref",  85, "center"),
                ("SSIM VNTBC",90, "center"),
                ("SSIM Δ",    80, "center"),
            ]

        def _build_row(parent, cells, fg, font, bold=False):
            """Build a row using pack with fixed-width labels."""
            row = ctk.CTkFrame(parent, fg_color=fg, corner_radius=4)
            row.pack(fill="x", pady=1)
            for i, (text, color, width, anchor) in enumerate(cells):
                f = ("Segoe UI", 11, "bold") if bold else font
                lbl = ctk.CTkLabel(row, text=text, font=f, anchor=anchor,
                                   text_color=color)
                if width == 0:
                    lbl.pack(side="left", fill="x", expand=True, padx=(10, 4), pady=4)
                else:
                    lbl.pack(side="left", padx=4, pady=4)
                    lbl.configure(width=width)
            return row

        # Header
        hdr_cells = [(h, FG_TEXT, w, a) for h, w, a in col_defs]
        _build_row(self.table_frame, hdr_cells, "#2A2A2A", FONT_MONO, bold=True)

        # Data rows
        for r in results:
            delta_color = SUCCESS_COLOR if r.psnr_delta >= -0.5 else "#FFA726" if r.psnr_delta >= -2.0 else ERROR_COLOR
            data = [
                (r.name, FG_TEXT),
                (r.fmt, BC1_COLOR if r.fmt == "BC1" else BC4_COLOR),
                (f"{r.psnr_ref:.2f}", FG_TEXT),
                (f"{r.psnr_ntbc:.2f}", FG_TEXT),
                (f"{r.psnr_delta:+.2f}", delta_color),
            ]
            if has_ssim:
                data += [
                    (f"{r.ssim_ref:.4f}" if r.ssim_ref is not None else "—", FG_TEXT),
                    (f"{r.ssim_ntbc:.4f}" if r.ssim_ntbc is not None else "—", FG_TEXT),
                    (f"{r.ssim_delta:+.4f}" if r.ssim_delta is not None else "—", FG_DIM),
                ]
            cells = [(text, color, col_defs[i][1], col_defs[i][2])
                     for i, (text, color) in enumerate(data)]
            _build_row(self.table_frame, cells, BG_CARD, FONT_MONO)

    def _open(self):
        if self._output_dir and Path(self._output_dir).exists():
            os.startfile(self._output_dir)


# ---------------------------------------------------------------------------
# Main Application
# ---------------------------------------------------------------------------
class VNTBCApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("VNTBC — Variable Bit Rate Neural Texture Block Compression")
        self.geometry("1050x900")
        self.minsize(800, 600)

        self.settings = load_settings()
        self.cancel_event = threading.Event()
        self.pipeline_thread: Optional[threading.Thread] = None

        # Header
        header = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=56)
        header.pack(fill="x"); header.pack_propagate(False)
        ctk.CTkLabel(header, text="VNTBC", font=FONT_TITLE, text_color=ACCENT).pack(side="left", padx=20)
        ctk.CTkLabel(header, text="Variable Bit Rate Neural Texture Block Compression", font=FONT_BODY, text_color=FG_TEXT).pack(side="left")
        ctk.CTkButton(header, text="Settings", font=FONT_BODY, width=90, fg_color="#313436",
                       hover_color="#2A2A2A", command=self._open_settings).pack(side="right", padx=20)

        # Mode selector
        mode_bar = ctk.CTkFrame(self, fg_color=BG_CARD, corner_radius=0, height=40)
        mode_bar.pack(fill="x"); mode_bar.pack_propagate(False)
        self.mode_var = ctk.StringVar(value="full")
        ctk.CTkSegmentedButton(
            mode_bar, values=["Full Pipeline", "Inference Only"],
            variable=self.mode_var, font=FONT_BODY,
            fg_color="#18181A", selected_color=ACCENT,
            selected_hover_color=ACCENT_HOVER,
            unselected_color="#2A2A2A", unselected_hover_color="#3A3A3A",
            command=self._on_mode_change,
        ).pack(side="left", padx=20, pady=6)
        self.mode_var.set("Full Pipeline")

        # Content
        self.content = ctk.CTkFrame(self, fg_color="transparent")
        self.content.pack(fill="both", expand=True)
        self.setup_page = SetupPage(self.content, self)
        self.infer_setup_page = InferenceSetupPage(self.content, self)
        self.progress_page = ProgressPage(self.content, self)
        self.results_page = ResultsPage(self.content, self)
        self.pages = {
            "setup": self.setup_page,
            "infer_setup": self.infer_setup_page,
            "progress": self.progress_page,
            "results": self.results_page,
        }
        self.show_page("setup")

        self.after(300, self._check_compressonator)

    def _on_mode_change(self, value):
        if value == "Full Pipeline":
            self.show_page("setup")
        else:
            self.show_page("infer_setup")

    def show_page(self, name):
        for p in self.pages.values(): p.pack_forget()
        self.pages[name].pack(fill="both", expand=True)

    def _check_compressonator(self):
        cli = self.settings.get("compressonator_cli", "")
        if not cli or not Path(cli).is_file():
            self._open_settings()

    def _open_settings(self):
        dlg = SettingsDialog(self, self.settings)
        self.wait_window(dlg)
        if dlg.result:
            self.settings.update(dlg.result)
            save_settings(self.settings)

    def start_pipeline(self, job: PipelineJob):
        self.cancel_event.clear()
        stages = build_stages(job)
        self.progress_page.setup_stages(stages)
        self.show_page("progress")

        def _stage(idx, s): self.after(0, lambda: self.progress_page.update_stage(idx, s))
        def _log(line): self.after(0, lambda: self.progress_page.append_log(line))
        def _done(ok, msg, evals):
            def _show():
                if ok:
                    self.results_page.show_success(job.output_dir, evals)
                    try: os.startfile(job.output_dir)
                    except: pass
                else:
                    self.results_page.show_error(msg, job.output_dir, evals)
                self.show_page("results")
            self.after(0, _show)

        self.pipeline_thread = threading.Thread(target=run_pipeline, args=(job, _stage, _log, _done, self.cancel_event), daemon=True)
        self.pipeline_thread.start()

    def start_inference_pipeline(self, job: InferenceJob):
        self.cancel_event.clear()
        stages = build_inference_stages(job)
        self.progress_page.setup_stages(stages)
        self.show_page("progress")

        def _stage(idx, s): self.after(0, lambda: self.progress_page.update_stage(idx, s))
        def _log(line): self.after(0, lambda: self.progress_page.append_log(line))
        def _done(ok, msg, evals):
            def _show():
                if ok:
                    self.results_page.show_success(job.output_dir, evals)
                    try: os.startfile(job.output_dir)
                    except: pass
                else:
                    self.results_page.show_error(msg, job.output_dir, evals)
                self.show_page("results")
            self.after(0, _show)

        self.pipeline_thread = threading.Thread(target=run_inference_pipeline, args=(job, _stage, _log, _done, self.cancel_event), daemon=True)
        self.pipeline_thread.start()

    def cancel_pipeline(self):
        self.cancel_event.set()


def main():
    VNTBCApp().mainloop()

if __name__ == "__main__":
    main()
