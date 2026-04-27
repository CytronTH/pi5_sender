#!/usr/bin/env python3
"""
Image Preprocessor - Windows GUI
==================================
Standalone desktop application for running the pi5_sender image
pre-processing pipeline on a folder of images.

Usage:
  python gui_app.py
  (or double-click the compiled .exe on Windows)
"""

import os
import sys
import threading
import zipfile
import shutil
import json
import time
import tkinter as tk
from tkinter import ttk, filedialog, messagebox

# ── Resolve base directory (works both as script and PyInstaller bundle) ──────
if getattr(sys, 'frozen', False):
    BASE_DIR = os.path.dirname(sys.executable)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

sys.path.insert(0, BASE_DIR)

# ── Colour palette ─────────────────────────────────────────────────────────────
BG          = "#0f1117"
SURFACE     = "#1a1d27"
SURFACE2    = "#22263a"
ACCENT      = "#6366f1"   # indigo
ACCENT2     = "#a855f7"   # purple
SUCCESS     = "#10b981"   # green
WARNING     = "#f59e0b"   # amber
DANGER      = "#ef4444"   # red
TEXT        = "#e2e8f0"
TEXT_MUTED  = "#64748b"
BORDER      = "#2d3748"


# ── Helpers ────────────────────────────────────────────────────────────────────

def detect_cam_id_from_zip(zip_path: str) -> str:
    """Try to detect cam_id from zip contents; default cam0."""
    try:
        with zipfile.ZipFile(zip_path) as zf:
            names = zf.namelist()
        for n in names:
            base = os.path.basename(n)
            if base.startswith("cam1_"):
                return "cam1"
    except Exception:
        pass
    return "cam0"


def extract_calibration_zip(zip_path: str, dest_configs: str, log_fn=None):
    """
    Extract calibration_camX.zip into dest_configs/.
    The zip has the structure: configs/... so we strip the leading 'configs/' prefix.
    """
    os.makedirs(dest_configs, exist_ok=True)
    os.makedirs(os.path.join(dest_configs, "templates"), exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as zf:
        for member in zf.infolist():
            # Strip leading 'configs/' prefix from arc name
            arc = member.filename.replace("\\", "/")
            if arc.startswith("configs/"):
                rel = arc[len("configs/"):]
            else:
                rel = arc

            if not rel or rel.endswith("/"):
                continue  # skip directories

            dest_path = os.path.join(dest_configs, rel)
            os.makedirs(os.path.dirname(dest_path), exist_ok=True)
            with zf.open(member) as src, open(dest_path, "wb") as dst:
                shutil.copyfileobj(src, dst)

            if log_fn:
                log_fn(f"  Extracted: {rel}")


# ── Main Application ───────────────────────────────────────────────────────────

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Image Preprocessor")
        self.geometry("820x720")
        self.minsize(720, 600)
        self.configure(bg=BG)
        self.resizable(True, True)

        # Try to set icon (fails silently if not available)
        try:
            icon_path = os.path.join(BASE_DIR, "icon.ico")
            if os.path.exists(icon_path):
                self.iconbitmap(icon_path)
        except Exception:
            pass

        # State variables
        self._zip_path     = tk.StringVar()
        self._input_dir    = tk.StringVar()
        self._output_dir   = tk.StringVar()
        self._cam_id       = tk.StringVar(value="cam0")
        self._jpeg_quality = tk.IntVar(value=90)
        self._step_align   = tk.BooleanVar(value=True)
        self._step_shadow  = tk.BooleanVar(value=True)
        self._step_gray    = tk.BooleanVar(value=True)
        self._step_clahe   = tk.BooleanVar(value=True)
        self._step_crop    = tk.BooleanVar(value=True)
        self._step_precrop = tk.BooleanVar(value=False)
        self._step_timestamp = tk.BooleanVar(value=False)
        self._skip_raw     = tk.BooleanVar(value=True)
        self._debug        = tk.BooleanVar(value=False)

        self._processing = False
        self._worker: threading.Thread = None

        self._build_ui()
        self._load_last_session()

    # ── UI Construction ────────────────────────────────────────────────────────

    def _build_ui(self):
        self._apply_style()

        # ── Header ────────────────────────────────────────────────────────────
        hdr = tk.Frame(self, bg=SURFACE, pady=16)
        hdr.pack(fill="x")
        tk.Label(hdr, text="🖼  Image Preprocessor",
                 bg=SURFACE, fg=TEXT,
                 font=("Segoe UI", 18, "bold")).pack()
        tk.Label(hdr, text="pi5_sender · Standalone Processing Tool",
                 bg=SURFACE, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack()

        # ── Body (scrollable canvas) ───────────────────────────────────────────
        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=20, pady=12)

        # Left column: inputs + options
        left = tk.Frame(body, bg=BG)
        left.pack(side="left", fill="both", expand=True)

        # ── Step 1: Calibration ZIP ────────────────────────────────────────────
        self._section(left, "📦  Step 1 — Calibration File")
        f1 = tk.Frame(left, bg=BG)
        f1.pack(fill="x", pady=(0, 8))
        tk.Entry(f1, textvariable=self._zip_path,
                 bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 10),
                 highlightthickness=1, highlightcolor=BORDER,
                 highlightbackground=BORDER).pack(side="left", fill="x", expand=True, ipady=6, ipadx=8)
        self._btn(f1, "Browse…", self._browse_zip).pack(side="left", padx=(8, 0))

        hint1 = tk.Label(left,
            text="Download calibration_camX.zip from WebUI → 🎯 Calibrate → 📦 Download Calib Files",
            bg=BG, fg=TEXT_MUTED, font=("Segoe UI", 9), wraplength=500, justify="left")
        hint1.pack(anchor="w", pady=(0, 12))

        # Camera ID auto-detect label
        self._cam_label = tk.Label(left, text="", bg=BG, fg=SUCCESS,
                                   font=("Segoe UI", 9, "bold"))
        self._cam_label.pack(anchor="w", pady=(0, 4))

        # ── Step 2: Input & Output Folders ────────────────────────────────────
        self._section(left, "📂  Step 2 — Image Folders")

        # Input
        fi = tk.Frame(left, bg=BG)
        fi.pack(fill="x", pady=(0, 6))
        tk.Label(fi, text="Input Folder:", bg=BG, fg=TEXT,
                 font=("Segoe UI", 10), width=14, anchor="w").pack(side="left")
        tk.Entry(fi, textvariable=self._input_dir,
                 bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 10),
                 highlightthickness=1, highlightcolor=BORDER,
                 highlightbackground=BORDER).pack(side="left", fill="x", expand=True, ipady=6, ipadx=8)
        self._btn(fi, "Browse…", self._browse_input).pack(side="left", padx=(8, 0))

        # Output
        fo = tk.Frame(left, bg=BG)
        fo.pack(fill="x", pady=(0, 12))
        tk.Label(fo, text="Output Folder:", bg=BG, fg=TEXT,
                 font=("Segoe UI", 10), width=14, anchor="w").pack(side="left")
        tk.Entry(fo, textvariable=self._output_dir,
                 bg=SURFACE2, fg=TEXT, insertbackground=TEXT,
                 relief="flat", font=("Segoe UI", 10),
                 highlightthickness=1, highlightcolor=BORDER,
                 highlightbackground=BORDER).pack(side="left", fill="x", expand=True, ipady=6, ipadx=8)
        self._btn(fo, "Browse…", self._browse_output).pack(side="left", padx=(8, 0))

        # ── Step 3: Options ───────────────────────────────────────────────────
        self._section(left, "⚙️  Step 3 — Processing Options")

        opts_frame = tk.Frame(left, bg=SURFACE2,
                              highlightthickness=1, highlightbackground=BORDER)
        opts_frame.pack(fill="x", pady=(0, 12))

        # Pipeline steps (two columns)
        steps_frame = tk.Frame(opts_frame, bg=SURFACE2, padx=12, pady=10)
        steps_frame.pack(fill="x")

        steps = [
            ("Alignment (template matching)",     self._step_align),
            ("Shadow Removal (divisive norm.)",   self._step_shadow),
            ("Grayscale + CLAHE",                 self._step_gray),
            ("CLAHE Enhancement",                 self._step_clahe),
            ("Mask-based Cropping (ROIs)",        self._step_crop),
            ("Pre-Crop (trim edges)",             self._step_precrop),
            ("Add Timestamp on Raw Image",        self._step_timestamp),
        ]

        col_l = tk.Frame(steps_frame, bg=SURFACE2)
        col_l.pack(side="left", fill="x", expand=True)
        col_r = tk.Frame(steps_frame, bg=SURFACE2)
        col_r.pack(side="right", fill="x", expand=True)

        for i, (label, var) in enumerate(steps):
            parent = col_l if i % 2 == 0 else col_r
            cb = tk.Checkbutton(parent, text=label, variable=var,
                                bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
                                activebackground=SURFACE2, activeforeground=TEXT,
                                font=("Segoe UI", 9), anchor="w")
            cb.pack(fill="x", pady=2)

        # Separator
        tk.Frame(opts_frame, bg=BORDER, height=1).pack(fill="x", padx=12)

        # Extra options row
        extra = tk.Frame(opts_frame, bg=SURFACE2, padx=12, pady=8)
        extra.pack(fill="x")

        tk.Checkbutton(extra, text="Skip saving 'raw' output",
                       variable=self._skip_raw,
                       bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
                       activebackground=SURFACE2, activeforeground=TEXT,
                       font=("Segoe UI", 9)).pack(side="left")

        tk.Checkbutton(extra, text="Save debug images",
                       variable=self._debug,
                       bg=SURFACE2, fg=TEXT, selectcolor=SURFACE,
                       activebackground=SURFACE2, activeforeground=TEXT,
                       font=("Segoe UI", 9)).pack(side="left", padx=(16, 0))

        tk.Label(extra, text="JPEG Quality:", bg=SURFACE2, fg=TEXT_MUTED,
                 font=("Segoe UI", 9)).pack(side="right", padx=(0, 4))
        tk.Spinbox(extra, from_=50, to=100, textvariable=self._jpeg_quality,
                   bg=SURFACE2, fg=TEXT, buttonbackground=SURFACE,
                   relief="flat", width=5, font=("Segoe UI", 9)).pack(side="right")

        # ── Run Button ────────────────────────────────────────────────────────
        run_frame = tk.Frame(left, bg=BG)
        run_frame.pack(fill="x", pady=(4, 0))
        self._run_btn = tk.Button(
            run_frame, text="▶  Run Pre-processing",
            command=self._start_processing,
            bg=ACCENT, fg="white", activebackground=ACCENT2,
            activeforeground="white", relief="flat",
            font=("Segoe UI", 12, "bold"), cursor="hand2",
            pady=10
        )
        self._run_btn.pack(fill="x")

        # ── Progress ──────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready.")
        tk.Label(left, textvariable=self._status_var,
                 bg=BG, fg=TEXT_MUTED, font=("Segoe UI", 9)).pack(anchor="w", pady=(6, 2))

        self._progress = ttk.Progressbar(left, mode="determinate", style="Green.Horizontal.TProgressbar")
        self._progress.pack(fill="x", pady=(0, 8))

        # ── Log Panel ─────────────────────────────────────────────────────────
        self._section(left, "📋  Log")
        log_frame = tk.Frame(left, bg=SURFACE2,
                             highlightthickness=1, highlightbackground=BORDER)
        log_frame.pack(fill="both", expand=True)

        self._log = tk.Text(log_frame, bg=SURFACE2, fg=TEXT,
                            font=("Consolas", 9), relief="flat",
                            state="disabled", wrap="word",
                            insertbackground=TEXT)
        log_scroll = tk.Scrollbar(log_frame, command=self._log.yview,
                                  bg=SURFACE2, troughcolor=SURFACE2,
                                  relief="flat")
        self._log.configure(yscrollcommand=log_scroll.set)
        log_scroll.pack(side="right", fill="y")
        self._log.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        # Tag colouring for log
        self._log.tag_configure("ok",      foreground=SUCCESS)
        self._log.tag_configure("warn",    foreground=WARNING)
        self._log.tag_configure("err",     foreground=DANGER)
        self._log.tag_configure("muted",   foreground=TEXT_MUTED)
        self._log.tag_configure("header",  foreground=ACCENT, font=("Consolas", 9, "bold"))

        # Clear log button
        self._btn(left, "Clear Log", self._clear_log, color=TEXT_MUTED).pack(anchor="e", pady=(4, 0))

    def _apply_style(self):
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Green.Horizontal.TProgressbar",
                        troughcolor=SURFACE2, background=SUCCESS,
                        lightcolor=SUCCESS, darkcolor=SUCCESS,
                        bordercolor=SURFACE2, thickness=8)

    def _section(self, parent, title):
        f = tk.Frame(parent, bg=BG)
        f.pack(fill="x", pady=(8, 4))
        tk.Label(f, text=title, bg=BG, fg=ACCENT2,
                 font=("Segoe UI", 10, "bold")).pack(side="left")
        tk.Frame(f, bg=BORDER, height=1).pack(side="left", fill="x", expand=True, padx=(8, 0), pady=8)

    def _btn(self, parent, text, cmd, color=ACCENT):
        return tk.Button(parent, text=text, command=cmd,
                         bg=SURFACE2, fg=color, activebackground=BORDER,
                         activeforeground=TEXT, relief="flat",
                         font=("Segoe UI", 9), cursor="hand2",
                         padx=10, pady=6)

    # ── Browse Callbacks ───────────────────────────────────────────────────────

    def _browse_zip(self):
        path = filedialog.askopenfilename(
            title="Select Calibration ZIP",
            filetypes=[("ZIP files", "*.zip"), ("All files", "*.*")]
        )
        if path:
            self._zip_path.set(path)
            cam = detect_cam_id_from_zip(path)
            self._cam_id.set(cam)
            self._cam_label.config(text=f"✓ Detected camera: {cam}")
            self._save_session()

    def _browse_input(self):
        path = filedialog.askdirectory(title="Select Input Image Folder")
        if path:
            self._input_dir.set(path)
            # Auto-set output folder beside input
            if not self._output_dir.get():
                self._output_dir.set(os.path.join(path, "preprocessed"))
            self._save_session()

    def _browse_output(self):
        path = filedialog.askdirectory(title="Select Output Folder")
        if path:
            self._output_dir.set(path)
            self._save_session()

    # ── Session Persistence ────────────────────────────────────────────────────

    def _session_path(self):
        return os.path.join(BASE_DIR, ".last_session.json")

    def _save_session(self):
        try:
            data = {
                "zip_path":   self._zip_path.get(),
                "input_dir":  self._input_dir.get(),
                "output_dir": self._output_dir.get(),
            }
            with open(self._session_path(), "w") as f:
                json.dump(data, f)
        except Exception:
            pass

    def _load_last_session(self):
        try:
            with open(self._session_path()) as f:
                data = json.load(f)
            if data.get("zip_path") and os.path.exists(data["zip_path"]):
                self._zip_path.set(data["zip_path"])
                cam = detect_cam_id_from_zip(data["zip_path"])
                self._cam_id.set(cam)
                self._cam_label.config(text=f"✓ Detected camera: {cam}")
            if data.get("input_dir"):
                self._input_dir.set(data["input_dir"])
            if data.get("output_dir"):
                self._output_dir.set(data["output_dir"])
        except Exception:
            pass

    # ── Log Helpers ────────────────────────────────────────────────────────────

    def _log_write(self, msg, tag=""):
        def _do():
            self._log.configure(state="normal")
            self._log.insert("end", msg + "\n", tag)
            self._log.see("end")
            self._log.configure(state="disabled")
        self.after(0, _do)

    def _clear_log(self):
        self._log.configure(state="normal")
        self._log.delete("1.0", "end")
        self._log.configure(state="disabled")

    def _set_status(self, msg, color=TEXT_MUTED):
        self.after(0, lambda: self._status_var.set(msg))

    def _set_progress(self, value, maximum=100):
        def _do():
            self._progress["maximum"] = maximum
            self._progress["value"] = value
        self.after(0, _do)

    # ── Processing ─────────────────────────────────────────────────────────────

    def _start_processing(self):
        if self._processing:
            return

        zip_path  = self._zip_path.get().strip()
        input_dir = self._input_dir.get().strip()
        output_dir = self._output_dir.get().strip()

        # Validation
        if not zip_path:
            messagebox.showwarning("Missing File", "Please select a Calibration ZIP file.")
            return
        if not os.path.exists(zip_path):
            messagebox.showerror("File Not Found", f"ZIP file not found:\n{zip_path}")
            return
        if not input_dir:
            messagebox.showwarning("Missing Folder", "Please select an Input Image Folder.")
            return
        if not os.path.isdir(input_dir):
            messagebox.showerror("Folder Not Found", f"Input folder not found:\n{input_dir}")
            return
        if not output_dir:
            output_dir = os.path.join(input_dir, "preprocessed")
            self._output_dir.set(output_dir)

        self._save_session()
        self._processing = True
        self._run_btn.config(text="⏳  Processing...", state="disabled", bg=SURFACE2)
        self._clear_log()
        self._set_progress(0)

        self._worker = threading.Thread(
            target=self._run_pipeline,
            args=(zip_path, input_dir, output_dir),
            daemon=True
        )
        self._worker.start()

    def _run_pipeline(self, zip_path, input_dir, output_dir):
        import glob

        IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tiff")

        try:
            # ── 1. Extract calibration zip ────────────────────────────────────
            self._log_write("=" * 50, "header")
            self._log_write("📦 Extracting calibration files...", "header")
            configs_dir = os.path.join(BASE_DIR, "configs")
            extract_calibration_zip(zip_path, configs_dir, log_fn=lambda m: self._log_write(m, "muted"))
            self._log_write("✓ Calibration files ready.", "ok")

            # ── 2. Build prep_config from UI toggles ──────────────────────────
            prep_config = {
                "enable_alignment":      self._step_align.get(),
                "enable_shadow_removal": self._step_shadow.get(),
                "enable_grayscale":      self._step_gray.get(),
                "enable_clahe":          self._step_clahe.get(),
                "enable_box_cropping":   self._step_crop.get(),
                "enable_pre_crop":       self._step_precrop.get(),
                "enable_timestamp_on_raw": self._step_timestamp.get(),
                "pre_crop":              {"top": 0, "bottom": 0, "left": 0, "right": 0},
            }
            cam_id  = self._cam_id.get()
            cam_num = int(cam_id.replace("cam", ""))

            # ── 3. Initialize pipeline ────────────────────────────────────────
            self._log_write(f"\n🔧 Initializing pipeline for {cam_id}...", "header")
            try:
                from src.image_pipeline import ImagePipeline
                import cv2
            except ImportError as e:
                self._log_write(f"❌ Import error: {e}", "err")
                self._log_write("   Make sure opencv-python is installed: pip install opencv-python", "warn")
                return

            pipeline = ImagePipeline(camera_id=cam_num, base_dir=BASE_DIR)
            try:
                pipeline.load_configs(
                    enable_align=prep_config["enable_alignment"],
                    enable_crop=prep_config["enable_box_cropping"]
                )
                self._log_write("✓ Pipeline initialized.", "ok")
            except RuntimeError as e:
                self._log_write(f"⚠ Config warning: {e}", "warn")
                self._log_write("  Processing will continue in raw/fallback mode.", "muted")

            # ── 4. Collect images ─────────────────────────────────────────────
            image_paths = []
            for ext in IMAGE_EXTS:
                image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
                image_paths.extend(glob.glob(os.path.join(input_dir, f"*{ext.upper()}")))
            image_paths = sorted(set(image_paths))

            if not image_paths:
                self._log_write(f"\n⚠ No images found in: {input_dir}", "warn")
                self._set_status("No images found.")
                return

            self._log_write(f"\n📂 Found {len(image_paths)} image(s) in: {input_dir}", "header")
            os.makedirs(output_dir, exist_ok=True)

            jpeg_q  = self._jpeg_quality.get()
            skip_raw= self._skip_raw.get()
            debug   = self._debug.get()

            success_count = 0
            skip_count    = 0
            error_count   = 0
            t_start = time.time()

            # ── 5. Process each image ─────────────────────────────────────────
            for i, img_path in enumerate(image_paths, 1):
                base_name = os.path.splitext(os.path.basename(img_path))[0]
                self._set_status(f"Processing {i}/{len(image_paths)}: {base_name}")
                self._set_progress(i - 1, len(image_paths))

                frame = cv2.imread(img_path)
                if frame is None:
                    self._log_write(f"[{i:>4}/{len(image_paths)}] SKIP — cannot read: {base_name}", "warn")
                    skip_count += 1
                    continue

                try:
                    t0 = time.time()
                    results = pipeline.process_frame(
                        frame=frame,
                        prep_config=prep_config,
                        debug=debug,
                        mock_name=base_name,
                        disable_clahe=not prep_config["enable_clahe"],
                    )
                    elapsed = time.time() - t0

                    saved_count = 0
                    for img_id, img_data in results:
                        if img_data is None or img_data.size == 0:
                            continue
                        if skip_raw and img_id == "reference_raw":
                            continue
                        if img_id == "debug_align" and not debug:
                            continue
                        out_name = f"{base_name}__{img_id}.jpg"
                        out_path = os.path.join(output_dir, out_name)
                        cv2.imwrite(out_path, img_data,
                                    [int(cv2.IMWRITE_JPEG_QUALITY), jpeg_q])
                        saved_count += 1

                    ids = [r[0] for r in results
                           if r[0] not in ("reference_raw",) or not skip_raw]
                    self._log_write(
                        f"[{i:>4}/{len(image_paths)}] ✓ {base_name}  "
                        f"({elapsed:.2f}s, {saved_count} file(s))", "ok")
                    success_count += 1

                except ValueError as e:
                    self._log_write(f"[{i:>4}/{len(image_paths)}] ⚠ SKIP — {base_name}: {e}", "warn")
                    skip_count += 1
                except Exception as e:
                    self._log_write(f"[{i:>4}/{len(image_paths)}] ✗ ERROR — {base_name}: {e}", "err")
                    error_count += 1

            # ── 6. Done ───────────────────────────────────────────────────────
            total_time = time.time() - t_start
            self._set_progress(len(image_paths), len(image_paths))

            summary_color = "ok" if error_count == 0 else "warn"
            self._log_write("\n" + "=" * 50, "header")
            self._log_write(
                f"✅ Done in {total_time:.1f}s  —  "
                f"Success: {success_count}  |  Skipped: {skip_count}  |  Errors: {error_count}",
                summary_color
            )
            self._log_write(f"📁 Outputs saved to: {output_dir}", "ok")

            status_msg = f"Done! {success_count} processed, {skip_count} skipped, {error_count} errors."
            self._set_status(status_msg)

            if success_count == 0 and skip_count > 0:
                self.after(0, lambda: messagebox.showwarning(
                    "All Images Skipped",
                    f"⚠️ All {skip_count} images were skipped!\n\n"
                    "Please check the log panel for the exact reasons.\n"
                    "Common causes:\n"
                    "- ❌ Wrong calibration ZIP (e.g. using cam0's zip for cam1)\n"
                    "- ❌ Camera was moved or zoom level changed drastically\n"
                    "- ❌ The image could not be read or is corrupted"
                ))
            elif success_count > 0:
                msg = f"Processing complete!\n\n✅ {success_count} images processed\n📁 Output folder: {output_dir}"
                if skip_count > 0:
                    msg += f"\n\n⚠️ {skip_count} images were skipped. See log panel for reasons."
                
                self.after(0, lambda text=msg: messagebox.showinfo(
                    "Complete", text
                ))

        except Exception as e:
            import traceback
            self._log_write(f"\n❌ Unexpected error: {e}", "err")
            self._log_write(traceback.format_exc(), "err")
            self._set_status(f"Error: {e}")
        finally:
            self._processing = False
            self.after(0, lambda: self._run_btn.config(
                text="▶  Run Pre-processing",
                state="normal",
                bg=ACCENT
            ))


if __name__ == "__main__":
    app = App()
    app.mainloop()
