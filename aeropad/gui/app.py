"""
aeropad.gui.app
===============
Tkinter front-end for the aeropad polar-reconstruction module.

Run with::

    python -m aeropad.gui.app

Workflow: load a polar CSV → map columns → set airfoil/flow parameters
→ pick a route (or Auto) → Run. The reconstruction executes on a worker
thread so the interface stays responsive during Kriging kernel
selection; results render into an embedded matplotlib canvas with
metrics and export options.
"""

from __future__ import annotations

import queue
import threading
import traceback
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from ..config import (
    AirfoilSpec, CaseConfig, DataSpec, FlowSpec, PipelineSpec,
)
from ..polar.reconstruct import recommend, reconstruct_polar

_NONE = "(none)"


class AeropadApp(tk.Tk):
    """Main application window."""

    def __init__(self) -> None:
        super().__init__()
        self.title("aeropad — Aircraft Preliminary Design Toolkit")
        self.geometry("1280x800")
        self._set_icon()
        self.minsize(1080, 680)

        self.df: pd.DataFrame | None = None
        self.result = None
        self._queue: queue.Queue = queue.Queue()

        self._build_layout()

    def _set_icon(self) -> None:
        import os
        assets = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "assets")
        try:
            if os.name == "nt":
                # Give the process its own taskbar identity, otherwise
                # Windows groups the window under the generic Python icon.
                import ctypes
                ctypes.windll.shell32.\
                    SetCurrentProcessExplicitAppUserModelID(
                        "USTH.aeropad.gui")
                self.iconbitmap(os.path.join(assets, "aeropad.ico"))
            else:
                img = tk.PhotoImage(
                    file=os.path.join(assets, "aeropad_icon.png"))
                self.iconphoto(True, img)
                self._icon_ref = img   # keep a reference alive
        except Exception:
            pass   # icon is cosmetic; never block startup

    # ── Layout ───────────────────────────────────────────────────────

    def _build_layout(self) -> None:
        nb = ttk.Notebook(self)
        nb.pack(fill="both", expand=True)

        polar_tab = ttk.Frame(nb, padding=8)
        nb.add(polar_tab, text="  Polar Reconstruction  ")
        self._build_polar_tab(polar_tab)

        sizing_tab = SizingTab(nb)
        nb.add(sizing_tab, text="  Sizing by Statistics  ")

    def _build_polar_tab(self, root: ttk.Frame) -> None:

        controls = ttk.Frame(root, width=360)
        controls.pack(side="left", fill="y", padx=(0, 8))
        controls.pack_propagate(False)

        plot_area = ttk.Frame(root)
        plot_area.pack(side="right", fill="both", expand=True)

        # -- data section --------------------------------------------
        sec_data = ttk.LabelFrame(controls, text="1 · Data", padding=6)
        sec_data.pack(fill="x", pady=(0, 6))

        ttk.Button(sec_data, text="Load polar CSV…",
                   command=self._load_csv).pack(fill="x")
        self.lbl_file = ttk.Label(sec_data, text="no file loaded",
                                  foreground="gray")
        self.lbl_file.pack(anchor="w", pady=(2, 4))

        self.col_vars: dict[str, tk.StringVar] = {}
        self.col_combos: dict[str, ttk.Combobox] = {}
        for key, label in (
                ("aoa", "AoA column"),
                ("lf_cl", "LF CL column"),
                ("lf_cd", "LF CD column"),
                ("hf_cl", "HF CL column (optional)"),
                ("hf_cd", "HF CD column (optional)")):
            row = ttk.Frame(sec_data)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=label, width=22).pack(side="left")
            var = tk.StringVar(value=_NONE)
            combo = ttk.Combobox(row, textvariable=var, state="disabled",
                                 width=14)
            combo.pack(side="right", fill="x", expand=True)
            self.col_vars[key] = var
            self.col_combos[key] = combo

        ttk.Checkbutton(sec_data, text="Flip AoA sign",
                        variable=self._mkvar("flip", False)
                        ).pack(anchor="w")

        # -- airfoil section -----------------------------------------
        sec_air = ttk.LabelFrame(controls, text="2 · Airfoil", padding=6)
        sec_air.pack(fill="x", pady=(0, 6))

        ttk.Button(sec_air, text="Load geometry from .dat…",
                   command=self._load_dat_geometry).pack(fill="x",
                                                         pady=(0, 3))

        row = ttk.Frame(sec_air); row.pack(fill="x", pady=1)
        ttk.Label(row, text="Symmetry", width=22).pack(side="left")
        self.var_sym = tk.StringVar(value="asymmetric")
        ttk.Combobox(row, textvariable=self.var_sym, state="readonly",
                     values=("asymmetric", "symmetric"), width=14
                     ).pack(side="right", fill="x", expand=True)

        self.air_entries: dict[str, tk.StringVar] = {}
        for key, label, default in (
                ("TC", "Thickness/chord", "0.12"),
                ("alpha_s_pos", "α_stall + (deg)", "15.0"),
                ("CL_s_pos", "CL_stall +", "1.45"),
                ("alpha_s_neg", "α_stall − (deg)", "-11.0"),
                ("CL_s_neg", "CL_stall −", "-0.85"),
                ("delta_nose", "Nose wedge (deg)", "0.0"),
                ("delta_tail", "Tail wedge (deg)", "0.0")):
            self._entry_row(sec_air, key, label, default,
                            self.air_entries)

        # -- flow section --------------------------------------------
        sec_flow = ttk.LabelFrame(controls, text="3 · Flow", padding=6)
        sec_flow.pack(fill="x", pady=(0, 6))
        self.flow_entries: dict[str, tk.StringVar] = {}
        self._entry_row(sec_flow, "Re", "Reynolds number", "2.0e6",
                        self.flow_entries)
        self._entry_row(sec_flow, "M", "Mach number", "0.16",
                        self.flow_entries)

        # -- route section -------------------------------------------
        sec_route = ttk.LabelFrame(controls, text="4 · Reconstruction",
                                   padding=6)
        sec_route.pack(fill="x", pady=(0, 6))

        self.var_route = tk.StringVar(value="auto")
        for val, label in (("auto", "Auto (advisor)"),
                           ("semi-empirical", "Semi-empirical extrapolation"),
                           ("kriging", "Kriging surrogate")):
            ttk.Radiobutton(sec_route, text=label, value=val,
                            variable=self.var_route).pack(anchor="w")

        row = ttk.Frame(sec_route); row.pack(fill="x", pady=(4, 1))
        ttk.Label(row, text="Method (semi-emp.)", width=22).pack(side="left")
        self.var_method = tk.StringVar(value="auto")
        ttk.Combobox(row, textvariable=self.var_method, state="readonly",
                     values=("auto", "battisti", "aerodas",
                             "montgomerie", "lindenburg"), width=14
                     ).pack(side="right", fill="x", expand=True)

        self.pipe_entries: dict[str, tk.StringVar] = {}
        self._entry_row(sec_route, "PM_cutoff", "LF cutoff (deg)", "20.0",
                        self.pipe_entries)
        self._entry_row(sec_route, "blend_end", "Blend end (deg)", "45.0",
                        self.pipe_entries)
        self._entry_row(sec_route, "spacing", "Kriging spacing (deg)",
                        "20.0", self.pipe_entries)

        row = ttk.Frame(sec_route)
        row.pack(fill="x", pady=(1, 0))
        ttk.Label(row, text="Stall-bracket stations", width=22).pack(
            side="left")
        self.var_bracket = tk.StringVar(value="auto")
        ttk.Combobox(row, textvariable=self.var_bracket,
                     state="readonly", values=("auto", "on", "off"),
                     width=14).pack(side="right", fill="x", expand=True)

        # -- actions --------------------------------------------------
        sec_act = ttk.Frame(controls)
        sec_act.pack(fill="x", pady=(0, 6))
        self.btn_advise = ttk.Button(sec_act, text="Advise",
                                     command=self._advise)
        self.btn_advise.pack(side="left", fill="x", expand=True,
                             padx=(0, 3))
        self.btn_run = ttk.Button(sec_act, text="Run",
                                  command=self._run)
        self.btn_run.pack(side="right", fill="x", expand=True,
                          padx=(3, 0))

        sec_exp = ttk.Frame(controls)
        sec_exp.pack(fill="x", pady=(0, 6))
        ttk.Button(sec_exp, text="Export CSV…",
                   command=self._export_csv).pack(
            side="left", fill="x", expand=True, padx=(0, 3))
        ttk.Button(sec_exp, text="Save figure…",
                   command=self._export_png).pack(
            side="right", fill="x", expand=True, padx=(3, 0))

        # -- status / metrics ----------------------------------------
        self.txt_status = tk.Text(controls, height=9, wrap="word",
                                  state="disabled", relief="solid",
                                  borderwidth=1)
        self.txt_status.pack(fill="both", expand=True)

        # -- plot area ------------------------------------------------
        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax_cl = self.fig.add_subplot(211)
        self.ax_cd = self.fig.add_subplot(212)
        self.fig.tight_layout(pad=2.5)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self._draw_placeholder()

    # ── Small helpers ────────────────────────────────────────────────

    def _mkvar(self, name: str, default) -> tk.Variable:
        var = tk.BooleanVar(value=default) if isinstance(default, bool) \
            else tk.StringVar(value=str(default))
        setattr(self, f"var_{name}", var)
        return var

    def _entry_row(self, parent, key, label, default, store) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=1)
        ttk.Label(row, text=label, width=22).pack(side="left")
        var = tk.StringVar(value=default)
        ttk.Entry(row, textvariable=var, width=14).pack(
            side="right", fill="x", expand=True)
        store[key] = var

    def _log(self, text: str, clear: bool = False) -> None:
        self.txt_status.configure(state="normal")
        if clear:
            self.txt_status.delete("1.0", "end")
        self.txt_status.insert("end", text + "\n")
        self.txt_status.see("end")
        self.txt_status.configure(state="disabled")

    def _draw_placeholder(self) -> None:
        for ax, label in ((self.ax_cl, "$C_L$"), (self.ax_cd, "$C_D$")):
            ax.clear()
            ax.set_xlim(-185, 185)
            ax.set_xlabel("AoA [deg]")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
            ax.text(0.5, 0.5, "load data and run",
                    transform=ax.transAxes, ha="center",
                    color="gray")
        self.canvas.draw_idle()

    # ── Data loading ─────────────────────────────────────────────────

    def _load_dat_geometry(self) -> None:
        """Fill the geometric airfoil fields from a coordinate file."""
        path = filedialog.askopenfilename(
            title="Select airfoil coordinate file (.dat)",
            filetypes=[("Airfoil dat", "*.dat"), ("All files", "*.*")])
        if not path:
            return
        try:
            from ..polar.geometry import analyze_dat
            geo = analyze_dat(path)
        except Exception as exc:
            messagebox.showerror("Geometry analysis failed", str(exc))
            return
        self.var_sym.set(geo.symmetry)
        self.air_entries["TC"].set(f"{geo.TC:.4f}")
        self.air_entries["delta_nose"].set(f"{geo.delta_nose_deg:.2f}")
        self.air_entries["delta_tail"].set(f"{geo.delta_tail_deg:.2f}")
        self._log("── Geometry from .dat ──", clear=True)
        self._log(geo.summary())
        self._log("\nFilled: symmetry, TC, nose/tail delta. Stall "
                  "characteristics (α_stall, CL_stall) are aerodynamic "
                  "inputs — set those from data or estimates.")

    def _load_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select polar CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            df = pd.read_csv(path)
            df.columns = [c.strip().replace("\ufeff", "")
                          for c in df.columns]
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.df = df
        self.lbl_file.configure(text=path.split("/")[-1],
                                foreground="black")
        cols = list(df.columns)
        options = cols + [_NONE]
        for key, combo in self.col_combos.items():
            combo.configure(state="readonly", values=options)
        # naive auto-mapping
        self._auto_map(cols)
        self._log(f"Loaded {len(df)} rows, columns: {', '.join(cols)}",
                  clear=True)

    def _auto_map(self, cols: list) -> None:
        def pick(*needles, exclude=()):
            for c in cols:
                lc = c.lower()
                if any(n in lc for n in needles) and \
                        not any(x in lc for x in exclude):
                    return c
            return _NONE
        self.col_vars["aoa"].set(pick("aoa", "alpha"))
        self.col_vars["lf_cl"].set(pick("cl_pm", "cl_xfoil", "cl_lf"))
        self.col_vars["lf_cd"].set(pick("cd_pm", "cd_xfoil", "cd_lf"))
        self.col_vars["hf_cl"].set(pick("cl_cfd", "cl_hf"))
        self.col_vars["hf_cd"].set(pick("cd_cfd", "cd_hf"))

    # ── Config assembly ──────────────────────────────────────────────

    def _make_config(self) -> CaseConfig:
        def fnum(var, default=None):
            s = var.get().strip()
            return default if not s else float(s)

        def col(key):
            v = self.col_vars[key].get()
            return None if v == _NONE else v

        return CaseConfig(
            airfoil=AirfoilSpec(
                name="gui_case",
                symmetry=self.var_sym.get(),
                TC=fnum(self.air_entries["TC"], 0.12),
                alpha_s_2D_pos=fnum(self.air_entries["alpha_s_pos"]),
                CL_s_2D_pos=fnum(self.air_entries["CL_s_pos"]),
                alpha_s_2D_neg=fnum(self.air_entries["alpha_s_neg"]),
                CL_s_2D_neg=fnum(self.air_entries["CL_s_neg"]),
                delta_nose_deg=fnum(self.air_entries["delta_nose"], 0.0),
                delta_tail_deg=fnum(self.air_entries["delta_tail"], 0.0)),
            flow=FlowSpec(
                Re=fnum(self.flow_entries["Re"], 1e6),
                M=fnum(self.flow_entries["M"], 0.1)),
            data=DataSpec(
                LF_CL_column=col("lf_cl") or "CL_LF",
                LF_CD_column=col("lf_cd") or "CD_LF",
                HF_CL_column=col("hf_cl"),
                HF_CD_column=col("hf_cd"),
                flip_aoa_sign=bool(self.var_flip.get())),
            pipeline=PipelineSpec(
                extrapolator=self.var_method.get(),
                PM_cutoff=fnum(self.pipe_entries["PM_cutoff"], 20.0),
                blend_end=fnum(self.pipe_entries["blend_end"], 45.0),
                kriging_spacing=fnum(self.pipe_entries["spacing"], 20.0)))

    def _prepared_df(self, config: CaseConfig) -> pd.DataFrame:
        aoa_col = self.col_vars["aoa"].get()
        if aoa_col == _NONE:
            raise ValueError("AoA column not selected.")
        df = self.df.rename(columns={aoa_col: "AoA"})
        return df

    # ── Actions ──────────────────────────────────────────────────────

    def _advise(self) -> None:
        if self.df is None:
            messagebox.showinfo("No data", "Load a polar CSV first.")
            return
        try:
            config = self._make_config()
            df = self._prepared_df(config)
            hf = config.data.HF_CL_column
            budget = int(df[hf].notna().sum()) \
                if hf and hf in df.columns else None
            adv = recommend(config, hf_budget=budget)
        except Exception as exc:
            messagebox.showerror("Advise failed", str(exc))
            return
        self._log("── Advisor ──", clear=True)
        self._log(f"Available HF samples: {budget}")
        self._log(f"Recommended route: {adv['route']}")
        if adv["method_CL"]:
            self._log(f"Method CL: {adv['method_CL']} · "
                      f"CD: {adv['method_CD']}")
        if adv["bracket_stall"]:
            self._log("Supplementary ±stall stations advised (low Mach).")
        self._log(adv["notes"])

    def _run(self) -> None:
        if self.df is None:
            messagebox.showinfo("No data", "Load a polar CSV first.")
            return
        try:
            config = self._make_config()
            df = self._prepared_df(config)
        except Exception as exc:
            messagebox.showerror("Configuration error", str(exc))
            return

        self.btn_run.configure(state="disabled", text="Running…")
        self._log("Running reconstruction…", clear=True)

        route = self.var_route.get()
        bracket = {"auto": None, "on": True,
                   "off": False}[self.var_bracket.get()]

        def work():
            try:
                res = reconstruct_polar(df, config, route=route,
                                        bracket_stall=bracket)
                self._queue.put(("ok", res, df, config))
            except Exception:
                self._queue.put(("err", traceback.format_exc(),
                                 None, None))

        threading.Thread(target=work, daemon=True).start()
        self.after(150, self._poll)

    def _poll(self) -> None:
        try:
            status, payload, df, config = self._queue.get_nowait()
        except queue.Empty:
            self.after(150, self._poll)
            return

        self.btn_run.configure(state="normal", text="Run")
        if status == "err":
            self._log("FAILED:\n" + payload)
            return

        self.result = payload
        self._log("── Result ──", clear=True)
        self._log(self.result.summary())
        if self.result.advisory.get("notes"):
            self._log("\nAdvisor: " + self.result.advisory["notes"])
        self._plot(df, config)

    # ── Plotting ─────────────────────────────────────────────────────

    def _plot(self, df: pd.DataFrame, config: CaseConfig) -> None:
        res = self.result
        d = config.data
        polar = res.polar

        for ax, coeff, curve_col, lf_col, hf_col in (
                (self.ax_cl, "$C_L$", "CL_full",
                 d.LF_CL_column, d.HF_CL_column),
                (self.ax_cd, "$C_D$", "CD_full",
                 d.LF_CD_column, d.HF_CD_column)):
            ax.clear()
            if lf_col and lf_col in df.columns:
                lf = df.dropna(subset=[lf_col])
                ax.scatter(lf["AoA"], lf[lf_col], s=10, color="0.75",
                           marker=".", label="LF data", zorder=2)
            if hf_col and hf_col in df.columns:
                hf = df.dropna(subset=[hf_col])
                ax.scatter(hf["AoA"], hf[hf_col], s=16, color="0.35",
                           marker="o", label="HF reference", zorder=3)
            ax.plot(polar["AoA"], polar[curve_col], color="C3",
                    linewidth=1.8, zorder=4,
                    label=f"Reconstruction ({res.route})")
            if res.route == "kriging":
                key = "train_" + curve_col.split("_")[0]
                train = res.models.get(key)
                if train is not None:
                    col = hf_col
                    ax.scatter(train["AoA"], train[col], s=46,
                               facecolor="none", edgecolor="C0",
                               linewidth=1.2, zorder=5,
                               label="Training stations")
            ax.set_xlim(-185, 185)
            ax.set_xlabel("AoA [deg]")
            ax.set_ylabel(coeff)
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best", fontsize=8)
        self.fig.tight_layout(pad=2.5)
        self.canvas.draw_idle()

    # ── Export ───────────────────────────────────────────────────────

    def _export_csv(self) -> None:
        if self.result is None:
            messagebox.showinfo("Nothing to export", "Run first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")])
        if path:
            self.result.polar.to_csv(path, index=False)
            self._log(f"Exported polar -> {path}")

    def _export_png(self) -> None:
        if self.result is None:
            messagebox.showinfo("Nothing to export", "Run first.")
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG image", "*.png")])
        if path:
            self.fig.savefig(path, dpi=200, bbox_inches="tight")
            self._log(f"Saved figure -> {path}")


class SizingTab(ttk.Frame):
    """Sizing-by-statistics tab: dataset exploration, family fitting,
    cross-family comparison, and point prediction."""

    def __init__(self, master) -> None:
        super().__init__(master, padding=8)
        self.df: pd.DataFrame | None = None
        self.model = None
        self._queue: queue.Queue = queue.Queue()
        self._predict_vars: dict[str, tk.StringVar] = {}
        self._build()

    # ── layout ───────────────────────────────────────────────────────

    def _build(self) -> None:
        controls = ttk.Frame(self, width=360)
        controls.pack(side="left", fill="y", padx=(0, 8))
        controls.pack_propagate(False)

        plot_area = ttk.Frame(self)
        plot_area.pack(side="right", fill="both", expand=True)

        # data
        sec = ttk.LabelFrame(controls, text="1 · Dataset", padding=6)
        sec.pack(fill="x", pady=(0, 6))
        ttk.Button(sec, text="Load dataset CSV…",
                   command=self._load_csv).pack(fill="x")
        self.lbl_file = ttk.Label(sec, text="no file loaded",
                                  foreground="gray")
        self.lbl_file.pack(anchor="w", pady=(2, 2))

        # features / target
        sec2 = ttk.LabelFrame(controls, text="2 · Relation", padding=6)
        sec2.pack(fill="x", pady=(0, 6))
        ttk.Label(sec2, text="Features (multi-select):").pack(anchor="w")
        lb_frame = ttk.Frame(sec2)
        lb_frame.pack(fill="x")
        self.lst_features = tk.Listbox(lb_frame, selectmode="multiple",
                                       height=6, exportselection=False)
        self.lst_features.pack(side="left", fill="x", expand=True)
        sb = ttk.Scrollbar(lb_frame, orient="vertical",
                           command=self.lst_features.yview)
        sb.pack(side="right", fill="y")
        self.lst_features.configure(yscrollcommand=sb.set)

        row = ttk.Frame(sec2)
        row.pack(fill="x", pady=(4, 1))
        ttk.Label(row, text="Target", width=10).pack(side="left")
        self.var_target = tk.StringVar()
        self.cmb_target = ttk.Combobox(row, textvariable=self.var_target,
                                       state="disabled")
        self.cmb_target.pack(side="right", fill="x", expand=True)

        # family
        sec3 = ttk.LabelFrame(controls, text="3 · Regression family",
                              padding=6)
        sec3.pack(fill="x", pady=(0, 6))
        from ..sizing.models import FAMILIES, symbolic_available
        fams = [f for f in FAMILIES
                if f != "symbolic" or symbolic_available()]
        self.var_family = tk.StringVar(value="power_law")
        ttk.Combobox(sec3, textvariable=self.var_family,
                     state="readonly", values=fams).pack(fill="x")
        if "symbolic" not in fams:
            ttk.Label(sec3, text="(symbolic: PySR/Julia not detected)",
                      foreground="gray", font=("", 8)).pack(anchor="w")
        else:
            ttk.Label(sec3,
                      text="(symbolic runs via Fit only — excluded "
                           "from Compare all for speed)",
                      foreground="gray", font=("", 8)).pack(anchor="w")

        # actions
        act = ttk.Frame(controls)
        act.pack(fill="x", pady=(0, 6))
        ttk.Button(act, text="Heatmap",
                   command=self._heatmap).pack(side="left", fill="x",
                                               expand=True, padx=(0, 2))
        self.btn_fit = ttk.Button(act, text="Fit", command=self._fit)
        self.btn_fit.pack(side="left", fill="x", expand=True, padx=2)
        self.btn_cmp = ttk.Button(act, text="Compare all",
                                  command=self._compare)
        self.btn_cmp.pack(side="right", fill="x", expand=True,
                          padx=(2, 0))

        # prediction
        self.sec_pred = ttk.LabelFrame(controls,
                                       text="4 · Predict a design",
                                       padding=6)
        self.sec_pred.pack(fill="x", pady=(0, 6))
        self.pred_holder = ttk.Frame(self.sec_pred)
        self.pred_holder.pack(fill="x")
        ttk.Button(self.sec_pred, text="Predict",
                   command=self._predict).pack(fill="x", pady=(4, 0))

        # status
        self.txt = tk.Text(controls, height=10, wrap="word",
                           state="disabled", relief="solid",
                           borderwidth=1)
        self.txt.pack(fill="both", expand=True)

        # canvas
        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_area)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        ax = self.fig.add_subplot(111)
        ax.text(0.5, 0.5, "load a dataset", transform=ax.transAxes,
                ha="center", color="gray")
        ax.set_axis_off()
        self.canvas.draw_idle()

    # ── helpers ──────────────────────────────────────────────────────

    def _log(self, text: str, clear: bool = False) -> None:
        self.txt.configure(state="normal")
        if clear:
            self.txt.delete("1.0", "end")
        self.txt.insert("end", text + "\n")
        self.txt.see("end")
        self.txt.configure(state="disabled")

    def _numeric_columns(self) -> list:
        num = self.df.apply(
            lambda s: pd.to_numeric(s, errors="coerce"))
        keep = [c for c in num.columns
                if num[c].notna().sum() >= 0.5 * len(num)]
        return keep

    def _selected_features(self) -> list:
        return [self.lst_features.get(i)
                for i in self.lst_features.curselection()]

    def _coerced_df(self) -> pd.DataFrame:
        out = self.df.copy()
        for c in self._numeric_columns():
            out[c] = pd.to_numeric(out[c], errors="coerce")
        return out

    # ── actions ──────────────────────────────────────────────────────

    def _load_csv(self) -> None:
        path = filedialog.askopenfilename(
            title="Select sizing dataset CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")])
        if not path:
            return
        try:
            self.df = pd.read_csv(path)
            self.df.columns = [c.strip().replace("\ufeff", "")
                               for c in self.df.columns]
        except Exception as exc:
            messagebox.showerror("Load failed", str(exc))
            return
        self.lbl_file.configure(text=path.split("/")[-1],
                                foreground="black")
        cols = self._numeric_columns()
        self.lst_features.delete(0, "end")
        for c in cols:
            self.lst_features.insert("end", c)
        self.cmb_target.configure(state="readonly", values=cols)
        if cols:
            self.var_target.set(cols[-1])
        self._log(f"Loaded {len(self.df)} rows. Numeric columns: "
                  f"{', '.join(cols)}", clear=True)

    def _heatmap(self) -> None:
        if self.df is None:
            messagebox.showinfo("No data", "Load a dataset CSV first.")
            return
        import numpy as np
        num = self._coerced_df()[self._numeric_columns()]
        corr = num.corr()
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        mask = np.triu(np.ones_like(corr, dtype=bool), k=1)
        data = corr.where(~mask)
        im = ax.imshow(data, cmap="coolwarm", vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr)), corr.columns, rotation=45,
                      ha="right", fontsize=8)
        ax.set_yticks(range(len(corr)), corr.columns, fontsize=8)
        for i in range(len(corr)):
            for j in range(i + 1):
                ax.text(j, i, f"{corr.iloc[i, j]:.2f}", ha="center",
                        va="center", fontsize=8)
        self.fig.colorbar(im, ax=ax, label="Pearson correlation")
        ax.set_title("Correlation heatmap")
        self.fig.tight_layout()
        self.canvas.draw_idle()
        self._log("Correlation heatmap drawn.", clear=True)

    def _fit(self) -> None:
        self._run_workers(mode="fit")

    def _compare(self) -> None:
        self._run_workers(mode="compare")

    def _run_workers(self, mode: str) -> None:
        """Run fits in worker *processes* so the UI never starves.

        Python threads share the GIL: a GridSearchCV storm of small
        numpy operations starves the Tk event loop ("Not Responding").
        Worker processes isolate that completely, and one-future-per-
        family gives natural progress reporting.
        """
        if self.df is None:
            messagebox.showinfo("No data", "Load a dataset CSV first.")
            return
        features = self._selected_features()
        target = self.var_target.get()
        if not features or not target or target in features:
            messagebox.showinfo(
                "Selection needed",
                "Select at least one feature and a distinct target.")
            return
        df = self._coerced_df()
        family = self.var_family.get()

        from ..sizing.models import FAMILIES
        if mode == "fit":
            fams = [family]
        else:
            # Symbolic regression is excluded from batch comparison by
            # design: its evolutionary search (plus Julia compilation
            # on the first-ever run) is an order of magnitude slower
            # than the other families. Run it deliberately via Fit.
            fams = [f for f in FAMILIES if f != "symbolic"]

        try:
            import concurrent.futures as cf
            self._executor = cf.ProcessPoolExecutor(max_workers=1)
            from ..sizing import worker
            self._futures = {
                self._executor.submit(worker.fit_family, f, df,
                                      features, target): f
                for f in fams}
            engine = "process pool"
        except Exception:
            # Fallback: threads (responsiveness may suffer)
            import threading
            self._executor = None
            self._futures = {}
            self._thread_results: list = []

            def run_all():
                from ..sizing.worker import fit_family
                for f in fams:
                    self._thread_results.append(
                        fit_family(f, df, features, target))

            threading.Thread(target=run_all, daemon=True).start()
            engine = "thread (fallback)"

        self._mode = mode
        self._n_total = len(fams)
        self._results: list = []
        self.btn_fit.configure(state="disabled")
        self.btn_cmp.configure(state="disabled")
        note = ""
        if family == "symbolic" and mode == "fit":
            note = ("\nNote: PySR's first-ever run compiles its Julia "
                    "backend and can take several minutes; subsequent "
                    "runs are much faster.")
        self._log(f"Running {mode} on {self._n_total} family(ies) "
                  f"[{engine}]…{note}", clear=True)
        self.after(300, self._poll_workers)

    def _poll_workers(self) -> None:
        if self._executor is not None:
            done = [f for f in self._futures if f.done()]
            for f in done:
                fam = self._futures.pop(f)
                try:
                    self._results.append(f.result())
                except Exception as exc:
                    self._results.append((fam, None, str(exc)))
                self._log(f"  [{len(self._results)}/{self._n_total}] "
                          f"{fam}: "
                          f"{'ok' if self._results[-1][1] else 'failed'}")
            finished = not self._futures
        else:
            self._results = list(self._thread_results)
            finished = len(self._results) >= self._n_total

        if not finished:
            self.after(400, self._poll_workers)
            return

        if self._executor is not None:
            self._executor.shutdown(wait=False)
            self._executor = None
        self.btn_fit.configure(state="normal")
        self.btn_cmp.configure(state="normal")
        self._finish_workers()

    def _finish_workers(self) -> None:
        ok = {fam: m for fam, m, err in self._results if m is not None}
        errs = {fam: err for fam, m, err in self._results
                if m is None}

        if not ok:
            self._log("All fits failed:\n" + "\n".join(
                f"  {f}: {e}" for f, e in errs.items()))
            return

        if self._mode == "fit":
            self.model = next(iter(ok.values()))
            self._log("── Fit result ──", clear=True)
            self._log(self.model.summary())
            self._draw_fit()
            self._build_predict_entries()
            return

        # compare: assemble the table from per-family metrics
        rows = []
        for fam, m in ok.items():
            rows.append({k: v for k, v in m.metrics.items()
                         if k != "best_params"})
        table = pd.DataFrame(rows).set_index("family")
        best = table["R2_test"].astype(float).idxmax()
        self.model = ok[best]
        self._log("── Family comparison ──", clear=True)
        self._log(table[["R2_test", "RMSE_test", "MAE_test"]]
                  .round(4).to_string())
        for fam, err in errs.items():
            self._log(f"  {fam}: SKIPPED ({err})")
        self._log(f"\nBest by test R²: {best} "
                  f"(kept as active model for prediction)")
        self._draw_compare(table)
        self._build_predict_entries()

    # ── drawing ──────────────────────────────────────────────────────

    def _draw_fit(self) -> None:
        import numpy as np
        m = self.model
        self.fig.clear()
        two_feat = len(m.features) == 2

        ax1 = self.fig.add_subplot(1, 2 if two_feat else 1, 1)
        _, X_te, _, y_te = m._split
        y_hat = m.predict(X_te)
        lo = min(y_te.min(), y_hat.min())
        hi = max(y_te.max(), y_hat.max())
        ax1.plot([lo, hi], [lo, hi], "r:", linewidth=1.2)
        ax1.scatter(y_te, y_hat, s=22, alpha=0.8)
        ax1.set_xlabel(f"Actual {m.target}")
        ax1.set_ylabel(f"Predicted {m.target}")
        ax1.set_title(f"{m.family} "
                      f"(R²={m.metrics['R2_test']:.3f})", fontsize=10)
        ax1.grid(True, alpha=0.3)

        if two_feat:
            f1, f2 = m.features
            df = self._coerced_df()[[f1, f2, m.target]].dropna()
            g1 = np.linspace(df[f1].min(), df[f1].max(), 40)
            g2 = np.linspace(df[f2].min(), df[f2].max(), 40)
            G1, G2 = np.meshgrid(g1, g2)
            Z = m.predict(np.column_stack([G1.ravel(), G2.ravel()])) \
                .reshape(G1.shape)
            ax2 = self.fig.add_subplot(122, projection="3d")
            ax2.plot_surface(G1, G2, Z, cmap="viridis", alpha=0.75,
                             linewidth=0)
            ax2.scatter(df[f1], df[f2], df[m.target], color="black",
                        s=10, alpha=0.6)
            ax2.set_xlabel(f1, fontsize=8)
            ax2.set_ylabel(f2, fontsize=8)
            ax2.set_zlabel(m.target, fontsize=8)
            ax2.set_title("Fitted surface", fontsize=10)

        self.fig.tight_layout()
        self.canvas.draw_idle()

    def _draw_compare(self, table) -> None:
        t = table.dropna(subset=["R2_test"]) \
            if "R2_test" in table.columns else table
        self.fig.clear()
        ax = self.fig.add_subplot(111)
        vals = t["R2_test"].astype(float)
        ax.barh(list(vals.index), vals.values, color="C0", alpha=0.85)
        ax.set_xlabel("Test R²")
        ax.set_xlim(0, 1)
        for i, v in enumerate(vals.values):
            ax.text(v + 0.01, i, f"{v:.3f}", va="center", fontsize=9)
        ax.set_title("Family comparison (held-out test R²)")
        ax.grid(True, axis="x", alpha=0.3)
        self.fig.tight_layout()
        self.canvas.draw_idle()

    # ── prediction ───────────────────────────────────────────────────

    def _build_predict_entries(self) -> None:
        for w in self.pred_holder.winfo_children():
            w.destroy()
        self._predict_vars.clear()
        for f in self.model.features:
            row = ttk.Frame(self.pred_holder)
            row.pack(fill="x", pady=1)
            ttk.Label(row, text=f, width=22).pack(side="left")
            var = tk.StringVar()
            ttk.Entry(row, textvariable=var, width=12).pack(
                side="right", fill="x", expand=True)
            self._predict_vars[f] = var

    def _predict(self) -> None:
        if self.model is None:
            messagebox.showinfo("No model", "Fit a family first.")
            return
        try:
            q = {f: float(v.get()) for f, v in self._predict_vars.items()}
        except ValueError:
            messagebox.showerror("Invalid input",
                                 "All feature values must be numeric.")
            return
        try:
            y = self.model.predict(q)[0]
        except Exception as exc:
            messagebox.showerror("Prediction failed", str(exc))
            return
        self._log(f"\nPrediction [{self.model.family}]: "
                  f"{self.model.target} = {y:.4g}  for {q}")
        eq = self.model.equation()
        if eq:
            self._log(f"Equation: {eq}")


def main() -> None:
    app = AeropadApp()
    app.mainloop()


if __name__ == "__main__":
    main()
