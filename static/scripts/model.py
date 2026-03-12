#!/usr/bin/env python3
"""
Sim-Only 3D Finger Shape Predictor  (Direct-Vertices Model)
============================================================
Keyboard-driven virtual servo control using a direct-vertices ML model
in a single Open3D window.  No physical hardware required.

Controls (click the Open3D window to focus it):
  <- / ->     Virtual Servo 1  +/-100  ticks
  up / dn     Virtual Servo 2  +/-100  ticks
  A / D       Virtual Servo 1  +/-1000 ticks
  W / S       Virtual Servo 2  +/-1000 ticks
  R           Set current position as new zero reference
  Space       Return to zero reference
  C           Reset both servos to absolute zero (origin)
  Q / Esc     Quit

Status is printed to the terminal after every update.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import open3d as o3d

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent.parent   # …/sim_scripts/

MODEL_PRIORITY = [
    BASE_DIR / "model_direct/mlp_direct_vertices_predictor.pth",
]

REF_MESH_PATH = (
    BASE_DIR
    / "output/dataset_20251123_022003/amp_0/step_000/mesh.obj"
)

CALIB_PATH   = BASE_DIR / "real/input_calibration.npz"
CORRNET_PATH = BASE_DIR / "ml_training/models/direct/correction_net.pth"

# ─────────────────────────────────────────────────────────────────────────────
# Motion parameters
# ─────────────────────────────────────────────────────────────────────────────

STEP_SMALL =   100
STEP_LARGE =  1000
POS_MIN    = -10_000
POS_MAX    =  10_000

# ─────────────────────────────────────────────────────────────────────────────
# GLFW key codes
# ─────────────────────────────────────────────────────────────────────────────

KEY_RIGHT = 262
KEY_LEFT  = 263
KEY_DOWN  = 264
KEY_UP    = 265
KEY_SPACE = 32
KEY_ESC   = 256
KEY_Q     = ord("Q")
KEY_R     = ord("R")
KEY_C     = ord("C")
KEY_A     = ord("A")
KEY_D     = ord("D")
KEY_W     = ord("W")
KEY_S     = ord("S")


# ═════════════════════════════════════════════════════════════════════════════
# Neural networks
# ═════════════════════════════════════════════════════════════════════════════

class FingerMLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dims, dropout=0.0):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h),
                       nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class CorrNet(nn.Module):
    """Additive per-vertex correction: sim_in (2,) -> delta (N_VERTICES, 3)."""
    def __init__(self, n_vertices, hidden=(64, 256, 256, 64), dropout=0.1):
        super().__init__()
        self.n_vertices = n_vertices
        layers, d = [], 2
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h),
                       nn.GELU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, n_vertices * 3))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).view(x.shape[0], self.n_vertices, 3)


# ═════════════════════════════════════════════════════════════════════════════
# Model loading
# ═════════════════════════════════════════════════════════════════════════════

def load_model(override: Path | None = None):
    candidates = [override] if override is not None else MODEL_PRIORITY
    for path in candidates:
        if not path.exists():
            print(f"[Model] Not found: {path}")
            continue
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        cfg  = ckpt["config"]
        mdl  = FingerMLP(cfg["input_dim"], cfg["output_dim"],
                         cfg["hidden_layers"], dropout=cfg.get("dropout", 0.0))
        mdl.load_state_dict(ckpt["model_state"])
        mdl.eval()
        print(f"[Model] {path.name}  "
              f"(in={cfg['input_dim']} -> out={cfg['output_dim']})")
        return mdl, ckpt
    print("[Model] No checkpoint found — viewer will show reference mesh only.")
    return None, None


def load_input_calib() -> dict | None:
    if not CALIB_PATH.exists():
        print("[InCalib] not found — using default scale mapping")
        return None
    d = dict(np.load(str(CALIB_PATH)))
    calib = {"A": d["A"].astype(np.float32), "b": d["b"].astype(np.float32)}
    print(f"[InCalib] loaded  A={calib['A'].tolist()}")
    return calib


def load_correction_net() -> tuple[CorrNet, dict] | tuple[None, None]:
    if not CORRNET_PATH.exists():
        print("[CorrNet] not found — running without shape correction")
        return None, None
    ckpt = torch.load(str(CORRNET_PATH), map_location="cpu", weights_only=False)
    cfg  = ckpt["config"]
    net  = CorrNet(n_vertices=cfg["n_vertices"],
                   hidden=cfg["hidden"], dropout=cfg["dropout"])
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    print(f"[CorrNet] loaded  val_cd={ckpt.get('best_val_cd', '?'):.3f} mm  "
          f"epoch={ckpt.get('epoch', '?')}")
    return net, ckpt


# ═════════════════════════════════════════════════════════════════════════════
# Inference
# ═════════════════════════════════════════════════════════════════════════════

def predict_vertices(rel1: float, rel2: float,
                     model: FingerMLP, ckpt: dict,
                     input_calib: dict | None = None,
                     corrnet: CorrNet | None = None,
                     corrnet_ckpt: dict | None = None) -> np.ndarray | None:
    if model is None:
        return None

    mean_i = np.asarray(ckpt["input_stats"]["mean"], dtype=np.float32)
    std_i  = np.asarray(ckpt["input_stats"]["std"],  dtype=np.float32)
    rel    = np.array([rel1, rel2], dtype=np.float32)

    if input_calib is not None:
        x = input_calib["A"] @ rel + input_calib["b"]
    else:
        scale = 3.0 * std_i / POS_MAX
        x = rel * scale

    x_norm = (x - mean_i) / (std_i + 1e-8)

    with torch.no_grad():
        p_norm = model(torch.FloatTensor(x_norm).unsqueeze(0)).numpy()[0]

    output_flat = p_norm * ckpt["output_stats"]["std"] + ckpt["output_stats"]["mean"]
    verts = output_flat.reshape(-1, 3)   # (N_verts, 3)

    if corrnet is not None and corrnet_ckpt is not None and input_calib is not None:
        cn_mean = corrnet_ckpt["sim_in_mean"].astype(np.float32)
        cn_std  = corrnet_ckpt["sim_in_std"].astype(np.float32)
        x_corr  = (x - cn_mean) / (cn_std + 1e-8)
        with torch.no_grad():
            delta = corrnet(torch.FloatTensor(x_corr).unsqueeze(0)).numpy()[0]
        verts = verts + delta

    return verts


# ═════════════════════════════════════════════════════════════════════════════
# Application
# ═════════════════════════════════════════════════════════════════════════════

class SimApp:
    def __init__(self, model, ckpt, ref_mesh_path: Path,
                 input_calib: dict | None = None,
                 corrnet: CorrNet | None = None,
                 corrnet_ckpt: dict | None = None):
        self.model        = model
        self.ckpt         = ckpt
        self.input_calib  = input_calib
        self.corrnet      = corrnet
        self.corrnet_ckpt = corrnet_ckpt

        ref_mesh = o3d.io.read_triangle_mesh(str(ref_mesh_path))
        self.rest_verts = np.asarray(ref_mesh.vertices)

        self.pos1 = 0;  self.pos2 = 0
        self.zero_pos1 = 0;  self.zero_pos2 = 0

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(
            window_name="Sim Finger Predictor  |  arrows/WASD move · R zero · Space home · Q quit",
            width=1100, height=800,
        )
        ropt = self.vis.get_render_option()
        ropt.background_color = np.array([0.07, 0.07, 0.12])
        ropt.point_size       = 3.0

        self.display_pcd = o3d.geometry.PointCloud()
        self.display_pcd.points = o3d.utility.Vector3dVector(self.rest_verts)
        self.display_pcd.paint_uniform_color([0.55, 0.55, 0.60])
        self.vis.add_geometry(self.display_pcd)
        self.vis.reset_view_point(True)
        self._predicted_once = False

        self.vis.register_key_callback(KEY_LEFT,  lambda v: self._step(-STEP_SMALL, 0))
        self.vis.register_key_callback(KEY_RIGHT, lambda v: self._step(+STEP_SMALL, 0))
        self.vis.register_key_callback(KEY_UP,    lambda v: self._step(0, +STEP_SMALL))
        self.vis.register_key_callback(KEY_DOWN,  lambda v: self._step(0, -STEP_SMALL))
        self.vis.register_key_callback(KEY_A,     lambda v: self._step(-STEP_LARGE, 0))
        self.vis.register_key_callback(KEY_D,     lambda v: self._step(+STEP_LARGE, 0))
        self.vis.register_key_callback(KEY_W,     lambda v: self._step(0, +STEP_LARGE))
        self.vis.register_key_callback(KEY_S,     lambda v: self._step(0, -STEP_LARGE))
        self.vis.register_key_callback(KEY_SPACE, lambda v: self._goto_zero())
        self.vis.register_key_callback(KEY_R,     lambda v: self._set_zero())
        self.vis.register_key_callback(KEY_C,     lambda v: self._reset_origin())
        self.vis.register_key_callback(KEY_Q,     lambda v: self._quit())
        self.vis.register_key_callback(KEY_ESC,   lambda v: self._quit())

        self._print_status("Ready — grey cloud = rest/reference shape")

    def _clamp(self, v):
        return max(POS_MIN, min(POS_MAX, v))

    def _step(self, d1, d2):
        self.pos1 = self._clamp(self.pos1 + d1)
        self.pos2 = self._clamp(self.pos2 + d2)
        self._update_mesh()
        return False

    def _goto_zero(self):
        self.pos1 = self.zero_pos1
        self.pos2 = self.zero_pos2
        self._update_mesh()
        return False

    def _set_zero(self):
        self.zero_pos1 = self.pos1
        self.zero_pos2 = self.pos2
        self._print_status(f"Zero reference set -> S1={self.zero_pos1}  S2={self.zero_pos2}")
        return False

    def _reset_origin(self):
        self.pos1 = 0;  self.pos2 = 0
        self.zero_pos1 = 0; self.zero_pos2 = 0
        self._update_mesh()
        return False

    def _quit(self):
        self.vis.close()
        return False

    def _update_mesh(self):
        rel1 = float(self.pos1 - self.zero_pos1)
        rel2 = float(self.pos2 - self.zero_pos2)

        verts = predict_vertices(rel1, rel2, self.model, self.ckpt,
                                 self.input_calib, self.corrnet, self.corrnet_ckpt)
        if verts is not None:
            display_verts = verts.copy()
            display_verts[:, 0] *= -1
            self.display_pcd.points = o3d.utility.Vector3dVector(display_verts)
            if not self._predicted_once:
                self.display_pcd.paint_uniform_color([0.20, 0.78, 0.50])
                self._predicted_once = True
            self.vis.update_geometry(self.display_pcd)

        self._print_status()

    def _print_status(self, note=""):
        rel1 = self.pos1 - self.zero_pos1
        rel2 = self.pos2 - self.zero_pos2
        pipe = ("CAL" if self.input_calib is not None else "---")
        pipe += ("+" if self.input_calib is not None and self.corrnet is not None else "-")
        pipe += ("CRR" if self.corrnet is not None else "---")
        line = (
            f"  S1={self.pos1:+6d} (D{rel1:+6d} / {rel1/4096:+.2f} turns)  "
            f"S2={self.pos2:+6d} (D{rel2:+6d} / {rel2/4096:+.2f} turns)"
            f"  [{pipe}]"
            + (f"  [{note}]" if note else "")
        )
        print(f"\r{line:<110}", end="", flush=True)

    def run(self):
        self.vis.run()
        self.vis.destroy_window()
        print()


# ═════════════════════════════════════════════════════════════════════════════
# Entry point
# ═════════════════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Sim-only 3D finger predictor (direct model)")
    parser.add_argument("--model", type=Path, default=None,
                        help="Path to a direct-vertices .pth checkpoint.")
    parser.add_argument("--no-correction", action="store_true", default=False,
                        help="Disable auto-loading of input_calibration.npz + correction_net.pth.")
    args = parser.parse_args()

    print("=" * 60)
    print("  SIM-ONLY 3D FINGER SHAPE PREDICTOR  (direct model)")
    print("=" * 60)

    if not REF_MESH_PATH.exists():
        sys.exit(
            f"[ERROR] Reference mesh not found:\n  {REF_MESH_PATH}\n"
            "Edit REF_MESH_PATH at the top of this script."
        )

    model, ckpt = load_model(args.model)

    input_calib  = None
    corrnet      = None
    corrnet_ckpt = None
    if not args.no_correction:
        input_calib           = load_input_calib()
        corrnet, corrnet_ckpt = load_correction_net()
        if corrnet is not None and input_calib is None:
            print("[CorrNet] WARNING: CorrNet found but no input_calib — correction skipped.")
            corrnet = None; corrnet_ckpt = None

    print("\nControls (focus the Open3D window):")
    print("  <- -> up dn   Small step (+/-100 ticks)")
    print("  A D W S       Large step (+/-1000 ticks)")
    print("  R             Set zero reference")
    print("  Space         Return to zero")
    print("  C             Reset to absolute origin")
    print("  Q / Esc       Quit\n")

    app = SimApp(model, ckpt, REF_MESH_PATH,
                 input_calib=input_calib,
                 corrnet=corrnet, corrnet_ckpt=corrnet_ckpt)
    app.run()
    print("Done.")


if __name__ == "__main__":
    main()
