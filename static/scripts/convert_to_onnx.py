#!/usr/bin/env python3
"""
Convert SOFTMAP PyTorch models to ONNX + JSON for browser demo.
================================================================
Reads from ./assets/ and writes to ../demo_assets/.

Usage:
  cd static/scripts
  python convert_to_onnx.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# ─────────────────────────────────────────────────────────────────
# Paths (relative to this script)
# ─────────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
ASSETS_DIR = SCRIPT_DIR / "assets"
OUT_DIR    = SCRIPT_DIR.parent / "demo_assets"

MODEL_PATH   = ASSETS_DIR / "mlp_direct_vertices_predictor.pth"
CORRNET_PATH = ASSETS_DIR / "correction_net.pth"
CALIB_PATH   = ASSETS_DIR / "input_calibration.npz"


# ─────────────────────────────────────────────────────────────────
# Network definitions (copied from model.py to be self-contained)
# ─────────────────────────────────────────────────────────────────
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


# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────
def to_list(x):
    if isinstance(x, (np.ndarray, torch.Tensor)):
        return x.tolist()
    return x


def load_main_model():
    if not MODEL_PATH.exists():
        sys.exit(f"[ERROR] Main model not found: {MODEL_PATH}")
    ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = FingerMLP(cfg["input_dim"], cfg["output_dim"],
                      cfg["hidden_layers"], dropout=cfg.get("dropout", 0.0))
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"[OK] Loaded main model: {MODEL_PATH.name}  "
          f"(in={cfg['input_dim']} -> out={cfg['output_dim']})")
    return model, ckpt


def load_correction_net():
    if not CORRNET_PATH.exists():
        print(f"[--] CorrNet not found: {CORRNET_PATH.name}  (skipping)")
        return None, None
    ckpt = torch.load(CORRNET_PATH, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    net = CorrNet(n_vertices=cfg["n_vertices"],
                  hidden=cfg["hidden"], dropout=cfg["dropout"])
    net.load_state_dict(ckpt["model_state"])
    net.eval()
    print(f"[OK] Loaded CorrNet: {CORRNET_PATH.name}  "
          f"(val_cd={ckpt.get('best_val_cd', '?'):.3f} mm)")
    return net, ckpt


def load_input_calib():
    if not CALIB_PATH.exists():
        print(f"[--] Calibration not found: {CALIB_PATH.name}  (skipping)")
        return None
    d = dict(np.load(str(CALIB_PATH)))
    calib = {"A": d["A"].astype(np.float32), "b": d["b"].astype(np.float32)}
    print(f"[OK] Loaded calibration: {CALIB_PATH.name}")
    return calib


# ─────────────────────────────────────────────────────────────────
# Export functions
# ─────────────────────────────────────────────────────────────────
def export_onnx(model, path, input_name, output_name):
    dummy = torch.randn(1, 2)
    torch.onnx.export(
        model, dummy, str(path),
        input_names=[input_name],
        output_names=[output_name],
        dynamic_axes={input_name: {0: "batch"}, output_name: {0: "batch"}},
        opset_version=17,
    )
    print(f"     -> {path.name}  ({path.stat().st_size / 1024:.0f} KB)")


def export_stats(ckpt, input_calib, corrnet_ckpt, out_dir):
    stats = {
        "input_mean":  to_list(ckpt["input_stats"]["mean"]),
        "input_std":   to_list(ckpt["input_stats"]["std"]),
        "output_mean": to_list(ckpt["output_stats"]["mean"]),
        "output_std":  to_list(ckpt["output_stats"]["std"]),
        "n_vertices":  int(np.asarray(ckpt["output_stats"]["mean"]).size // 3),
    }
    if input_calib is not None:
        stats["calib_A"] = to_list(input_calib["A"])
        stats["calib_b"] = to_list(input_calib["b"])
    if corrnet_ckpt is not None:
        stats["corrnet_input_mean"] = to_list(corrnet_ckpt["sim_in_mean"])
        stats["corrnet_input_std"]  = to_list(corrnet_ckpt["sim_in_std"])

    path = out_dir / "model_stats.json"
    with open(path, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"     -> {path.name}")


# ─────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────
def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Assets dir : {ASSETS_DIR}")
    print(f"Output dir : {OUT_DIR}\n")

    # 1. Main model
    model, ckpt = load_main_model()
    export_onnx(model, OUT_DIR / "main_model.onnx", "servo_input", "vertex_output")

    # 2. Input calibration
    input_calib = load_input_calib()

    # 3. Correction net
    corrnet, corrnet_ckpt = load_correction_net()
    if corrnet is not None:
        export_onnx(corrnet, OUT_DIR / "correction_net.onnx", "sim_input", "delta_output")

    # 4. Stats JSON
    export_stats(ckpt, input_calib, corrnet_ckpt, OUT_DIR)

    print(f"\nDone! Demo assets written to: {OUT_DIR}/")
    print("Files:")
    for f in sorted(OUT_DIR.iterdir()):
        print(f"  {f.name}  ({f.stat().st_size / 1024:.1f} KB)")


if __name__ == "__main__":
    main()
