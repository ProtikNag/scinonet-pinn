"""Regenerate the plate layout + results table for a finished neighborhood run.

Rebuilds only what the new figures need (unique (x,y) + role/nbhd from the CSV,
and the saved metrics.json) — no model load, no retraining.

    python experiments/exp3_neighborhood/scripts/regen_figs.py \
        --csv  experiments/exp3_neighborhood/data/dataset_nbhd_1pct_n6_3ply_fullsignal_6001steps.csv \
        --out  experiments/exp3_neighborhood/outputs/nbhd1_n6_t10_silu_F256_h256x256x256_a1
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "experiments/exp2_layerwise/scripts"))
import scinonet_pinn as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    meta_path = args.csv.replace(".csv", "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}

    # minimal geometry: unique (x,y) with their neighborhood id and role
    print("[regen] reading unique (x,y) / nbhd / role ...", flush=True)
    df = pd.read_csv(args.csv, usecols=["x", "y", "nbhd", "role"])
    xy_df = df.drop_duplicates(subset=["x", "y"]).sort_values(["x", "y"]).reset_index(drop=True)
    xy_points = xy_df[["x", "y"]].to_numpy()
    nbhd_of_point = xy_df["nbhd"].to_numpy()
    roles = xy_df["role"].to_numpy().astype(str)
    data = {"xy_points": xy_points, "nbhd_of_point": nbhd_of_point,
            "inside_held_idx": np.where(roles == "inside_held")[0].tolist(),
            "outside_held_idx": np.where(roles == "outside_held")[0].tolist()}
    pct = meta.get("pct_train", "?"); n_nbhd = meta.get("n_nbhd", "?")
    print(f"[regen] xy={len(xy_points)} train={(roles=='train').sum()} "
          f"inside_held={len(data['inside_held_idx'])} outside_held={len(data['outside_held_idx'])}",
          flush=True)

    P.plot_plate_nbhd(data, save_stem=os.path.join(args.out, "plate_layout"),
                      centers=meta.get("centers_xy"),
                      title=f"Neighborhood sampling ({pct}% train, {n_nbhd} neighborhoods)")

    # results table from the saved metrics
    m = json.load(open(os.path.join(args.out, "metrics.json")))["eval"]
    cfg = json.load(open(os.path.join(args.out, "config.json")))
    name = (f"PINN ({cfg.get('activation')}, F{cfg.get('num_freq')}, "
            f"{'x'.join(str(h) for h in cfg.get('hidden', []))}, a={cfg.get('balance_alpha')})")
    P.plot_results_table([{
        "name": name,
        "e_r": "wave (phi,psi) + gauge + IC",
        "e_b": "Dirichlet u=v=w=0" if cfg.get("bdry") else "None",
        "recon": m["seen_temporal"], "interp": m["unseen_inside"],
        "extrap": m["unseen_outside"],
    }], save_stem=os.path.join(args.out, "results_table"))
    print("[regen] done", flush=True)


if __name__ == "__main__":
    main()
