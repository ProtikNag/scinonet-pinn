"""Evaluate a layer-wise run from its checkpoint.pt WITHOUT resuming training.

Used when a run is killed at the SLURM wall limit before it could write
metrics.json / model.pt / figures. Rebuilds the exact dataset from config.json,
loads the best-by-validation weights from checkpoint.pt, then runs the same
evaluate() + figure pipeline as run_exp.py.

    python experiments/exp2_layerwise/scripts/eval_from_ckpt.py \
        --out experiments/exp2_layerwise/outputs/sp20_t10_silu_F256_h256x256x256_a1_hpc
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import scinonet_pinn as P  # noqa: E402
from run_exp import evaluate  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="run output dir with config.json + checkpoint.pt")
    ap.add_argument("--which", choices=["best", "current"], default="best",
                    help="best=best-by-validation state (default), current=last epoch")
    ap.add_argument("--n-plot", type=int, default=5)
    args = ap.parse_args()

    out = args.out
    cfg = json.load(open(os.path.join(out, "config.json")))
    ckpt = torch.load(os.path.join(out, "checkpoint.pt"), map_location="cpu")
    ep = ckpt.get("epoch"); best_val = ckpt.get("best")
    print(f"[eval] checkpoint epoch={ep} best_val={best_val} which={args.which}", flush=True)

    # push the exact training configuration onto the core module
    P.set_dtype(cfg["dtype"])
    P.set_seed(cfg["seed"])
    P.ACTIVATION = cfg["activation"]
    P.NUM_FREQ = cfg["num_freq"]
    P.HIDDEN_SIZES = list(cfg["hidden"])
    P.BALANCE_ALPHA = cfg["balance_alpha"]
    P.SUBSAMPLE_KEEP = cfg["temporal_keep"]
    P.N_HELD_SPATIAL = cfg["n_held_spatial"]
    P.BATCH_SIZE = cfg["batch_size"]
    P.DATA_ONLY_EPOCHS = cfg["data_only_epochs"]
    P.BDRY_ENABLE = bool(cfg["bdry"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed = cfg["seed"]

    df = P.load_full_signal(cfg["csv"])
    data = P.build_dataset(df, P.SUBSAMPLE_KEEP, P.N_HELD_SPATIAL, seed)
    sc = data["scalers"]
    print(f"[eval] train_rows={len(data['Xtr']):,} held={len(data['held_spatial_idx'])} "
          f"xy_points={len(data['xy_points'])}", flush=True)

    model = P.make_net(sc, seed)
    state = ckpt["best_state"] if args.which == "best" else ckpt["model"]
    model.load_state_dict(state)
    model.to(device).eval()

    hist = ckpt.get("hist")

    # cheap figures first
    pct = cfg.get("pct_spatial", "?")
    P.plot_plate(data, save_stem=os.path.join(out, "plate_layout"),
                 title=f"Sampled points on the plate ({pct}% spatial, 3 plies)")
    if hist:
        P.plot_loss_curves(hist, P.DATA_ONLY_EPOCHS, save_stem=os.path.join(out, "loss"))

    # evaluation + reconstruction / held-out figures
    eval_metrics = evaluate(model, data, device, seed)
    out_metrics = {"epochs_run": ep, "best_val": best_val,
                   "stopped_reason": "wall_limit_eval_from_ckpt", "eval": eval_metrics}
    json.dump(out_metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    torch.save({"state_dict": model.state_dict(), "history": hist,
                "metrics": out_metrics, "config": cfg, "scalers": vars(sc)},
               os.path.join(out, "model.pt"))

    seen_xy = P.select_seen_points(data, k=args.n_plot, seed=seed)
    recs_seen = [P.reconstruct_xy(model, data, i, P.SURFACE_Z, device) for i in seen_xy]
    P.plot_reconstruction(recs_seen, save_stem=os.path.join(out, "reconstruction"))
    held_xy = P.select_heldout_points(data, k=args.n_plot, seed=seed)
    recs_held = [P.reconstruct_xy(model, data, i, P.SURFACE_Z, device) for i in held_xy]
    P.plot_heldout_prediction(recs_held, save_stem=os.path.join(out, "heldout_prediction"))

    us = eval_metrics["unseen_spatial"]; se = eval_metrics["seen_temporal"]
    print(f"[eval] DONE epoch={ep} | unseen(spatial) median={us['median']} "
          f"| seen(temporal) median={se['median']}", flush=True)


if __name__ == "__main__":
    main()
