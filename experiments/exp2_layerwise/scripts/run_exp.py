"""Experiment 1 runner: layer-wise spatial sampling x 10% temporal, with ablations.

Drives the config-driven core (`scinonet_pinn.py`). One invocation trains a single
configuration on a generated layer-wise dataset and writes, under
``outputs/<run_tag>/``:

    loss_<comp>.{png,svg}        loss-parameter curves (current viz style)
    reconstruction.{png,svg}     seen spatial points (10% temporal seen)
    heldout_prediction.{png,svg} unseen spatial points (no temporal point seen)
    model.pt                     state_dict + history + metrics + scalers + config
    metrics.json                 seen / unseen relative-L2 summaries
    config.json                  the exact resolved configuration + dataset path

The training dataset itself lives under ``data/`` and is referenced (not copied)
so prediction code can be re-run later without retraining.

    python experiments/exp2_layerwise/scripts/run_exp.py \
        --csv experiments/exp2_layerwise/data/dataset_layerwise_1pct_3ply_fullsignal_6001steps.csv \
        --activation tanh --num-freq 160 --hidden 256 256 256 \
        --balance-alpha 0.3 --epochs 150 --patience 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import scinonet_pinn as P  # noqa: E402


def _run_tag(args, pct):
    h = "x".join(str(w) for w in args.hidden)
    sp = f"sp{pct:g}".replace(".", "p")
    return f"{sp}_t{int(args.temporal*100)}_{args.activation}_F{args.num_freq}_h{h}_a{args.balance_alpha:g}{args.tag_suffix}"


def evaluate(model, data, device, seed, n_seen_eval=40):
    """relative-L2 on unseen spatial points (full signal) and seen points (held timesteps)."""
    # unseen spatial points: all held-out, full-signal reconstruction
    unseen_idx = list(data["held_spatial_idx"])
    unseen_err = []
    for i in unseen_idx:
        rec = P.reconstruct_xy(model, data, i, P.SURFACE_Z, device)
        valid = ~np.isnan(rec["gt"]).any(axis=1)
        unseen_err.append(P.relative_l2(rec["pred"][valid], rec["gt"][valid]))

    # seen spatial points: temporally held-out timesteps (subset for speed)
    seen_all = P.select_seen_points(data, k=n_seen_eval, seed=seed)
    seen_err = []
    for i in seen_all:
        rec = P.reconstruct_xy(model, data, i, P.SURFACE_Z, device)
        ti = rec["test_idx"]
        if len(ti) == 0:
            continue
        valid = ~np.isnan(rec["gt"][ti]).any(axis=1)
        if valid.sum() == 0:
            continue
        seen_err.append(P.relative_l2(rec["pred"][ti][valid], rec["gt"][ti][valid]))

    def summ(a):
        a = np.array(a, float)
        return {"median": float(np.median(a)) if a.size else None,
                "mean": float(np.mean(a)) if a.size else None,
                "n": int(a.size), "per_point": a.tolist()}
    return {"unseen_spatial": summ(unseen_err), "seen_temporal": summ(seen_err)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--temporal", type=float, default=0.10, help="temporal keep fraction")
    ap.add_argument("--n-held-spatial", type=int, default=10)
    ap.add_argument("--activation", default="tanh", choices=list(P._ACTIVATIONS.keys()))
    ap.add_argument("--num-freq", type=int, default=160)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--balance-alpha", type=float, default=0.3)
    ap.add_argument("--epochs", type=int, default=2000, help="hard epoch cap")
    ap.add_argument("--patience", type=int, default=180,
                    help="early-stop patience on training loss (spec: 180)")
    ap.add_argument("--data-only-epochs", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bdry", type=int, default=1, help="1=Dirichlet BC on, 0=off")
    ap.add_argument("--drop-data-after-warmup", action="store_true",
                    help="after data-only warmup, optimize physics ONLY (no data loss)")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64",
                    help="float32 is ~2x faster on a V100; validate the physics residual")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--ckpt-every", type=int, default=5,
                    help="save checkpoint.pt every N epochs (for resume after a time limit)")
    ap.add_argument("--resume", action="store_true",
                    help="resume from outputs/<tag>/checkpoint.pt if present")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--n-plot", type=int, default=5)
    args = ap.parse_args()

    meta_path = args.csv.replace(".csv", "_meta.json")
    pct = json.load(open(meta_path)).get("pct") if os.path.exists(meta_path) else 0
    tag = _run_tag(args, pct)
    out = os.path.join(ROOT, "experiments/exp2_layerwise/outputs", tag)
    os.makedirs(out, exist_ok=True)

    # push configuration onto the core module
    P.set_dtype(args.dtype)                 # float32 (fast on GPU) or float64
    P.set_seed(args.seed)
    P.ACTIVATION = args.activation
    P.NUM_FREQ = args.num_freq
    P.HIDDEN_SIZES = list(args.hidden)
    P.BALANCE_ALPHA = args.balance_alpha
    P.SUBSAMPLE_KEEP = args.temporal
    P.N_HELD_SPATIAL = args.n_held_spatial
    P.BATCH_SIZE = args.batch_size
    P.EPOCHS = args.epochs
    P.EARLY_STOP_PATIENCE = args.patience
    # Stop on the *validation* data loss (the held-out metric we report and the one
    # best_state is kept by), not the combined train loss. Monitoring train_total
    # is unreliable once physics turns on: it includes phys_w*phys, so it jumps at
    # the data->physics switch and the warmup data-loss minimum becomes unbeatable,
    # making the plateau counter climb for structural reasons rather than real
    # stagnation. val_data is comparable across the switch and self-corrects.
    P.EARLY_STOP_METRIC = "val"
    P.DATA_ONLY_EPOCHS = args.data_only_epochs
    P.BDRY_ENABLE = bool(args.bdry)
    P.DROP_DATA_AFTER_WARMUP = bool(args.drop_data_after_warmup)
    P.LOG_EVERY = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(out, "checkpoint.pt")
    progress_path = os.path.join(out, "progress.csv")
    print(f"[run] tag={tag} device={device} dtype={args.dtype} resume={args.resume} "
          f"csv={os.path.basename(args.csv)}", flush=True)

    df = P.load_full_signal(args.csv)
    data = P.build_dataset(df, P.SUBSAMPLE_KEEP, P.N_HELD_SPATIAL, args.seed)
    sc = data["scalers"]
    print(f"[run] beta_y={sc.beta_y:.3f} beta_z={sc.beta_z:.3f} gamma={sc.gamma:.3f} "
          f"train_rows={len(data['Xtr']):,} test_rows={len(data['Xte']):,} "
          f"xy_points={len(data['xy_points'])} held={len(data['held_spatial_idx'])}", flush=True)

    config = {"csv": args.csv, "pct_spatial": pct, "temporal_keep": args.temporal,
              "activation": args.activation, "num_freq": args.num_freq,
              "hidden": list(args.hidden), "balance_alpha": args.balance_alpha,
              "epochs_cap": args.epochs, "patience": args.patience,
              "data_only_epochs": args.data_only_epochs, "bdry": bool(args.bdry),
              "drop_data_after_warmup": bool(args.drop_data_after_warmup),
              "dtype": args.dtype, "batch_size": args.batch_size,
              "seed": args.seed, "n_held_spatial": args.n_held_spatial,
              "beta_y": sc.beta_y, "beta_z": sc.beta_z, "gamma": sc.gamma,
              "train_rows": int(len(data["Xtr"]))}
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)

    model = P.make_net(sc, args.seed)
    history, metrics = P.train(model, data, device, ckpt_path=ckpt_path,
                               ckpt_every=args.ckpt_every, resume=args.resume,
                               progress_path=progress_path)

    # save the model immediately (survives even if eval/figures are interrupted)
    torch.save({"state_dict": model.state_dict(), "history": history,
                "metrics": metrics, "config": config, "scalers": vars(sc)},
               os.path.join(out, "model.pt"))
    # cheap figures first (loss curves + plate, from history/data)
    P.plot_plate(data, save_stem=os.path.join(out, "plate_layout"),
                 title=f"Sampled points on the plate ({pct}% spatial, 3 plies)")
    P.plot_loss_curves(history, P.DATA_ONLY_EPOCHS, save_stem=os.path.join(out, "loss"))

    # evaluation + the reconstruction/held-out figures (slower)
    eval_metrics = evaluate(model, data, device, args.seed)
    out_metrics = {**metrics, "eval": eval_metrics}
    json.dump(out_metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)
    torch.save({"state_dict": model.state_dict(), "history": history,
                "metrics": out_metrics, "config": config,
                "scalers": vars(sc)}, os.path.join(out, "model.pt"))
    seen_xy = P.select_seen_points(data, k=args.n_plot, seed=args.seed)
    recs_seen = [P.reconstruct_xy(model, data, i, P.SURFACE_Z, device) for i in seen_xy]
    P.plot_reconstruction(recs_seen, save_stem=os.path.join(out, "reconstruction"))
    held_xy = P.select_heldout_points(data, k=args.n_plot, seed=args.seed)
    recs_held = [P.reconstruct_xy(model, data, i, P.SURFACE_Z, device) for i in held_xy]
    P.plot_heldout_prediction(recs_held, save_stem=os.path.join(out, "heldout_prediction"))

    us = eval_metrics["unseen_spatial"]; se = eval_metrics["seen_temporal"]
    print(f"[run] DONE {tag} | epochs_run={metrics.get('epochs_run')} "
          f"| unseen(spatial) median={us['median']} | seen(temporal) median={se['median']}", flush=True)


if __name__ == "__main__":
    main()
