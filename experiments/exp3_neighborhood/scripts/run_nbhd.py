"""Experiment 2 runner: neighborhood-confined sampling.

Trains one PINN on the neighborhood dataset (see ``gen_neighborhood_dataset.py``)
with the same working configuration as Experiment 1, then evaluates and visualizes
generalization to the two held-out categories the spec asks for:

    * unseen points INSIDE a neighborhood   (held members of the dense local regions)
    * unseen points OUTSIDE every neighborhood (spatial extrapolation)

Writes, under ``outputs/<run_tag>/`` (identical visualization style to Experiment 1):

    plate_layout.{png,svg}           train / inside-unseen / outside-unseen on the plate
    loss_<comp>.{png,svg}            loss-parameter curves
    reconstruction.{png,svg}         seen spatial points (10% temporal seen)
    unseen_inside.{png,svg}          unseen points inside the neighborhoods
    unseen_outside.{png,svg}         unseen points outside the neighborhoods
    model.pt / metrics.json / config.json / progress.csv

    python experiments/exp3_neighborhood/scripts/run_nbhd.py \
        --csv experiments/exp3_neighborhood/data/dataset_nbhd_1pct_n6_3ply_fullsignal_6001steps.csv \
        --activation silu --num-freq 256 --hidden 256 256 256 --balance-alpha 1.0
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
# reuse the shared core module that lives with Experiment 1
sys.path.insert(0, os.path.join(ROOT, "experiments/exp2_layerwise/scripts"))

import scinonet_pinn as P  # noqa: E402


def _run_tag(args, pct, n_nbhd):
    h = "x".join(str(w) for w in args.hidden)
    sp = f"nbhd{pct:g}".replace(".", "p")
    return (f"{sp}_n{n_nbhd}_t{int(args.temporal*100)}_{args.activation}"
            f"_F{args.num_freq}_h{h}_a{args.balance_alpha:g}{args.tag_suffix}")


def _summ(a):
    a = np.array(a, float)
    return {"median": float(np.median(a)) if a.size else None,
            "mean": float(np.mean(a)) if a.size else None,
            "n": int(a.size), "per_point": a.tolist()}


def _full_signal_err(model, data, idx_list, device):
    errs = []
    for i in idx_list:
        rec = P.reconstruct_xy(model, data, i, P.SURFACE_Z, device)
        valid = ~np.isnan(rec["gt"]).any(axis=1)
        if valid.sum() == 0:
            continue
        errs.append(P.relative_l2(rec["pred"][valid], rec["gt"][valid]))
    return errs


def evaluate(model, data, device, seed, n_seen_eval=40):
    """relative-L2 on unseen-inside, unseen-outside (full signal) and seen (held timesteps)."""
    inside_err = _full_signal_err(model, data, list(data["inside_held_idx"]), device)
    outside_err = _full_signal_err(model, data, list(data["outside_held_idx"]), device)

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
    return {"unseen_inside": _summ(inside_err), "unseen_outside": _summ(outside_err),
            "seen_temporal": _summ(seen_err)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--temporal", type=float, default=0.10, help="temporal keep fraction")
    ap.add_argument("--activation", default="silu", choices=list(P._ACTIVATIONS.keys()))
    ap.add_argument("--num-freq", type=int, default=256)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--balance-alpha", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=2000, help="hard epoch cap")
    ap.add_argument("--patience", type=int, default=50,
                    help="early-stop patience on the validation data loss")
    ap.add_argument("--data-only-epochs", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--bdry", type=int, default=1, help="1=Dirichlet BC on, 0=off")
    ap.add_argument("--dtype", choices=["float64", "float32"], default="float64")
    ap.add_argument("--batch-size", type=int, default=16384)
    ap.add_argument("--ckpt-every", type=int, default=5)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--n-plot", type=int, default=5)
    args = ap.parse_args()

    meta_path = args.csv.replace(".csv", "_meta.json")
    meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
    pct = meta.get("pct_train", 0)
    n_nbhd = meta.get("n_nbhd", 0)
    tag = _run_tag(args, pct, n_nbhd)
    out = os.path.join(ROOT, "experiments/exp3_neighborhood/outputs", tag)
    os.makedirs(out, exist_ok=True)

    # push configuration onto the shared core module
    P.set_dtype(args.dtype)
    P.set_seed(args.seed)
    P.ACTIVATION = args.activation
    P.NUM_FREQ = args.num_freq
    P.HIDDEN_SIZES = list(args.hidden)
    P.BALANCE_ALPHA = args.balance_alpha
    P.SUBSAMPLE_KEEP = args.temporal
    P.BATCH_SIZE = args.batch_size
    P.EPOCHS = args.epochs
    P.EARLY_STOP_PATIENCE = args.patience
    P.EARLY_STOP_METRIC = "val"      # stop on validation data loss (see Experiment 1)
    P.DATA_ONLY_EPOCHS = args.data_only_epochs
    P.BDRY_ENABLE = bool(args.bdry)
    P.DROP_DATA_AFTER_WARMUP = False
    P.LOG_EVERY = 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt_path = os.path.join(out, "checkpoint.pt")
    progress_path = os.path.join(out, "progress.csv")
    print(f"[run] tag={tag} device={device} dtype={args.dtype} resume={args.resume} "
          f"csv={os.path.basename(args.csv)}", flush=True)

    df = P.load_full_signal(args.csv)
    data = P.build_dataset_nbhd(df, P.SUBSAMPLE_KEEP, args.seed)
    sc = data["scalers"]
    print(f"[run] beta_y={sc.beta_y:.3f} beta_z={sc.beta_z:.3f} gamma={sc.gamma:.3f} "
          f"train_rows={len(data['Xtr']):,} test_rows={len(data['Xte']):,} "
          f"xy_points={len(data['xy_points'])} inside_held={len(data['inside_held_idx'])} "
          f"outside_held={len(data['outside_held_idx'])}", flush=True)

    config = {"csv": args.csv, "experiment": "neighborhood", "pct_train": pct,
              "n_nbhd": n_nbhd, "temporal_keep": args.temporal,
              "activation": args.activation, "num_freq": args.num_freq,
              "hidden": list(args.hidden), "balance_alpha": args.balance_alpha,
              "epochs_cap": args.epochs, "patience": args.patience,
              "data_only_epochs": args.data_only_epochs, "bdry": bool(args.bdry),
              "dtype": args.dtype, "batch_size": args.batch_size, "seed": args.seed,
              "beta_y": sc.beta_y, "beta_z": sc.beta_z, "gamma": sc.gamma,
              "train_rows": int(len(data["Xtr"])),
              "n_inside_held": len(data["inside_held_idx"]),
              "n_outside_held": len(data["outside_held_idx"])}
    json.dump(config, open(os.path.join(out, "config.json"), "w"), indent=2)

    model = P.make_net(sc, args.seed)
    history, metrics = P.train(model, data, device, ckpt_path=ckpt_path,
                               ckpt_every=args.ckpt_every, resume=args.resume,
                               progress_path=progress_path)

    torch.save({"state_dict": model.state_dict(), "history": history,
                "metrics": metrics, "config": config, "scalers": vars(sc)},
               os.path.join(out, "model.pt"))

    # cheap figures first (plate + loss curves)
    P.plot_plate_nbhd(data, save_stem=os.path.join(out, "plate_layout"),
                      centers=meta.get("centers_xy"),
                      title=f"Neighborhood sampling ({pct}% train, {n_nbhd} neighborhoods)")
    P.plot_loss_curves(history, P.DATA_ONLY_EPOCHS, save_stem=os.path.join(out, "loss"))

    # evaluation + reconstruction / held-out figures (slower)
    eval_metrics = evaluate(model, data, device, args.seed)
    out_metrics = {**metrics, "eval": eval_metrics}
    json.dump(out_metrics, open(os.path.join(out, "metrics.json"), "w"), indent=2)

    # results table: interpolation (unseen inside) vs extrapolation (unseen outside)
    P.plot_results_table([{
        "name": f"PINN ({args.activation}, F{args.num_freq}, "
                f"{'x'.join(str(h) for h in args.hidden)}, a={args.balance_alpha:g})",
        "e_r": "wave (phi,psi) + gauge + IC",
        "e_b": "Dirichlet u=v=w=0" if args.bdry else "None",
        "recon": eval_metrics["seen_temporal"],
        "interp": eval_metrics["unseen_inside"],
        "extrap": eval_metrics["unseen_outside"],
    }], save_stem=os.path.join(out, "results_table"))
    torch.save({"state_dict": model.state_dict(), "history": history,
                "metrics": out_metrics, "config": config,
                "scalers": vars(sc)}, os.path.join(out, "model.pt"))

    # seen spatial points (10% temporal seen)
    seen_xy = P.select_seen_points(data, k=args.n_plot, seed=args.seed)
    recs_seen = [P.reconstruct_xy(model, data, i, P.SURFACE_Z, device) for i in seen_xy]
    P.plot_reconstruction(recs_seen, save_stem=os.path.join(out, "reconstruction"))

    # unseen INSIDE the neighborhoods
    inside_xy = P._spread_pick(list(data["inside_held_idx"]), args.n_plot,
                               data["nbhd_of_point"], args.seed)
    recs_in = [P.reconstruct_xy(model, data, i, P.SURFACE_Z, device) for i in inside_xy]
    P.plot_heldout_prediction(recs_in, save_stem=os.path.join(out, "unseen_inside"),
                              title="Unseen point INSIDE neighborhood (u/v/w)")

    # unseen OUTSIDE every neighborhood
    outside_xy = P._spread_pick(list(data["outside_held_idx"]), args.n_plot,
                                data["nbhd_of_point"], args.seed)
    recs_out = [P.reconstruct_xy(model, data, i, P.SURFACE_Z, device) for i in outside_xy]
    P.plot_heldout_prediction(recs_out, save_stem=os.path.join(out, "unseen_outside"),
                              title="Unseen point OUTSIDE neighborhood (u/v/w)")

    ui = eval_metrics["unseen_inside"]; uo = eval_metrics["unseen_outside"]
    se = eval_metrics["seen_temporal"]
    print(f"[run] DONE {tag} | epochs_run={metrics.get('epochs_run')} "
          f"| unseen_inside median={ui['median']} | unseen_outside median={uo['median']} "
          f"| seen(temporal) median={se['median']}", flush=True)


if __name__ == "__main__":
    main()
