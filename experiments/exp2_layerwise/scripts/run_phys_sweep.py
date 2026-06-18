"""Physics-loss weight sweep (spec: make the model physics-aware).

The physics weight is the gradient-balancing factor ``BALANCE_ALPHA`` (physics
gradient as a fraction of the data gradient). The spec asks to sweep it and pick
the *highest* value that keeps the model usable. This trains a short run per alpha
on a given dataset, records the held-out data MSE and the wave/gauge/IC/BC
residuals, and recommends the largest alpha whose val data MSE stays within
``--tol`` x the best (so physics is maximized without collapsing the data fit).

Writes ``outputs/phys_sweep_<dataset>/sweep.{json,csv}`` and a summary plot.

    python experiments/exp2_layerwise/scripts/run_phys_sweep.py \
        --csv experiments/exp2_layerwise/data/dataset_layerwise_1pct_3ply_fullsignal_6001steps.csv \
        --alphas 0.3 0.5 1.0 2.0 4.0 --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import scinonet_pinn as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 0.5, 1.0, 2.0, 4.0])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--temporal", type=float, default=0.10)
    ap.add_argument("--n-held-spatial", type=int, default=10)
    ap.add_argument("--activation", default="tanh")
    ap.add_argument("--num-freq", type=int, default=160)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256, 256, 256])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data-only-epochs", type=int, default=8)
    ap.add_argument("--tol", type=float, default=2.0,
                    help="accept alpha if val_data <= tol x best val_data")
    args = ap.parse_args()

    base = os.path.basename(args.csv).replace(".csv", "")
    out = os.path.join(ROOT, "experiments/exp2_layerwise/outputs", f"phys_sweep_{base}")
    os.makedirs(out, exist_ok=True)

    P.ACTIVATION = args.activation
    P.NUM_FREQ = args.num_freq
    P.HIDDEN_SIZES = list(args.hidden)
    P.SUBSAMPLE_KEEP = args.temporal
    P.N_HELD_SPATIAL = args.n_held_spatial
    P.EPOCHS = args.epochs
    P.EARLY_STOP_PATIENCE = args.epochs        # no early stop during the scan
    P.DATA_ONLY_EPOCHS = args.data_only_epochs
    P.LOG_EVERY = max(5, args.epochs // 4)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = P.load_full_signal(args.csv)
    data = P.build_dataset(df, P.SUBSAMPLE_KEEP, P.N_HELD_SPATIAL, args.seed)
    print(f"[sweep] {len(args.alphas)} alphas x {args.epochs} epochs | device={device} "
          f"train_rows={len(data['Xtr']):,}", flush=True)

    rows = []
    for a in args.alphas:
        P.BALANCE_ALPHA = a
        P.set_seed(args.seed)
        net = P.make_net(data["scalers"], args.seed)
        hist, metrics = P.train(net, data, device, verbose=True)
        row = {"alpha": a, "val_data_mse": metrics["val_data_mse"],
               "final_wave": hist["val_wave"][-1], "final_gauge": hist["val_gauge"][-1],
               "final_ic": hist["val_ic"][-1], "final_bdry": hist["val_bdry"][-1],
               "phys_w": hist["phys_w"][-1]}
        rows.append(row)
        print(f"  alpha={a:>4} -> val_data={row['val_data_mse']:.3e} "
              f"wave={row['final_wave']:.2e} gauge={row['final_gauge']:.2e} "
              f"ic={row['final_ic']:.2e} bdry={row['final_bdry']:.2e}", flush=True)

    res = pd.DataFrame(rows)
    best_val = res["val_data_mse"].min()
    ok = res[res["val_data_mse"] <= args.tol * best_val]
    recommended = float(ok["alpha"].max()) if len(ok) else float(res.loc[res["val_data_mse"].idxmin(), "alpha"])

    res.to_csv(os.path.join(out, "sweep.csv"), index=False)
    json.dump({"rows": rows, "best_val_data": float(best_val),
               "tol": args.tol, "recommended_alpha": recommended},
              open(os.path.join(out, "sweep.json"), "w"), indent=2)

    # summary plot in the current style
    import matplotlib.pyplot as plt
    P.apply_style()
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.plot(res["alpha"], res["val_data_mse"], color=P.AC["blue"], lw=3.0, marker="o", ms=10, label="val data MSE")
    ax.plot(res["alpha"], res["final_wave"], color=P.AC["amber"], lw=3.0, marker="s", ms=10, label="wave residual")
    ax.axvline(recommended, color=P.AC["red"], lw=2.0, ls=":", label=f"chosen alpha={recommended:g}")
    ax.set_yscale("log"); ax.set_xlabel("Physics weight (balance alpha)", fontsize=28)
    ax.set_ylabel("MSE (standardized)", fontsize=28); ax.set_title("Physics-weight sweep", fontsize=32)
    ax.tick_params(axis="both", labelsize=22); ax.legend(frameon=False, fontsize=20)
    fig.tight_layout(); P.save_fig(fig, os.path.join(out, "phys_sweep")); plt.close(fig)

    print(f"\n[sweep] recommended (highest workable) alpha = {recommended:g}")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()
