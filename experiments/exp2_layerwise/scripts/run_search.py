"""Config search on a small dataset: determine activation, physics weight, and
architecture before committing to the 1% run.

Two staged phases on a small generated dataset (the split is built once and reused
so configs are compared fairly):

  Phase 1  activation x physics-weight(alpha) grid  (fixed F, hidden)
  Phase 2  with the best (activation, alpha): sweep num_freq and hidden

Each trial does a short training run and a light evaluation (relative L2 on the
unseen spatial holdout and on a small set of seen points' held-out timesteps),
plus the final physics residuals. Results are ranked and the recommended full
config is printed. Writes outputs/search_<dataset>/search.{csv,json} and a plot.

    python experiments/exp2_layerwise/scripts/run_search.py \
        --csv experiments/exp2_layerwise/data/dataset_layerwise_0p2pct_3ply_fullsignal_6001steps.csv \
        --epochs 40
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd
import torch

HERE = os.path.dirname(__file__)
ROOT = os.path.abspath(os.path.join(HERE, "..", "..", ".."))
sys.path.insert(0, HERE)

import scinonet_pinn as P  # noqa: E402


def light_eval(model, data, device, n_seen=15, seed=42):
    """relative L2: unseen spatial (all held) and a small seen-point subset."""
    unseen = []
    for i in data["held_spatial_idx"]:
        rec = P.reconstruct_xy(model, data, i, P.SURFACE_Z, device)
        valid = ~np.isnan(rec["gt"]).any(axis=1)
        unseen.append(P.relative_l2(rec["pred"][valid], rec["gt"][valid]))
    seen_idx = P.select_seen_points(data, k=n_seen, seed=seed)
    seen = []
    for i in seen_idx:
        rec = P.reconstruct_xy(model, data, i, P.SURFACE_Z, device)
        ti = rec["test_idx"]
        if len(ti) == 0:
            continue
        valid = ~np.isnan(rec["gt"][ti]).any(axis=1)
        if valid.sum():
            seen.append(P.relative_l2(rec["pred"][ti][valid], rec["gt"][ti][valid]))
    med = lambda a: float(np.median(a)) if len(a) else float("nan")
    return med(unseen), med(seen)


def trial(data, device, activation, alpha, num_freq, hidden, epochs, data_only, seed):
    P.ACTIVATION = activation
    P.BALANCE_ALPHA = alpha
    P.NUM_FREQ = num_freq
    P.HIDDEN_SIZES = list(hidden)
    P.EPOCHS = epochs
    P.EARLY_STOP_PATIENCE = epochs
    P.DATA_ONLY_EPOCHS = data_only
    P.LOG_EVERY = max(10, epochs)
    P.set_seed(seed)
    t0 = time.time()
    try:
        net = P.make_net(data["scalers"], seed)
        hist, metrics = P.train(net, data, device, verbose=False)
        unseen_med, seen_med = light_eval(net, data, device, seed=seed)
        row = {"activation": activation, "alpha": alpha, "num_freq": num_freq,
               "hidden": "x".join(map(str, hidden)),
               "val_data_mse": metrics["val_data_mse"], "epochs_run": metrics["epochs_run"],
               "unseen_median": unseen_med, "seen_median": seen_med,
               "wave": hist["val_wave"][-1], "gauge": hist["val_gauge"][-1],
               "ic": hist["val_ic"][-1], "bdry": hist["val_bdry"][-1],
               "phys_w": hist["phys_w"][-1], "minutes": (time.time() - t0) / 60.0}
    except Exception as e:                                   # divergence (e.g. raw sin)
        row = {"activation": activation, "alpha": alpha, "num_freq": num_freq,
               "hidden": "x".join(map(str, hidden)), "val_data_mse": float("nan"),
               "epochs_run": 0, "unseen_median": float("nan"), "seen_median": float("nan"),
               "wave": float("nan"), "gauge": float("nan"), "ic": float("nan"),
               "bdry": float("nan"), "phys_w": float("nan"),
               "minutes": (time.time() - t0) / 60.0, "error": str(e)[:120]}
    print(f"  [{activation:>4} a={alpha:<3} F={num_freq:<3} h={row['hidden']:>11}] "
          f"val={row['val_data_mse']:.3e} unseen={row['unseen_median']:.3f} "
          f"seen={row['seen_median']:.3f} ({row['minutes']:.1f}m)", flush=True)
    return row


def rank_key(r):
    """Lower is better; push NaNs to the bottom. Primary unseen, tiebreak seen."""
    u = r["unseen_median"]; s = r["seen_median"]
    u = u if np.isfinite(u) else 1e9
    s = s if np.isfinite(s) else 1e9
    return (u, s)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--activations", nargs="+", default=["tanh", "sin", "gelu", "silu"])
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.3, 1.0, 3.0])
    ap.add_argument("--num-freqs", type=int, nargs="+", default=[96, 256])
    ap.add_argument("--hiddens", nargs="+", default=["128x128x128", "256x256x256x256"])
    ap.add_argument("--base-freq", type=int, default=160)
    ap.add_argument("--base-hidden", default="256x256x256")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--data-only-epochs", type=int, default=6)
    ap.add_argument("--temporal", type=float, default=0.10)
    ap.add_argument("--n-held-spatial", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    def parse_h(s): return [int(w) for w in s.split("x")]
    base_hidden = parse_h(args.base_hidden)

    base = os.path.basename(args.csv).replace(".csv", "")
    out = os.path.join(ROOT, "experiments/exp2_layerwise/outputs", f"search_{base}")
    os.makedirs(out, exist_ok=True)

    P.SUBSAMPLE_KEEP = args.temporal
    P.N_HELD_SPATIAL = args.n_held_spatial
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # build the split ONCE; reused by every config for a fair comparison
    df = P.load_full_signal(args.csv)
    data = P.build_dataset(df, args.temporal, args.n_held_spatial, args.seed)
    print(f"[search] device={device} train_rows={len(data['Xtr']):,} "
          f"xy_points={len(data['xy_points'])} held={len(data['held_spatial_idx'])} "
          f"beta_z={data['scalers'].beta_z:.1f}", flush=True)

    rows = []
    # ── Phase 1: activation x alpha ──
    print(f"\n[phase 1] activation x alpha "
          f"({len(args.activations)}x{len(args.alphas)} trials)", flush=True)
    for act in args.activations:
        for a in args.alphas:
            r = trial(data, device, act, a, args.base_freq, base_hidden,
                      args.epochs, args.data_only_epochs, args.seed)
            r["phase"] = 1
            rows.append(r)

    p1 = sorted([r for r in rows if r["phase"] == 1], key=rank_key)
    best_act = p1[0]["activation"]; best_alpha = p1[0]["alpha"]
    print(f"\n[phase 1 best] activation={best_act} alpha={best_alpha} "
          f"unseen={p1[0]['unseen_median']:.3f} seen={p1[0]['seen_median']:.3f}", flush=True)

    # ── Phase 2: num_freq and hidden around the best (act, alpha) ──
    print(f"\n[phase 2] architecture sweep at act={best_act} alpha={best_alpha}", flush=True)
    for F in args.num_freqs:
        r = trial(data, device, best_act, best_alpha, F, base_hidden,
                  args.epochs, args.data_only_epochs, args.seed)
        r["phase"] = 2; rows.append(r)
    for hs in args.hiddens:
        r = trial(data, device, best_act, best_alpha, args.base_freq, parse_h(hs),
                  args.epochs, args.data_only_epochs, args.seed)
        r["phase"] = 2; rows.append(r)

    # ── rank everything, recommend ──
    res = pd.DataFrame(rows)
    res_sorted = res.reindex(sorted(res.index, key=lambda i: rank_key(res.loc[i])))
    best = res_sorted.iloc[0]
    res_sorted.to_csv(os.path.join(out, "search.csv"), index=False)
    recommendation = {"activation": best["activation"], "alpha": float(best["alpha"]),
                      "num_freq": int(best["num_freq"]), "hidden": best["hidden"],
                      "unseen_median": float(best["unseen_median"]),
                      "seen_median": float(best["seen_median"]),
                      "val_data_mse": float(best["val_data_mse"])}
    json.dump({"recommendation": recommendation, "rows": rows},
              open(os.path.join(out, "search.json"), "w"), indent=2)

    print("\n[search] ranked (best first):")
    print(res_sorted[["phase", "activation", "alpha", "num_freq", "hidden",
                      "val_data_mse", "unseen_median", "seen_median", "wave"]].to_string(index=False))
    print(f"\n[search] RECOMMENDED config: {json.dumps(recommendation, indent=2)}")
    print(f"[search] -> outputs/search_{base}/")


if __name__ == "__main__":
    main()
