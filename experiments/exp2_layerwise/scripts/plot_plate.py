"""Generate the plate-layout figure for an existing run (or any dataset).

Rebuilds the exact train / boundary / held-out split from a run's config.json
(same csv, seed, temporal keep, n_held_spatial) and writes plate_layout.{png,svg}
into that run folder, in the current visualization style.

    # for a finished run
    python experiments/exp2_layerwise/scripts/plot_plate.py \
        --run experiments/exp2_layerwise/outputs/sp1_t10_silu_F256_h256x256x256_a3_best
    # or directly for a dataset
    python experiments/exp2_layerwise/scripts/plot_plate.py --csv <dataset.csv>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(__file__)
sys.path.insert(0, HERE)
import scinonet_pinn as P  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=None, help="a run output folder containing config.json")
    ap.add_argument("--csv", default=None, help="dataset CSV (if not using --run)")
    ap.add_argument("--temporal", type=float, default=0.10)
    ap.add_argument("--n-held-spatial", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.run:
        cfg = json.load(open(os.path.join(args.run, "config.json")))
        csv = cfg["csv"]; temporal = cfg["temporal_keep"]
        n_held = cfg["n_held_spatial"]; seed = cfg["seed"]; pct = cfg.get("pct_spatial")
        out_stem = args.out or os.path.join(args.run, "plate_layout")
    else:
        csv = args.csv; temporal = args.temporal
        n_held = args.n_held_spatial; seed = args.seed
        meta = csv.replace(".csv", "_meta.json")
        pct = json.load(open(meta)).get("pct") if os.path.exists(meta) else "?"
        out_stem = args.out or csv.replace(".csv", "_plate")

    P.set_seed(seed)
    df = P.load_full_signal(csv)
    data = P.build_dataset(df, temporal, n_held, seed)
    P.plot_plate(data, save_stem=out_stem,
                 title=f"Sampled points on the plate ({pct}% spatial, 3 plies)")
    print(f"[plate] {len(data['xy_points'])} unique (x,y) | "
          f"boundary={int(data['is_boundary_point'].sum())} | "
          f"held-out={len(data['held_spatial_idx'])} -> {out_stem}.png/.svg")


if __name__ == "__main__":
    main()
