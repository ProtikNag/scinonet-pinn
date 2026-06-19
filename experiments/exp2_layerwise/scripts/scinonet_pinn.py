"""SciNoNet Helmholtz-potential PINN — config-driven port of june16_ffn_signal_reconstruction.

Faithful port of the current setup (`june16_ffn_signal_reconstruction_(1).py`):
same non-dimensionalization, Fourier features, four-potential network, Helmholtz
displacement, wave + Coulomb-gauge + IC + Dirichlet-BC physics, gradient-norm
balancing, and the exact visualization style. The only changes for Experiment 1:

  * module-level config globals (no top-level execution) so a runner / sweep can
    drive width, depth, NUM_FREQ, activation, physics weight, epochs, etc.;
  * `sin` activation added (SIREN-style), alongside tanh/gelu/silu;
  * early stopping monitors the *training* loss (stop after EARLY_STOP_PATIENCE
    epochs without improvement), best checkpoint still chosen by validation;
  * boundary points (`is_boundary=1`) are never placed in the spatial holdout, so
    the training set keeps enough boundary coverage.

Set globals on this module from the runner, then call build_dataset / make_net /
train / the plot_* helpers. The visualization functions are unchanged.
"""

from __future__ import annotations

import copy
import os
import random
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib as mpl
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

# Compute dtype. float64 is the safe default (validated); float32 is ~2x faster on
# a V100 and is selected on GPU via set_dtype("float32"). Validate the physics
# residual when switching, the beta_z^2 multiplier stresses precision.
DTYPE = torch.float64
torch.set_default_dtype(DTYPE)


def set_dtype(name) -> "torch.dtype":
    global DTYPE
    DTYPE = torch.float32 if str(name) == "float32" else torch.float64
    torch.set_default_dtype(DTYPE)
    return DTYPE


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ── Config (overridable globals; defaults mirror june16) ────────────────────────
DT = 1e-8
N_T = 6001
CP_MM_PER_S = 6.30e6
CS_MM_PER_S = 3.20e6

K_MAX_SPATIAL = 0.13
SPATIAL_SCALE = 2.0
F_MAX_HZ = 300e3
NUM_FREQ = 160

HIDDEN_SIZES = [256, 256, 256]
ACTIVATION = "tanh"
CONCAT_RAW = True

N_HELD_SPATIAL = 10
SUBSAMPLE_KEEP = 0.10

EPOCHS = 80
BATCH_SIZE = 16384
N_COLLOC = 2048
N_IC = 1024
LR = 2e-3
GRAD_CLIP = 1.0
DATA_ONLY_EPOCHS = 12
BALANCE_ALPHA = 0.3
BALANCE_EMA = 0.9
W_GAUGE = 1.0
W_IC = 1.0
SCHED_FACTOR = 0.5
SCHED_PATIENCE = 15
MIN_LR = 1e-5
EARLY_STOP_PATIENCE = EPOCHS
EARLY_STOP_METRIC = "train"      # "train" -> stop on training-loss plateau (spec)
DROP_DATA_AFTER_WARMUP = False   # if True: after DATA_ONLY_EPOCHS, optimize physics ONLY
LOG_EVERY = 1

SURFACE_Z = 0.0
FIELD_COLS = ["u", "v", "w"]

# Cap the validation/test tensors built in build_dataset. Training only samples a
# 16k val subset per epoch, so materializing the full (possibly ~100M-row) test
# set wastes memory; at high spatial % this keeps RAM bounded. Held-out spatial
# points are evaluated separately from the full dataframe and are unaffected.
MAX_TEST_ROWS = 300_000

# Dirichlet boundary condition
BDRY_ENABLE = True
BDRY_EDGES = ["x_min", "x_max", "y_min", "y_max"]
BDRY_COMPONENTS = ["u", "v", "w"]
BDRY_VALUE = 0.0
W_BDRY = 1.0
N_BDRY = 1024

NBHD_COL = None

# ── Plot style (academic palette) — unchanged from june16 ───────────────────────
AC = {
    "blue": "#2563EB", "amber": "#D97706", "green": "#059669", "red": "#DC2626",
    "axis": "#212529", "grid": "#E9ECEF", "text": "#212529", "muted": "#212529",
    "ink": "#212529",
}

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
    "font.weight": "regular",
    "axes.titleweight": "regular",
    "axes.labelweight": "regular",
})


def apply_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "Helvetica", "Arial", "DejaVu Sans"],
        "axes.spines.top": True, "axes.spines.right": True,
        "axes.spines.left": True, "axes.spines.bottom": True,
        "axes.grid": False,
        "axes.edgecolor": AC["axis"], "axes.linewidth": 1.4,
        "font.size": 20, "axes.titlesize": 26, "axes.labelsize": 22,
        "axes.titleweight": "regular", "axes.labelweight": "regular",
        "axes.labelcolor": AC["text"], "axes.titlecolor": AC["text"],
        "xtick.labelsize": 18, "ytick.labelsize": 18,
        "xtick.color": AC["text"], "ytick.color": AC["text"],
        "legend.fontsize": 18, "figure.titlesize": 30,
        "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    })


def save_fig(fig, save_stem):
    if save_stem is None:
        return
    os.makedirs(os.path.dirname(save_stem) or ".", exist_ok=True)
    fig.savefig(f"{save_stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{save_stem}.svg", bbox_inches="tight", facecolor="white")
    print(f"saved: {save_stem}.png / .svg")


apply_style()

# ── Data ────────────────────────────────────────────────────────────────────────
_NBHD_CANDIDATES = ["nbhd", "neighborhood", "neighbourhood", "neighbor",
                    "neighbour", "nbr", "group", "region", "cluster", "patch",
                    "block", "zone"]


def _detect_nbhd_col(columns):
    reserved = {"x", "y", "z", "t", "u", "v", "w", "t_idx", "is_boundary"}
    lower = {c.lower(): c for c in columns}
    for cand in _NBHD_CANDIDATES:
        if cand in lower and lower[cand].lower() not in reserved:
            return lower[cand]
    for c in columns:
        cl = c.lower()
        if ("nbhd" in cl or "neigh" in cl) and cl not in reserved:
            return c
    return None


def load_full_signal(csv_path: str) -> pd.DataFrame:
    global NBHD_COL
    df = pd.read_csv(csv_path)
    NBHD_COL = _detect_nbhd_col(df.columns)
    sort_cols = ["x", "y", "z", "t"]
    df = df.sort_values(sort_cols).reset_index(drop=True)
    df["t_idx"] = np.round(df["t"].to_numpy() / DT - 1).astype(int)
    if "is_boundary" not in df.columns:
        df["is_boundary"] = 0
    if NBHD_COL is None:
        df["nbhd"] = 0
        NBHD_COL = "nbhd"
        print("neighborhood column: none found, using a single group")
    else:
        print(f"neighborhood column: '{NBHD_COL}' ({df[NBHD_COL].nunique()} groups)")
    return df


class PotentialScalers:
    """Per-axis non-dimensionalization (x_hat=(x-mu)/L_axis, t_hat=(t-mu)/tau)."""

    def __init__(self, mu_x, mu_y, Lx, mu_y_, Ly, mu_z, Lz, mu_t, tau, s_f,
                 cp_mm_per_s, cs_mm_per_s):
        self.mu_x = mu_x; self.Lx = Lx
        self.mu_y = mu_y; self.Ly = Ly
        self.mu_z = mu_z; self.Lz = Lz
        self.mu_t = mu_t; self.tau = tau
        self.s_f = s_f
        self.cp = cp_mm_per_s; self.cs = cs_mm_per_s

    @classmethod
    def fit(cls, frame, cp_mm_per_s, cs_mm_per_s):
        Lx = float(frame["x"].std() + 1e-12)
        Ly = float(frame["y"].std() + 1e-12)
        Lz = float(frame["z"].std() + 1e-12)
        if Lz < 1e-9:
            Lz = Lx
        tau = Lx / cp_mm_per_s
        uvw = np.concatenate([frame["u"].to_numpy(), frame["v"].to_numpy(),
                              frame["w"].to_numpy()])
        s_f = float(uvw.std() + 1e-30)
        return cls(mu_x=float(frame["x"].mean()), Lx=Lx,
                   mu_y=float(frame["y"].mean()), mu_y_=None, Ly=Ly,
                   mu_z=float(frame["z"].mean()), Lz=Lz,
                   mu_t=float(frame["t"].mean()), tau=tau,
                   s_f=s_f, cp_mm_per_s=cp_mm_per_s, cs_mm_per_s=cs_mm_per_s)

    @property
    def beta_y(self): return self.Lx / self.Ly
    @property
    def beta_z(self): return self.Lx / self.Lz
    @property
    def gamma(self): return self.cs / self.cp

    def encode(self, x, y, z, t):
        return np.stack([(x - self.mu_x) / self.Lx, (y - self.mu_y) / self.Ly,
                         (z - self.mu_z) / self.Lz, (t - self.mu_t) / self.tau], axis=1)

    def decode_fields(self, uvw_hat):
        return uvw_hat * self.s_f


def _choose_spatial_holdout(xy_points, nbhd_of_point, n_held, seed, exclude=None):
    """Pick n_held unique (x,y) indices to exclude, spread across groups.

    `exclude` is a set of indices that must never be held out (boundary points),
    so the training set keeps its boundary coverage.
    """
    exclude = exclude or set()
    rng = np.random.RandomState(seed)
    groups = {}
    for idx, g in enumerate(nbhd_of_point):
        if idx in exclude:
            continue
        groups.setdefault(g, []).append(idx)
    for g in groups:
        rng.shuffle(groups[g])
    order = sorted(groups.keys(), key=lambda g: str(g))
    held, gi = [], 0
    n_held = min(n_held, sum(len(v) for v in groups.values()))
    while len(held) < n_held:
        g = order[gi % len(order)]
        if groups[g]:
            held.append(groups[g].pop())
        gi += 1
        if gi > len(order) * (len(xy_points) + 1):
            break
    return sorted(held)


def build_dataset(df, subsample_keep, n_held_spatial, seed):
    df = df.copy()
    xy_df = df[["x", "y", NBHD_COL, "is_boundary"]].drop_duplicates(subset=["x", "y"])
    xy_df = xy_df.sort_values(["x", "y"]).reset_index(drop=True)
    xy_points = xy_df[["x", "y"]].to_numpy()
    nbhd_of_point = xy_df[NBHD_COL].to_numpy()
    is_boundary_point = xy_df["is_boundary"].to_numpy().astype(bool)
    z_values = np.sort(df["z"].unique())

    # stage 1: spatial holdout (exclude whole signals), never holding out boundary
    exclude = set(np.where(is_boundary_point)[0].tolist())
    held_idx = _choose_spatial_holdout(xy_points, nbhd_of_point, n_held_spatial,
                                       seed, exclude=exclude)
    held_xy = set(map(tuple, xy_points[held_idx]))
    point_key = list(zip(df["x"].to_numpy(), df["y"].to_numpy()))
    held_spatial = np.array([k in held_xy for k in point_key])
    df["held_spatial"] = held_spatial

    # stage 2: temporal subsample on the trained points only
    rng = np.random.RandomState(seed + 1)
    keep = (rng.rand(len(df)) < subsample_keep) & (~held_spatial)
    df["is_train"] = keep

    tr = df[keep]
    te = df[(~keep) & (~held_spatial)]
    # bound the validation tensor footprint for high-% datasets
    if len(te) > MAX_TEST_ROWS:
        te = te.sample(n=MAX_TEST_ROWS, random_state=seed + 2)
    sc = PotentialScalers.fit(tr, CP_MM_PER_S, CS_MM_PER_S)

    def to_xy(frame):
        X = sc.encode(frame["x"].to_numpy(), frame["y"].to_numpy(),
                      frame["z"].to_numpy(), frame["t"].to_numpy())
        Y = frame[FIELD_COLS].to_numpy() / sc.s_f
        return torch.tensor(X, dtype=DTYPE), torch.tensor(Y, dtype=DTYPE)

    Xtr, Ytr = to_xy(tr)
    Xte, Yte = to_xy(te)
    allc = sc.encode(df["x"].to_numpy(), df["y"].to_numpy(),
                     df["z"].to_numpy(), df["t"].to_numpy())
    colloc_lo = torch.tensor(allc.min(0), dtype=DTYPE)
    colloc_hi = torch.tensor(allc.max(0), dtype=DTYPE)
    t0_std = float((0.0 - sc.mu_t) / sc.tau)
    return {"Xtr": Xtr, "Ytr": Ytr, "Xte": Xte, "Yte": Yte, "scalers": sc,
            "xy_points": xy_points, "nbhd_of_point": nbhd_of_point,
            "is_boundary_point": is_boundary_point,
            "held_spatial_idx": held_idx, "z_values": z_values, "df": df,
            "colloc_lo": colloc_lo, "colloc_hi": colloc_hi, "t0_std": t0_std}


def build_dataset_nbhd(df, subsample_keep, seed, role_col="role"):
    """Experiment-2 dataset builder honoring a predefined ``role`` column.

    Unlike :func:`build_dataset` (which derives a random spatial holdout), the split
    is baked into the data by the neighborhood generator:

        role=train         -> training pool (temporally subsampled at `subsample_keep`)
        role=inside_held   -> unseen point INSIDE a neighborhood  (full-signal holdout)
        role=outside_held  -> unseen point OUTSIDE every neighborhood (full-signal holdout)

    Returns the same dict shape as ``build_dataset`` plus ``inside_held_idx`` /
    ``outside_held_idx`` (indices into ``xy_points``), so every existing plotting /
    reconstruction helper works unchanged.
    """
    df = df.copy()
    if role_col not in df.columns:
        raise ValueError(f"build_dataset_nbhd needs a '{role_col}' column")
    xy_df = (df[["x", "y", NBHD_COL, "is_boundary", role_col]]
             .drop_duplicates(subset=["x", "y"]).sort_values(["x", "y"]).reset_index(drop=True))
    xy_points = xy_df[["x", "y"]].to_numpy()
    nbhd_of_point = xy_df[NBHD_COL].to_numpy()
    is_boundary_point = xy_df["is_boundary"].to_numpy().astype(bool)
    roles = xy_df[role_col].to_numpy().astype(str)
    z_values = np.sort(df["z"].unique())

    inside_held_idx = np.where(roles == "inside_held")[0].tolist()
    outside_held_idx = np.where(roles == "outside_held")[0].tolist()
    held_idx = sorted(inside_held_idx + outside_held_idx)

    # stage 1: a point is spatially held out iff it is not a training-pool point
    is_train_pool = (roles == "train")
    train_keys = set(map(tuple, xy_points[is_train_pool]))
    point_key = list(zip(df["x"].to_numpy(), df["y"].to_numpy()))
    held_spatial = np.array([k not in train_keys for k in point_key])
    df["held_spatial"] = held_spatial

    # stage 2: temporal subsample on the training-pool points only
    rng = np.random.RandomState(seed + 1)
    keep = (rng.rand(len(df)) < subsample_keep) & (~held_spatial)
    df["is_train"] = keep

    tr = df[keep]
    te = df[(~keep) & (~held_spatial)]          # seen spatial, unseen temporal (validation)
    if len(te) > MAX_TEST_ROWS:
        te = te.sample(n=MAX_TEST_ROWS, random_state=seed + 2)
    sc = PotentialScalers.fit(tr, CP_MM_PER_S, CS_MM_PER_S)

    def to_xy(frame):
        X = sc.encode(frame["x"].to_numpy(), frame["y"].to_numpy(),
                      frame["z"].to_numpy(), frame["t"].to_numpy())
        Y = frame[FIELD_COLS].to_numpy() / sc.s_f
        return torch.tensor(X, dtype=DTYPE), torch.tensor(Y, dtype=DTYPE)

    Xtr, Ytr = to_xy(tr)
    Xte, Yte = to_xy(te)
    allc = sc.encode(df["x"].to_numpy(), df["y"].to_numpy(),
                     df["z"].to_numpy(), df["t"].to_numpy())
    colloc_lo = torch.tensor(allc.min(0), dtype=DTYPE)
    colloc_hi = torch.tensor(allc.max(0), dtype=DTYPE)
    t0_std = float((0.0 - sc.mu_t) / sc.tau)
    return {"Xtr": Xtr, "Ytr": Ytr, "Xte": Xte, "Yte": Yte, "scalers": sc,
            "xy_points": xy_points, "nbhd_of_point": nbhd_of_point,
            "is_boundary_point": is_boundary_point,
            "held_spatial_idx": held_idx, "inside_held_idx": sorted(inside_held_idx),
            "outside_held_idx": sorted(outside_held_idx),
            "z_values": z_values, "df": df,
            "colloc_lo": colloc_lo, "colloc_hi": colloc_hi, "t0_std": t0_std}


# ── Fourier features ────────────────────────────────────────────────────────────
class SpecializedFourierFeatures(nn.Module):
    def __init__(self, lo, hi, num_frequencies, seed=0):
        super().__init__()
        self.in_features = lo.numel()
        self.num_frequencies = num_frequencies
        lo = lo.to(DTYPE).unsqueeze(1)
        hi = hi.to(DTYPE).unsqueeze(1)
        gen = torch.Generator().manual_seed(seed)
        u = torch.rand(self.in_features, num_frequencies, generator=gen, dtype=DTYPE)
        self.B = nn.Parameter(lo + u * (hi - lo))
        self.A = nn.Parameter(torch.ones(num_frequencies, dtype=DTYPE))
        self.register_buffer("B_min", lo)
        self.register_buffer("B_max", hi)

    @property
    def out_features(self): return 2 * self.num_frequencies

    def clamp_B(self):
        with torch.no_grad():
            self.B.clamp_(self.B_min, self.B_max)

    def forward(self, x):
        proj = 2.0 * np.pi * (x @ self.B.to(x))
        amp = self.A.to(x)
        return torch.cat([amp * torch.sin(proj), amp * torch.cos(proj)], dim=-1)


def make_features(sc, seed):
    ks = K_MAX_SPATIAL * SPATIAL_SCALE
    lo = torch.tensor([-ks * sc.Lx, -ks * sc.Ly, -ks * sc.Lz, 0.0], dtype=DTYPE)
    hi = torch.tensor([ks * sc.Lx, ks * sc.Ly, ks * sc.Lz, F_MAX_HZ * sc.tau], dtype=DTYPE)
    return SpecializedFourierFeatures(lo, hi, NUM_FREQ, seed=seed)


# ── Model ───────────────────────────────────────────────────────────────────────
class Sin(nn.Module):
    """SIREN-style sinusoidal activation."""
    def forward(self, x):
        return torch.sin(x)


_ACTIVATIONS = {"gelu": nn.GELU, "tanh": nn.Tanh, "silu": nn.SiLU, "sin": Sin}


class PotentialNet(nn.Module):
    def __init__(self, features, hidden_sizes, gamma_init, activation="tanh", concat_raw=True):
        super().__init__()
        self.features = features
        self.concat_raw = concat_raw
        act = _ACTIVATIONS[activation]
        in_dim = features.out_features + (features.in_features if concat_raw else 0)
        layers, prev = [], in_dim
        for width in hidden_sizes:
            layers += [nn.Linear(prev, width), act()]
            prev = width
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 4)
        self.log_gamma = nn.Parameter(torch.tensor(float(np.log(gamma_init)), dtype=DTYPE))

    @property
    def gamma(self): return torch.exp(self.log_gamma)

    def forward(self, x):
        e = self.features(x)
        if self.concat_raw:
            e = torch.cat([e, x], dim=-1)
        return self.head(self.backbone(e))


def make_net(sc, seed):
    net = PotentialNet(make_features(sc, seed), HIDDEN_SIZES, gamma_init=sc.gamma,
                       activation=ACTIVATION, concat_raw=CONCAT_RAW)
    net.log_gamma.requires_grad_(False)
    return net


# ── Physics ─────────────────────────────────────────────────────────────────────
def _grad(out, x):
    return torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=True)[0]


def displacement(model, x, beta_y, beta_z):
    p = model(x)
    gphi = _grad(p[:, 0], x); gpx = _grad(p[:, 1], x)
    gpy = _grad(p[:, 2], x);  gpz = _grad(p[:, 3], x)
    u = gphi[:, 0] + beta_y * gpz[:, 1] - beta_z * gpy[:, 2]
    v = beta_y * gphi[:, 1] + beta_z * gpx[:, 2] - gpz[:, 0]
    w = beta_z * gphi[:, 2] + gpy[:, 0] - beta_y * gpx[:, 1]
    return torch.stack([u, v, w], dim=1)


def _wave_residual(q, x, coeff2, beta_y, beta_z):
    g = _grad(q, x)
    q_xx = _grad(g[:, 0], x)[:, 0]
    q_yy = _grad(g[:, 1], x)[:, 1]
    q_zz = _grad(g[:, 2], x)[:, 2]
    q_tt = _grad(g[:, 3], x)[:, 3]
    laplacian = q_xx + beta_y ** 2 * q_yy + beta_z ** 2 * q_zz
    return coeff2 * laplacian - q_tt


def physics_residuals(model, x, beta_y, beta_z):
    p = model(x)
    gamma2 = model.gamma ** 2
    one = torch.ones_like(gamma2)
    r_phi = _wave_residual(p[:, 0], x, one, beta_y, beta_z)
    r_px = _wave_residual(p[:, 1], x, gamma2, beta_y, beta_z)
    r_py = _wave_residual(p[:, 2], x, gamma2, beta_y, beta_z)
    r_pz = _wave_residual(p[:, 3], x, gamma2, beta_y, beta_z)
    gauge = (_grad(p[:, 1], x)[:, 0] + beta_y * _grad(p[:, 2], x)[:, 1]
             + beta_z * _grad(p[:, 3], x)[:, 2])
    return torch.stack([r_phi, r_px, r_py, r_pz], dim=1), gauge


def ic_residual(model, x):
    p = model(x)
    p_t = torch.stack([_grad(p[:, k], x)[:, 3] for k in range(4)], dim=1)
    return torch.cat([p, p_t], dim=1)


# ── Dirichlet boundary ──────────────────────────────────────────────────────────
_EDGE_AXIS = {"x_min": (0, "lo"), "x_max": (0, "hi"), "y_min": (1, "lo"),
              "y_max": (1, "hi"), "z_min": (2, "lo"), "z_max": (2, "hi")}
_COMP_IDX = {"u": 0, "v": 1, "w": 2}


def sample_boundary(data, n, device, dtype, edges=None):
    edges = BDRY_EDGES if edges is None else edges
    lo, hi = data["colloc_lo"], data["colloc_hi"]
    lo = lo.to(device, dtype); hi = hi.to(device, dtype)
    if not edges:
        return torch.empty(0, 4, device=device, dtype=dtype, requires_grad=True)
    per = max(1, n // len(edges))
    chunks = []
    for edge in edges:
        axis, side = _EDGE_AXIS[edge]
        u = torch.rand(per, 4, device=device, dtype=dtype)
        x = lo + u * (hi - lo)
        x[:, axis] = lo[axis] if side == "lo" else hi[axis]
        chunks.append(x)
    return torch.cat(chunks, dim=0).requires_grad_(True)


def boundary_residual(model, x, sc, components=None, value=None):
    components = BDRY_COMPONENTS if components is None else components
    value = BDRY_VALUE if value is None else value
    if x.shape[0] == 0:
        return torch.zeros(0, len(components), device=x.device, dtype=x.dtype)
    disp = displacement(model, x, sc.beta_y, sc.beta_z)
    idx = [_COMP_IDX[c] for c in components]
    return disp[:, idx] - value / sc.s_f


def sample_colloc(lo, hi, n, device, dtype):
    u = torch.rand(n, lo.numel(), device=device, dtype=dtype)
    return (lo.to(device, dtype) + u * (hi - lo).to(device, dtype)).requires_grad_(True)


def sample_ic(data, n, device, dtype):
    lo, hi = data["colloc_lo"], data["colloc_hi"]
    u = torch.rand(n, 4, device=device, dtype=dtype)
    x = lo.to(device, dtype) + u * (hi - lo).to(device, dtype)
    x[:, 3] = data["t0_std"]
    return x.requires_grad_(True)


# ── Training (early stop on training loss per spec) ─────────────────────────────
def train(model, data, device, verbose=True, ckpt_path=None, ckpt_every=5,
          resume=False, progress_path=None):
    import signal
    dtype = next(model.parameters()).dtype
    model = model.to(device)
    sc = data["scalers"]
    beta_y, beta_z = sc.beta_y, sc.beta_z
    # keep the training tensors resident on the device and batch by index (avoids
    # per-batch host->device copies; ~1.2 GB float64 / 0.6 GB float32 at 20%)
    Xtr = data["Xtr"].to(device, dtype); Ytr = data["Ytr"].to(device, dtype)
    n_train = Xtr.shape[0]
    n_val = min(len(data["Xte"]), 16384)
    vperm = torch.randperm(len(data["Xte"]))[:n_val]
    Xte = data["Xte"][vperm].to(device, dtype)
    Yte = data["Yte"][vperm].to(device, dtype)

    opt = torch.optim.Adam(model.parameters(), lr=LR)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=SCHED_FACTOR, patience=SCHED_PATIENCE, min_lr=MIN_LR)
    mse = nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]

    def gnorm(loss):
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.sqrt(sum((gi.detach() ** 2).sum() for gi in g if gi is not None))

    n_mon = min(N_COLLOC, 1024)
    mon_c = sample_colloc(data["colloc_lo"], data["colloc_hi"], n_mon, device, dtype)
    mon_i = sample_ic(data, min(N_IC, 512), device, dtype)
    val_c = sample_colloc(data["colloc_lo"], data["colloc_hi"], n_mon, device, dtype)
    val_i = sample_ic(data, min(N_IC, 512), device, dtype)
    mon_b = sample_boundary(data, min(N_BDRY, 512), device, dtype) if BDRY_ENABLE else None
    val_b = sample_boundary(data, min(N_BDRY, 512), device, dtype) if BDRY_ENABLE else None

    def phys_terms(xc, xi, xb):
        wave, gauge = physics_residuals(model, xc, beta_y, beta_z)
        w_l = mse(wave, torch.zeros_like(wave)).item()
        g_l = mse(gauge, torch.zeros_like(gauge)).item()
        i_l = mse(ic_residual(model, xi),
                  torch.zeros(xi.shape[0], 8, device=device, dtype=dtype)).item()
        if BDRY_ENABLE and xb is not None and xb.shape[0] > 0:
            br = boundary_residual(model, xb, sc)
            b_l = mse(br, torch.zeros_like(br)).item()
        else:
            b_l = 0.0
        return w_l, g_l, i_l, b_l

    keys = ["train_data", "train_wave", "train_gauge", "train_ic", "train_bdry",
            "train_total", "val_data", "val_wave", "val_gauge", "val_ic",
            "val_bdry", "val_total", "phys_w", "gamma"]
    hist = {k: [] for k in keys}
    best = float("inf"); best_state = copy.deepcopy(model.state_dict())
    best_train = float("inf"); bad = 0
    phys_w = 0.0
    start_epoch = 1

    # resume from a checkpoint if asked (model + optimizer + history + counters)
    if resume and ckpt_path and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        try:
            sch.load_state_dict(ck["sched"])
        except Exception:
            pass
        hist = ck["hist"]; best = ck["best"]; best_state = ck["best_state"]
        best_train = ck.get("best_train", float("inf")); bad = ck.get("bad", 0)
        phys_w = ck.get("phys_w", 0.0); start_epoch = ck["epoch"] + 1
        if verbose:
            print(f"[train] RESUMED from epoch {ck['epoch']} | best val {best:.2e}", flush=True)

    # save a checkpoint on SLURM time-limit (SIGTERM) or Ctrl-C (SIGINT) so a long
    # run never loses everything; also checkpoint every ckpt_every epochs.
    stop = {"flag": False}
    def _handler(signum, frame):
        stop["flag"] = True
    try:
        signal.signal(signal.SIGTERM, _handler); signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass

    def save_ckpt(ep_now):
        if not ckpt_path:
            return
        os.makedirs(os.path.dirname(ckpt_path) or ".", exist_ok=True)
        torch.save({"epoch": ep_now, "model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sch.state_dict(), "hist": hist, "best": best,
                    "best_state": best_state, "best_train": best_train, "bad": bad,
                    "phys_w": phys_w, "dtype": str(dtype)}, ckpt_path)

    if progress_path and not (resume and os.path.exists(progress_path)):
        os.makedirs(os.path.dirname(progress_path) or ".", exist_ok=True)
        with open(progress_path, "w") as f:
            f.write("epoch,train_data,wave,gauge,ic,bdry,val_data,best_val,phys_w,bad,epoch_sec,unix_ts\n")

    start = time.time()
    if verbose:
        print(f"[train] device={device} dtype={dtype} beta_y={beta_y:.3f} beta_z={beta_z:.3f} "
              f"gamma={sc.gamma:.3f} act={ACTIVATION} alpha={BALANCE_ALPHA} batch={BATCH_SIZE} "
              f"bdry={'on' if BDRY_ENABLE else 'off'} "
              f"params={sum(p.numel() for p in model.parameters()):,}", flush=True)

    ep = start_epoch - 1
    for ep in range(start_epoch, EPOCHS + 1):
        ep_t0 = time.time()
        physics_on = ep > DATA_ONLY_EPOCHS
        model.train()
        dsum = wsum = gsum = isum = bsum = 0.0; nb = 0
        perm = torch.randperm(n_train, device=device)
        for bi in range(0, n_train, BATCH_SIZE):
            idx = perm[bi:bi + BATCH_SIZE]
            xb_data = Xtr[idx].requires_grad_(True)
            yb = Ytr[idx]
            opt.zero_grad()
            data_loss = mse(displacement(model, xb_data, beta_y, beta_z), yb)
            if physics_on:
                xc = sample_colloc(data["colloc_lo"], data["colloc_hi"], N_COLLOC, device, dtype)
                wave, gauge = physics_residuals(model, xc, beta_y, beta_z)
                wave_loss = mse(wave, torch.zeros_like(wave))
                gauge_loss = mse(gauge, torch.zeros_like(gauge))
                xi = sample_ic(data, N_IC, device, dtype)
                ic_loss = mse(ic_residual(model, xi),
                              torch.zeros(N_IC, 8, device=device, dtype=dtype))
                if BDRY_ENABLE:
                    xbd = sample_boundary(data, N_BDRY, device, dtype)
                    br = boundary_residual(model, xbd, sc)
                    bdry_loss = mse(br, torch.zeros_like(br))
                else:
                    bdry_loss = torch.zeros((), device=device, dtype=dtype)
                phys = wave_loss + W_GAUGE * gauge_loss + W_IC * ic_loss + W_BDRY * bdry_loss
                if DROP_DATA_AFTER_WARMUP:
                    # physics-only refinement: the data loss no longer participates
                    phys_w = 1.0
                    loss = phys
                else:
                    if nb == 0:
                        gd, gp = gnorm(data_loss), gnorm(phys)
                        tgt = BALANCE_ALPHA * float(gd / (gp + 1e-12))
                        phys_w = (BALANCE_EMA * phys_w + (1 - BALANCE_EMA) * tgt) if phys_w > 0 else tgt
                    loss = data_loss + phys_w * phys
            else:
                wave_loss = gauge_loss = ic_loss = bdry_loss = torch.zeros((), device=device, dtype=dtype)
                loss = data_loss
            loss.backward()
            if GRAD_CLIP:
                nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            opt.step()
            if hasattr(model.features, "clamp_B"):
                model.features.clamp_B()
            dsum += data_loss.item(); wsum += float(wave_loss.item())
            gsum += float(gauge_loss.item()); isum += float(ic_loss.item())
            bsum += float(bdry_loss.item()); nb += 1

        model.eval()
        Xv = Xte.clone().requires_grad_(True)
        val_data = mse(displacement(model, Xv, beta_y, beta_z), Yte).item()
        sch.step(val_data)
        tw, tg, ti, tb = phys_terms(mon_c, mon_i, mon_b)
        vw, vg, vi, vb = phys_terms(val_c, val_i, val_b)
        tphys = tw + W_GAUGE * tg + W_IC * ti + W_BDRY * tb
        vphys = vw + W_GAUGE * vg + W_IC * vi + W_BDRY * vb
        # in drop mode the optimized training loss is physics-only after warmup
        if DROP_DATA_AFTER_WARMUP and physics_on:
            train_total = tphys
        else:
            train_total = dsum / nb + phys_w * tphys
        for k, val in [("train_data", dsum / nb), ("train_wave", tw), ("train_gauge", tg),
                       ("train_ic", ti), ("train_bdry", tb), ("train_total", train_total),
                       ("val_data", val_data), ("val_wave", vw), ("val_gauge", vg),
                       ("val_ic", vi), ("val_bdry", vb), ("val_total", val_data + phys_w * vphys),
                       ("phys_w", phys_w), ("gamma", model.gamma.item())]:
            hist[k].append(val)

        # best checkpoint by validation data MSE (kept for reference / pre-drop best)
        if val_data < best - 1e-12:
            best = val_data; best_state = copy.deepcopy(model.state_dict())
        # in drop mode, reset the plateau counter at the data->physics switch so the
        # (much larger) physics loss is not immediately read as "no improvement"
        if DROP_DATA_AFTER_WARMUP and ep == DATA_ONLY_EPOCHS + 1:
            best_train = float("inf"); bad = 0
        monitored = train_total if EARLY_STOP_METRIC == "train" else val_data
        if monitored < best_train - 1e-12:
            best_train = monitored; bad = 0
        else:
            bad += 1
        ep_sec = time.time() - ep_t0
        if progress_path:
            with open(progress_path, "a") as f:
                f.write(f"{ep},{dsum/nb:.6e},{tw:.6e},{tg:.6e},{ti:.6e},{tb:.6e},"
                        f"{val_data:.6e},{best:.6e},{phys_w:.4f},{bad},{ep_sec:.1f},{time.time():.0f}\n")
        if verbose and (ep % LOG_EVERY == 0 or ep == start_epoch):
            done = ep - start_epoch + 1
            eta = (time.time() - start) / max(done, 1) * (EPOCHS - ep) / 60.0
            print(f"ep {ep:04d}/{EPOCHS} | data {dsum/nb:.2e} | wave {wsum/nb:.2e} "
                  f"| ic {isum/nb:.2e} | bdry {bsum/nb:.2e} | val {val_data:.2e} "
                  f"| best {best:.2e} | w {phys_w:.3f} | bad {bad} | {ep_sec:.0f}s | ETA {eta:.0f}m",
                  flush=True)
        if ep % ckpt_every == 0:
            save_ckpt(ep)
        if stop["flag"]:
            if verbose:
                print(f"[train] signal received -> checkpoint at epoch {ep} and stop", flush=True)
            save_ckpt(ep)
            break
        if bad >= EARLY_STOP_PATIENCE:
            if verbose:
                print(f"early stop {ep} ({EARLY_STOP_METRIC} loss plateau {EARLY_STOP_PATIENCE})", flush=True)
            break

    save_ckpt(ep)
    # in drop mode keep the FINAL (physics-only refined) weights so the experiment
    # shows that outcome; otherwise restore the best-by-validation checkpoint.
    if not DROP_DATA_AFTER_WARMUP:
        model.load_state_dict(best_state)
    final_val = hist["val_data"][-1] if hist["val_data"] else float("nan")
    if verbose:
        tail = f"final val {final_val:.2e}" if DROP_DATA_AFTER_WARMUP else f"best val {best:.2e}"
        print(f"[train] done {(time.time()-start)/60:.2f} min | {tail} | epochs {ep}", flush=True)
    return hist, {"val_data_mse": best, "final_val_data": final_val,
                  "data_dropped": bool(DROP_DATA_AFTER_WARMUP), "epochs_run": ep}


# ── Visualization (unchanged style) ─────────────────────────────────────────────
def plot_loss_curves(hist, data_only_epochs, save_stem=None):
    comps = [("data", "Data loss"), ("wave", "Wave residual"),
             ("gauge", "Gauge div(psi)"), ("ic", "Initial condition"),
             ("bdry", "Boundary (Dirichlet)"), ("total", "Total loss")]
    if not any(hist.get("train_bdry", [])):
        comps = [c for c in comps if c[0] != "bdry"]
    for key, title in comps:
        fig, ax = plt.subplots(figsize=(9, 7))
        tr = np.maximum(hist[f"train_{key}"], 1e-30)
        ax.plot(range(1, len(tr) + 1), tr, color=AC["blue"], lw=3.0, label="Train")
        if f"val_{key}" in hist:
            va = np.maximum(hist[f"val_{key}"], 1e-30)
            ax.plot(range(1, len(va) + 1), va, color=AC["amber"], lw=3.0, label="Validation")
        if 0 < data_only_epochs < len(tr):
            ax.axvline(data_only_epochs, color=AC["red"], lw=2.0, ls=":", label="physics on")
        ax.set_yscale("log")
        ax.set_xlabel("Epoch", fontsize=30); ax.set_ylabel("MSE (standardized)", fontsize=30)
        ax.set_title(title, fontsize=34); ax.tick_params(axis="both", labelsize=24)
        ax.legend(frameon=False, fontsize=24); fig.tight_layout()
        save_fig(fig, f"{save_stem}_{key}" if save_stem else None)
        plt.close(fig)


def plot_plate(data, save_stem=None, title="Sampled points on the 300 x 200 mm plate",
               plate_xlim=(-149.5, 149.5), plate_ylim=(-199.5, -0.5)):
    """Map the unique sampled (x, y) points on the plate footprint (current style).

    Train interior points (blue), train boundary points kept in training
    (green squares), and the spatially held-out / unseen points (red diamonds).
    """
    xy = data["xy_points"]
    n = len(xy)
    held = set(data["held_spatial_idx"])
    is_b = np.asarray(data["is_boundary_point"], bool)
    is_held = np.array([i in held for i in range(n)])
    interior = (~is_held) & (~is_b)
    boundary = (~is_held) & is_b

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_patch(plt.Rectangle((plate_xlim[0], plate_ylim[0]),
                               plate_xlim[1] - plate_xlim[0],
                               plate_ylim[1] - plate_ylim[0],
                               fill=False, edgecolor=AC["axis"], lw=1.8))
    ax.scatter(xy[interior, 0], xy[interior, 1], s=22, color=AC["blue"],
               label=f"train interior ({int(interior.sum())})", zorder=3)
    if boundary.any():
        ax.scatter(xy[boundary, 0], xy[boundary, 1], s=34, color=AC["green"],
                   marker="s", edgecolor="white", lw=0.4,
                   label=f"train boundary ({int(boundary.sum())})", zorder=4)
    if is_held.any():
        ax.scatter(xy[is_held, 0], xy[is_held, 1], s=130, color=AC["red"], marker="D",
                   edgecolor="white", lw=0.8,
                   label=f"held-out / unseen ({int(is_held.sum())})", zorder=5)
    ax.set_xlabel("x [mm]", fontsize=26)
    ax.set_ylabel("y [mm]", fontsize=26)
    ax.set_title(title, fontsize=28)
    ax.tick_params(axis="both", labelsize=20)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=18, loc="upper right")
    fig.tight_layout()
    save_fig(fig, save_stem)
    plt.close(fig)


def plot_plate_nbhd(data, save_stem=None,
                    title="Neighborhood sampling on the 300 x 200 mm plate",
                    plate_xlim=(-149.5, 149.5), plate_ylim=(-199.5, -0.5)):
    """Plate footprint for Experiment 2 (same style as ``plot_plate``).

    Train points inside neighborhoods (blue), unseen points held out INSIDE the
    neighborhoods (amber diamonds), and unseen points OUTSIDE every neighborhood
    (red diamonds). Faint circles mark each neighborhood for context.
    """
    xy = data["xy_points"]; n = len(xy)
    inside_held = set(data.get("inside_held_idx", []))
    outside_held = set(data.get("outside_held_idx", []))
    held = inside_held | outside_held
    is_train = np.array([i not in held for i in range(n)])
    in_h = np.array([i in inside_held for i in range(n)])
    out_h = np.array([i in outside_held for i in range(n)])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.add_patch(plt.Rectangle((plate_xlim[0], plate_ylim[0]),
                               plate_xlim[1] - plate_xlim[0],
                               plate_ylim[1] - plate_ylim[0],
                               fill=False, edgecolor=AC["axis"], lw=1.8))
    ax.scatter(xy[is_train, 0], xy[is_train, 1], s=22, color=AC["blue"],
               label=f"train (in neighborhood) ({int(is_train.sum())})", zorder=3)
    if in_h.any():
        ax.scatter(xy[in_h, 0], xy[in_h, 1], s=120, color=AC["amber"], marker="D",
                   edgecolor="white", lw=0.8,
                   label=f"unseen inside ({int(in_h.sum())})", zorder=5)
    if out_h.any():
        ax.scatter(xy[out_h, 0], xy[out_h, 1], s=120, color=AC["red"], marker="D",
                   edgecolor="white", lw=0.8,
                   label=f"unseen outside ({int(out_h.sum())})", zorder=5)
    ax.set_xlabel("x [mm]", fontsize=26); ax.set_ylabel("y [mm]", fontsize=26)
    ax.set_title(title, fontsize=28); ax.tick_params(axis="both", labelsize=20)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(frameon=False, fontsize=16, loc="upper right")
    fig.tight_layout()
    save_fig(fig, save_stem)
    plt.close(fig)


def reconstruct_xy(model, data, xy_index, z, device, n_t=N_T):
    sc = data["scalers"]
    beta_y, beta_z = sc.beta_y, sc.beta_z
    px, py = data["xy_points"][xy_index]
    t_full = np.arange(1, n_t + 1) * DT
    X = sc.encode(np.full(n_t, px), np.full(n_t, py), np.full(n_t, z), t_full)
    Xg = torch.tensor(X, dtype=next(model.parameters()).dtype, device=device).requires_grad_(True)
    model.eval()
    uvw_hat = displacement(model, Xg, beta_y, beta_z).detach().cpu().numpy()
    pred = sc.decode_fields(uvw_hat)
    frame = data["df"]
    mask = ((np.abs(frame["x"].to_numpy() - px) < 1e-6) & (np.abs(frame["y"].to_numpy() - py) < 1e-6)
            & (np.abs(frame["z"].to_numpy() - z) < 1e-6))
    grp = frame[mask].sort_values("t")
    gt = np.full((n_t, 3), np.nan)
    gt[grp["t_idx"].to_numpy()] = grp[FIELD_COLS].to_numpy()
    train_keep = np.zeros(n_t, bool)
    train_keep[grp["t_idx"].to_numpy()] = grp["is_train"].to_numpy()
    return {"pred": pred, "gt": gt, "point": (px, py, z),
            "train_idx": np.where(train_keep)[0], "test_idx": np.where(~train_keep)[0]}


def relative_l2(pred, target):
    d = np.linalg.norm(target)
    return float(np.linalg.norm(pred - target) / d) if d > 0 else float(np.linalg.norm(pred - target))


def plot_reconstruction(recs, save_stem=None,
                        title="Helmholtz-potential PINN reconstruction (u/v/w)"):
    n = len(recs); comps = ["u", "v", "w"]
    fig, axes = plt.subplots(n, 3, figsize=(22, 4.2 * n), squeeze=False)
    t = np.arange(len(recs[0]["pred"]))
    for i in range(n):
        px, py, _ = recs[i]["point"]
        for ci, comp in enumerate(comps):
            ax = axes[i][ci]
            ax.plot(t, recs[i]["gt"][:, ci], color=AC["ink"], lw=2.0, label="Ground truth")
            ax.plot(t, recs[i]["pred"][:, ci], color=AC["red"], lw=2.2, ls="--", alpha=0.9, label="PINN")
            if ci == 0:
                ax.set_ylabel(f"x={px:.1f}\ny={py:.1f}")
            if i == 0:
                ax.set_title(f"component {comp}")
            if i == 0 and ci == 2:
                ax.legend(frameon=False, loc="upper right")
    for ci in range(3):
        axes[-1][ci].set_xlabel("Timestep index")
    fig.suptitle(title, y=1.005); fig.tight_layout()
    save_fig(fig, save_stem); plt.close(fig)


def _spread_pick(indices, k, nbhd, seed):
    if len(indices) <= k:
        return sorted(indices)
    rng = np.random.RandomState(seed)
    by_group = {}
    for i in indices:
        by_group.setdefault(nbhd[i], []).append(i)
    for g in by_group:
        rng.shuffle(by_group[g])
    order = sorted(by_group.keys(), key=lambda g: str(g))
    chosen, gi = [], 0
    while len(chosen) < k:
        g = order[gi % len(order)]
        if by_group[g]:
            chosen.append(by_group[g].pop())
        gi += 1
    return sorted(chosen)


def select_heldout_points(data, k=5, seed=42):
    held = list(data["held_spatial_idx"])
    return _spread_pick(held, k, data["nbhd_of_point"], seed)


def plot_heldout_prediction(recs, save_stem=None,
                            title="Held-out spatial point prediction (u/v/w)"):
    n = len(recs); comps = ["u", "v", "w"]
    fig, axes = plt.subplots(n, 3, figsize=(22, 4.2 * n), squeeze=False)
    t = np.arange(len(recs[0]["pred"]))
    for i in range(n):
        px, py, _ = recs[i]["point"]
        for ci, comp in enumerate(comps):
            ax = axes[i][ci]
            ax.plot(t, recs[i]["gt"][:, ci], color=AC["ink"], lw=2.0, label="Ground truth")
            ax.plot(t, recs[i]["pred"][:, ci], color=AC["red"], lw=2.2, ls="--", alpha=0.9,
                    label="PINN (unseen point)")
            if ci == 0:
                ax.set_ylabel(f"x={px:.1f}\ny={py:.1f}")
            if i == 0:
                ax.set_title(f"component {comp}")
            if i == 0 and ci == 2:
                ax.legend(frameon=False, loc="upper right")
    for ci in range(3):
        axes[-1][ci].set_xlabel("Timestep index")
    fig.suptitle(title, y=1.005); fig.tight_layout()
    save_fig(fig, save_stem); plt.close(fig)


def select_seen_points(data, k=5, seed=42):
    df = data["df"]; xy = data["xy_points"]
    held = set(data["held_spatial_idx"])
    key_to_idx = {(px, py): i for i, (px, py) in enumerate(map(tuple, xy))}
    trained_keys = (df[df["is_train"]][["x", "y"]].drop_duplicates()
                    .itertuples(index=False, name=None))
    seen = sorted({key_to_idx[k_] for k_ in trained_keys
                   if k_ in key_to_idx and key_to_idx[k_] not in held})
    return _spread_pick(seen, k, data["nbhd_of_point"], seed)
