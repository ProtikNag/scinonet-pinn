"""3D Helmholtz-potential PINN (option B): solve the wave equations on potentials.

The network predicts four scalar potentials ``(phi, psi_x, psi_y, psi_z)`` from
standardized coordinates ``(x, y, z, t)``. The displacement is the Helmholtz
combination

    u = phi_x + psi_z,y - psi_y,z
    v = phi_y + psi_x,z - psi_z,x
    w = phi_z + psi_y,x - psi_x,y

and each potential obeys a scalar wave equation (P-speed c_p for phi, S-speed c_s
for psi), which together carry both the symmetric (S0) and antisymmetric (A0)
Lamb modes. The gauge freedom of psi is removed by the Coulomb gauge
``div(psi)=0`` plus a rest initial condition; see HANDOFF / docs for the
non-dimensionalization.

Isotropic in-plane scale ``L`` (for x, y), separate through-thickness scale
``Lz`` (for z, the plate is thin), time scale ``s_t``, common field scale ``s_f``
for (u, v, w). The in-plane/through-thickness ratio is ``rho = L / Lz``; it weighs
every z-derivative term below.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from .data import DT, FIELD_COLS, N_T, load_full_signal
from .features import SpecializedFourierFeatures
from .models import _ACTIVATIONS

# Bulk wave speeds for structural aluminum (E~69 GPa, nu~0.33, rho~2700 kg/m^3),
# standard handbook values. Units mm/s (c_p/c_s ~ 2.0).
CS_MM_PER_S = 3.15e6
CP_MM_PER_S = 6.30e6

# Through-thickness non-dimensionalization mode (see PotentialScalers.fit):
#   "physical" -> Lz = 1 mm (physical ply spacing; rho = L, large)   [default]
#   "inplane"  -> Lz = L    (rho = 1; the earlier well-conditioned change of vars)
LZ_MODE = "physical"


@dataclass
class PotentialScalers:
    mu_x: float; mu_y: float; L: float
    mu_z: float; Lz: float
    mu_t: float; s_t: float
    s_f: float

    @classmethod
    def fit(cls, df: pd.DataFrame) -> "PotentialScalers":
        xy = np.concatenate([df["x"].to_numpy(), df["y"].to_numpy()])
        uvw = np.concatenate([df["u"].to_numpy(), df["v"].to_numpy(), df["w"].to_numpy()])
        L = float(xy.std() + 1e-12)
        # Through-thickness scale, selected by the module-level LZ_MODE:
        #   "physical" -> Lz = 1 mm (physical ply spacing; rho = L large, the
        #                 z-derivative terms carry physical weight but are stiff)
        #   "inplane"  -> Lz = L   (rho = 1; the earlier well-conditioned change of
        #                 variables that kept the displacement/Laplacian balanced)
        Lz = 1.0 if LZ_MODE == "physical" else L
        return cls(
            mu_x=float(df["x"].mean()), mu_y=float(df["y"].mean()), L=L,
            mu_z=float(df["z"].mean()), Lz=Lz,
            mu_t=float(df["t"].mean()), s_t=float(df["t"].std() + 1e-12),
            s_f=float(uvw.std() + 1e-30))

    @property
    def rho(self) -> float:
        return self.L / self.Lz

    def chat(self, c_mm_per_s: float) -> float:
        return c_mm_per_s * self.s_t / self.L          # in-plane dimensionless speed

    def encode(self, x, y, z, t) -> np.ndarray:
        return np.stack([(x - self.mu_x) / self.L, (y - self.mu_y) / self.L,
                         (z - self.mu_z) / self.Lz, (t - self.mu_t) / self.s_t], axis=1)

    def decode_fields(self, uvw_hat: np.ndarray) -> np.ndarray:
        return uvw_hat * self.s_f                       # common field scale (zero-mean)


@dataclass
class PotentialDataset:
    Xtr: torch.Tensor; Ytr: torch.Tensor
    Xte: torch.Tensor; Yte: torch.Tensor
    scalers: PotentialScalers
    xy_points: np.ndarray          # unique (x, y) in mm, [P, 2]
    z_values: np.ndarray           # the through-thickness z levels [nz]
    df: pd.DataFrame
    holdout_xy_indices: list[int]
    colloc_lo: torch.Tensor; colloc_hi: torch.Tensor   # standardized [x,y,z,t]
    t0_std: float
    field_cols = FIELD_COLS


def build_potential_dataset(csv_path, subsample_keep=0.5, seed=42,
                            n_holdout_xy=0, holdout_indices=None) -> PotentialDataset:
    """Load the 3-ply CSV and build standardized 4D tensors.

    Spatial holdout removes whole (x, y) columns (all z and t). Temporal holdout
    keeps a random fraction of timesteps per point.
    """
    df = load_full_signal(csv_path)
    xy = df[["x", "y"]].drop_duplicates().to_numpy()
    z_values = np.sort(df["z"].unique())

    rng = np.random.RandomState(seed)
    keep = rng.rand(len(df)) < subsample_keep

    held: list[int] = []
    if holdout_indices is not None:
        held = sorted(int(i) for i in holdout_indices)
    elif n_holdout_xy > 0:
        rng2 = np.random.RandomState(seed + 1)
        held = sorted(rng2.choice(len(xy), n_holdout_xy, replace=False).tolist())
    if held:
        held_set = {tuple(np.round(xy[i], 6)) for i in held}
        rowxy = list(map(tuple, np.round(df[["x", "y"]].to_numpy(), 6)))
        is_held = np.array([p in held_set for p in rowxy])
        keep = keep & ~is_held

    df = df.copy(); df["is_train"] = keep
    tr, te = df[keep], df[~keep]
    sc = PotentialScalers.fit(tr)

    def to_xy(frame):
        X = sc.encode(frame["x"].to_numpy(), frame["y"].to_numpy(),
                      frame["z"].to_numpy(), frame["t"].to_numpy())
        Y = frame[FIELD_COLS].to_numpy() / sc.s_f
        return torch.tensor(X, dtype=torch.float64), torch.tensor(Y, dtype=torch.float64)

    Xtr, Ytr = to_xy(tr); Xte, Yte = to_xy(te)
    allc = sc.encode(df["x"].to_numpy(), df["y"].to_numpy(),
                     df["z"].to_numpy(), df["t"].to_numpy())
    lo = torch.tensor(allc.min(0), dtype=torch.float64)
    hi = torch.tensor(allc.max(0), dtype=torch.float64)
    t0 = float((0.0 - sc.mu_t) / sc.s_t)
    return PotentialDataset(Xtr, Ytr, Xte, Yte, sc, xy, z_values, df, held, lo, hi, t0)


class PotentialNet(nn.Module):
    """Fourier-feature MLP -> 4 potentials, with learnable standardized speeds."""

    def __init__(self, features: SpecializedFourierFeatures, hidden_sizes,
                 chatp_init: float, chats_init: float, activation="tanh",
                 concat_raw=True):
        super().__init__()
        self.features = features
        self.concat_raw = concat_raw
        act = _ACTIVATIONS[activation]
        in_dim = features.out_features + (features.in_features if concat_raw else 0)
        layers, prev = [], in_dim
        for wdt in hidden_sizes:
            layers += [nn.Linear(prev, wdt), act()]; prev = wdt
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 4)                      # phi, psi_x, psi_y, psi_z
        self.log_chatp = nn.Parameter(torch.tensor(float(np.log(chatp_init)), dtype=torch.float64))
        self.log_chats = nn.Parameter(torch.tensor(float(np.log(chats_init)), dtype=torch.float64))

    @property
    def chatp(self): return torch.exp(self.log_chatp)
    @property
    def chats(self): return torch.exp(self.log_chats)

    def forward(self, x):
        e = self.features(x)
        if self.concat_raw:
            e = torch.cat([e, x], dim=-1)
        return self.head(self.backbone(e))


def _grad(out, x):
    return torch.autograd.grad(out, x, grad_outputs=torch.ones_like(out), create_graph=True)[0]


def displacement(model: PotentialNet, x: torch.Tensor, rho: float):
    """Standardized (u,v,w) from the potential first derivatives. x requires grad."""
    p = model(x)
    gphi = _grad(p[:, 0], x); gpx = _grad(p[:, 1], x)
    gpy = _grad(p[:, 2], x);  gpz = _grad(p[:, 3], x)
    # columns of g are d/dxhat, d/dyhat, d/dzhat, d/dthat
    u = gphi[:, 0] + gpz[:, 1] - rho * gpy[:, 2]
    v = gphi[:, 1] + rho * gpx[:, 2] - gpz[:, 0]
    w = rho * gphi[:, 2] + gpy[:, 0] - gpx[:, 1]
    return torch.stack([u, v, w], dim=1)


def _laplacian_tt(q, x, chat, rho):
    """Return normalized wave residual q_tt/(chat^2 rho^2) - (1/rho^2)(q_xx+q_yy) - q_zz."""
    g = _grad(q, x)
    q_xx = _grad(g[:, 0], x)[:, 0]
    q_yy = _grad(g[:, 1], x)[:, 1]
    q_zz = _grad(g[:, 2], x)[:, 2]
    q_tt = _grad(g[:, 3], x)[:, 3]
    return q_tt / (chat ** 2 * rho ** 2) - (q_xx + q_yy) / rho ** 2 - q_zz


def physics_residuals(model: PotentialNet, x: torch.Tensor, rho: float):
    """Wave residuals (phi at c_p, psi at c_s) and the Coulomb-gauge residual.

    Returns ``(wave [N,4], gauge [N])``. x requires grad.
    """
    p = model(x)
    cp, cs = model.chatp, model.chats
    r_phi = _laplacian_tt(p[:, 0], x, cp, rho)
    r_px = _laplacian_tt(p[:, 1], x, cs, rho)
    r_py = _laplacian_tt(p[:, 2], x, cs, rho)
    r_pz = _laplacian_tt(p[:, 3], x, cs, rho)
    # Coulomb gauge div(psi) = psi_x,x + psi_y,y + rho psi_z,z (standardized)
    gauge = (_grad(p[:, 1], x)[:, 0] + _grad(p[:, 2], x)[:, 1]
             + rho * _grad(p[:, 3], x)[:, 2])
    return torch.stack([r_phi, r_px, r_py, r_pz], dim=1), gauge


def ic_residual(model: PotentialNet, x: torch.Tensor):
    """Rest IC at t=0: all four potentials and their time-derivatives vanish.

    Returns ``[N, 8]`` = (phi, psi_x, psi_y, psi_z, phi_t, ...). x at t0, requires grad.
    """
    p = model(x)
    p_t = torch.stack([_grad(p[:, k], x)[:, 3] for k in range(4)], dim=1)
    return torch.cat([p, p_t], dim=1)


@dataclass
class PotentialTrainConfig:
    epochs: int = 200
    batch_size: int = 16384
    n_colloc: int = 2048
    n_ic: int = 1024
    lr: float = 2e-3
    grad_clip: float = 1.0
    data_only_epochs: int = 15        # short warmup; physics on early to pin potentials
    balance_alpha: float = 0.3
    balance_ema: float = 0.9
    w_gauge: float = 0.0          # Coulomb-gauge loss removed from the objective
    w_ic: float = 1.0
    scheduler_factor: float = 0.5
    scheduler_patience: int = 15
    min_lr: float = 1e-5
    early_stop_patience: int = 80
    log_every: int = 20


def _sample(lo, hi, n, device, dtype):
    u = torch.rand(n, lo.numel(), device=device, dtype=dtype)
    return (lo.to(device, dtype) + u * (hi - lo).to(device, dtype)).requires_grad_(True)


def _sample_ic(data, n, device, dtype):
    lo, hi = data.colloc_lo, data.colloc_hi
    u = torch.rand(n, 4, device=device, dtype=dtype)
    x = lo.to(device, dtype) + u * (hi - lo).to(device, dtype)
    x[:, 3] = data.t0_std
    return x.requires_grad_(True)


def train_potential(model: PotentialNet, data: PotentialDataset,
                    config: PotentialTrainConfig, device, verbose=True):
    """Train: data (via displacement) + wave residuals + gauge + IC, grad-balanced."""
    import copy, time
    from torch.utils.data import DataLoader, TensorDataset

    dtype = next(model.parameters()).dtype
    model = model.to(device)
    rho = data.scalers.rho
    loader = DataLoader(TensorDataset(data.Xtr, data.Ytr),
                        batch_size=config.batch_size, shuffle=True)
    # cap the per-epoch validation subset (displacement needs autograd -> memory)
    n_val = min(len(data.Xte), 8192)
    vperm = torch.randperm(len(data.Xte))[:n_val]
    Xte = data.Xte[vperm].to(device, dtype); Yte = data.Yte[vperm].to(device, dtype)
    opt = torch.optim.Adam(model.parameters(), lr=config.lr)
    sch = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=config.scheduler_factor,
        patience=config.scheduler_patience, min_lr=config.min_lr)
    mse = nn.MSELoss()
    params = [p for p in model.parameters() if p.requires_grad]

    def gnorm(loss):
        g = torch.autograd.grad(loss, params, retain_graph=True, allow_unused=True)
        return torch.sqrt(sum((gi.detach() ** 2).sum() for gi in g if gi is not None))

    # Fixed monitoring collocation sets (train + val) so the physics curves are
    # real and continuous from epoch 1 -- not the log(0) floor during warmup.
    n_mon = min(config.n_colloc, 1024)
    mon_c = _sample(data.colloc_lo, data.colloc_hi, n_mon, device, dtype)
    mon_i = _sample_ic(data, min(config.n_ic, 512), device, dtype)
    val_c = _sample(data.colloc_lo, data.colloc_hi, n_mon, device, dtype)
    val_i = _sample_ic(data, min(config.n_ic, 512), device, dtype)

    def phys_terms(xc, xi):
        wave, gauge = physics_residuals(model, xc, rho)
        w_l = mse(wave, torch.zeros_like(wave)).item()
        g_l = mse(gauge, torch.zeros_like(gauge)).item()
        i_l = mse(ic_residual(model, xi), torch.zeros(xi.shape[0], 8, device=device, dtype=dtype)).item()
        return w_l, g_l, i_l

    hist = {k: [] for k in ["train_data", "train_wave", "train_gauge", "train_ic", "train_total",
                            "val_data", "val_wave", "val_gauge", "val_ic", "val_total",
                            "phys_w", "chatp", "chats"]}
    best = float("inf"); best_state = copy.deepcopy(model.state_dict()); bad = 0
    phys_w = 0.0
    start = time.time()
    if verbose:
        print(f"[pot] device={device} rho={rho:.2f} chatp0={model.chatp.item():.1f} "
              f"chats0={model.chats.item():.1f} train_rows={len(data.Xtr)} "
              f"val_rows={len(data.Xte)} params={sum(p.numel() for p in model.parameters()):,}")
    for ep in range(1, config.epochs + 1):
        physics_on = ep > config.data_only_epochs
        model.train()
        dsum = wsum = gsum = isum = 0.0; nb = 0
        for xb, yb in loader:
            xb = xb.to(device, dtype).requires_grad_(True); yb = yb.to(device, dtype)
            opt.zero_grad()
            data_loss = mse(displacement(model, xb, rho), yb)
            if physics_on:
                xc = _sample(data.colloc_lo, data.colloc_hi, config.n_colloc, device, dtype)
                wave, gauge = physics_residuals(model, xc, rho)
                wave_loss = mse(wave, torch.zeros_like(wave))
                gauge_loss = mse(gauge, torch.zeros_like(gauge))
                xi = _sample_ic(data, config.n_ic, device, dtype)
                ic_loss = mse(ic_residual(model, xi), torch.zeros(config.n_ic, 8, device=device, dtype=dtype))
                # Gauge enters the objective only when w_gauge > 0 (default 0.0 = removed).
                phys = wave_loss + config.w_gauge * gauge_loss + config.w_ic * ic_loss
                if nb == 0:
                    gd, gp = gnorm(data_loss), gnorm(phys)
                    tgt = config.balance_alpha * float(gd / (gp + 1e-12))
                    phys_w = (config.balance_ema * phys_w + (1 - config.balance_ema) * tgt) if phys_w > 0 else tgt
                loss = data_loss + phys_w * phys
            else:
                wave_loss = gauge_loss = ic_loss = torch.zeros((), device=device, dtype=dtype)
                loss = data_loss
            loss.backward()
            if config.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            opt.step()
            if hasattr(model.features, "clamp_B"):
                model.features.clamp_B()
            dsum += data_loss.item(); wsum += float(wave_loss.item())
            gsum += float(gauge_loss.item()); isum += float(ic_loss.item()); nb += 1
        model.eval()
        Xv = Xte.clone().requires_grad_(True)
        val_data = mse(displacement(model, Xv, rho), Yte).item()
        sch.step(val_data)
        # monitored physics residuals (computed every epoch, including warmup)
        tw, tg, ti = phys_terms(mon_c, mon_i)
        vw, vg, vi = phys_terms(val_c, val_i)
        tphys = tw + config.w_gauge * tg + config.w_ic * ti
        vphys = vw + config.w_gauge * vg + config.w_ic * vi
        for k, val in [("train_data", dsum/nb), ("train_wave", tw), ("train_gauge", tg),
                       ("train_ic", ti), ("train_total", dsum/nb + phys_w*tphys),
                       ("val_data", val_data), ("val_wave", vw), ("val_gauge", vg),
                       ("val_ic", vi), ("val_total", val_data + phys_w*vphys),
                       ("phys_w", phys_w), ("chatp", model.chatp.item()), ("chats", model.chats.item())]:
            hist[k].append(val)
        if val_data < best - 1e-12:
            best = val_data; best_state = copy.deepcopy(model.state_dict()); bad = 0
        else:
            bad += 1
        if verbose and (ep % config.log_every == 0 or ep == 1):
            print(f"ep {ep:04d} | data {dsum/nb:.2e} | wave {wsum/nb:.2e} | gauge {gsum/nb:.2e} "
                  f"| ic {isum/nb:.2e} | val {val_data:.2e} | best {best:.2e} | w {phys_w:.3f} "
                  f"| cp {model.chatp.item():.1f} cs {model.chats.item():.1f}")
        if bad >= config.early_stop_patience:
            if verbose: print(f"early stop {ep}")
            break
    model.load_state_dict(best_state)
    if verbose:
        print(f"[pot] done {(time.time()-start)/60:.2f} min | best val {best:.2e}")
    return hist, {"val_data_mse": best}


def relative_l2(pred, target):
    d = np.linalg.norm(target)
    return float(np.linalg.norm(pred - target) / d) if d > 0 else float(np.linalg.norm(pred - target))


def reconstruct_xy(model: PotentialNet, data: PotentialDataset, xy_index, z, device, n_t=N_T):
    """Reconstruct physical (u,v,w) over full time at one (x,y) and chosen z level."""
    sc = data.scalers; rho = sc.rho
    px, py = data.xy_points[xy_index]
    t_full = np.arange(1, n_t + 1) * DT
    X = sc.encode(np.full(n_t, px), np.full(n_t, py), np.full(n_t, z), t_full)
    Xg = torch.tensor(X, dtype=next(model.parameters()).dtype, device=device).requires_grad_(True)
    model.eval()
    uvw_hat = displacement(model, Xg, rho).detach().cpu().numpy()
    pred = sc.decode_fields(uvw_hat)
    df = data.df
    mask = ((np.abs(df["x"].to_numpy() - px) < 1e-6) & (np.abs(df["y"].to_numpy() - py) < 1e-6)
            & (np.abs(df["z"].to_numpy() - z) < 1e-6))
    grp = df[mask].sort_values("t")
    gt = np.full((n_t, 3), np.nan)
    gt[grp["t_idx"].to_numpy()] = grp[FIELD_COLS].to_numpy()
    train_keep = np.zeros(n_t, bool)
    train_keep[grp["t_idx"].to_numpy()] = grp["is_train"].to_numpy()
    return {"pred": pred, "gt": gt, "point": (px, py, z),
            "train_idx": np.where(train_keep)[0], "test_idx": np.where(~train_keep)[0]}


def evaluate_holdout(model, data, device, xy_indices, z=0.0):
    errs = []
    for i in xy_indices:
        rec = reconstruct_xy(model, data, i, z, device)
        ti = rec["test_idx"]
        if len(ti) == 0: continue
        valid = ~np.isnan(rec["gt"][ti]).any(axis=1)
        errs.append(relative_l2(rec["pred"][ti][valid], rec["gt"][ti][valid]))
    errs = np.array(errs)
    return {"median": float(np.median(errs)), "mean": float(np.mean(errs)), "per_point": errs.tolist()}
