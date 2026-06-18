# SciNoNet Handoff

Running log of project state, decisions, findings, and the road forward. Keep
current as work progresses.

Last updated: 2026-05-30

---

## 1. Goal

Reconstruct the full 6001-timestep elastic-wave displacement signal at points on
a 200x300 mm plate from a temporally partial observation, with a Fourier-feature
network, then add physics (PINN). Demonstrate two regimes:

- **Temporal holdout**: a *seen* point, reconstruct timesteps not in training.
- **Spatial holdout**: a *never-seen* point, reconstruct its whole signal.

---

## 2. Data

Source: `data/3D_Pristine.mat` (16 GB, HDF5). Keys: `Disp_x/y/z`
(180000 x 6001), `X/Y/Z_zero_coord_ply` (180000), `dt=1e-8`, `TotTim`. First ply =
first 60000 rows; coords and displacements are in metres (multiply by 1000 for
mm). Plate spans x in [-149.5, 149.5], y in [-199.5, -0.5] on a 1 mm grid.

Working dataset (generated here):
`data/dataset_4nbhd_50pts_fullsignal_6001steps.csv` — 4 dense neighborhoods of 50
points each (200 points), centered at `(+/-70, -60)` and `(+/-70, -140)` mm, each
a ~1 mm cluster (r < 4 mm). Build it with
`scripts/generate_neighbor_dataset.py --centers="-70,-60;70,-60;-70,-140;70,-140"
--per-center 50` (reads only coords + the selected rows, never the full 16 GB
arrays). The earlier single 100-point origin cluster is also available via
`--centers="0,0" --per-center 100`.

Note: with spread neighborhoods the isotropic spatial scale `L` is large (~76 mm),
so the standardized speed `chat = c*s_t/L` is small (~0.7) and the PDE residual is
dominated by the high spatial-frequency features needed to index 1 mm-spaced
points. Gradient balancing keeps training stable in this regime.

Spectral content (w): dominant ~167 kHz, 99% energy by ~217 kHz; shear wavelength
~20 mm. The dense 1 mm spacing gives good spatial coverage.

Legacy files (`*_70ratio_*`, `*_rec_neighbor_*`) are an earlier mismatched pair;
see git history. They are superseded by the `.mat`-derived dataset.

---

## 3. Bugs fixed from the original notebooks

1. **File mismatch** — the original partial/full CSVs were different point sets.
2. **Output magnitude floor** — `sign * 10**Softplus(mag)` could not represent
   ~85% of standardized targets. Now predict the field directly.
3. **Fourier bandwidth off ~1000x** — physical Hz applied to standardized inputs.
   Fixed via `features.bandwidth_from_physics` (a physical freq `f` maps to
   `f * std` cycles per standardized unit).
4. **PDE residual dimensionally inconsistent** — see §6 for the corrected,
   non-dimensionalized residual.
5. **Data/physics disconnect** — the plotted branch was untouched by the PDE.
   Fixed: the same predicted field feeds both data and physics losses.

---

## 4. Data-only baseline (validated, working)

Fourier-feature MLP predicting standardized `(u, v, w)` directly.

- **Temporal holdout**: reconstructs full signals well. On the 70ratio stand-in,
  held-out relL2 ~0.004 at 50% retention, ~0.01 at 10%, with a clean cliff below
  ~5% (see git history; `comparison_retention_2to50`).
- **Spatial holdout**: fails (relL2 ~1.0) — a data-only model cannot synthesize a
  never-seen point's waveform. This motivated the PINN.

Entry point: `scripts/run_experiment.py --config configs/default.yaml`.

---

## 5. PINN (physics-informed) — current state

### 5.1 Governing equation, validated against data

A finite-difference check on the dense 1 mm grid (`u_tt / laplacian(u)` at
interior points) shows each component approximately obeys a **scalar wave
equation** with an effective speed near the shear speed:

    measured c:  u ~ 2.1e6,  v ~ 3.0e6,  w ~ 3.1e6 mm/s   (cs = 3.2e6, cp = 6.3e6)

So the components are ~decoupled at the dominant shear/Lamb mode. The PINN
therefore predicts `(u, v, w)` **directly** and enforces, per component,

    f_tt = c_f^2 (f_xx + f_yy)

with a **learnable** effective speed `c_f` (inverse-problem parameter). Predicting
the field directly (not Helmholtz potentials) avoids the gauge freedom that made
the potential residual explode (the data constrains only derivative combinations
of the potentials, leaving their second derivatives unconstrained).

### 5.2 Non-dimensionalization

Isotropic spatial scale `L`, per-component field scale, time scale `s_t`. The
standardized field obeys the same equation with `chat = c * s_t / L`. The residual
is divided by `chat^2` so temporal and spatial terms are O(1) (otherwise the
spatial term dominates by ~chat^2 ~ 1e3 and the optimizer satisfies physics with a
trivial zero field). Implemented in `src/scinonet/pinn.py`.

### 5.2b Specialized Fourier features (dispersion-informed B)

Replaced the random Gaussian frequency matrix with
`features.SpecializedFourierFeatures` — the corrected version of the notebooks'
`FourierFeatures`: a **learnable** `B` **clamped** to per-dimension frequency
ranges from the Lamb-wave dispersion plot, with the amplitude **outside** the
sinusoid: `gamma = [A*sin(2pi x B), A*cos(2pi x B)]`. The notebooks put `a*B`
inside the sinusoid (which rescales the frequency and lets it escape the clamp);
`A` outside is a true amplitude and `B` stays a pure, clamped frequency.

Bands are read off the frequency-wavenumber figure (Fig. A.4): energy at
`kappa <= ~0.8 rad/mm` and `f <= ~300 kHz`, so `k_max ~ 0.8/(2pi) ~ 0.13 cyc/mm`
and `f_max ~ 300 kHz`, converted to standardized cyc/unit via the coordinate std
(`B_xy in [-k_max*L, k_max*L]`, `B_t in [0, f_max*s_t]`). Using the physical
wavenumber band (vs the earlier 0.5 cyc/mm) drops the PDE residual scale from
~1e10 to ~1e6 (less stiff). `clamp_B()` is called after each optimizer step.

### 5.2c Initial and boundary conditions

- **IC (rest at t=0):** `u=v=w=0` and `u_t=v_t=w_t=0` at the standardized t=0
  plane, sampled over the data region (`initial_condition_residual`).
- **BC (Dirichlet):** `u=v=w=0` on the four physical plate edges
  (`x=+/-149.5`, `y in {-199.5,-0.5}`) over the full time range
  (`boundary_condition_residual`).

The physics term is now `L_pde + w_ic*L_ic + w_bc*L_bc`, tracked as separate
components in the history and plotted as separate loss-curve panels (this is why
the earlier single physics curve "looked flat": physics is off during warmup and
the balanced weight is small, so total ~ data).

### 5.3 Training stability — gradient-norm balancing

Naive weighting collapses to the trivial (zero-field) solution the instant
physics turns on: the second-derivative physics gradient is ~1000x the data
gradient, so under gradient clipping it erases the data fit. Fix: **gradient-norm
balancing** (Wang et al.) — each epoch, rescale the physics weight so
`||g_phys|| = balance_alpha * ||g_data||` (EMA-smoothed). With `balance_alpha`
~0.3-0.6 the data fit survives and physics still acts. Plus a data-only warmup
(`data_only_epochs`) to fit the data before physics engages.

### 5.4 Saved artifacts and visualizations

Every run saves a self-contained folder under `outputs/pinn/<mode>/<phys_off|phys_on>/`:
`model.pt` (state_dict + scalers + history + metrics + config), `history.json`,
`metrics.json`, and `loss_curves.{png,svg}` (data / physics / total loss, train vs
validation, with the physics-on epoch marked). Each mode also writes
`plate_layout.{png,svg}` (the selected points on the 300x200 mm plate, held-out
points in red) and `compare_<mode>_<comp>.{png,svg}`. Temporal training uses **1%**
of timesteps per point.

### 5.4b Results

Held-out relative L2 (4-neighborhood data; amplitude-outside embedding with the
Fig-A.4 wavenumber band):

| Setting | data-only | PINN (PDE+IC+BC) |
|---|---|---|
| Temporal, 1% samples (median, 200 pts) | 0.56 | 0.27 |
| Spatial, unseen points (median, 8 pts) | 0.36 | **0.06** |

- **Spatial holdout** now reconstructs never-seen interior points faithfully:
  every held-out point reaches 5-11% error with physics (grid figure
  `outputs/pinn/spatial/all_holdout_uvw.{png,svg}`, all points x u,v,w). What
  unlocked this: (a) amplitude outside the sinusoid so B stays a pure clamped
  frequency, and (b) the physically-correct wavenumber band from the Lamb
  dispersion plot (kappa <= 0.8 rad/mm), which stops the network indexing points
  with non-physical high spatial frequencies. The PDE+IC+BC then clean up the
  residual oscillations.
- **Temporal 1%**: physics roughly halves the error (0.56 -> 0.27); the IC anchors
  the quiet pre-arrival window.

**Read.** With the corrected embedding the per-component scalar-wave PINN plus
IC/BC generalizes to never-seen points well. A small residual gap remains only for
the lowest-amplitude components, where a source term (next step) would help.

Entry points: `scripts/pinn_final.py` (both figures), `scripts/pinn_ab.py`
(controlled spatial A/B), `scripts/run_pinn.py` (single-mode compare).

---

## 5.5 Helmholtz-potential model (option B, current best)

`src/scinonet/potential.py` + `scripts/run_potential.py`. The faithful two-mode
formulation: predict the four potentials `(phi, psi_x, psi_y, psi_z)` from 3D
inputs `(x,y,z,t)` (the 3-ply data `dataset_1nbhd_100pts_3ply_*.csv`), derive the
displacement by the Helmholtz combination `u = grad(phi) + curl(psi)`, and solve
`phi_tt = cp^2 lap(phi)`, `psi_tt = cs^2 lap(psi)` (bulk `cp=6.3e6`, `cs=3.15e6`
mm/s, ratio ~2 for nu=0.33, held fixed so cp/cs does not collapse). The gauge
freedom is removed by the **Coulomb gauge** `div(psi)=0` plus the **rest IC**
(`phi=psi=phi_t=psi_t=0` at t=0). The two Lamb modes S0/A0 emerge from the
superposition; the three plies supply the through-thickness structure (in-plane
scale L >> through-thickness Lz, ratio rho weights every z-derivative).

Result (8 unseen interior points, surface z=0): held-out median **0.114 -> 0.021**
with physics; every point 0.017-0.034 across u,v,w. The gauge and IC residuals
fall to ~1e-4 alongside the wave residual, so the network finds the physical
potentials. Figures: `outputs/potential/{all_holdout_uvw, plate_layout,
phys_*/loss_curves}.{png,svg}`. This is at least as good as the direct model
(option A) and is the formulation matching the data's elastodynamics.

## 6. Next steps to strengthen spatial reconstruction

1. **Source / boundary information.** The excitation enters as a source term or
   IC/BC, not the homogeneous equation. Adding the known source (location/time)
   or near-source Dirichlet data would pin the amplitude the homogeneous residual
   cannot.
2. **Stronger / scheduled physics.** Push `balance_alpha` higher with a slower
   warmup; or curriculum/causal weighting (enforce physics from early time
   outward) which is known to help wave PINNs.
3. **Per-component learned speeds unfrozen** + mild anisotropy; possibly a
   dispersive correction since Lamb waves are not single-speed.
4. **Local inductive bias.** A local interpolation prior (e.g. distance-weighted
   features) would help the dense-grid interpolation the global MLP underuses.
5. **Float64 + more collocation near held-out points**, longer training. Current
   runs are CPU-bound (MPS lacks float64 double-backward).
6. **More points / larger neighborhood** from the `.mat` for richer coverage.

---

## 7. Repo map

See `README.md`. Originals preserved untouched in `notebooks/`. Outputs are
organized as `outputs/{data_only,pinn,comparisons}/`.
