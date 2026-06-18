"""Data loading, standardization and temporal train/test construction.

The proof-of-concept trains on a temporally *partial* view of each point's
signal (a set of kept time windows) and is evaluated on the held-out gaps. The
full signal CSV is the self-contained ground truth, so the held-out gaps give an
honest reconstruction error.

Column convention throughout: ``[x, y, z, t, u, v, w]`` where ``(x, y, z)`` are mm,
``t`` is seconds and ``(u, v, w)`` are displacement components.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import torch

COORD_COLS = ["x", "y", "z", "t"]
FIELD_COLS = ["u", "v", "w"]
DT = 1e-8  # simulation timestep [s]
N_T = 6001  # timesteps per point in the full signal

# Kept time-index windows (training); the complement is held out for evaluation.
# Mirrors the windows used in the original data-acquisition notebook.
DEFAULT_TIME_RANGES: list[tuple[int, int]] = [
    (20, 280),
    (900, 1100),
    (1800, 3000),
    (4000, 5500),
    (5990, 6000),
]


def load_full_signal(csv_path: str) -> pd.DataFrame:
    """Load the full-signal CSV and attach a per-point integer timestep index.

    Args:
        csv_path: Path to the full-signal CSV with columns ``[x, y, z, t, u, v, w]``.

    Returns:
        The DataFrame sorted by ``(x, y, z, t)`` with an added integer column
        ``t_idx`` giving each row's timestep index within its spatial point.
    """
    df = pd.read_csv(csv_path)
    df = df.sort_values(["x", "y", "z", "t"]).reset_index(drop=True)
    # Recover the timestep index from t directly so it is robust to row order.
    df["t_idx"] = np.round(df["t"].to_numpy() / DT - 1).astype(int)
    return df


def time_range_mask(t_idx: np.ndarray, time_ranges: list[tuple[int, int]]) -> np.ndarray:
    """Boolean mask of rows whose timestep index falls inside any kept window.

    Args:
        t_idx: Integer timestep indices.
        time_ranges: Inclusive ``(start, end)`` index windows kept for training.

    Returns:
        Boolean array, ``True`` where ``t_idx`` is inside a kept window.
    """
    keep = np.zeros_like(t_idx, dtype=bool)
    for start, end in time_ranges:
        keep |= (t_idx >= start) & (t_idx <= end)
    return keep


@dataclass
class Standardizer:
    """Z-score standardizer fitted on training rows only (no leakage).

    A zero standard deviation (e.g. the constant ``z`` column) is replaced by 1
    so the transform is the identity for that column.
    """

    mean: torch.Tensor
    std: torch.Tensor
    columns: list[str]

    @classmethod
    def fit(cls, df: pd.DataFrame, columns: list[str]) -> "Standardizer":
        mean = torch.tensor(df[columns].mean().to_numpy(), dtype=torch.float64)
        std = torch.tensor(df[columns].std(ddof=1).to_numpy(), dtype=torch.float64)
        std = torch.where(std == 0, torch.ones_like(std), std)
        return cls(mean=mean, std=std, columns=list(columns))

    def transform(self, arr: torch.Tensor) -> torch.Tensor:
        return (arr - self.mean.to(arr)) / self.std.to(arr)

    def inverse(self, arr: torch.Tensor) -> torch.Tensor:
        return arr * self.std.to(arr) + self.mean.to(arr)


@dataclass
class WaveDataset:
    """Container for standardized tensors and the metadata needed downstream."""

    X_train: torch.Tensor
    y_train: torch.Tensor
    X_test: torch.Tensor
    y_test: torch.Tensor
    coord_scaler: Standardizer
    field_scaler: Standardizer
    points: np.ndarray  # unique (x, y, z) in physical units, shape [P, 3]
    df: pd.DataFrame
    holdout_point_indices: list[int] = field(default_factory=list)

    @property
    def n_points(self) -> int:
        return len(self.points)


def _training_keep_mask(
    df: pd.DataFrame,
    time_ranges: list[tuple[int, int]],
    split_mode: str,
    subsample_keep: float,
    seed: int,
) -> np.ndarray:
    """Boolean mask selecting the training rows under a given split mode.

    Modes:
        ``"shared"``: every point keeps the same time windows. The wide gaps then
            contain no data at any point, so filling them needs physics. Hardest.
        ``"mixed"``: half the points (group A) keep the windows; the other half
            (group B) keep the complement. The union covers all timesteps, so a
            spatially-generalizing model can reconstruct each point's held-out
            part from neighbours. Matches the original acquisition design.
        ``"subsample"``: each point keeps a random fraction of its timesteps. The
            resulting gaps are small, so a data-only model interpolates them
            well. Easiest; a sanity check that the pipeline reconstructs signals.

    Args:
        df: Full-signal frame with a ``t_idx`` column.
        time_ranges: Kept windows for the window-based modes.
        split_mode: One of ``"shared"``, ``"mixed"``, ``"subsample"``.
        subsample_keep: Fraction kept per point in ``"subsample"`` mode.
        seed: RNG seed controlling point grouping / subsampling.

    Returns:
        Boolean training mask aligned to ``df`` rows.
    """
    t_idx = df["t_idx"].to_numpy()
    in_window = time_range_mask(t_idx, time_ranges)

    if split_mode == "shared":
        return in_window

    if split_mode == "subsample":
        rng = np.random.RandomState(seed)
        return rng.rand(len(df)) < subsample_keep

    if split_mode == "mixed":
        rng = np.random.RandomState(seed)
        pts = df[["x", "y"]].drop_duplicates().to_numpy()
        order = rng.permutation(len(pts))
        n_a = len(pts) // 2
        group_a = set(map(tuple, np.round(pts[order[:n_a]], 6)))
        xy = list(map(tuple, np.round(df[["x", "y"]].to_numpy(), 6)))
        is_a = np.array([p in group_a for p in xy])
        # Group A keeps windows; group B keeps the complement.
        return (is_a & in_window) | (~is_a & ~in_window)

    raise ValueError(f"unknown split_mode: {split_mode}")


def build_dataset(
    csv_path: str,
    time_ranges: list[tuple[int, int]] | None = None,
    split_mode: str = "shared",
    subsample_keep: float = 0.5,
    seed: int = 42,
    n_holdout_points: int = 0,
    holdout_indices: list[int] | None = None,
) -> WaveDataset:
    """Construct standardized train/test tensors for temporal reconstruction.

    Training rows are selected by ``split_mode`` (see :func:`_training_keep_mask`);
    test rows are the per-row complement. Standardizers are fitted on training
    rows only.

    Spatial holdout: if ``n_holdout_points > 0``, that many entire spatial points
    are excluded from training (all their timesteps land in the test set),
    measuring generalization to never-seen locations rather than to held-out
    timesteps at seen locations.

    Args:
        csv_path: Path to the full-signal CSV.
        time_ranges: Kept training windows. Defaults to :data:`DEFAULT_TIME_RANGES`.
        split_mode: ``"shared"``, ``"mixed"`` or ``"subsample"``.
        subsample_keep: Fraction kept per point for ``"subsample"`` mode.
        seed: RNG seed for grouping/subsampling and spatial holdout.
        n_holdout_points: Number of spatial points fully excluded from training.

    Returns:
        A :class:`WaveDataset` with standardized tensors and metadata. The
        attribute ``holdout_point_indices`` lists the held-out point rows.
    """
    time_ranges = time_ranges or DEFAULT_TIME_RANGES
    df = load_full_signal(csv_path)

    keep = _training_keep_mask(df, time_ranges, split_mode, subsample_keep, seed)
    df = df.copy()

    points = df[["x", "y", "z"]].drop_duplicates().to_numpy()
    holdout_idx: list[int] = []
    if holdout_indices is not None:
        holdout_idx = sorted(int(i) for i in holdout_indices)
    elif n_holdout_points > 0:
        rng = np.random.RandomState(seed + 1)
        holdout_idx = sorted(rng.choice(len(points), n_holdout_points, replace=False).tolist())
    if holdout_idx:
        held = {tuple(np.round(points[i, :2], 6)) for i in holdout_idx}
        xy = list(map(tuple, np.round(df[["x", "y"]].to_numpy(), 6)))
        is_held_row = np.array([p in held for p in xy])
        keep = keep & ~is_held_row  # never train on held-out points

    df["is_train"] = keep
    df_train = df[keep]
    df_test = df[~keep]

    coord_scaler = Standardizer.fit(df_train, COORD_COLS)
    field_scaler = Standardizer.fit(df_train, FIELD_COLS)

    def to_tensors(frame: pd.DataFrame) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(frame[COORD_COLS].to_numpy(), dtype=torch.float64)
        y = torch.tensor(frame[FIELD_COLS].to_numpy(), dtype=torch.float64)
        return coord_scaler.transform(x), field_scaler.transform(y)

    X_train, y_train = to_tensors(df_train)
    X_test, y_test = to_tensors(df_test)

    points = df[["x", "y", "z"]].drop_duplicates().to_numpy()

    return WaveDataset(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        coord_scaler=coord_scaler,
        field_scaler=field_scaler,
        points=points,
        df=df,
        holdout_point_indices=holdout_idx,
    )


def full_grid_inputs(
    coord_scaler: Standardizer,
    point_xyz: np.ndarray,
    n_t: int = N_T,
) -> torch.Tensor:
    """Standardized input grid for the full time signal at one spatial point.

    Args:
        coord_scaler: Coordinate standardizer used during training.
        point_xyz: Physical ``(x, y, z)`` of the point.
        n_t: Number of timesteps to generate (defaults to the full record).

    Returns:
        Standardized input tensor of shape ``[n_t, 4]``.
    """
    t_full = np.arange(1, n_t + 1) * DT
    grid = np.zeros((n_t, 4), dtype=np.float64)
    grid[:, 0] = point_xyz[0]
    grid[:, 1] = point_xyz[1]
    grid[:, 2] = point_xyz[2]
    grid[:, 3] = t_full
    return coord_scaler.transform(torch.tensor(grid, dtype=torch.float64))
