#!/usr/bin/env python3
"""ANN ensemble reconstruction with unweighted training and an expanded MLP hyperparameter grid."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.interpolate import UnivariateSpline
from scipy import linalg
from sklearn.compose import TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.model_selection import GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
import warnings

try:
    from tqdm.auto import tqdm
except ModuleNotFoundError:
    tqdm = None


def parse_args() -> argparse.Namespace:
    script_dir = Path(__file__).resolve().parent
    data_dir = script_dir / "data"
    output_dir = script_dir / "output" / "mb_ann_unweighted_expandgrid_run"
    default_frame_max = 2.5
    default_frame_size = 100

    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct Pantheon+ corrected apparent magnitudes m_B(z) with a "
            "scikit-learn MLP ensemble on a supplied redshift frame, using unweighted training "
            "and an expanded hyperparameter search space."
        ),
        epilog=(
            "Parallel control: --n-jobs sets how many realizations or CV fits are run in "
            "parallel. Low-level BLAS/OpenMP threads can be limited separately with "
            "OMP_NUM_THREADS, OPENBLAS_NUM_THREADS, and MKL_NUM_THREADS."
        ),
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--data",
        type=Path,
        default=data_dir / "Pantheon+SH0ES.dat",
        help="Pantheon+SH0ES data table.",
    )
    parser.add_argument(
        "--cov",
        type=Path,
        default=data_dir / "Pantheon+SH0ES_STAT+SYS.cov",
        help="Full Pantheon+ covariance for corrected apparent magnitudes.",
    )
    parser.add_argument(
        "--frame",
        type=Path,
        default=None,
        help=(
            "Optional target redshift frame (.npy). If omitted, use 100 uniformly spaced "
            "points on [min(zHD), 2.5] after the sample selection."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=output_dir,
        help="Directory for reconstruction tables, covariance products, and best-fit CV settings.",
    )
    parser.add_argument(
        "--n-realizations",
        type=int,
        default=200,
        help="Number of Monte Carlo realizations from N(m_B, C).",
    )
    parser.add_argument(
        "--cv-folds",
        type=int,
        default=5,
        help="Number of folds in the MLP hyperparameter cross-validation search.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=4000,
        help="Maximum iterations for each MLP fit.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Number of parallel jobs for CV and ensemble fitting.",
    )
    parser.add_argument(
        "--blas-threads",
        type=int,
        default=1,
        help="BLAS/OpenMP threads used inside each worker.",
    )
    parser.add_argument(
        "--derivative-weight",
        type=float,
        default=1.0,
        help="Relative weight of dm_B/dz in the cross-validation score.",
    )
    parser.add_argument(
        "--curvature-penalty-weight",
        type=float,
        default=0.05,
        help="Weight of the high-order curvature penalty on dm_B/dz in model selection.",
    )
    parser.add_argument(
        "--distance-curvature-penalty-weight",
        type=float,
        default=0.1,
        help=(
            "Weight of the high-order curvature penalty on "
            "10^(m_B/5)/(1+z)^2 in model selection."
        ),
    )
    parser.add_argument(
        "--derivative-smoothing",
        type=float,
        default=0.2,
        help="Smoothing strength used when building derivative targets from m_B(z) samples.",
    )
    parser.add_argument(
        "--density-weight-strength",
        type=float,
        default=1.0,
        help="Additional strength applied to inverse-density weights.",
    )
    parser.add_argument(
        "--density-weight-power",
        type=float,
        default=1.0,
        help="Power-law index applied to the inverse-density weight profile.",
    )
    parser.add_argument(
        "--density-k",
        type=int,
        default=25,
        help="Neighbor rank used to estimate the local redshift spacing for inverse-density weighting.",
    )
    parser.add_argument(
        "--min-density-spacing",
        type=float,
        default=1.0e-3,
        help="Lower floor on the local spacing used in inverse-density weighting.",
    )
    args = parser.parse_args()
    args.default_frame_max = default_frame_max
    args.default_frame_size = default_frame_size
    return args


def load_pantheon_table(path: Path) -> pd.DataFrame:
    table = pd.read_csv(path, sep=r"\s+")
    required_columns = {"zHD", "m_b_corr"}
    missing = required_columns - set(table.columns)
    if missing:
        raise ValueError(f"Missing required columns in {path}: {sorted(missing)}")
    return table


def load_covariance(path: Path) -> np.ndarray:
    with path.open("r", encoding="utf-8") as handle:
        size = int(handle.readline().strip())
        values = np.loadtxt(handle)

    if values.size != size * size:
        raise ValueError(
            f"Covariance size mismatch in {path}: expected {size * size}, got {values.size}"
        )

    covariance = values.reshape(size, size)
    covariance = 0.5 * (covariance + covariance.T)
    return covariance


def stabilize_covariance(covariance: np.ndarray) -> np.ndarray:
    eigvals = np.linalg.eigvalsh(covariance)
    min_eig = eigvals.min()
    if min_eig > 0.0:
        return covariance

    jitter = abs(min_eig) + 1.0e-10
    return covariance + jitter * np.eye(covariance.shape[0])


def build_selection_mask(table: pd.DataFrame) -> np.ndarray:
    mask = np.ones(len(table), dtype=bool)
    if "IS_CALIBRATOR" in table.columns:
        mask &= table["IS_CALIBRATOR"].to_numpy() == 0
    return mask


def load_target_frame(path: Path) -> np.ndarray:
    frame = np.load(path)
    frame = np.asarray(frame, dtype=float).reshape(-1)
    frame = np.unique(np.sort(frame))
    return frame


def baseline_magnitude_shape(z_values: np.ndarray) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=float)
    z_safe = np.clip(z_values, 1.0e-8, None)
    return 5.0 * np.log10(z_safe)


def baseline_magnitude_derivative(z_values: np.ndarray) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=float)
    z_safe = np.clip(z_values, 1.0e-8, None)
    return 5.0 / (np.log(10.0) * z_safe)


def build_default_target_frame(
    z_min: float,
    z_max: float = 2.5,
    n_points: int = 100,
) -> np.ndarray:
    return np.linspace(z_min, z_max, n_points, dtype=float)


def make_inverse_density_sample_weight(
    z_values: np.ndarray,
    strength: float,
    power: float,
    k: int,
    min_spacing: float,
) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=float)
    order = np.argsort(z_values)
    z_sorted = z_values[order]
    z_unique, inverse, counts = np.unique(z_sorted, return_inverse=True, return_counts=True)

    if z_unique.size < 2:
        local_spacing_unique = np.full_like(z_unique, fill_value=min_spacing, dtype=float)
    else:
        k_index = min(max(int(k), 1), z_unique.size - 1)
        local_spacing_unique = np.empty_like(z_unique, dtype=float)
        for idx, z0 in enumerate(z_unique):
            distances = np.abs(z_unique - z0)
            local_spacing_unique[idx] = max(
                float(min_spacing),
                float(np.partition(distances, k_index)[k_index]),
            )

    spacing_sorted = local_spacing_unique[inverse]
    local_spacing = np.empty_like(spacing_sorted)
    local_spacing[order] = spacing_sorted

    reference_spacing = float(np.median(local_spacing))
    reference_spacing = max(reference_spacing, float(min_spacing))
    relative_sparsity = np.clip(local_spacing / reference_spacing, 1.0e-8, None)
    return 1.0 + strength * np.power(relative_sparsity, power)


def estimate_derivative_targets(
    z_obs: np.ndarray,
    m_values: np.ndarray,
    smoothing_strength: float,
) -> np.ndarray:
    order = np.argsort(z_obs)
    z_sorted = z_obs[order]
    m_sorted = m_values[order]

    z_unique, inverse, counts = np.unique(z_sorted, return_inverse=True, return_counts=True)
    m_unique = np.bincount(inverse, weights=m_sorted) / counts

    if z_unique.size < 4:
        derivative_unique = np.gradient(m_unique, z_unique, edge_order=1)
    else:
        smoothing = smoothing_strength * z_unique.size
        spline = UnivariateSpline(z_unique, m_unique, s=smoothing, k=min(3, z_unique.size - 1))
        derivative_unique = spline.derivative()(z_unique)

    derivative_sorted = derivative_unique[inverse]
    derivative_targets = np.empty_like(derivative_sorted)
    derivative_targets[order] = derivative_sorted
    return derivative_targets


def build_training_targets(
    z_obs: np.ndarray,
    m_values: np.ndarray,
    derivative_smoothing: float,
) -> np.ndarray:
    dm_dz = estimate_derivative_targets(
        z_obs=z_obs,
        m_values=m_values,
        smoothing_strength=derivative_smoothing,
    )
    residual_m = m_values - baseline_magnitude_shape(z_obs)
    residual_dm_dz = dm_dz - baseline_magnitude_derivative(z_obs)
    return np.column_stack([residual_m, residual_dm_dz])


def distance_shape_proxy(z_values: np.ndarray, m_values: np.ndarray) -> np.ndarray:
    z_values = np.asarray(z_values, dtype=float)
    m_values = np.asarray(m_values, dtype=float)
    return np.power(10.0, m_values / 5.0) / np.power(1.0 + z_values, 2.0)


def estimate_loss_scales(z_obs: np.ndarray, train_targets: np.ndarray) -> dict[str, float]:
    mb_scale = float(np.std(train_targets[:, 0], ddof=1))
    dmb_scale = float(np.std(train_targets[:, 1], ddof=1))
    physical_m = train_targets[:, 0] + baseline_magnitude_shape(z_obs)
    shape_values = distance_shape_proxy(z_obs, physical_m)
    shape_scale = float(np.std(shape_values, ddof=1))
    return {
        "mb_scale": max(mb_scale, 1.0e-8),
        "dmb_scale": max(dmb_scale, 1.0e-8),
        "shape_scale": max(shape_scale, 1.0e-8),
    }


def default_param_grid(max_iter: int, seed: int) -> list[dict[str, object]]:
    common_hidden = [(32, 32), (64, 32), (64, 64), (128, 64)]
    common_activation = ["tanh", "relu"]
    common_alpha = [1.0e-5, 1.0e-4, 1.0e-3, 1.0e-2]
    return [
        {
            "regressor__mlp__hidden_layer_sizes": common_hidden,
            "regressor__mlp__activation": common_activation,
            "regressor__mlp__solver": ["lbfgs"],
            "regressor__mlp__alpha": common_alpha,
            "regressor__mlp__max_iter": [max_iter],
            "regressor__mlp__random_state": [seed],
        },
        {
            "regressor__mlp__hidden_layer_sizes": common_hidden,
            "regressor__mlp__activation": common_activation,
            "regressor__mlp__solver": ["adam"],
            "regressor__mlp__alpha": common_alpha,
            "regressor__mlp__learning_rate": ["constant", "adaptive"],
            "regressor__mlp__max_iter": [max_iter],
            "regressor__mlp__random_state": [seed],
        },
    ]


def build_regressor(
    max_iter: int,
    seed: int,
    model_params: dict[str, object] | None = None,
) -> TransformedTargetRegressor:
    model_params = model_params or {}
    mlp = MLPRegressor(
        hidden_layer_sizes=model_params.get("hidden_layer_sizes", (32, 32)),
        activation=model_params.get("activation", "tanh"),
        solver=model_params.get("solver", "lbfgs"),
        alpha=model_params.get("alpha", 1.0e-3),
        learning_rate=model_params.get("learning_rate", "constant"),
        max_iter=max_iter,
        random_state=seed,
    )
    model = Pipeline(
        steps=[
            ("x_scaler", StandardScaler()),
            ("mlp", mlp),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        transformer=StandardScaler(),
    )


def extract_best_model_params(best_params: dict[str, object]) -> dict[str, object]:
    extracted = {}
    for key, value in best_params.items():
        prefix = "regressor__mlp__"
        if key.startswith(prefix):
            extracted[key.removeprefix(prefix)] = value
    return extracted


def curvature_penalty(z_values: np.ndarray, derivative_values: np.ndarray) -> float:
    z_values = np.asarray(z_values, dtype=float)
    derivative_values = np.asarray(derivative_values, dtype=float)

    z_unique, inverse, counts = np.unique(z_values, return_inverse=True, return_counts=True)
    if z_unique.size < 5:
        return 0.0

    derivative_unique = np.bincount(inverse, weights=derivative_values) / counts
    first = np.gradient(derivative_unique, z_unique, edge_order=1)
    second = np.gradient(first, z_unique, edge_order=1)
    return float(np.mean(second**2))


def make_cv_scorer(args: argparse.Namespace, loss_scales: dict[str, float]):
    def scorer(estimator, x_val, y_val) -> float:
        pred = estimator.predict(x_val)
        mb_resid = (pred[:, 0] - y_val[:, 0]) / loss_scales["mb_scale"]
        dmb_resid = (pred[:, 1] - y_val[:, 1]) / loss_scales["dmb_scale"]
        mb_mse = float(np.mean(mb_resid**2))
        dmb_mse = float(np.mean(dmb_resid**2))

        order = np.argsort(x_val[:, 0])
        z_sorted = x_val[order, 0]
        dmb_sorted = pred[order, 1] / loss_scales["dmb_scale"]
        derivative_penalty = curvature_penalty(z_sorted, dmb_sorted)
        physical_m_sorted = pred[order, 0] + baseline_magnitude_shape(z_sorted)
        shape_sorted = distance_shape_proxy(z_sorted, physical_m_sorted) / loss_scales["shape_scale"]
        distance_penalty = curvature_penalty(z_sorted, shape_sorted)

        total_loss = (
            mb_mse
            + args.derivative_weight * dmb_mse
            + args.curvature_penalty_weight * derivative_penalty
            + args.distance_curvature_penalty_weight * distance_penalty
        )
        return -total_loss

    return scorer


def cross_validate_hyperparameters(
    z_obs: np.ndarray,
    train_targets: np.ndarray,
    loss_scales: dict[str, float],
    args: argparse.Namespace,
) -> tuple[dict[str, object], float]:
    x_train = z_obs.reshape(-1, 1)
    estimator = build_regressor(max_iter=args.max_iter, seed=args.seed)
    param_grid = default_param_grid(max_iter=args.max_iter, seed=args.seed)
    cv = KFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    search = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=make_cv_scorer(args, loss_scales),
        cv=cv,
        n_jobs=args.n_jobs,
        refit=True,
    )
    with threadpool_limits(limits=args.blas_threads):
        search.fit(x_train, train_targets)
    return extract_best_model_params(search.best_params_), float(search.best_score_)


def sample_realizations(
    m_obs: np.ndarray,
    covariance: np.ndarray,
    n_realizations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chol = linalg.cholesky(stabilize_covariance(covariance), lower=True, check_finite=True)
    normals = rng.standard_normal((m_obs.size, n_realizations))
    return m_obs[:, None] + chol @ normals


def fit_single_realization(
    idx: int,
    z_obs: np.ndarray,
    m_sample: np.ndarray,
    z_frame: np.ndarray,
    args: argparse.Namespace,
    model_params: dict[str, object],
    seed: int,
) -> tuple[int, np.ndarray, np.ndarray, bool]:
    x_train = z_obs.reshape(-1, 1)
    x_frame = z_frame.reshape(-1, 1)
    train_targets = build_training_targets(
        z_obs=z_obs,
        m_values=m_sample,
        derivative_smoothing=args.derivative_smoothing,
    )

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        with threadpool_limits(limits=args.blas_threads):
            model = build_regressor(
                max_iter=args.max_iter,
                seed=seed,
                model_params=model_params,
            )
            model.fit(x_train, train_targets)
            prediction_residual = model.predict(x_frame)

    unconverged = any(issubclass(w.category, ConvergenceWarning) for w in caught)
    mb_pred = prediction_residual[:, 0] + baseline_magnitude_shape(z_frame)
    dmb_pred = prediction_residual[:, 1] + baseline_magnitude_derivative(z_frame)
    return idx, mb_pred, dmb_pred, unconverged


class SimpleProgress:
    def __init__(self, total: int, desc: str, unit: str) -> None:
        self.total = total
        self.desc = desc
        self.unit = unit
        self.count = 0
        self.step = max(1, total // 20)
        print(f"{self.desc}: 0/{self.total} {self.unit}")

    def update(self, n: int = 1) -> None:
        self.count += n
        if self.count == self.total or self.count % self.step == 0:
            print(f"{self.desc}: {self.count}/{self.total} {self.unit}")

    def __enter__(self) -> "SimpleProgress":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None


def reconstruct_mb_grid(
    z_obs: np.ndarray,
    m_samples: np.ndarray,
    z_frame: np.ndarray,
    args: argparse.Namespace,
    model_params: dict[str, object],
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mb_predictions = np.empty((m_samples.shape[1], z_frame.size), dtype=float)
    dmb_predictions = np.empty((m_samples.shape[1], z_frame.size), dtype=float)
    unconverged = np.zeros(m_samples.shape[1], dtype=bool)
    seeds = rng.integers(0, 2**31 - 1, size=m_samples.shape[1], dtype=np.int64)

    tasks = (
        delayed(fit_single_realization)(
            idx=idx,
            z_obs=z_obs,
            m_sample=m_samples[:, idx],
            z_frame=z_frame,
            args=args,
            model_params=model_params,
            seed=int(seeds[idx]),
        )
        for idx in range(m_samples.shape[1])
    )

    progress_cls = tqdm if tqdm is not None else SimpleProgress
    with progress_cls(total=m_samples.shape[1], desc="Training realizations", unit="net") as pbar:
        parallel = Parallel(n_jobs=args.n_jobs, prefer="processes", return_as="generator")
        for idx, mb_pred, dmb_pred, unconverged_flag in parallel(tasks):
            mb_predictions[idx] = mb_pred
            dmb_predictions[idx] = dmb_pred
            unconverged[idx] = unconverged_flag
            pbar.update(1)

    return mb_predictions, dmb_predictions, unconverged


def build_outputs(
    z_frame: np.ndarray,
    mb_frame_samples: np.ndarray,
    dmb_frame_samples: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    mb_mean = mb_frame_samples.mean(axis=0)
    mb_cov = np.cov(mb_frame_samples, rowvar=False, ddof=1)
    mb_sigma = np.sqrt(np.clip(np.diag(mb_cov), 0.0, None))
    dmb_mean = dmb_frame_samples.mean(axis=0)
    dmb_cov = np.cov(dmb_frame_samples, rowvar=False, ddof=1)
    dmb_sigma = np.sqrt(np.clip(np.diag(dmb_cov), 0.0, None))

    output = pd.DataFrame(
        {
            "z": z_frame,
            "m_b_mean": mb_mean,
            "m_b_sigma": mb_sigma,
            "dm_b_dz_mean": dmb_mean,
            "dm_b_dz_sigma": dmb_sigma,
        }
    )
    return output, mb_cov, dmb_cov


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    pantheon = load_pantheon_table(args.data)
    covariance_full = load_covariance(args.cov)
    if covariance_full.shape[0] != len(pantheon):
        raise ValueError(
            "Covariance dimension does not match Pantheon+SH0ES table length: "
            f"{covariance_full.shape[0]} vs {len(pantheon)}"
        )

    selection_mask = build_selection_mask(pantheon)
    selected = pantheon.loc[selection_mask].reset_index(drop=True)
    covariance = covariance_full[np.ix_(selection_mask, selection_mask)]

    z_obs = selected["zHD"].to_numpy(dtype=float)
    m_obs = selected["m_b_corr"].to_numpy(dtype=float)
    m_diag_sigma = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    train_targets = build_training_targets(
        z_obs=z_obs,
        m_values=m_obs,
        derivative_smoothing=args.derivative_smoothing,
    )
    loss_scales = estimate_loss_scales(z_obs=z_obs, train_targets=train_targets)
    if args.frame is None:
        default_frame_min = float(z_obs.min())
        z_frame = build_default_target_frame(
            z_min=default_frame_min,
            z_max=args.default_frame_max,
            n_points=args.default_frame_size,
        )
        frame_label = (
            f"generated uniform frame ({args.default_frame_size} points on "
            f"[{default_frame_min:.5f}, {args.default_frame_max:.5f}])"
        )
    else:
        z_frame = load_target_frame(args.frame)
        frame_label = str(args.frame)
    rng = np.random.default_rng(args.seed)

    print(
        "Loaded Pantheon+ table:",
        f"N_total={len(pantheon)}, N_selected={len(selected)}, "
        f"z in [{z_obs.min():.5f}, {z_obs.max():.5f}]",
    )
    print(
        "Selection counts:",
        f"excluded={len(pantheon) - len(selected)} "
        "(SH0ES calibrators removed when the column is present)",
    )
    print(
        "Target frame:",
        f"{frame_label} with {z_frame.size} redshift points in "
        f"[{z_frame.min():.5f}, {z_frame.max():.5f}]",
    )
    print(
        "Parallel settings:",
        f"n_jobs={args.n_jobs}, blas_threads={args.blas_threads}, "
        f"OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS', 'unset')}, "
        f"OPENBLAS_NUM_THREADS={os.environ.get('OPENBLAS_NUM_THREADS', 'unset')}, "
        f"MKL_NUM_THREADS={os.environ.get('MKL_NUM_THREADS', 'unset')}",
    )
    print(
        "Loss normalization scales:",
        f"residual m_B std={loss_scales['mb_scale']:.6f}, "
        f"residual dm_B/dz std={loss_scales['dmb_scale']:.6f}, "
        f"10^(m_B/5)/(1+z)^2 std={loss_scales['shape_scale']:.6f}",
    )

    print("Running cross-validation over MLP hyperparameters...")
    best_model_params, best_cv_score = cross_validate_hyperparameters(
        z_obs=z_obs,
        train_targets=train_targets,
        loss_scales=loss_scales,
        args=args,
    )
    print("Best CV settings:", best_model_params)
    print("Best CV score (neg composite loss, expanded grid):", best_cv_score)

    print(
        "Generating Monte Carlo realizations:",
        f"{args.n_realizations} samples from N(m_B, C_mB)",
    )
    m_samples = sample_realizations(
        m_obs=m_obs,
        covariance=covariance,
        n_realizations=args.n_realizations,
        rng=rng,
    )

    print("Training joint MLP ensemble for m_B and dm_B/dz and evaluating on the target frame...")
    mb_frame_samples, dmb_frame_samples, unconverged = reconstruct_mb_grid(
        z_obs=z_obs,
        m_samples=m_samples,
        z_frame=z_frame,
        args=args,
        model_params=best_model_params,
        rng=rng,
    )

    result_table, mb_cov, dmb_cov = build_outputs(
        z_frame=z_frame,
        mb_frame_samples=mb_frame_samples,
        dmb_frame_samples=dmb_frame_samples,
    )

    csv_path = args.output_dir / "pantheon_mb_ann_reconstruction.csv"
    npz_path = args.output_dir / "pantheon_mb_ann_reconstruction.npz"
    json_path = args.output_dir / "pantheon_mb_ann_best_params.json"

    result_table.to_csv(csv_path, index=False)
    np.savez_compressed(
        npz_path,
        z_frame=z_frame,
        z_obs=z_obs,
        m_b_obs=m_obs,
        m_b_diag_sigma=m_diag_sigma,
        selection_mask=selection_mask,
        m_b_realizations=m_samples,
        m_b_frame_realizations=mb_frame_samples,
        dm_b_dz_frame_realizations=dmb_frame_samples,
        m_b_cov=mb_cov,
        dm_b_dz_cov=dmb_cov,
        unconverged_mask=unconverged,
    )
    json_path.write_text(
        json.dumps(
            {
                "best_model_params": best_model_params,
                "best_cv_score_neg_composite_loss": best_cv_score,
                "variant": "unweighted_expandgrid",
                "target_transform": "residual relative to 5*log10(z) and its derivative",
                "loss_scales": loss_scales,
                "density_weight_strength": args.density_weight_strength,
                "density_weight_power": args.density_weight_power,
                "density_k": args.density_k,
                "min_density_spacing": args.min_density_spacing,
                "derivative_weight": args.derivative_weight,
                "curvature_penalty_weight": args.curvature_penalty_weight,
                "distance_curvature_penalty_weight": args.distance_curvature_penalty_weight,
                "derivative_smoothing": args.derivative_smoothing,
                "n_selected_rows": int(len(selected)),
                "n_total_rows": int(len(pantheon)),
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Saved reconstructed m_B grid to: {csv_path}")
    print(f"Saved covariance products to: {npz_path}")
    print(f"Saved best CV settings to: {json_path}")
    print(
        "Convergence summary:",
        f"{unconverged.sum()}/{args.n_realizations} realizations emitted ConvergenceWarning",
    )
    print(
        "Example output:",
        result_table.head(3).to_string(index=False),
    )


if __name__ == "__main__":
    main()
