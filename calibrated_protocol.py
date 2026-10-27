import argparse
import json
import os
import platform
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

import baseline as baselines
from garnn_temporal import GarnnTemporalBranch, masked_reconstruction_mae
from lstm_predictor import LSTMPredictor, load_lstm

WINDOW = 30
INPUT_FEATURES = 7
LABEL_HORIZON = 50
LABEL_DIMS = 2
DT = 0.1
PENETRATION_RATES = (10, 20, 30)
DEFAULT_TEST_SEEDS = (17, 18, 19, 20)


@dataclass
class CriterionResult:
    name: str
    status: str
    value: Optional[float]
    threshold: str
    notes: str


def _seed_names(seeds: Sequence[int]) -> set:
    return {f"run_seed_{seed}.npz" for seed in seeds}


def _list_seed_files(data_root: Path, penetration: int, seeds: Sequence[int]) -> List[Path]:
    scenario_dir = data_root / f"penetration_{penetration}"
    if not scenario_dir.exists():
        raise FileNotFoundError(f"Missing scenario directory: {scenario_dir}")

    valid_names = _seed_names(seeds)
    files = sorted([p for p in scenario_dir.glob("*.npz") if p.name in valid_names])
    if not files:
        raise FileNotFoundError(
            f"No seed files found in {scenario_dir} for seeds {sorted(seeds)}"
        )
    return files


def _load_npz(path: Path) -> Dict[str, np.ndarray]:
    with np.load(path, allow_pickle=True) as data:
        return {k: data[k] for k in data.files}


def _validate_shapes(x: np.ndarray, y: np.ndarray) -> None:
    if x.ndim != 3 or x.shape[1:] != (WINDOW, INPUT_FEATURES):
        raise ValueError(f"Expected X shape (N, {WINDOW}, {INPUT_FEATURES}), got {x.shape}")
    if y.ndim != 3 or y.shape[1:] != (LABEL_HORIZON, LABEL_DIMS):
        raise ValueError(f"Expected Y shape (N, {LABEL_HORIZON}, {LABEL_DIMS}), got {y.shape}")


def _euclidean_errors(pred_xy: np.ndarray, true_xy: np.ndarray) -> np.ndarray:
    return np.sqrt(np.sum((pred_xy - true_xy) ** 2, axis=2))


def compute_ade_fde(pred_xy: np.ndarray, true_xy: np.ndarray) -> Tuple[float, float]:
    dists = _euclidean_errors(pred_xy, true_xy)
    return float(np.mean(dists)), float(np.mean(dists[:, -1]))


def cv_dead_reckoning(
    x_input: np.ndarray,
    horizon_steps: int = LABEL_HORIZON,
    dt: float = DT,
    x_idx: int = 2,
    y_idx: int = 3,
) -> np.ndarray:
    if x_input.shape[1] < 2:
        raise ValueError("Need at least 2 timesteps for constant-velocity extrapolation")

    pos_now = x_input[:, -1, [x_idx, y_idx]]
    pos_prev = x_input[:, -2, [x_idx, y_idx]]
    vel = (pos_now - pos_prev) / dt

    time_offsets = np.arange(1, horizon_steps + 1, dtype=np.float32)[None, :, None] * dt
    return pos_now[:, None, :] + vel[:, None, :] * time_offsets


def _predict_lstm_physical(model: torch.nn.Module, x_physical: np.ndarray, device: torch.device) -> np.ndarray:
    sx_m = model.scaler_X_mean.detach().cpu().numpy()
    sx_s = model.scaler_X_std.detach().cpu().numpy()
    sy_m = model.scaler_Y_mean.detach().cpu().numpy()
    sy_s = model.scaler_Y_std.detach().cpu().numpy()

    x_clean = np.nan_to_num(x_physical.astype(np.float32), nan=0.0)
    x_norm = (x_clean - sx_m) / sx_s

    preds: List[np.ndarray] = []
    batch_size = 1024
    with torch.no_grad():
        for start in range(0, x_norm.shape[0], batch_size):
            bx = torch.from_numpy(x_norm[start : start + batch_size]).to(device)
            pred_norm = model(bx).detach().cpu().numpy()
            preds.append(pred_norm * sy_s + sy_m)
    return np.concatenate(preds, axis=0)


def _required_keys_for_conditions() -> Dict[str, Sequence[str]]:
    return {
        "Baseline 1: Zero-fill": ("X_corrupted",),
        "Baseline 2: Forward-fill": ("X_corrupted",),
        "Baseline 3: Mean-fill": ("X_corrupted",),
        "Baseline 4: Linear": ("X_corrupted",),
        "Baseline 5: Spline": ("X_corrupted",),
        "Condition B: Channel-1": ("X_ch1_imputed",),
        "Condition C: Dual-channel": ("X_fully_imputed",),
    }


def _build_condition_arrays(npz_data: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    x_corr = npz_data["X_corrupted"].astype(np.float32)
    x_ch1 = npz_data.get("X_ch1_imputed")
    x_dual = npz_data.get("X_fully_imputed")

    conditions = {
        "Baseline 1: Zero-fill": baselines.zero_fill(x_corr),
        "Baseline 2: Forward-fill": baselines.forward_fill(x_corr),
        "Baseline 3: Mean-fill": baselines.mean_fill(x_corr),
        "Baseline 4: Linear": baselines.linear_interp(x_corr),
        "Baseline 5: Spline": baselines.spline_interp(x_corr),
    }

    if x_ch1 is not None:
        conditions["Condition B: Channel-1"] = x_ch1.astype(np.float32)
    if x_dual is not None:
        conditions["Condition C: Dual-channel"] = x_dual.astype(np.float32)

    return conditions


def run_cv_baseline_suite(
    data_root: Path,
    seeds: Sequence[int],
    output_dir: Path,
) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []

    for pen in PENETRATION_RATES:
        seed_files = _list_seed_files(data_root, pen, seeds)
        per_seed_by_condition: Dict[str, List[Tuple[float, float]]] = {}

        for seed_path in seed_files:
            data = _load_npz(seed_path)
            if "Y" not in data:
                raise KeyError(f"Missing Y in {seed_path}")

            y = data["Y"].astype(np.float32)
            conditions = _build_condition_arrays(data)

            for condition_name, x_input in conditions.items():
                _validate_shapes(x_input, y)
                pred = cv_dead_reckoning(np.nan_to_num(x_input, nan=0.0))
                ade, fde = compute_ade_fde(pred, y)
                per_seed_by_condition.setdefault(condition_name, []).append((ade, fde))

        for condition_name, metrics in sorted(per_seed_by_condition.items()):
            ade_values = np.array([m[0] for m in metrics], dtype=np.float64)
            fde_values = np.array([m[1] for m in metrics], dtype=np.float64)
            rows.append(
                {
                    "penetration_rate": pen,
                    "condition": condition_name,
                    "seed_count": int(len(metrics)),
                    "ade_mean_m": float(np.mean(ade_values)),
                    "ade_seed_sd_m": float(np.std(ade_values, ddof=0)),
                    "fde_mean_m": float(np.mean(fde_values)),
                    "fde_seed_sd_m": float(np.std(fde_values, ddof=0)),
                }
            )

    df = pd.DataFrame(rows)
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "cv_baseline_suite.csv", index=False)
    return df


def _denormalisation_roundtrip_ok(model: torch.nn.Module) -> bool:
    y_mean = model.scaler_Y_mean.detach().cpu().numpy().astype(np.float32)
    y_std = model.scaler_Y_std.detach().cpu().numpy().astype(np.float32)
    probe = np.array([[1.5, -2.0]], dtype=np.float32)
    denorm = probe * y_std + y_mean
    renorm = (denorm - y_mean) / y_std
    return bool(np.allclose(probe, renorm, atol=1e-5))


def _xy_ordering_check(x_clean: np.ndarray, y_true: np.ndarray) -> Dict[str, object]:
    # Compare continuity error for expected ordering [x_idx=2, y_idx=3] vs swapped.
    expected = np.mean(np.abs(y_true[:, 0, 0] - x_clean[:, -1, 2]) + np.abs(y_true[:, 0, 1] - x_clean[:, -1, 3]))
    swapped = np.mean(np.abs(y_true[:, 0, 0] - x_clean[:, -1, 3]) + np.abs(y_true[:, 0, 1] - x_clean[:, -1, 2]))
    return {
        "expected_continuity_error": float(expected),
        "swapped_continuity_error": float(swapped),
        "passed": bool(expected <= swapped),
    }


def characterize_lstm_on_clean_input(
    model: torch.nn.Module,
    device: torch.device,
    clean_x: np.ndarray,
    clean_y: np.ndarray,
    output_dir: Path,
    random_seed: int,
) -> Dict[str, object]:
    _validate_shapes(clean_x, clean_y)

    preds = _predict_lstm_physical(model, clean_x, device)
    dists = _euclidean_errors(preds, clean_y)

    checkpoints = {
        "0.1s": 0,
        "0.5s": 4,
        "1.0s": 9,
        "2.0s": 19,
        "5.0s": 49,
    }
    per_step = {t: float(np.mean(dists[:, idx])) for t, idx in checkpoints.items()}

    output_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(random_seed)
    sample_count = min(20, clean_x.shape[0])
    indices = rng.choice(clean_x.shape[0], size=sample_count, replace=False)
    overlay_path = output_dir / "clean_lstm_overlays.png"

    fig, axes = plt.subplots(4, 5, figsize=(16, 12), constrained_layout=True)
    axes_flat = axes.flatten()
    for plot_idx, sample_idx in enumerate(indices):
        ax = axes_flat[plot_idx]
        ax.plot(clean_y[sample_idx, :, 0], clean_y[sample_idx, :, 1], label="true", lw=1.5)
        ax.plot(preds[sample_idx, :, 0], preds[sample_idx, :, 1], label="pred", lw=1.2, linestyle="--")
        ax.set_title(f"window {sample_idx}")
    if sample_count > 0:
        axes_flat[0].legend(loc="best")
    for k in range(sample_count, len(axes_flat)):
        axes_flat[k].axis("off")
    fig.savefig(overlay_path, dpi=200)
    plt.close(fig)

    xy_check = _xy_ordering_check(clean_x, clean_y)

    return {
        "per_step_ade_m": per_step,
        "denormalisation_roundtrip_ok": _denormalisation_roundtrip_ok(model),
        "xy_ordering_check": xy_check,
        "overlay_plot": str(overlay_path),
        "overall_clean_ade_m": float(np.mean(dists)),
        "overall_clean_fde_m": float(np.mean(dists[:, -1])),
        "preds": preds,
    }


def evaluate_calibration_criteria(
    clean_y: np.ndarray,
    lstm_preds: np.ndarray,
    clean_x: np.ndarray,
    lane_width_m: float,
) -> Tuple[List[CriterionResult], str, Dict[str, object]]:
    dists = _euclidean_errors(lstm_preds, clean_y)
    clean_ade_01 = float(np.mean(dists[:, 0]))

    criteria: List[CriterionResult] = []
    criteria.append(
        CriterionResult(
            name="clean_input_ade_at_0.1s",
            status="pass" if clean_ade_01 <= 0.5 else "fail",
            value=clean_ade_01,
            threshold="<= 0.5 m",
            notes="Computed on clean GT windows.",
        )
    )

    lane_change_mask = np.abs(clean_y[:, -1, 1] - clean_y[:, 0, 1]) > (lane_width_m / 2.0)
    lane_n = int(np.sum(lane_change_mask))

    if lane_n < 200:
        criteria.append(
            CriterionResult(
                name="lateral_endpoint_sigma_ratio",
                status="inconclusive",
                value=None,
                threshold=">= 30% with n>=200",
                notes=f"lane_change_n={lane_n} < 200",
            )
        )
        sigma_ratio = None
    else:
        pred_sigma = float(np.std(lstm_preds[lane_change_mask, -1, 1]))
        true_sigma = float(np.std(clean_y[lane_change_mask, -1, 1]))
        if true_sigma <= 1e-9:
            sigma_ratio = None
            sigma_status = "inconclusive"
            sigma_note = "Ground-truth endpoint sigma is near zero."
        else:
            sigma_ratio = (pred_sigma / true_sigma) * 100.0
            sigma_status = "pass" if sigma_ratio >= 30.0 else "fail"
            sigma_note = "Computed on lane-change subset."
        criteria.append(
            CriterionResult(
                name="lateral_endpoint_sigma_ratio",
                status=sigma_status,
                value=sigma_ratio,
                threshold=">= 30% with n>=200",
                notes=f"lane_change_n={lane_n}; {sigma_note}",
            )
        )

    cv_preds = cv_dead_reckoning(clean_x)
    cv_ade, _ = compute_ade_fde(cv_preds, clean_y)
    lstm_ade, _ = compute_ade_fde(lstm_preds, clean_y)
    delta = lstm_ade - cv_ade
    cv_status = "fail" if delta > 5.0 else "pass"
    criteria.append(
        CriterionResult(
            name="clean_cv_not_beating_lstm_by_more_than_5m",
            status=cv_status,
            value=float(delta),
            threshold="(LSTM_ADE - CV_ADE) <= 5.0 m",
            notes=f"CV_ADE={cv_ade:.3f}m, LSTM_ADE={lstm_ade:.3f}m",
        )
    )

    determinate = [c for c in criteria if c.status in {"pass", "fail"}]
    failed = [c for c in determinate if c.status == "fail"]
    verdict = "branch_b_required" if failed else "calibration_pass"

    aux = {
        "lane_change_n": lane_n,
        "sigma_ratio_percent": sigma_ratio,
        "clean_cv_ade_m": cv_ade,
        "clean_lstm_ade_m": lstm_ade,
    }
    return criteria, verdict, aux


def _spline_ego_with_neighbour_source(
    x_corr: np.ndarray,
    neighbour_source: np.ndarray,
) -> np.ndarray:
    out = x_corr.copy().astype(np.float32)
    spline_full = baselines.spline_interp(x_corr)
    out[:, :, 0:4] = spline_full[:, :, 0:4]
    out[:, :, 4:7] = neighbour_source[:, :, 4:7]
    return out


def build_matched_control_table(npz_data: Dict[str, np.ndarray]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    x_corr = npz_data["X_corrupted"].astype(np.float32)
    x_ch1 = npz_data.get("X_ch1_imputed")
    if x_ch1 is None:
        raise KeyError("X_ch1_imputed is required for matched-control construction")

    spline_ego_ch1_neighbour = _spline_ego_with_neighbour_source(x_corr, x_ch1.astype(np.float32))
    spline_ego_zero_neighbour = _spline_ego_with_neighbour_source(
        x_corr,
        np.concatenate([x_corr[:, :, 0:4], np.zeros_like(x_corr[:, :, 4:7])], axis=2),
    )

    conditions = pd.DataFrame(
        [
            {
                "condition": "spline_ego_plus_ch1_neighbour",
                "ego_imputation": "spline",
                "neighbour_imputation": "channel1",
                "x": spline_ego_ch1_neighbour,
            },
            {
                "condition": "spline_ego_plus_zero_neighbour",
                "ego_imputation": "spline",
                "neighbour_imputation": "zero",
                "x": spline_ego_zero_neighbour,
            },
        ]
    )

    matched_factors = pd.DataFrame(
        [
            {
                "comparator": "spline_ego_plus_ch1_neighbour",
                "candidate": "spline_ego_plus_zero_neighbour",
                "fixed_factors": "dataset split, seeds, LSTM checkpoint, normalization, ego imputation=spline, metrics (ADE/FDE)",
                "difference": "neighbour imputation only",
                "exactly_one_difference": True,
            }
        ]
    )

    return conditions, matched_factors


def run_week7_garnn_checkpoint(
    npz_data: Dict[str, np.ndarray],
    checkpoint_path: Optional[Path],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    required = ["X_gt", "X_corrupted", "X_mask"]
    for key in required:
        if key not in npz_data:
            raise KeyError(f"Missing {key} for GARNN reconstruction")

    x_gt = npz_data["X_gt"].astype(np.float32)
    x_corr = np.nan_to_num(npz_data["X_corrupted"].astype(np.float32), nan=0.0)
    x_mask = npz_data["X_mask"].astype(np.float32)

    model = GarnnTemporalBranch(feature_dim=x_gt.shape[2], hidden_dim=64).to(device)
    if checkpoint_path is not None and checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))

    model.eval()
    with torch.no_grad():
        inp = torch.from_numpy(x_corr).to(device)
        msk = torch.from_numpy(x_mask).to(device)
        tgt = torch.from_numpy(x_gt).to(device)
        recon = model(inp, msk)
        mae = masked_reconstruction_mae(recon, tgt, msk).item()

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics = {
        "mask_condition": "single_selected_npz",
        "untuned_masked_reconstruction_mae": float(mae),
        "checkpoint_used": str(checkpoint_path) if checkpoint_path and checkpoint_path.exists() else None,
    }
    with open(output_dir / "week7_garnn_reconstruction.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    return metrics


def run_branch_b_diagnosis(
    model: torch.nn.Module,
    clean_x: np.ndarray,
    clean_y: np.ndarray,
    data_root: Path,
    train_seeds: Sequence[int],
    device: torch.device,
    output_dir: Path,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1) Normalization audit
    seed_paths: List[Path] = []
    for pen in PENETRATION_RATES:
        try:
            seed_paths.extend(_list_seed_files(data_root, pen, train_seeds))
        except FileNotFoundError:
            continue

    norm_audit = {"status": "inconclusive", "reason": "insufficient training files"}
    if seed_paths:
        xs = []
        ys = []
        for p in seed_paths:
            d = _load_npz(p)
            x_key = "X_gt" if "X_gt" in d else "X"
            if x_key in d and "Y" in d:
                xs.append(d[x_key].astype(np.float32))
                ys.append(d["Y"].astype(np.float32))
        if xs and ys:
            x_all = np.concatenate(xs, axis=0)
            y_all = np.concatenate(ys, axis=0)
            calc_x_mean = np.nanmean(x_all, axis=(0, 1))
            calc_x_std = np.nanstd(x_all, axis=(0, 1))
            calc_y_mean = np.mean(y_all, axis=(0, 1))
            calc_y_std = np.std(y_all, axis=(0, 1))

            ck_x_mean = model.scaler_X_mean.detach().cpu().numpy()
            ck_x_std = model.scaler_X_std.detach().cpu().numpy()
            ck_y_mean = model.scaler_Y_mean.detach().cpu().numpy()
            ck_y_std = model.scaler_Y_std.detach().cpu().numpy()

            norm_audit = {
                "status": "completed",
                "x_mean_linf": float(np.max(np.abs(calc_x_mean - ck_x_mean))),
                "x_std_linf": float(np.max(np.abs(calc_x_std - ck_x_std))),
                "y_mean_linf": float(np.max(np.abs(calc_y_mean - ck_y_mean))),
                "y_std_linf": float(np.max(np.abs(calc_y_std - ck_y_std))),
            }

    # 2) x/y ordering check
    xy_check = _xy_ordering_check(clean_x, clean_y)

    # 3) Overfit 100-window memorization check
    n_overfit = min(100, clean_x.shape[0])
    overfit = {"status": "inconclusive", "reason": "not enough windows"}
    if n_overfit >= 10:
        x_small = clean_x[:n_overfit]
        y_small = clean_y[:n_overfit]

        sx_m = model.scaler_X_mean.detach().cpu().numpy()
        sx_s = model.scaler_X_std.detach().cpu().numpy()
        sy_m = model.scaler_Y_mean.detach().cpu().numpy()
        sy_s = model.scaler_Y_std.detach().cpu().numpy()

        x_norm = (np.nan_to_num(x_small, nan=0.0) - sx_m) / sx_s
        y_norm = (y_small - sy_m) / sy_s

        x_t = torch.from_numpy(x_norm.astype(np.float32)).to(device)
        y_t = torch.from_numpy(y_norm.astype(np.float32)).to(device)

        probe = LSTMPredictor(
            input_dim=7,
            hidden_dim=128,
            num_layers=2,
            scaler_X_mean=torch.from_numpy(sx_m.astype(np.float32)),
            scaler_X_std=torch.from_numpy(sx_s.astype(np.float32)),
            scaler_Y_mean=torch.from_numpy(sy_m.astype(np.float32)),
            scaler_Y_std=torch.from_numpy(sy_s.astype(np.float32)),
        ).to(device)
        probe.train()
        opt = torch.optim.Adam(probe.parameters(), lr=1e-3)
        mse = torch.nn.MSELoss()

        for _ in range(30):
            opt.zero_grad()
            pred = probe(x_t)
            loss = mse(pred, y_t)
            loss.backward()
            opt.step()

        probe.eval()
        with torch.no_grad():
            pred_norm = probe(x_t).detach().cpu().numpy()
            pred_phys = pred_norm * sy_s + sy_m
        ade, fde = compute_ade_fde(pred_phys, y_small)
        overfit = {
            "status": "completed",
            "window_count": n_overfit,
            "post_overfit_ade_m": float(ade),
            "post_overfit_fde_m": float(fde),
            "likely_memorised": bool(ade < 1.0),
        }

    repair_spec = {
        "normalization_fix": "Recompute training-only scalers and ensure train/val/test leakage is impossible.",
        "xy_order_fix": "Enforce explicit [x,y] index mapping checks during dataset load and inference.",
        "model_fit_fix": "If overfit test fails, inspect label alignment and optimiser setup before full retraining.",
    }

    diagnosis = {
        "normalization_audit": norm_audit,
        "xy_ordering_check": xy_check,
        "overfit_100_window": overfit,
        "repair_specification": repair_spec,
    }

    with open(output_dir / "branch_b_diagnosis.json", "w", encoding="utf-8") as f:
        json.dump(diagnosis, f, indent=2)
    return diagnosis


def _write_manifest(output_dir: Path, args: argparse.Namespace, artifacts: Dict[str, object]) -> None:
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "torch": torch.__version__,
        "config": {
            "window": WINDOW,
            "input_shape": [WINDOW, INPUT_FEATURES],
            "label_shape": [LABEL_HORIZON, LABEL_DIMS],
            "sampling_hz": 10,
            "penetration_rates": list(PENETRATION_RATES),
            "test_seeds": list(args.test_seeds),
            "train_seeds": list(args.train_seeds),
            "lane_change_min_n": 200,
            "lane_width_m": args.lane_width_m,
        },
        "input_artifacts": artifacts,
    }
    with open(output_dir / "run_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _collect_clean_windows(data_root: Path, penetration: int, seeds: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    x_list = []
    y_list = []
    for p in _list_seed_files(data_root, penetration, seeds):
        d = _load_npz(p)
        x_key = "X_gt" if "X_gt" in d else "X"
        if x_key not in d or "Y" not in d:
            raise KeyError(f"Missing clean arrays in {p}")
        x_list.append(d[x_key].astype(np.float32))
        y_list.append(d["Y"].astype(np.float32))

    x_all = np.concatenate(x_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    _validate_shapes(x_all, y_all)
    return x_all, y_all


def _write_calibration_report(
    output_dir: Path,
    criteria: Sequence[CriterionResult],
    verdict: str,
    aux: Dict[str, object],
    char_summary: Dict[str, object],
    branch_b: Optional[Dict[str, object]],
) -> None:
    payload = {
        "verdict": verdict,
        "criteria": [c.__dict__ for c in criteria],
        "auxiliary": aux,
        "characterization": {
            "per_step_ade_m": char_summary["per_step_ade_m"],
            "denormalisation_roundtrip_ok": char_summary["denormalisation_roundtrip_ok"],
            "xy_ordering_check": char_summary["xy_ordering_check"],
            "overlay_plot": char_summary["overlay_plot"],
        },
        "branch_b": branch_b,
    }
    with open(output_dir / "calibration_report.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    lines = [
        "# Calibration Report",
        "",
        f"Verdict: **{verdict}**",
        "",
        "## Pre-registered criteria",
    ]
    for c in criteria:
        value_str = "n/a" if c.value is None else f"{c.value:.6f}"
        lines.append(f"- `{c.name}`: **{c.status}** (value={value_str}, threshold={c.threshold}) — {c.notes}")

    lines += [
        "",
        "## Characterization",
        f"- Per-step ADE (m): {char_summary['per_step_ade_m']}",
        f"- Denormalisation roundtrip check: {char_summary['denormalisation_roundtrip_ok']}",
        f"- X/Y ordering check: {char_summary['xy_ordering_check']}",
        f"- Overlay plot: `{char_summary['overlay_plot']}`",
        "",
        "## Auxiliary",
        f"- Lane-change n: {aux.get('lane_change_n')}",
        f"- Sigma ratio (%): {aux.get('sigma_ratio_percent')}",
        f"- Clean CV ADE (m): {aux.get('clean_cv_ade_m')}",
        f"- Clean LSTM ADE (m): {aux.get('clean_lstm_ade_m')}",
    ]

    if branch_b is not None:
        lines += [
            "",
            "## Branch B diagnosis",
            "Branch B triggered due to at least one determinate criterion failure.",
            f"- Normalization audit: {branch_b.get('normalization_audit')}",
            f"- X/Y ordering check: {branch_b.get('xy_ordering_check')}",
            f"- 100-window overfit test: {branch_b.get('overfit_100_window')}",
            f"- Repair specification: {branch_b.get('repair_specification')}",
        ]

    with open(output_dir / "calibration_report.md", "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _write_argument_and_limitations(output_dir: Path) -> None:
    docs_dir = output_dir / "docs"
    docs_dir.mkdir(parents=True, exist_ok=True)

    argument = """# Why graph attention is not applicable under SFM

Under Structural Feature Missingness (SFM), the neighbour-speed columns are absent because they were never observed by any node in the first place; they are not noisy observations, they are structurally unobserved values. In the GARNN formulation, Eqs. (12)-(14) couple temporal states with a spatial regulariser through neighbour messages. That regulariser assumes at least one observed anchor exists in the graph so the latent state can be pulled toward a physically grounded target. With SFM this anchor is missing: for the affected feature column, every candidate neighbour message is also unobserved, so attention weights can only redistribute uncertainty, not introduce evidence.

Concretely, if we write the spatial update as a weighted sum of neighbour embeddings and then apply a residual correction (the pattern used in Eqs. 12-14), the weighted sum has no observed component for that column under SFM. The residual branch then becomes self-referential and may appear numerically stable but is not identifiable from data. This is why the implemented Week-7 baseline keeps only the forward temporal branch (value/mask split, fusion, decay gate, residual estimate) and excludes graph attention. The temporal model can still leverage each node's observed history and missingness mask, while avoiding a spatial term that has no empirical anchor for the missing feature.
"""

    limitations = """# What we could not verify

- Full multi-node graph-attention branch: omitted because SFM leaves no observed anchor for the affected feature column, making attention-based spatial regularisation unidentifiable in this dataset setting.
- Tuned GARNN training recipe: not verified here because no dedicated GARNN checkpoint/training manifest was provided in-repo.
- End-to-end E1 secondary row with GARNN: reported only if calibration passes and required artifacts exist; otherwise left unreported to avoid fabricated numbers.
"""

    with open(docs_dir / "sfm_graph_attention_argument.md", "w", encoding="utf-8") as f:
        f.write(argument)
    with open(docs_dir / "what_we_could_not_verify.md", "w", encoding="utf-8") as f:
        f.write(limitations)


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Calibrated baseline protocol and matched-control ablation runner")
    p.add_argument("--data-root", type=Path, default=Path("data_corrupted"))
    p.add_argument("--checkpoint", type=Path, default=Path("lstm_baseline.pt"))
    p.add_argument("--output-dir", type=Path, default=Path("reports/calibrated_protocol"))
    p.add_argument("--garnn-checkpoint", type=Path, default=None)
    p.add_argument("--test-seeds", type=int, nargs="+", default=list(DEFAULT_TEST_SEEDS))
    p.add_argument("--train-seeds", type=int, nargs="+", default=list(range(1, 13)))
    p.add_argument("--clean-penetration", type=int, default=20, choices=list(PENETRATION_RATES))
    p.add_argument("--lane-width-m", type=float, default=3.7)
    p.add_argument("--random-seed", type=int, default=123)
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Missing LSTM checkpoint: {args.checkpoint}")

    model = load_lstm(str(args.checkpoint), device=str(device))

    cv_df = run_cv_baseline_suite(args.data_root, args.test_seeds, args.output_dir)

    clean_x, clean_y = _collect_clean_windows(args.data_root, args.clean_penetration, args.test_seeds)
    char = characterize_lstm_on_clean_input(
        model=model,
        device=device,
        clean_x=clean_x,
        clean_y=clean_y,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )

    criteria, verdict, aux = evaluate_calibration_criteria(
        clean_y=clean_y,
        lstm_preds=char["preds"],
        clean_x=clean_x,
        lane_width_m=args.lane_width_m,
    )

    branch_b = None
    if verdict == "branch_b_required":
        branch_b = run_branch_b_diagnosis(
            model=model,
            clean_x=clean_x,
            clean_y=clean_y,
            data_root=args.data_root,
            train_seeds=args.train_seeds,
            device=device,
            output_dir=args.output_dir,
        )

    # Matched-control table from first available clean-penetration file.
    sample_file = _list_seed_files(args.data_root, args.clean_penetration, args.test_seeds)[0]
    sample_data = _load_npz(sample_file)
    matched_conditions, matched_pairs = build_matched_control_table(sample_data)
    matched_pairs.to_csv(args.output_dir / "table3_matched_control_factors.csv", index=False)

    # Optional: compute secondary E1 row only when calibration passes.
    if verdict == "calibration_pass":
        rows = []
        y = sample_data["Y"].astype(np.float32)
        for _, row in matched_conditions.iterrows():
            x = np.nan_to_num(row["x"], nan=0.0)
            pred = _predict_lstm_physical(model, x, device)
            ade, fde = compute_ade_fde(pred, y)
            rows.append({"condition": row["condition"], "ade_m": ade, "fde_m": fde})
        pd.DataFrame(rows).to_csv(args.output_dir / "table3_matched_control_metrics.csv", index=False)

    # Week-7 checkpoint support for GARNN temporal branch.
    if all(k in sample_data for k in ["X_gt", "X_corrupted", "X_mask"]):
        run_week7_garnn_checkpoint(
            sample_data,
            checkpoint_path=args.garnn_checkpoint,
            device=device,
            output_dir=args.output_dir,
        )

    _write_calibration_report(args.output_dir, criteria, verdict, aux, char, branch_b)
    _write_argument_and_limitations(args.output_dir)

    artifacts = {
        "data_root": str(args.data_root),
        "checkpoint": str(args.checkpoint),
        "sample_file": str(sample_file),
        "cv_rows": int(len(cv_df)),
    }
    _write_manifest(args.output_dir, args, artifacts)

    # Remove large in-memory predictions from summary path.
    char.pop("preds", None)

    summary = {
        "status": "ok",
        "verdict": verdict,
        "output_dir": str(args.output_dir),
    }
    with open(args.output_dir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
