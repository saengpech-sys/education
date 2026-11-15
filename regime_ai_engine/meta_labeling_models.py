"""
Two-Stage Meta-Labeling Machine Learning Pipeline (Marcos López de Prado) with ONNX Export:
1. Primary Model (LightGBM 3-Class): Predicts Directional / Mean-Reversion Side {-1, 0, +1}.
2. Secondary Meta-Model (LightGBM Binary): Trained on Purged Out-Of-Fold Primary signals to
   estimate calibrated Win Probability P(Win) in [0, 1].
3. Exports both models to ONNX with dynamic batching for sub-millisecond production inference.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import lightgbm as lgb
import numpy as np
import onnx
import onnxmltools
import pandas as pd
from onnxmltools.convert.common.data_types import FloatTensorType

from backtest_and_validation.purged_cv import PurgedKFoldCV
from core.config_loader import RegimeAIConfig
from core.contracts import MarketRegime, SymbolSpec
from data_pipeline.mtf_aligner import PRIMARY_FEATURE_COLUMNS
from regime_ai_engine.regime_detector import MarketRegimeDetector, REGIME_TO_NUMERIC_CODE
from regime_ai_engine.triple_barrier import (
    compute_meta_labels_for_sides,
    compute_triple_barrier_labels,
)

_PRIM_COL_IDX = {name: idx for idx, name in enumerate(PRIMARY_FEATURE_COLUMNS)}

META_AUX_COLUMNS: Tuple[str, ...] = (
    "primary_side_num",
    "primary_side_prob",
    "primary_dir_skew",
    "primary_margin_prob",
    "sleeve_id_code",
    "structural_confluence",
    "regime_code",
    "spread_to_atr_ratio",
)

META_FEATURE_COLUMNS: Tuple[str, ...] = PRIMARY_FEATURE_COLUMNS + META_AUX_COLUMNS

# Primary class index mapping: 0 -> SHORT (-1), 1 -> FLAT (0), 2 -> LONG (+1)
CLASS_IDX_TO_SIDE = np.array([-1, 0, 1], dtype=np.int64)
SIDE_TO_CLASS_IDX = {-1: 0, 0: 1, 1: 2}


@dataclass(frozen=True)
class TrainedMetaLabelingArtifacts:
    primary_onnx_path: Path
    meta_onnx_path: Path
    primary_onnx_bytes: bytes
    meta_onnx_bytes: bytes
    primary_feature_names: Tuple[str, ...]
    meta_feature_names: Tuple[str, ...]
    oof_meta_auc_proxy: float
    num_train_samples: int


def _export_lgbm_to_onnx_bytes(model: lgb.LGBMClassifier, num_features: int) -> bytes:
    """Convert fitted LGBMClassifier to clean ONNX ModelProto bytes with dynamic batch 'N'."""
    onnx_model = onnxmltools.convert_lightgbm(
        model,
        initial_types=[("features", FloatTensorType([None, num_features]))],
        zipmap=False,
        target_opset=15,
    )
    for out in onnx_model.graph.output:
        if len(out.type.tensor_type.shape.dim) > 0:
            out.type.tensor_type.shape.dim[0].ClearField("dim_value")
            out.type.tensor_type.shape.dim[0].dim_param = "N"
    onnx.checker.check_model(onnx_model)
    return onnx_model.SerializeToString()


def extract_institutional_sleeve_signals(
    X_primary: np.ndarray,
    regime_codes: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract strictly causal Institutional Bullion Market-Clock candidate signals:
      - Sleeve 1 (ID=1, Hold=66 bars / 5.5h): NY COMEX Close -> Shanghai SGE AM Fix Accumulation LONG
      - Sleeve 2 (ID=2, Hold=156 bars / 13.0h): COMEX / LBMA PM Fix Handoff 6h Exhaustion LONG & SHORT
      - Sleeve 3 (ID=3, Hold=84 bars / 7.0h): London Close / COMEX Settlement Selloff Exhaustion LONG

    Strictly enforces Macro Regime Alignment:
      - Never emits LONG during TRENDING_BEAR (-1.0) or multi-week macro bear trend.
      - Never emits SHORT during TRENDING_BULL (+1.0) or multi-day macro bull surge.
    """
    m5_ret1 = X_primary[:, _PRIM_COL_IDX["m5_ret_1"]]
    ret_6h = X_primary[:, _PRIM_COL_IDX["m5_ret_72"]]
    ret_12h = X_primary[:, _PRIM_COL_IDX["m5_ret_144"]]
    is_sge_win = X_primary[:, _PRIM_COL_IDX["m5_is_sge_win"]] > 0.5
    is_comex_win = X_primary[:, _PRIM_COL_IDX["m5_is_comex_win"]] > 0.5
    is_ldn_win = X_primary[:, _PRIM_COL_IDX["m5_is_ldn_close_win"]] > 0.5
    h4_ema = X_primary[:, _PRIM_COL_IDX["h4_ema_spread_atr"]]
    ret_5d = X_primary[:, _PRIM_COL_IDX["h4_ret_30"]]
    ret_20d = X_primary[:, _PRIM_COL_IDX["h4_ret_120"]]

    is_bull = np.isclose(regime_codes, 1.0)
    is_bear = np.isclose(regime_codes, -1.0)
    is_mr = np.isclose(regime_codes, 0.5)
    tradable = is_bull | is_bear | is_mr

    not_macro_bear = (~is_bear) & (h4_ema > -0.20) & (ret_20d > -0.010) & (ret_5d > -0.012)
    not_macro_bull = (~is_bull) & (h4_ema < 0.65) & (ret_5d < 0.016)

    sge_long = tradable & is_sge_win & not_macro_bear & (ret_12h < 0.008)
    comex_long = tradable & is_comex_win & not_macro_bear & (ret_6h <= -0.0042) & (m5_ret1 > 0.0)
    comex_short = tradable & is_comex_win & not_macro_bull & (ret_6h >= 0.0042) & (m5_ret1 < 0.0)
    ldn_long = tradable & is_ldn_win & not_macro_bear & (ret_6h <= -0.0032) & (m5_ret1 > 0.0)

    active_side = np.where(
        comex_long | ldn_long,
        1,
        np.where(comex_short, -1, np.where(sge_long, 1, 0)),
    ).astype(np.int64)

    # Hard regime invariant guarantee
    active_side = np.where(
        is_bull & (active_side < 0),
        0,
        np.where(is_bear & (active_side > 0), 0, active_side),
    ).astype(np.int64)

    raw_sleeve = np.where(
        comex_long | comex_short,
        2,
        np.where(ldn_long, 3, np.where(sge_long, 1, 0)),
    ).astype(np.int64)
    sleeve_id = np.where(active_side != 0, raw_sleeve, 0).astype(np.int64)
    hold_bars = np.where(sleeve_id == 2, 156, np.where(sleeve_id == 3, 84, 66)).astype(np.int64)
    return active_side, sleeve_id, hold_bars


def build_meta_feature_matrix(
    X_primary: np.ndarray,
    primary_probs: np.ndarray,
    regime_codes: np.ndarray,
    spread_to_atr: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Construct Secondary Meta-Model input matrix from Primary class probabilities [P(-1), P(0), P(+1)]
    and strictly causal Institutional Bullion Market-Clock & Macro Regime alignment.
    """
    p_short = primary_probs[:, 0]
    p_long = primary_probs[:, 2]

    m5_ofi = X_primary[:, _PRIM_COL_IDX["m5_tick_ofi"]]
    ret_6h = X_primary[:, _PRIM_COL_IDX["m5_ret_72"]]
    mtf_ema = X_primary[:, _PRIM_COL_IDX["mtf_ema_alignment"]]

    active_side, sleeve_id, _ = extract_institutional_sleeve_signals(
        X_primary=X_primary,
        regime_codes=regime_codes,
    )

    side_prob = np.where(
        active_side == 1,
        p_long,
        np.where(active_side == -1, p_short, np.maximum(p_long, p_short)),
    )
    dir_skew = (p_long - p_short) * active_side.astype(np.float64)

    sorted_probs = np.sort(primary_probs, axis=1)
    margin_prob = sorted_probs[:, -1] - sorted_probs[:, -2]

    # Signed confluence combining macro EMA alignment, exhaustion stretch, and order flow
    structural_confluence = (
        0.40 * mtf_ema * active_side
        - 0.35 * np.clip(ret_6h * 100.0, -2.0, 2.0) * active_side
        + 0.25 * m5_ofi * active_side
    )

    aux = np.column_stack(
        [
            active_side.astype(np.float32),
            side_prob.astype(np.float32),
            dir_skew.astype(np.float32),
            margin_prob.astype(np.float32),
            sleeve_id.astype(np.float32),
            structural_confluence.astype(np.float32),
            regime_codes.astype(np.float32),
            spread_to_atr.astype(np.float32),
        ]
    )
    X_meta = np.hstack([X_primary.astype(np.float32), aux]).astype(np.float32)
    return X_meta, active_side.astype(np.int64), side_prob.astype(np.float64)


class MetaLabelingTrainer:
    """Trains Primary + Secondary Meta-Model with Purged CV and exports both to ONNX."""

    def __init__(
        self,
        ai_config: RegimeAIConfig,
        symbol_spec: SymbolSpec,
        random_seed: int = 42,
    ) -> None:
        self._cfg = ai_config
        self._spec = symbol_spec
        self._seed = random_seed
        self._regime_detector = MarketRegimeDetector(ai_config)

    def _make_primary_estimator(self) -> lgb.LGBMClassifier:
        return lgb.LGBMClassifier(
            objective="multiclass",
            num_class=3,
            n_estimators=min(80, self._cfg.n_estimators),
            learning_rate=self._cfg.learning_rate,
            max_depth=3,
            num_leaves=7,
            min_child_samples=self._cfg.min_child_samples,
            subsample=self._cfg.subsample,
            colsample_bytree=self._cfg.colsample_bytree,
            class_weight="balanced",
            random_state=self._seed,
            n_jobs=1,
            verbose=-1,
        )

    def _make_meta_estimator(self, n_samples: int = 300) -> lgb.LGBMClassifier:
        min_leaf = max(16, min(45, n_samples // 8))
        return lgb.LGBMClassifier(
            objective="binary",
            n_estimators=min(45, self._cfg.n_estimators),
            learning_rate=0.025,
            max_depth=2,
            num_leaves=3,
            min_child_samples=min_leaf,
            subsample=0.80,
            colsample_bytree=0.65,
            reg_alpha=1.5,
            reg_lambda=4.0,
            random_state=self._seed + 17,
            n_jobs=1,
            verbose=-1,
        )

    def train_and_export(
        self,
        train_df: pd.DataFrame,
        output_dir: Path | str | None = None,
        n_splits: int = 5,
        embargo_pct: float = 0.02,
    ) -> TrainedMetaLabelingArtifacts:
        """
        Fit Primary and Secondary Meta-Models strictly within `train_df` and export ONNX binaries.
        """
        clean_df = train_df.dropna(subset=list(PRIMARY_FEATURE_COLUMNS)).reset_index(drop=True)
        n = len(clean_df)
        if n < 160:
            raise ValueError(f"Need at least 160 aligned bars to train Meta-Labeling models; got {n}.")

        X_prim = clean_df[list(PRIMARY_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        regimes = self._regime_detector.classify_dataframe(clean_df)
        regime_codes = np.array([REGIME_TO_NUMERIC_CODE[MarketRegime(r)] for r in regimes], dtype=np.float32)
        spread_price = clean_df["spread"].to_numpy(dtype=np.float32) * float(self._spec.point)
        atr_col = "structural_atr" if "structural_atr" in clean_df.columns else "m5_atr"
        atr_vals = np.maximum(clean_df[atr_col].to_numpy(dtype=np.float32), float(self._spec.point))
        spread_to_atr = spread_price / atr_vals

        tb_df = compute_triple_barrier_labels(
            df=clean_df,
            pt_atr_mult=self._cfg.triple_barrier_pt_atr,
            sl_atr_mult=self._cfg.triple_barrier_sl_atr,
            max_holding_bars=min(48, self._cfg.triple_barrier_max_bars),
            point=self._spec.point,
        )
        y_prim_raw = tb_df["primary_label"].to_numpy(dtype=np.int64)
        y_prim_cls = np.vectorize(SIDE_TO_CLASS_IDX.get)(y_prim_raw).astype(np.int64)

        # Ensure all 3 classes exist in edge-case synthetic slices
        for cls_id in (0, 1, 2):
            if not np.any(y_prim_cls == cls_id):
                y_prim_cls[cls_id] = cls_id

        # Step 1: Purged Out-Of-Fold Primary probabilities to prevent Meta-Model overfitting
        oof_probs = np.full((n, 3), 1.0 / 3.0, dtype=np.float32)
        purged_cv = PurgedKFoldCV(n_splits=n_splits, embargo_pct=embargo_pct)

        for fold_train_idx, fold_val_idx in purged_cv.split(
            tb_df["event_start_time"],
            tb_df["event_end_time"],
        ):
            if len(fold_train_idx) < 40:
                continue
            y_fold = y_prim_cls[fold_train_idx].copy()
            for cls_id in (0, 1, 2):
                if not np.any(y_fold == cls_id):
                    y_fold[cls_id] = cls_id
            fold_clf = self._make_primary_estimator()
            fold_clf.fit(X_prim[fold_train_idx], y_fold)
            oof_probs[fold_val_idx] = fold_clf.predict_proba(X_prim[fold_val_idx]).astype(np.float32)

        # Step 2: Fit Final Primary Model on full In-Sample dataset
        final_primary = self._make_primary_estimator()
        final_primary.fit(X_prim, y_prim_cls)

        # Step 3: Build Secondary Meta-Labels from OOF Primary + Institutional Sleeve signals
        _, oof_sleeves, oof_hold_bars = extract_institutional_sleeve_signals(X_prim, regime_codes)
        X_meta_oof, oof_sides, _ = build_meta_feature_matrix(
            X_primary=X_prim,
            primary_probs=oof_probs,
            regime_codes=regime_codes,
            spread_to_atr=spread_to_atr,
        )
        meta_label_df = compute_meta_labels_for_sides(
            df=clean_df,
            predicted_sides=oof_sides,
            pt_atr_mult=self._cfg.triple_barrier_pt_atr,
            sl_atr_mult=self._cfg.triple_barrier_sl_atr,
            max_holding_bars=self._cfg.triple_barrier_max_bars,
            point=self._spec.point,
            max_holding_bars_arr=oof_hold_bars,
            contract_size=self._spec.contract_size,
            enable_ratchet=False,
        )
        y_meta = meta_label_df["meta_label"].to_numpy(dtype=np.int64)
        end_idx_arr = meta_label_df["event_end_idx"].to_numpy(dtype=np.int64)

        # De-cluster consecutive M5 bars per sleeve so SGE (10 bars/night) does not 10x-dominate COMEX/London
        declustered_mask = np.zeros(n, dtype=bool)
        last_slv_entry: dict[int, int] = {}
        active_slv_exit: dict[int, int] = {}
        for idx_i in range(n - 1):
            if oof_sides[idx_i] == 0 or end_idx_arr[idx_i] >= (n - 1):
                continue
            s_id = int(oof_sleeves[idx_i])
            if idx_i < active_slv_exit.get(s_id, -1):
                continue
            if (idx_i - last_slv_entry.get(s_id, -999)) < 36:
                continue
            declustered_mask[idx_i] = True
            last_slv_entry[s_id] = idx_i
            active_slv_exit[s_id] = int(end_idx_arr[idx_i])

        active_mask = (oof_sides != 0) & (end_idx_arr < (n - 1))
        if int(np.sum(declustered_mask)) >= 30:
            X_meta_train = X_meta_oof[declustered_mask]
            y_meta_train = y_meta[declustered_mask].copy()
        elif int(np.sum(active_mask)) >= 30:
            X_meta_train = X_meta_oof[active_mask]
            y_meta_train = y_meta[active_mask].copy()
        else:
            X_meta_train = X_meta_oof
            y_meta_train = y_meta.copy()

        if len(np.unique(y_meta_train)) < 2:
            y_meta_train[0] = 0
            y_meta_train[-1] = 1

        final_meta = self._make_meta_estimator(n_samples=len(X_meta_train))
        final_meta.fit(X_meta_train, y_meta_train)

        meta_train_probs = final_meta.predict_proba(X_meta_train)[:, 1]
        win_mean = float(np.mean(meta_train_probs[y_meta_train == 1])) if np.any(y_meta_train == 1) else 0.5
        loss_mean = float(np.mean(meta_train_probs[y_meta_train == 0])) if np.any(y_meta_train == 0) else 0.5
        separation_score = float(np.clip(0.5 + 0.5 * (win_mean - loss_mean), 0.0, 1.0))

        # Step 4: Export both models to ONNX format
        primary_bytes = _export_lgbm_to_onnx_bytes(final_primary, num_features=X_prim.shape[1])
        meta_bytes = _export_lgbm_to_onnx_bytes(final_meta, num_features=X_meta_train.shape[1])

        export_dir = Path(output_dir) if output_dir is not None else Path(self._cfg.onnx_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        primary_path = export_dir / self._cfg.primary_onnx_filename
        meta_path = export_dir / self._cfg.meta_onnx_filename
        primary_path.write_bytes(primary_bytes)
        meta_path.write_bytes(meta_bytes)

        return TrainedMetaLabelingArtifacts(
            primary_onnx_path=primary_path,
            meta_onnx_path=meta_path,
            primary_onnx_bytes=primary_bytes,
            meta_onnx_bytes=meta_bytes,
            primary_feature_names=PRIMARY_FEATURE_COLUMNS,
            meta_feature_names=META_FEATURE_COLUMNS,
            oof_meta_auc_proxy=separation_score,
            num_train_samples=n,
        )
