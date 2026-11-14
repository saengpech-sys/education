"""
Sub-Millisecond ONNX Runtime Inference Engine for Two-Stage Meta-Labeling.

Combines:
1. MarketRegimeDetector (First-Principles Volatility & Noise Filter)
2. ONNX Primary Model (Directional / Mean-Reversion Signal)
3. ONNX Secondary Meta-Model (Calibrated Confidence Filter: P(Win) > Threshold)
"""

from __future__ import annotations

import time
import uuid
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnxruntime as ort
import pandas as pd

from core.config_loader import RegimeAIConfig, RiskAndScalingConfig
from core.contracts import (
    AISignalDecision,
    MarketRegime,
    MarketSnapshot,
    SignalSide,
    SymbolSpec,
)
from data_pipeline.mtf_aligner import PRIMARY_FEATURE_COLUMNS
from regime_ai_engine.meta_labeling_models import build_meta_feature_matrix
from regime_ai_engine.regime_detector import MarketRegimeDetector, REGIME_TO_NUMERIC_CODE


def _create_low_latency_session(model_source: Path | bytes) -> ort.InferenceSession:
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 1
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    raw_or_path = str(model_source) if isinstance(model_source, Path) else model_source
    return ort.InferenceSession(raw_or_path, sess_options=opts, providers=["CPUExecutionProvider"])


class ONNXMetaLabelingEngine:
    """Production ONNX Runtime engine executing Primary + Meta inference in microseconds."""

    def __init__(
        self,
        ai_config: RegimeAIConfig,
        risk_config: RiskAndScalingConfig,
        symbol_spec: SymbolSpec,
        primary_model_source: Path | bytes,
        meta_model_source: Path | bytes,
        confidence_threshold_override: Optional[float] = None,
    ) -> None:
        self._ai_cfg = ai_config
        self._risk_cfg = risk_config
        self._spec = symbol_spec
        self._regime_detector = MarketRegimeDetector(ai_config)
        self._threshold = (
            float(confidence_threshold_override)
            if confidence_threshold_override is not None
            else float(ai_config.meta_confidence_threshold)
        )

        self._primary_sess = _create_low_latency_session(primary_model_source)
        self._meta_sess = _create_low_latency_session(meta_model_source)
        self._primary_input_name = self._primary_sess.get_inputs()[0].name
        self._meta_input_name = self._meta_sess.get_inputs()[0].name

    @property
    def confidence_threshold(self) -> float:
        return self._threshold

    @property
    def reward_to_risk_ratio(self) -> float:
        return self._risk_cfg.atr_tp_multiplier / max(self._risk_cfg.atr_stop_multiplier, 1e-9)

    def predict_primary_proba(self, X_primary: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(X_primary, dtype=np.float32)
        probs = self._primary_sess.run(["probabilities"], {self._primary_input_name: arr})[0]
        return np.asarray(probs, dtype=np.float64)

    def predict_meta_proba(self, X_meta: np.ndarray) -> np.ndarray:
        arr = np.ascontiguousarray(X_meta, dtype=np.float32)
        probs = self._meta_sess.run(["probabilities"], {self._meta_input_name: arr})[0]
        return np.asarray(probs[:, 1], dtype=np.float64)

    def evaluate_snapshot(self, snapshot: MarketSnapshot) -> AISignalDecision:
        """Run full Regime + Primary ONNX + Meta ONNX inference on a single live MarketSnapshot."""
        t0 = time.perf_counter()
        signal_id = uuid.uuid4().hex[:16]
        rr_ratio = self.reward_to_risk_ratio

        regime = self._regime_detector.classify_snapshot(snapshot)
        if not regime.is_tradable:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return AISignalDecision(
                signal_id=signal_id,
                timestamp_utc=snapshot.timestamp_utc,
                symbol=snapshot.symbol,
                regime=regime,
                regime_tradable=False,
                primary_side=SignalSide.FLAT,
                primary_probability=0.0,
                meta_win_probability=0.0,
                confidence_threshold=self._threshold,
                reward_to_risk_ratio=rr_ratio,
                approved=False,
                rejection_reason=f"REGIME_FILTER_HALT:{regime.value}",
                inference_latency_ms=latency_ms,
                feature_snapshot=dict(snapshot.features),
            )

        X_prim = np.array(
            [[float(snapshot.features[col]) for col in PRIMARY_FEATURE_COLUMNS]],
            dtype=np.float32,
        )
        prim_probs = self.predict_primary_proba(X_prim)
        regime_code = np.array([REGIME_TO_NUMERIC_CODE[regime]], dtype=np.float32)
        spread_to_atr = np.array(
            [snapshot.spread_price / max(snapshot.atr, self._spec.point)],
            dtype=np.float32,
        )

        X_meta, active_sides, max_dir_probs = build_meta_feature_matrix(
            X_primary=X_prim,
            primary_probs=prim_probs,
            regime_codes=regime_code,
            spread_to_atr=spread_to_atr,
        )
        side_val = int(active_sides[0])
        prim_prob = float(max_dir_probs[0])

        if side_val == 0:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            return AISignalDecision(
                signal_id=signal_id,
                timestamp_utc=snapshot.timestamp_utc,
                symbol=snapshot.symbol,
                regime=regime,
                regime_tradable=True,
                primary_side=SignalSide.FLAT,
                primary_probability=prim_prob,
                meta_win_probability=0.0,
                confidence_threshold=self._threshold,
                reward_to_risk_ratio=rr_ratio,
                approved=False,
                rejection_reason="PRIMARY_SIGNAL_FLAT",
                inference_latency_ms=latency_ms,
                feature_snapshot=dict(snapshot.features),
            )

        meta_win_prob = float(self.predict_meta_proba(X_meta)[0])
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Require BOTH confidence threshold AND positive mathematical Kelly expectancy
        min_breakeven_prob = 1.0 / (1.0 + rr_ratio)
        effective_gate = max(self._threshold, min_breakeven_prob + 0.02)

        approved = meta_win_prob >= effective_gate
        rejection = (
            None
            if approved
            else f"META_CONFIDENCE_BELOW_THRESHOLD:{meta_win_prob:.4f}<{effective_gate:.4f}"
        )

        return AISignalDecision(
            signal_id=signal_id,
            timestamp_utc=snapshot.timestamp_utc,
            symbol=snapshot.symbol,
            regime=regime,
            regime_tradable=True,
            primary_side=SignalSide(side_val),
            primary_probability=prim_prob,
            meta_win_probability=meta_win_prob,
            confidence_threshold=effective_gate,
            reward_to_risk_ratio=rr_ratio,
            approved=approved,
            rejection_reason=rejection,
            inference_latency_ms=latency_ms,
            feature_snapshot=dict(snapshot.features),
        )

    def evaluate_batch_dataframe(
        self,
        df: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
        """
        Batch ONNX evaluation for fast Walk-Forward Out-Of-Sample validation.
        Returns (approved_sides, primary_probs, meta_win_probs, regimes).
        """
        regimes = self._regime_detector.classify_dataframe(df)
        tradable_mask = np.array([MarketRegime(r).is_tradable for r in regimes], dtype=bool)
        regime_codes = np.array(
            [REGIME_TO_NUMERIC_CODE[MarketRegime(r)] for r in regimes], dtype=np.float32
        )

        X_prim = df[list(PRIMARY_FEATURE_COLUMNS)].to_numpy(dtype=np.float32)
        prim_probs = self.predict_primary_proba(X_prim)
        spread_to_atr = (
            df["spread"].to_numpy(dtype=np.float32)
            * float(self._spec.point)
            / np.maximum(df["m5_atr"].to_numpy(dtype=np.float32), float(self._spec.point))
        )

        X_meta, active_sides, max_dir_probs = build_meta_feature_matrix(
            X_primary=X_prim,
            primary_probs=prim_probs,
            regime_codes=regime_codes,
            spread_to_atr=spread_to_atr,
        )
        meta_probs = self.predict_meta_proba(X_meta)

        rr_ratio = self.reward_to_risk_ratio
        effective_gate = max(self._threshold, (1.0 / (1.0 + rr_ratio)) + 0.02)
        approved_mask = tradable_mask & (active_sides != 0) & (meta_probs >= effective_gate)
        final_sides = np.where(approved_mask, active_sides, 0).astype(np.int64)
        return final_sides, max_dir_probs, meta_probs, regimes
