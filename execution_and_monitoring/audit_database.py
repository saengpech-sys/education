"""
AI Decision Audit Database & Post-Trade Feedback Loop (SQLite WAL / PostgreSQL).

Records every single AI evaluation (both approved executions and filtered rejections)
alongside full microstructure feature snapshots, latency, sizing math, and realized
post-trade PnL to power continuous probability recalibration and model retraining.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import pandas as pd

from core.config_loader import DatabaseConfig
from core.contracts import (
    AISignalDecision,
    CircuitBreakerStatus,
    OrderExecutionReport,
    PositionSizeResult,
)


@dataclass(frozen=True)
class CalibrationFeedbackSummary:
    closed_trades_count: int
    mean_predicted_win_prob: float
    realized_win_rate: float
    calibration_gap: float
    brier_score: float
    recommended_threshold_delta: float


class DecisionAuditDatabase:
    """Persistent SQL audit store supporting SQLite (WAL) and PostgreSQL backends."""

    def __init__(self, config: DatabaseConfig) -> None:
        self._cfg = config
        self._backend = config.backend.lower()
        self._placeholder = "?" if self._backend == "sqlite" else "%s"
        self._conn = self._connect()
        self._initialize_schema()

    def _connect(self):
        if self._backend == "sqlite":
            db_path = Path(self._cfg.sqlite_path)
            db_path.parent.mkdir(parents=True, exist_ok=True)
            conn = sqlite3.connect(str(db_path), check_same_thread=False)
            conn.execute("PRAGMA journal_mode=WAL;")
            conn.execute("PRAGMA synchronous=NORMAL;")
            conn.execute("PRAGMA foreign_keys=ON;")
            return conn
        if self._backend == "postgresql":
            if not self._cfg.postgres_dsn:
                raise ValueError("postgres_dsn is required when database.backend == 'postgresql'.")
            try:
                import psycopg  # type: ignore

                return psycopg.connect(self._cfg.postgres_dsn)
            except ImportError:
                import psycopg2  # type: ignore

                return psycopg2.connect(self._cfg.postgres_dsn)
        raise ValueError(f"Unsupported database backend: {self._backend}")

    def _initialize_schema(self) -> None:
        cur = self._conn.cursor()
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS ai_decision_logs (
                signal_id TEXT PRIMARY KEY,
                timestamp_utc TEXT NOT NULL,
                symbol TEXT NOT NULL,
                regime TEXT NOT NULL,
                regime_tradable INTEGER NOT NULL,
                primary_side INTEGER NOT NULL,
                primary_probability REAL NOT NULL,
                meta_win_probability REAL NOT NULL,
                confidence_threshold REAL NOT NULL,
                ai_approved INTEGER NOT NULL,
                cb_state TEXT NOT NULL,
                cb_daily_dd_pct REAL NOT NULL,
                cb_trailing_dd_pct REAL NOT NULL,
                sized_lots REAL NOT NULL,
                effective_risk_pct REAL NOT NULL,
                executed INTEGER NOT NULL,
                filled_lots REAL NOT NULL,
                vwap_fill_price REAL NOT NULL,
                avg_slippage_points REAL NOT NULL,
                inference_latency_ms REAL NOT NULL,
                status_reason TEXT NOT NULL,
                features_json TEXT NOT NULL
            )
            """
        )
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS post_trade_outcomes (
                signal_id TEXT PRIMARY KEY,
                exit_timestamp_utc TEXT NOT NULL,
                exit_price REAL NOT NULL,
                realized_pnl_usd REAL NOT NULL,
                realized_r_multiple REAL NOT NULL,
                actual_win INTEGER NOT NULL,
                exit_reason TEXT NOT NULL,
                FOREIGN KEY(signal_id) REFERENCES ai_decision_logs(signal_id)
            )
            """
        )
        self._conn.commit()

    def log_decision(
        self,
        ai_decision: AISignalDecision,
        cb_status: CircuitBreakerStatus,
        size_result: Optional[PositionSizeResult] = None,
        exec_report: Optional[OrderExecutionReport] = None,
    ) -> None:
        """Persist full AI decision, risk state, and execution outcome atomically."""
        sized_lots = size_result.final_lots if size_result is not None else 0.0
        eff_risk = size_result.effective_risk_pct if size_result is not None else 0.0
        executed = int(bool(exec_report and exec_report.executed))
        filled_lots = exec_report.filled_total_lots if exec_report is not None else 0.0
        vwap_px = exec_report.vwap_fill_price if exec_report is not None else 0.0
        avg_slip = exec_report.average_slippage_points if exec_report is not None else 0.0

        status_reason = "EXECUTED_OK"
        if not ai_decision.approved:
            status_reason = ai_decision.rejection_reason or "AI_REJECTED"
        elif size_result is not None and not size_result.approved:
            status_reason = size_result.rejection_reason or "RISK_SIZER_REJECTED"
        elif exec_report is not None and not exec_report.executed:
            status_reason = exec_report.abort_reason or "EXECUTION_ABORTED"
        elif exec_report is None:
            status_reason = "DRY_RUN_NO_EXECUTION"

        ph = self._placeholder
        sql = f"""
            INSERT INTO ai_decision_logs (
                signal_id, timestamp_utc, symbol, regime, regime_tradable,
                primary_side, primary_probability, meta_win_probability, confidence_threshold,
                ai_approved, cb_state, cb_daily_dd_pct, cb_trailing_dd_pct,
                sized_lots, effective_risk_pct, executed, filled_lots,
                vwap_fill_price, avg_slippage_points, inference_latency_ms,
                status_reason, features_json
            ) VALUES ({",".join([ph] * 22)})
        """
        params = (
            ai_decision.signal_id,
            ai_decision.timestamp_utc.isoformat(),
            ai_decision.symbol,
            ai_decision.regime.value,
            int(ai_decision.regime_tradable),
            int(ai_decision.primary_side),
            float(ai_decision.primary_probability),
            float(ai_decision.meta_win_probability),
            float(ai_decision.confidence_threshold),
            int(ai_decision.approved),
            cb_status.state.value,
            float(cb_status.daily_drawdown_pct),
            float(cb_status.trailing_drawdown_pct),
            float(sized_lots),
            float(eff_risk),
            executed,
            float(filled_lots),
            float(vwap_px),
            float(avg_slip),
            float(ai_decision.inference_latency_ms),
            status_reason,
            json.dumps(ai_decision.feature_snapshot, sort_keys=True),
        )
        cur = self._conn.cursor()
        cur.execute(sql, params)
        self._conn.commit()

    def record_post_trade_outcome(
        self,
        signal_id: str,
        exit_timestamp_utc: datetime,
        exit_price: float,
        realized_pnl_usd: float,
        realized_r_multiple: float,
        exit_reason: str,
    ) -> None:
        """Record realized trade outcome to close the Post-Trade Feedback Loop."""
        ph = self._placeholder
        actual_win = 1 if realized_pnl_usd > 0.0 else 0
        sql = f"""
            INSERT INTO post_trade_outcomes (
                signal_id, exit_timestamp_utc, exit_price,
                realized_pnl_usd, realized_r_multiple, actual_win, exit_reason
            ) VALUES ({",".join([ph] * 7)})
        """
        cur = self._conn.cursor()
        cur.execute(
            sql,
            (
                signal_id,
                exit_timestamp_utc.isoformat(),
                float(exit_price),
                float(realized_pnl_usd),
                float(realized_r_multiple),
                actual_win,
                exit_reason,
            ),
        )
        self._conn.commit()

    def export_feedback_dataset(self) -> pd.DataFrame:
        """Join logged AI features with realized outcomes for continuous Meta-Model retraining."""
        query = """
            SELECT
                d.signal_id,
                d.timestamp_utc,
                d.symbol,
                d.regime,
                d.primary_side,
                d.primary_probability,
                d.meta_win_probability,
                d.confidence_threshold,
                d.filled_lots,
                d.avg_slippage_points,
                d.features_json,
                o.realized_pnl_usd,
                o.realized_r_multiple,
                o.actual_win,
                o.exit_reason
            FROM ai_decision_logs d
            INNER JOIN post_trade_outcomes o ON d.signal_id = o.signal_id
            ORDER BY d.timestamp_utc ASC
        """
        cur = self._conn.cursor()
        cur.execute(query)
        rows = cur.fetchall()
        cols = [desc[0] for desc in cur.description]
        if not rows:
            return pd.DataFrame(columns=cols)

        base_df = pd.DataFrame(rows, columns=cols)
        parsed_features: list[Dict[str, Any]] = [
            json.loads(raw_json) for raw_json in base_df["features_json"]
        ]
        feat_df = pd.DataFrame(parsed_features)
        return pd.concat([base_df.drop(columns=["features_json"]), feat_df], axis=1)

    def compute_calibration_summary(self) -> CalibrationFeedbackSummary:
        """
        Compare Meta-Model predicted win probability against realized execution win rate.
        Recommends tightening confidence_threshold if live realization underperforms predictions.
        """
        df = self.export_feedback_dataset()
        if df.empty:
            return CalibrationFeedbackSummary(
                closed_trades_count=0,
                mean_predicted_win_prob=0.0,
                realized_win_rate=0.0,
                calibration_gap=0.0,
                brier_score=0.0,
                recommended_threshold_delta=0.0,
            )

        pred = df["meta_win_probability"].astype(float).to_numpy()
        actual = df["actual_win"].astype(float).to_numpy()
        mean_pred = float(pred.mean())
        win_rate = float(actual.mean())
        gap = win_rate - mean_pred
        brier = float(((pred - actual) ** 2).mean())

        # If realized win rate trails predicted probability by > 5%, tighten threshold upward
        if gap < -0.05:
            rec_delta = min(0.06, round(-0.5 * gap, 4))
        elif gap > 0.08:
            rec_delta = max(-0.02, round(-0.25 * gap, 4))
        else:
            rec_delta = 0.0

        return CalibrationFeedbackSummary(
            closed_trades_count=len(df),
            mean_predicted_win_prob=mean_pred,
            realized_win_rate=win_rate,
            calibration_gap=gap,
            brier_score=brier,
            recommended_threshold_delta=rec_delta,
        )

    def count_logged_decisions(self) -> int:
        cur = self._conn.cursor()
        cur.execute("SELECT COUNT(*) FROM ai_decision_logs")
        row = cur.fetchone()
        return int(row[0]) if row else 0

    def close(self) -> None:
        self._conn.close()
