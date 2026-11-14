# Antigravity AI Algorithmic Trading System (MT5 24/5)

Institutional-grade quantitative algorithmic trading architecture engineered around **Ultra-Low Equity Volatility** and **Geometric Compounding**.

## Core First Principles
1. **Volatility Targeting & Market Regime Detection**: Separates persistent trending (`H > 0.55`) and mean-reverting (`H < 0.45`) regimes from random-walk noise (`0.45 <= H <= 0.55`) and abnormal volatility shocks, halting execution immediately during unfavorable states.
2. **Risk-Adjusted Return Optimization (Zero Martingale)**: Combines Marcos López de Prado's **Two-Stage Meta-Labeling** (LightGBM -> ONNX sub-millisecond inference) with **Dynamic Fractional Kelly Criterion**, **Inverse-ATR & Spread Position Sizing**, **Hard Drawdown Circuit Breakers** (`1.5%` Daily DD / `4.0%` Max Trailing DD -> Cooldown Halt -> `0.5x` Probation Recovery), and **Liquidity-Aware TWAP / Iceberg Slicing**.

## Directory Structure
```text
.
├── config.yaml                          # Single source of truth configuration
├── pyproject.toml                       # Package & pytest configuration
├── requirements.txt                     # Runtime dependencies
├── main.py                              # 24/5 VPS Orchestrator & CLI entrypoint
├── core/
│   ├── contracts.py                     # Strict dataclasses, enums, & MT5 protocols
│   └── config_loader.py                 # Validated YAML loader & invariant checks
├── data_pipeline/
│   ├── mt5_data_client.py               # NativeMT5Gateway & SimulatedMT5Gateway (OHLCV + Ticks)
│   ├── feature_engineering.py           # Causal ATR, Parkinson Vol, Z-Score, Hurst, Volume Profile, OFI
│   └── mtf_aligner.py                   # Leak-free close_time MTF aligner (M5, M15, H1, H4)
├── regime_ai_engine/
│   ├── regime_detector.py               # Market Regime Classifier & Turbulence Halt Gate
│   ├── triple_barrier.py                # Volatility-scaled Triple-Barrier & Meta-Labels
│   ├── meta_labeling_models.py          # LightGBM Primary + Meta Trainer & ONNX Exporter
│   └── onnx_inference.py                # Sub-millisecond ONNX Runtime Inference Engine
├── risk_and_scaling_engine/
│   ├── position_sizer.py                # Dynamic Fractional Kelly + Vol Target + Inverse ATR/Spread Sizer
│   ├── circuit_breaker.py               # Equity Curve Control (1.5% Daily / 4.0% Trailing DD)
│   └── liquidity_scaler.py              # Max-Lot Ceiling & TWAP / Iceberg Order Slicer
├── backtest_and_validation/
│   ├── purged_cv.py                     # Purged K-Fold Cross-Validation with Embargo
│   ├── metrics.py                       # Sharpe (>2.0), Sortino, Calmar, MaxDD, Tail Ratio
│   ├── walk_forward.py                  # Rolling Walk-Forward Optimization Engine
│   └── monte_carlo.py                   # 1,000-Iteration Block-Bootstrap Tail-Risk Simulator
├── execution_and_monitoring/
│   ├── time_and_spread_filters.py       # SpreadGuard & Rollover / Weekend / News Blackout Filter
│   ├── mt5_executor.py                  # Auto-Reconnect, Transient Retry, & TWAP Slice Executor
│   └── audit_database.py                # SQLite (WAL) / PostgreSQL Audit & Feedback Loop
└── tests/                               # Unit & Integration Tests
```

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run full unit & integration test suite (12 tests)
pytest

# Run end-to-end Walk-Forward + 1,000x Monte Carlo + ONNX + SQLite pipeline (30,000 M5 bars)
python main.py --mode demo-pipeline --bars 30000

# Launch 24/5 live MetaTrader 5 VPS loop (Windows VPS with MT5 terminal)
export MT5_LOGIN=12345678
export MT5_PASSWORD="your_password"
export MT5_SERVER="YourBroker-Live"
python main.py --mode live-mt5 --config config.yaml
```
