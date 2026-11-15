# Antigravity AI Algorithmic Trading System (MT5 24/5)

Institutional-grade quantitative algorithmic trading architecture engineered around **Ultra-Low Equity Volatility** and **Geometric Compounding**.

## Core First Principles
1. **Volatility Targeting & Macro-Gated Market Regime Detection**: Combines Multi-Scale Hurst Exponent (`M5` + `H1`), 24-hour Volatility Z-Score/Ratio baselines, and Structural `H4` / 20-Day Macro Trend Gating to halt execution immediately during abnormal volatility shocks or random mid-session chop, while blocking counter-trend longs during macro bear trends (`h4_ema <= -0.20`) and counter-trend shorts during macro bull expansions (`h4_ema >= +0.30`).
2. **4-Sleeve Institutional Bullion Market-Clock & Breakout Alpha + Chandelier Trailing Stop + ONNX Meta-Labeling (Zero Martingale)**: Exploits structural physical/futures liquidity handoffs and trend breakouts in `XAUUSD` across four orthogonal sleeves — **Shanghai SGE Benchmark Fix Accumulation (`20:45–21:30 -> 02:30 UTC`)**, **COMEX / LBMA PM Fix 6h Exhaustion Reversal (`12:00–15:55 UTC`)**, **London-Close Settlement Exhaustion + `9h` Chandelier Bull-Runner (`16:30–19:55 UTC`)**, and **Post-LBMA PM Fix `12h` High Trend Breakout + `8h` Chandelier Trailing Stop (`15:00–16:25 UTC`)** — filtered by Marcos López de Prado's **Two-Stage LightGBM -> ONNX Meta-Labeling Engine**, **Dynamic Fractional-Kelly (`2.40%–3.50%` risk/trade) + Downside Semi-Volatility Targeting**, **Hard Drawdown Circuit Breakers** (`7.5%` Daily DD / `17.0%` Trailing DD `< 20.0%` -> Cooldown Halt -> `0.5x` Probation Recovery), and **Liquidity-Aware TWAP / Iceberg Slicing**.

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
│   ├── feature_engineering.py           # Causal 24h ATR, Parkinson Vol, Z-Score, Hurst, Market-Clock, VP, OFI
│   └── mtf_aligner.py                   # Leak-free close_time MTF aligner (M5, M15, H1, H4)
├── regime_ai_engine/
│   ├── regime_detector.py               # H4 Macro + Hurst Market Regime Classifier & Shock Gate
│   ├── triple_barrier.py                # Triple-Barrier, Chandelier Trailing-Stop & Causal Simulator
│   ├── meta_labeling_models.py          # 4-Sleeve Institutional + Chandelier Primary & Meta LightGBM -> ONNX
│   └── onnx_inference.py                # Sub-millisecond Calibrated ONNX Runtime Inference Engine
├── risk_and_scaling_engine/
│   ├── position_sizer.py                # Dynamic Fractional Kelly + Vol Target + Inverse ATR/Spread Sizer
│   ├── circuit_breaker.py               # Equity Curve Control (7.5% Daily / 17.0% Trailing DD < 20.0%)
│   └── liquidity_scaler.py              # Max-Lot Ceiling & TWAP / Iceberg Order Slicer
├── backtest_and_validation/
│   ├── purged_cv.py                     # Purged K-Fold Cross-Validation with Embargo
│   ├── metrics.py                       # Sharpe (>2.0), Sortino, Calmar, MaxDD, Tail Ratio
│   ├── walk_forward.py                  # Multi-Sleeve Rolling Walk-Forward Optimization Engine
│   └── monte_carlo.py                   # 1,000-Iteration Block-Bootstrap Tail-Risk Simulator
├── execution_and_monitoring/
│   ├── time_and_spread_filters.py       # SpreadGuard & Rollover / Weekend / News Blackout Filter
│   ├── mt5_executor.py                  # Auto-Reconnect, Transient Retry, & TWAP Slice Executor
│   └── audit_database.py                # SQLite (WAL) / PostgreSQL Audit & Feedback Loop
└── tests/                               # 16 Unit & Integration Tests
```

## Quick Start
```bash
# Install dependencies
pip install -r requirements.txt

# Run full unit & integration test suite (16 tests)
pytest

# Run end-to-end Walk-Forward + 1,000x Monte Carlo + ONNX + SQLite pipeline (30,000 M5 bars)
python main.py --mode demo-pipeline --bars 30000

# Launch 24/5 live MetaTrader 5 VPS loop (Windows VPS with MT5 terminal)
export MT5_LOGIN=12345678
export MT5_PASSWORD="your_password"
export MT5_SERVER="YourBroker-Live"
python main.py --mode live-mt5 --config config.yaml
```
