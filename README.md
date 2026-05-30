# Market Regime Detection Using HMM and Mixture of Experts

This project builds a multi-asset, regime-aware financial ML system for market direction prediction.

## Core Idea

Financial markets behave differently across regimes. A single model trained across all periods may underperform because bull markets, bear markets, and crisis periods have different return and risk characteristics.

This project detects latent regimes using Hidden Markov Models, then trains regime-aware expert classifiers using a Mixture of Experts setup.

## Assets

- S&P 500: `^GSPC`
- NASDAQ: `^IXIC`
- NIFTY 50: `^NSEI`
- US VIX: `^VIX`
- India VIX: `^INDIAVIX`

## Notebook Pipeline

| Notebook | Purpose |
|---|---|
| `01_download_market_data.ipynb` | Download raw Yahoo Finance market data |
| `02_feature_engineering.ipynb` | Build returns, volatility, momentum, moving average, drawdown features |
| `03_hmm_regime_detection.ipynb` | Detect Bull/Bear/Crisis regimes using HMM |
| `04_mixture_of_experts.ipynb` | First return-regression MoE experiment |
| `05_directional_moe_classifier.ipynb` | Direction classification MoE |
| `06_walk_forward_validation.ipynb` | Walk-forward validation |
| `07_final_research_report.ipynb` | Initial S&P-focused report |
| `08_all_assets_boosted_experts.ipynb` | All-asset Random Forest/XGBoost/LightGBM experiments |
| `09_final_all_asset_report.ipynb` | Final all-asset report |
| `10_transaction_costs_position_sizing.ipynb` | Transaction costs and volatility-based position sizing |
| `11_india_vix_macro_enrichment.ipynb` | India VIX, FRED, and RBI macro feature enrichment |
| `12_advanced_regime_and_deep_models.ipynb` | Markov-switching, neural gating, and transformer extensions |
| `13_macro_enriched_hmm_moe.ipynb` | Re-run HMM and MoE using asset-specific VIX and macro-enriched features |

## Best Results

The strongest all-asset model was:

```text
Random Forest + Soft-Gated MoE
All-asset strategy Sharpe: 2.329
Random Forest baseline Sharpe: 1.917
```

The most realistic transaction-cost test used volatility-based position sizing and 10 bps costs:

```text
Net strategy Sharpe: 2.298
Buy-and-hold Sharpe: 1.732
Net max drawdown: -5.02%
Buy-and-hold max drawdown: -10.85%
```

India VIX improved the financial correctness of the NIFTY setup and improved some NIFTY-specific regime results, but the best overall multi-asset model remained the original Random Forest Soft-Gated MoE from notebook 08.

## Setup

Use Python 3.11.

```powershell
cd D:\market_regime_detection
py -3.11 -m venv .venv
.\.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m ipykernel install --user --name market-regime --display-name "Python 3.11 Market Regime"
```

If `xgboost`, `lightgbm`, or `torch` fails to install, the core notebooks still run with Random Forest, HistGradientBoosting, HMM, and scikit-learn.

## Optional FRED Setup

For macro features, create a free FRED API key and set:

```powershell
setx FRED_API_KEY "your_api_key_here"
```

Restart VS Code or Jupyter after setting it.

## RBI Macro Data

RBI DBIE is the suggested source for Indian macro variables. The recommended workflow is to export selected RBI series manually as CSV into:

```text
data/external/rbi_macro.csv
```

Expected columns:

```text
Date, india_cpi, india_repo_rate, india_gdp_growth, india_10y_yield, usd_inr
```

## Final Research Conclusion

Regime-aware modeling improved risk-adjusted performance over single-model baselines in walk-forward testing. The strongest result came from combining HMM regime probabilities with specialized expert classifiers through a Soft-Gated Mixture of Experts. After transaction costs and volatility-based position sizing, the strategy still outperformed buy-and-hold on Sharpe ratio and maximum drawdown.

## Resume Bullet

Built a multi-asset regime-aware financial ML system using HMMs to detect Bull, Bear, and Crisis states across S&P 500, NASDAQ, and NIFTY, then trained Mixture-of-Experts classifiers with Random Forest, XGBoost, and LightGBM comparisons, improving walk-forward risk-adjusted performance after transaction costs and volatility-based position sizing.
