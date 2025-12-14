# Kurtay-Final-Project
This project introduces a robust risk forecasting framework that improves upon conventional volatility models. By integrating an **EGARCH(1,3)** volatility filter with **Pearson Type VII innovations** estimated via **L-moments**, this model addresses the heavy tails and volatility asymmetry often observed in financial markets.

The study analyzes 25 years of daily returns (2000–2024) for 24 major US equities, demonstrating that the proposed framework consistently outperforms standard Normal and Student-t benchmarks in Value-at-Risk (VaR) coverage and goodness-of-fit tests.

## Key Features
* **Volatility Filtering:** Implementation of the Exponential GARCH (EGARCH 1,3) model to capture leverage effects and volatility clustering.
* **Robust Estimation:** Utilization of the Method of L-Moments (Hosking, 1990) to estimate Pearson Type VII distribution parameters, offering superior stability against outliers compared to Maximum Likelihood Estimation (MLE).
* **Regime-Based Bootstrapping:** Validation of model stability across different monetary policy environments (High, Medium, and Low interest rate regimes).
* **Risk Forecasting:** Out-of-sample Value-at-Risk (VaR) backtesting (2020–2024) validated via the Kupiec Proportion of Failures (POF) test.

## Data
The dataset consists of daily closing prices for **24 US equities** equally split between the **Technology** and **Industrial** sectors.
* **Period:** January 1, 2000 – December 31, 2024.
* **Sectors:**
    * *Technology:* AAPL, ADBE, AMD, AMAT, CSCO, IBM, INTC, MSFT, NVDA, ORCL, QCOM, TXN.
    * *Industrials:* BA, CAT, DE, ETN, GE, GD, HON, LMT, MMM, RTX, UNP, UPS.
* **External Factors:** 3-Month Treasury Bill rates (`FRED_DGS3MO.csv`) used for regime stratification.

## Repository Structure

### Data & Preprocessing
* `Extract and Process Data`: Scripts for fetching raw equity data and calculating log-returns.
* `CLEANWRDS.csv`: Processed dataset containing clean return series.
* `FRED_DGS3MO.csv`: Historical 3-Month Treasury Bill rates.

### Core Analysis
* `Student_Residuals/L moments.py`: Python script implementing the L-moment estimation logic for Pearson Type VII parameters.
* `ProcessStudent-t`: Script for generating the benchmark Student-t EGARCH models and extracting standardized residuals.
* `import os.py`: Utility script for directory and path management.

### Results & Outputs
* `Student_Residuals/*.csv`: Standardized residual files for individual assets (e.g., `AAPL_resid.csv`, `MSFT_resid.csv`).
* `PearsonVII_Lmoments_Fit.csv`: Summary of fitted Pearson VII parameters (alpha, m) for all assets.
* `RateRegime_Boot_Lmoment.csv` & `RateRegime_Summary_Lmoment.csv`: Results from the block-bootstrap simulations across interest rate regimes.
* `ResidualOnly_Comparison_Lmoments.csv`: Goodness-of-fit comparison metrics (AIC/BIC).
* `var backtest`: Results from the out-of-sample Value-at-Risk backtesting.

### Visualizations
* `Student_Residuals/Residual visuals`: Generated plots of residual distributions.
* `Student_Residuals/Table1_Final.png`: Comparison table of AIC scores (Pearson vs. Benchmarks).
* `Student_Residuals/Table3_Bootstrap_Results.png`: Summary of bootstrap performance stability.

## Methodology Summary
1.  **Preprocessing:** Returns are winsorized (0.5%, 99.5%) and scaled.
2.  **Filtering:** An EGARCH(1,3) model removes serial correlation and conditional heteroskedasticity.
3.  **Innovation Modeling:** Standardized residuals are fitted to the Pearson Type VII distribution using L-moments (matching the first four L-moments: location, scale, L-skewness, L-kurtosis).
4.  **Benchmarking:** The model is compared against a standard Student-t EGARCH model estimated via MLE.

## Key Findings
* **Fit:** The Pearson Type VII specification produced the lowest AIC for **all 24 assets** in the sample.
* **VaR Accuracy:** The Pearson model achieved a significantly lower Mean Absolute Deviation from the 1% VaR target (0.0073) compared to the Student-t model (0.0079).
* **Stability:** Bootstrap simulations confirm the model's superiority holds across different interest rate regimes (High, Medium, Low).

## Dependencies
* Python 3.x
* `pandas`
* `numpy`
* `scipy`
* `statsmodels` (for GARCH/EGARCH implementations)
* `matplotlib` / `seaborn` (for visualization)
