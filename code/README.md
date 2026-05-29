# LightGBM Hybrid Tweedie Model (Scenario A, Honest Val)

This directory contains the self-contained codebase, configuration, and documentation for the **LightGBM Hybrid Tweedie** model under **Scenario A** utilizing the **Honest 3-Year Lag Validation Protocol**.

---

## 1. Model Design & Architecture

The **LightGBM Hybrid Tweedie** model is designed for predicting 3-year-ahead truck freight volume (`tons`) at the granular route level (or commodity level if configured). It addresses two key characteristics of freight data: the zero-inflated skewness of freight tonnage and the presence of volatile "shock" routes.

### Hybrid Blending System (Shock Routing Blending)
Instead of predicting all routes using a single machine learning model, the hybrid system splits the routes into **Stable** and **Shock** categories:
1. **Shock Routes Detection**: Based on the past 4 years of historical data (e.g., from `context_year - 3` to `context_year`), the model calculates the absolute Year-over-Year (YoY) variance. It applies an Exponentially Weighted Moving Average (EWMA) to place more weight on recent changes and flags the top 15% most volatile routes as "Shock" routes.
2. **ML routing**:
   - **Stable Routes**: The model bypasses the LightGBM regressor and falls back to a robust historical baseline (e.g., 3-year historical median target tonnage). This prevents overfitting to noise on stable, low-variance routes.
   - **Shock Routes**: The model applies a LightGBM Regressor (Tweedie objective, $p = 1.5$) to capture non-linear trend adjustments and economic growth patterns.
3. **Target Transformation**: The model predicts the **Ratio** between the target year tons and the baseline tons:
   $$\text{target} = \log(1 + y) - \log(1 + \text{base\_pred})$$
   Alphas ($\alpha$) are dynamically optimized at the validation stage to scale and blend the predicted ratios back to tons.

---

## 2. Honest 3-Year Lag Validation Setup

Traditional walk-forward splits suffer from temporal leakage if they use validation target years that occur after the test set context year. This "Honest" version resolves look-ahead bias through a strict validation horizon.

### Purging & Embargo Protocol
* **Context Year**: The anchor year from which we look forward to the future target.
* **Test Year**: $t_{\text{test}} = \text{context\_year} + 3$ (3-year lag forecast).
* **Validation Year**: $t_{\text{val}} = t_{\text{test}} - 3$ (which matches the context year).
* **Training Year Limit**: To predict the validation target $t_{\text{val}}$ out-of-sample, we can only train on historical targets up to $t_{\text{val}} - 2$ (allowing a 1-year purging buffer to ensure no target leakage).
* **Train Split Definition**:
  * **Split 1**: Context = 2019, Val = 2019, Test = 2022. Train range: `2015-2016`
  * **Split 2**: Context = 2020, Val = 2020, Test = 2023. Train range: `2015-2017`
  * **Split 3**: Context = 2021, Val = 2021, Test = 2024. Train range: `2015-2018`

---

## 3. Data & Feature Engineering

The model uses a total of **52 features** categorized into four groups:

1. **Context Attributes (8 features)**:
   * Label-encoded `origin` & `destination`.
   * Current volume metrics: `tons_ctx`, `value_ctx`, `tmiles_ctx`.
   * Local statistics: `tons_mean_c`, `tons_std_c`, `n_commodity`.
   * Era indicator: `is_faf5_era` (representing FAF5 data post-2017).
2. **Historical Time-Series Trends (5 features)**:
   * Extracted from `train.csv` historical records.
   * Features: `tons_recent_avg`, `tons_cagr`, `tons_volatility`, `tons_yoy_last`, `tons_trend_slope`.
3. **External Datasets (39 features)**:
   * State-level economic metrics (GDP, unemployment rate, population growth, median income, Net Migration, personal consumption expenditures (PCE) total/goods, Heating Degree Days, NOAA drought index PDSI, and USDA crop yields).
   * National-level energy and commodity indices (Natural gas electricity generation, FAO DAP & Urea fertilizer price indices).
4. **Derived Indicators (4 features)**:
   * `gdp_ratio`, `pop_ratio`, `income_ratio`, `freight_intensity`.

---

## 4. Experimental Progress & Results

Under Scenario A (where Multidimensional Bridging aligns historical FAF4 scales to FAF5 standards), this model achieved the best experimental performance.

### Best Run Metrics (`a82b8bc8f0c641888d8a09f9908c9281`):
* **Test WMAPE**: **4.21%** (Weighted Mean Absolute Percentage Error)
* **Test Skill Score vs. Naive**: **+26.08%** (improvement over naive prediction)
* **RMSE**: **1,946.7** (unweighted Route-level Root Mean Squared Error)
* **Weighted R² Score**: **0.865**

*Note: In earlier runs, RMSE was tons-weighted (wRMSE $\approx 24,365.6$). In the finalized protocol, unweighted RMSE is mapped to the `RMSE` metric, with the actual WMAPE remaining highly stable at 4.21%.*

---

## 5. Directory Structure & Files

```
export_honest_hybrid_tweedie/
├── README.md                 # This documentation
├── run_honest_experiment.py  # Pipeline runner (LightGBM Hybrid Tweedie and robust selection)
├── modeling_baseline.py      # Feature engineering and base evaluation helper
├── load_external.py         # External datasets processing and normalization
└── experiment_config.json    # Experiment configuration (MLflow config & seeds)
```

---

## 6. How to Run

### Prerequisite Environment
Ensure you have the required packages installed. You can set up the environment using `uv` (recommended) or `pip`:
```bash
pip install numpy pandas scikit-learn lightgbm mlflow
```

### Execution
To run the best experiment for **Scenario A** (Full History with multidimensional bridging):
```bash
python run_honest_experiment.py --scenario A --model LightGBM_Hybrid_Tweedie
```

To run under **Scenario B** (FAF5 data only, 2017+):
```bash
python run_honest_experiment.py --scenario B --model LightGBM_Hybrid_Tweedie
```

### Configurations
You can edit the `experiment_config.json` file to modify parameters such as:
* `mlflow.tracking_uri`: MLflow server tracking URL.
* `run_defaults.random_seed`: Seed used for reproducibility.
* `run_defaults.granularity`: Switch between `route` (origin-destination) and `commodity` (origin-destination-commodity) levels.
