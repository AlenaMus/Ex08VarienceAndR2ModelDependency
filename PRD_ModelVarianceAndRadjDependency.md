# PRD: ModelVarianceAndRadjDependency.py

## 1. Objective

To introduce the concept of **Adjusted R-squared (R²_adj)** and demonstrate its utility as a more robust metric for model evaluation, especially in the context of multiple regression. The script illustrates how Adjusted R² penalizes the model for the number of predictors included.

## 2. Requirements

*   **Language**: Python 3
*   **Libraries**:
    *   `numpy`
    *   `scikit-learn`
    *   `pandas`
    *   `matplotlib`
*   **Input Data**: None. The script generates its own synthetic data.

## 3. Process

1.  **Data Generation**:
    *   The script generates a synthetic dataset with 200 observations (`n`) and 70 predictor variables (`p`), similar to the other scripts.
    *   Unique random error (`epsilon_i`) is generated for each observation.
2.  **Model Fitting**:
    *   A `LinearRegression` model is fitted to the data.
3.  **Metrics Calculation**:
    *   The script calculates **SSE**, **SST**, and the standard **R-squared**.
    *   It then calculates **Adjusted R-squared** using its formula, which incorporates the number of observations (`n`) and the number of predictors (`p`):
        `R²_adj = 1 - [(1 - R²) * (n - 1) / (n - p - 1)]`
    *   The console output explicitly shows the calculation steps for Adjusted R².
4.  **Simulation**:
    *   A simulation is run across 40 different noise levels. For each level, SSE, R², and Adjusted R² are calculated and stored.
5.  **Visualization**:
    *   The script generates a comprehensive set of plots:
        1.  A primary scatter plot of **SSE vs. Adjusted R²**.
        2.  A comparative plot of **SSE vs. R²**.
        3.  A line plot comparing **R² and Adjusted R²** directly against SSE, highlighting the "penalty" gap between them.
        4.  A quality-region plot for Adjusted R² vs. SSE.

## 4. Output

*   **Console Output**:
    *   Details of the data generation and model fitting.
    *   Calculations for SSE, SST, and R².
    *   A detailed breakdown of the **Adjusted R² calculation**, showing the formula and the values used.
    *   The final values for both R² and Adjusted R², along with their difference.
*   **Visualizations**:
    *   A 2x2 grid of plots that visually articulate the relationship between SSE, R², and Adjusted R². The key plot directly compares the curves of R² and Adjusted R² as SSE changes, making the penalty for additional parameters visually explicit.

## 5. Key Demonstrated Concept

The central concept is the superiority of **Adjusted R-squared** over the standard R-squared when evaluating models with multiple predictors. The script demonstrates that:

1.  **R-squared** can be misleadingly high, as it tends to increase with the addition of any predictor, regardless of its actual significance.
2.  **Adjusted R-squared** provides a more honest assessment of model fit by imposing a penalty for the number of predictors (`p`) relative to the number of observations (`n`).
3.  The difference between R² and Adjusted R² represents this penalty. This gap widens as the number of predictors increases.
4.  Like R², Adjusted R² still maintains an inverse relationship with the model's error (SSE).
