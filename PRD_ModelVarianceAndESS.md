# PRD: ModelVarianceAndESS.py

## 1. Objective

To demonstrate and visualize the fundamental inverse relationship between the Sum of Squared Errors (SSE) and the R-squared (R²) coefficient of determination in a linear regression model.

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
    *   A synthetic dataset is created with 200 observations and 70 predictor variables.
    *   True `beta` parameters (coefficients) are randomly generated.
    *   A realistic level of random noise (`epsilon`) is added to the target variable `y`. This ensures the model is not a perfect fit.
2.  **Model Fitting**:
    *   A standard `LinearRegression` model from `scikit-learn` is fitted to the generated data (`X` and `y`).
3.  **Metrics Calculation**:
    *   The model makes predictions (`y_hat`).
    *   **SSE** is calculated as the sum of the squared differences between the actual `y` and predicted `y_hat` values.
    *   **SST** (Total Sum of Squares) is calculated as the sum of squared differences between `y` and its mean.
    *   **R-squared** is calculated using the formula: `R² = 1 - (SSE / SST)`.
4.  **Simulation**:
    *   The script iterates 40 times, each time generating a new `y` vector with a different level of random noise.
    *   For each iteration, a new model is fit, and the resulting SSE and R² are stored.
5.  **Visualization**:
    *   The script generates two plots to visualize the inverse relationship between SSE and R².

## 4. Output

*   **Console Output**:
    *   Details of the data generation process (number of samples, features, etc.).
    *   Estimated model parameters vs. true parameters.
    *   Calculated SSE, SST, and R² values.
    *   A summary table of the key metrics.
*   **Visualizations**:
    *   A plot of **Actual vs. Predicted** values.
    *   A **Residual Plot** to show the distribution of errors.
    *   A scatter plot of **SSE vs. R²**, where each point represents a model with a different noise level. This plot clearly shows that as SSE increases, R² decreases.

## 5. Key Demonstrated Concept

The primary takeaway is the direct mathematical relationship between a model's error and its explanatory power. A lower SSE indicates a better fit and results in a higher R², meaning the model explains a larger portion of the variance in the target variable. Conversely, a higher SSE (more error) leads to a lower R².
