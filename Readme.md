# Linear Regression: Variance, R-squared, and Adjusted R-squared

This repository contains a set of Python scripts designed to visually and mathematically demonstrate key concepts in linear regression. The scripts explore the relationships between model error (variance), the number of parameters, and the goodness-of-fit metrics R-squared and Adjusted R-squared.

Each script generates a synthetic dataset, fits a linear regression model, and produces visualizations to illustrate these concepts.

## Scripts

### 1. `ModelVarianceAndESS.py`

This script demonstrates the fundamental inverse relationship between the **Sum of Squared Errors (SSE)** and the **R-squared (R²)** coefficient.

- **Purpose**: To show that as the model's prediction errors increase, the R² value, which represents the proportion of variance explained by the model, decreases.
- **Method**: It generates data with a defined level of random noise, fits a model, and then runs a simulation with varying noise levels to plot the clear inverse correlation between SSE and R².

### 2. `ModelVarianceAndESSOnMoreParams.py`

This script reinforces the concepts from the first program using a model with a higher number of parameters and a greater level of noise.

- **Purpose**: To provide a more detailed look at how unique random error for each observation impacts the model's fit and to confirm the SSE vs. R² relationship in a different scenario.
- **Method**: It explicitly models a unique random error (`epsilon_i`) for each data point, simulating a more realistic data generation process. It then performs the same analysis as the first script, confirming the inverse relationship between SSE and R².

### 3. `ModelVarianceAndRadjDependency.py`

Building on the previous scripts, this program introduces the concept of **Adjusted R-squared (R²_adj)**.

- **Purpose**: To demonstrate how Adjusted R² provides a more conservative and realistic measure of model fit by penalizing the inclusion of numerous predictor variables.
- **Method**: It calculates both R² and Adjusted R² and visualizes the difference between them. The script shows that Adjusted R² is always less than or equal to R² and highlights the "penalty" for adding more features. It also confirms that Adjusted R² maintains an inverse relationship with SSE.
