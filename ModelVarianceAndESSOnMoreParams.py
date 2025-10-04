import numpy as np
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

print("=" * 80)
print("REALISTIC REGRESSION: 200 Observations with 70 Parameters")
print("=" * 80)

# ============================================
# SETUP: Generate REALISTIC data
# ============================================
n_samples = 200  # 200 observations (MORE than parameters!)
n_features = 70  # 70 predictor variables
n_betas = 71     # 71 total: beta_0 + beta_1...beta_70

print("\nWHY THIS IS REALISTIC:")
print(f"  * We have {n_samples} observations but only {n_betas} parameters")
print(f"  * Ratio: {n_samples}/{n_betas} = {n_samples/n_betas:.1f} observations per parameter")
print(f"  * This prevents overfitting and gives realistic R^2 < 1")
print(f"  * The model CANNOT perfectly fit all data points\n")

# Generate true parameters
beta_0_true = 5.0
beta_slopes_true = np.random.randn(n_features) * 0.8  # Reduced effect of features

print("Generated TRUE parameters:")
print(f"  beta_0 = {beta_0_true:.4f}")
print(f"  First 5 slopes: {beta_slopes_true[:5]}")

# Generate X matrix (200 samples × 70 features)
X = np.random.randn(n_samples, n_features) * 2 + 3

print(f"\nGenerated X matrix:")
print(f"  Shape: {X.shape} (200 observations × 70 features)")
print(f"  Mean: {X.mean():.2f}, Std: {X.std():.2f}")

# Generate DIFFERENT random error for EACH observation
epsilon = np.random.randn(n_samples) * 8.0  # 200 different random errors!

print(f"\nGenerated errors (epsilon) - DIFFERENT for each observation:")
print(f"  Shape: {epsilon.shape} (200 unique random values)")
print(f"  Mean: {epsilon.mean():.2f}, Std: {epsilon.std():.2f}")
print(f"  Noise level (sigma) = 8.0 (constant), but each epsilon_i is random")
print(f"\n  First 10 error values (all different):")
for i in range(10):
    print(f"    epsilon[{i}] = {epsilon[i]:7.3f}")
print(f"  ...")
print(f"  Last error: epsilon[199] = {epsilon[-1]:7.3f}")

# Generate original y values (200 values, EACH with its own random error)
y_original = beta_0_true + X @ beta_slopes_true + epsilon

print(f"\nGenerated y_original (each has DIFFERENT random error):")
print(f"  Shape: {y_original.shape} (200 observations)")
print(f"  Mean: {y_original.mean():.2f}, Std: {y_original.std():.2f}")
print(f"\n  Formula for each observation:")
print(f"    y_1 = beta_0 + sum(beta_j*x_1_j) + epsilon_1 = {beta_0_true} + {np.dot(X[0], beta_slopes_true):.3f} + {epsilon[0]:.3f} = {y_original[0]:.3f}")
print(f"    y_2 = beta_0 + sum(beta_j*x_2_j) + epsilon_2 = {beta_0_true} + {np.dot(X[1], beta_slopes_true):.3f} + {epsilon[1]:.3f} = {y_original[1]:.3f}")
print(f"    y_3 = beta_0 + sum(beta_j*x_3_j) + epsilon_3 = {beta_0_true} + {np.dot(X[2], beta_slopes_true):.3f} + {epsilon[2]:.3f} = {y_original[2]:.3f}")
print(f"    ...")
print(f"  Each y_i has its own unique epsilon_i!")

# ============================================
# FIT MODEL
# ============================================
print("\n" + "=" * 80)
print("FITTING REGRESSION MODEL")
print("=" * 80)

model = LinearRegression()
model.fit(X, y_original)

beta_0_est = model.intercept_
beta_slopes_est = model.coef_

print(f"\nEstimated parameters:")
print(f"  beta_hat_0 = {beta_0_est:.4f} (True: {beta_0_true:.4f})")
print(f"  First 5 estimated slopes: {beta_slopes_est[:5]}")
print(f"  First 5 true slopes: {beta_slopes_true[:5]}")
print(f"\nNote: Estimates differ from true values due to noise!")

# ============================================
# CALCULATE PREDICTED VALUES FOR ALL 200
# ============================================
print("\n" + "=" * 80)
print("PREDICTED VALUES: Calculated for ALL 200 instances")
print("=" * 80)

y_predicted = model.predict(X)

print(f"\ny_predicted shape: {y_predicted.shape} (200 predictions)")
print(f"\nFirst 10 comparisons (showing DIFFERENT errors for each):")
print(f"{ 'i':<4} { 'y_i (actual)':<14} { 'y_hat_i (predicted)':<16} { 'epsilon_i (true error)':<18} { 'e_i (residual)':<15}")
print("-" * 75)
for i in range(10):
    print(f"{i:<4} {y_original[i]:12.3f}  {y_predicted[i]:14.3f}  {epsilon[i]:16.3f}  {y_original[i]-y_predicted[i]:13.3f}")
print(f"\nNotice: Each observation has its OWN unique error epsilon_i!")

# ============================================
# CALCULATE SSE AND R^2
# ============================================
print("\n" + "=" * 80)
print("SSE AND R^2 CALCULATION")
print("=" * 80)

residuals = y_original - y_predicted
residuals_squared = residuals**2
SSE = np.sum(residuals_squared)

y_mean = np.mean(y_original)
SST = np.sum((y_original - y_mean)**2)
R_squared = 1 - (SSE / SST)

print(f"\nSSE = sum(y_i - y_hat_i)^2 = {SSE:.4f}")
print(f"SST = sum(y_i - y_bar)^2 = {SST:.4f}")
print(f"R^2 = 1 - (SSE/SST) = {R_squared:.6f}")
print(f"\nMSE (Mean Squared Error): {SSE/n_samples:.4f}")
print(f"RMSE (Root Mean Squared Error): {np.sqrt(SSE/n_samples):.4f}")

print(f"\nV R^2 = {R_squared:.4f} is REALISTIC!")
print(f"V The model explains {R_squared*100:.2f}% of variance")
print(f"V {(1-R_squared)*100:.2f}% remains unexplained (due to noise)")
print(f"\nThis represents real-world data where:")
print(f"  * There's significant measurement error")
print(f"  * Not all factors affecting y are captured")
print(f"  * Random variation exists in the system")

# ============================================
# VISUALIZATIONS
# ============================================
print("\n" + "=" * 80)
print("CREATING VISUALIZATIONS...")
print("=" * 80)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Sample of actual vs predicted
ax1 = axes[0, 0]
sample_size = 50  # Show first 50 for clarity
ax1.scatter(range(1, sample_size+1), y_original[:sample_size], alpha=0.6, s=50, label='y_i (actual)', color='blue')
ax1.scatter(range(1, sample_size+1), y_predicted[:sample_size], alpha=0.6, s=50, label='y_hat_i (predicted)', color='red')
for i in range(sample_size):
    ax1.plot([i+1, i+1], [y_original[i], y_predicted[i]], 'k-', alpha=0.2, linewidth=0.5)
ax1.set_xlabel('Observation index (first 50 of 200)')
ax1.set_ylabel('Value')
ax1.set_title('Sample: Actual vs Predicted Values')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Plot 2: Actual vs Predicted scatter
ax2 = axes[0, 1]
ax2.scatter(y_original, y_predicted, alpha=0.4, s=30)
ax2.plot([y_original.min(), y_original.max()], \
         [y_original.min(), y_original.max()], \
         'r--', linewidth=2, label='Perfect prediction')
ax2.set_xlabel('y_i (actual)')
ax2.set_ylabel('y_hat_i (predicted)')
ax2.set_title(f'All 200 Points: Actual vs Predicted\nR^2 = {R_squared:.4f}')
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Residuals
ax3 = axes[1, 0]
ax3.scatter(y_predicted, residuals, alpha=0.4, s=30)
ax3.axhline(0, color='r', linestyle='--', linewidth=2)
ax3.set_xlabel('y_hat_i (predicted)')
ax3.set_ylabel('e_i = y_i - y_hat_i (residuals)')
ax3.set_title(f'Residual Plot (200 observations)\nSSE = {SSE:.2f}')
ax3.grid(True, alpha=0.3)

# Plot 4: Histogram of residuals
ax4 = axes[1, 1]
ax4.hist(residuals, bins=30, alpha=0.7, color='orange', edgecolor='black')
ax4.axvline(0, color='r', linestyle='--', linewidth=2)
ax4.set_xlabel('Residuals (e_i)')
ax4.set_ylabel('Frequency')
ax4.set_title(f'Distribution of Residuals\nMean = {residuals.mean():.3f}, Std = {residuals.std():.3f}')
ax4.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# ============================================
# SSE vs R^2 RELATIONSHIP WITH VARYING NOISE
# ============================================
print("\n" + "=" * 80)
print("EXPLORING SSE vs R^2 RELATIONSHIP (Varying Noise Levels)")
print("=" * 80)

# Generate multiple models with different noise levels
noise_levels = np.linspace(2.0, 15, 40)
SSE_values = []
R2_values = []

print("\nGenerating 40 models with different noise levels...")

for noise in noise_levels:
    # Generate new y with different noise
    epsilon_varied = np.random.randn(n_samples) * noise
    y_varied = beta_0_true + X @ beta_slopes_true + epsilon_varied
    
    # Fit model
    model_varied = LinearRegression()
    model_varied.fit(X, y_varied)
    
    # Predict and calculate metrics
    y_pred_varied = model_varied.predict(X)
    residuals_varied = y_varied - y_pred_varied
    
    SSE_varied = np.sum(residuals_varied**2)
    SST_varied = np.sum((y_varied - np.mean(y_varied))**2)
    R2_varied = 1 - (SSE_varied / SST_varied)
    
    SSE_values.append(SSE_varied)
    R2_values.append(R2_varied)

print(f"SSE range: {min(SSE_values):.2f} to {max(SSE_values):.2f}")
print(f"R^2 range: {min(R2_values):.4f} to {max(R2_values):.4f}")

# Create visualization of SSE vs R^2 relationship
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: SSE (x-axis) vs R^2 (y-axis) scatter
ax1 = axes[0]
scatter = ax1.scatter(SSE_values, R2_values, c=noise_levels, cmap='viridis', s=100, alpha=0.7)
plt.colorbar(scatter, ax=ax1, label='Noise Level (sigma)')
ax1.set_xlabel('SSE (Sum of Squared Errors)', fontsize=12, fontweight='bold')
ax1.set_ylabel('R^2 (Coefficient of Determination)', fontsize=12, fontweight='bold')
ax1.set_title('SSE vs R^2 Relationship\n(Each point = different noise level)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)

# Annotate key points
min_sse_idx = np.argmin(SSE_values)
max_sse_idx = np.argmax(SSE_values)
ax1.scatter(SSE_values[min_sse_idx], R2_values[min_sse_idx], \
           color='green', s=300, marker='*', edgecolors='black', linewidths=2,\
           label=f'Low noise: R^2={R2_values[min_sse_idx]:.3f}', zorder=5)
ax1.scatter(SSE_values[max_sse_idx], R2_values[max_sse_idx], \
           color='red', s=300, marker='X', edgecolors='black', linewidths=2,\
           label=f'High noise: R^2={R2_values[max_sse_idx]:.3f}', zorder=5)

# Highlight our original model
ax1.scatter(SSE, R_squared, \
           color='blue', s=300, marker='D', edgecolors='white', linewidths=2,\
           label=f'Our model: R^2={R_squared:.3f}', zorder=5)
ax1.legend(loc='best')

# Add formula annotation
formula_text = 'R^2 = 1 - SSE/SST\n\nincreases SSE -> decreases R^2\ndecreases SSE -> increases R^2'
ax1.text(0.05, 0.95, formula_text, transform=ax1.transAxes,\
         fontsize=11, verticalalignment='top',\
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# Plot 2: Show the inverse relationship more clearly
ax2 = axes[1]
sorted_indices = np.argsort(SSE_values)
SSE_sorted = np.array(SSE_values)[sorted_indices]
R2_sorted = np.array(R2_values)[sorted_indices]

ax2.plot(SSE_sorted, R2_sorted, 'b-', linewidth=2.5, marker='o', markersize=4, alpha=0.7)
ax2.set_xlabel('SSE (Sum of Squared Errors)', fontsize=12, fontweight='bold')
ax2.set_ylabel('R^2 (Coefficient of Determination)', fontsize=12, fontweight='bold')
ax2.set_title('Inverse Relationship\n(As SSE increases, R^2 decreases)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)

# Highlight quality regions
ax2.axhline(y=0.9, color='green', linestyle='--', alpha=0.5, linewidth=2, label='Excellent fit (R^2 > 0.9)')
ax2.axhline(y=0.7, color='yellow', linestyle='--', alpha=0.5, linewidth=2, label='Good fit (R^2 > 0.7)')
ax2.axhline(y=0.5, color='orange', linestyle='--', alpha=0.5, linewidth=2, label='Moderate fit (R^2 > 0.5)')
ax2.axhline(y=0.3, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Poor fit (R^2 < 0.3)')

# Mark our model
# Find the simulated model that is closest to our original model's noise level (8.0)
our_model_noise = 8.0
closest_noise_idx = (np.abs(noise_levels - our_model_noise)).argmin()
ax2.scatter(SSE_values[closest_noise_idx], R2_values[closest_noise_idx], \
           color='blue', s=300, marker='D', edgecolors='white', linewidths=2, zorder=5, \
           label=f'Our model (noise~{our_model_noise:.1f}): R^2={R2_values[closest_noise_idx]:.3f}')

ax2.legend(loc='best', fontsize=9)

plt.tight_layout()
plt.show()

# ============================================
# MATHEMATICAL RELATIONSHIP
# ============================================
print("\n" + "=" * 80)
print("MATHEMATICAL RELATIONSHIP: SSE and R^2")
print("=" * 80)
print(f"""

The formula shows the inverse relationship:

    R^2 = 1 - (SSE / SST)

For our dataset (200 observations, 70 parameters):
    * SST = {SST:.4f} (constant - total variance in y)
    * SSE = {SSE:.4f} (varies with model fit)
    * R^2 = 1 - ({SSE:.4f}/{SST:.4f}) = {R_squared:.6f}

KEY INSIGHTS:
    * With {n_samples} observations and {n_features} features: realistic R^2
    * Increased noise (sigma=8.0) gives R^2 around 0.8
    * More noise -> Higher SSE -> Lower R^2
    * Less noise -> Lower SSE -> Higher R^2
    
REALISTIC DATA (R^2 approx 0.8):
    V We have 200 observations but only 71 parameters
    V High random noise (sigma = 8.0)
    V Model explains ~80% of variance
    V ~20% unexplained (inherent randomness)
    V This mimics real-world scenarios like:
      - Economic data with many unmeasured factors
      - Biological systems with natural variation
      - Survey data with measurement error
""")

# ============================================
# SUMMARY TABLE
# ============================================
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)

summary_df = pd.DataFrame({
    'Metric': ['Observations', 'Features', 'Parameters', 'Noise Level (sigma)', 
               'SSE', 'SST', 'R^2', 'Variance Explained'],
    'Value': [n_samples, n_features, n_betas, '8.0',
              f'{SSE:.2f}', f'{SST:.2f}', f'{R_squared:.4f}', f'{R_squared*100:.2f}%']
})

print(summary_df.to_string(index=False))

print(f"""

V REALISTIC model with R^2 approx 0.8 (target achieved!)
V Model explains about {R_squared*100:.1f}% of variance
V About {(1-R_squared)*100:.1f}% unexplained (realistic level of noise)
V SSE and R^2 show clear INVERSE relationship
V This represents typical real-world data quality
""")
