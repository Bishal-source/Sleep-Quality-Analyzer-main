import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import os

# Load dataset
data_path = "C:/Users/bishal/Desktop/Sleep-Quality-Analyzer-main/dataset/sleep_dataset_balanced_influence.csv"
output_dir = "C:/Users/bishal/Desktop/Sleep-Quality-Analyzer-main/test_new"

df = pd.read_csv(data_path)
print("Dataset shape:", df.shape)
print("\nColumns:", df.columns.tolist())
print("\nFirst few rows:")
print(df.head())

# Handle missing values
print("\nMissing values:")
print(df.isnull().sum())

# Drop rows with missing values for PCA
df_clean = df.dropna()
print(f"\nAfter removing missing values: {df_clean.shape[0]} rows")

# Separate features and target
feature_cols = ['Sleep Duration', 'Stress Level', 'Physical Activity', 'Caffeine Intake',
                'Screen Time', 'SpO2 Before', 'SpO2 After', 'Heart Rate Before', 'Heart Rate After']
target_col = 'Sleep Quality'

X = df_clean[feature_cols]
y = df_clean[target_col]

print(f"\nFeatures shape: {X.shape}")
print(f"Target shape: {y.shape}")

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Perform PCA
pca = PCA()
X_pca = pca.fit_transform(X_scaled)

# Explained variance
explained_variance = pca.explained_variance_ratio_
cumulative_variance = np.cumsum(explained_variance)

print("\n" + "="*50)
print("PCA RESULTS")
print("="*50)
print("\nExplained Variance Ratio by Component:")
for i, (var, cum) in enumerate(zip(explained_variance, cumulative_variance)):
    print(f"PC{i+1}: {var*100:.2f}% (Cumulative: {cum*100:.2f}%)")

# Save PCA results to CSV
pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
pca_df['Sleep Quality'] = y.values
pca_df.to_csv(os.path.join(output_dir, "pca_results.csv"), index=False)
print(f"\nPCA results saved to: {os.path.join(output_dir, 'pca_results.csv')}")

# Save explained variance
variance_df = pd.DataFrame({
    'Component': [f'PC{i+1}' for i in range(len(explained_variance))],
    'Explained_Variance': explained_variance,
    'Cumulative_Variance': cumulative_variance
})
variance_df.to_csv(os.path.join(output_dir, "explained_variance.csv"), index=False)
print(f"Explained variance saved to: {os.path.join(output_dir, 'explained_variance.csv')}")

# Create visualizations
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Scree plot
axes[0].bar(range(1, len(explained_variance)+1), explained_variance, alpha=0.7, label='Individual')
axes[0].plot(range(1, len(explained_variance)+1), cumulative_variance, 'ro-', label='Cumulative')
axes[0].axhline(y=0.95, color='g', linestyle='--', label='95% threshold')
axes[0].set_xlabel('Principal Component')
axes[0].set_ylabel('Explained Variance Ratio')
axes[0].set_title('Scree Plot')
axes[0].legend()
axes[0].set_xticks(range(1, len(explained_variance)+1))

# Plot 2: PCA scatter (first 2 components colored by Sleep Quality)
scatter = axes[1].scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, ax=axes[1], label='Sleep Quality')
axes[1].set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
axes[1].set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
axes[1].set_title('PCA: First Two Components')

plt.tight_layout()
plt.savefig(os.path.join(output_dir, "pca_visualization.png"), dpi=150)
print(f"Visualization saved to: {os.path.join(output_dir, 'pca_visualization.png')}")

# Component loadings
print("\n" + "="*50)
print("COMPONENT LOADINGS")
print("="*50)
loadings = pd.DataFrame(pca.components_.T, columns=[f'PC{i+1}' for i in range(len(explained_variance))], index=feature_cols)
print(loadings.round(4))

# Save loadings
loadings.to_csv(os.path.join(output_dir, "pca_loadings.csv"))
print(f"\nLoadings saved to: {os.path.join(output_dir, 'pca_loadings.csv')}")

print("\n" + "="*50)
print("PCA ANALYSIS COMPLETE")
print("="*50)