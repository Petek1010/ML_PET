import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# Generate Non-Scaled Data (Different Distributions)
np.random.seed(42)
x1 = np.random.normal(loc=10, scale=1, size=500)  # Normal Distribution
x2 = np.random.normal(loc=-5, scale=1, size=500)   # Different mean & scale

# Introduce Outliers in x1
x1_with_outliers = np.append(x1, [60, 60, 60])  # Adding extreme values
x2_with_outliers = np.append(x2, [5, 6, 7])      # Less extreme outliers
x2_short = np.random.normal(loc=-5, scale=1, size=500)
x2_outliers = np.random.binomial(200,0.9,250)
x2_join = np.concatenate((x2_short, x2_outliers))
# Reshape for StandardScaler
X_original = np.column_stack((x1_with_outliers, x2_with_outliers))

# Apply Z-score Normalization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_original)

X_firstExample = np.column_stack((x1, x2))
X_firstExample = scaler.fit_transform(X_firstExample)

# Create Plots
fig, axes = plt.subplots(2, 3, figsize=(12, 6))

# Plot Original Data (Before Scaling)
sns.kdeplot(x=x1, label="x1", linestyle="--", color="brown", ax=axes[0, 0])
sns.kdeplot(x=x2, label="x2", linestyle=":", color="blue", ax=axes[0, 0])
axes[0, 0].set_title("Nonscaled data")
axes[0, 0].legend()

sns.kdeplot(x=x1, label="x1", linestyle="--", color="brown", ax=axes[0, 1])
sns.kdeplot(x=x2_join, label="x2", linestyle=":", color="blue", ax=axes[0, 1])
axes[0, 1].set_title("Izobčenci")
axes[0, 1].annotate("Outliers", xy=(90, 0.02), xytext=(60, 0.05), arrowprops=dict(arrowstyle="->"))
axes[0, 1].legend()

sns.kdeplot(x=x1, label="x1", linestyle="--", color="brown", ax=axes[0, 2])
sns.kdeplot(x=x2, label="x2", linestyle=":", color="blue", ax=axes[0, 2])
axes[0, 2].set_title("Nonscaled data")
axes[0, 2].legend()

# Plot Standard Scaled Data
sns.kdeplot(x=X_firstExample[:, 0], label="x1", linestyle="--", color="brown", ax=axes[1, 0])
sns.kdeplot(x=X_firstExample[:, 1], label="x2", linestyle=":", color="blue", ax=axes[1, 0])
axes[1, 0].set_title("Z-vrednost normalizacija")
axes[1, 0].legend()

sns.kdeplot(x=X_scaled[:, 0], label="x1", linestyle="--", color="brown", ax=axes[1, 1])
sns.kdeplot(x=X_scaled[:, 1], label="x2", linestyle=":", color="blue", ax=axes[1, 1])
axes[1, 1].set_title("Standard Scaler")
axes[1, 1].legend()

sns.kdeplot(x=X_scaled[:, 0], label="x1", linestyle="--", color="brown", ax=axes[1, 2])
sns.kdeplot(x=X_scaled[:, 1], label="x2", linestyle=":", color="blue", ax=axes[1, 2])
axes[1, 2].set_title("Standard Scaler")
axes[1, 2].legend()

plt.tight_layout()
plt.show()
