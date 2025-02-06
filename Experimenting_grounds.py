from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

# Sample dataset
X = [[1, 2], [3, 4], [5, 6], [7, 8]]
y = [0, 1, 0, 1]

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize scaler
scaler = StandardScaler()

# 🔹 Fit & transform training data (computes mean/std, then scales)
X_train_scaled = scaler.fit_transform(X_train)

# 🔹 Transform test data using the same scaling parameters (NO fitting)
X_test_scaled = scaler.transform(X_test)

print("Original X_train:", X_train)
print("Scaled X_train:", X_train_scaled)
print("Original X_test:", X_test)
print("Scaled X_test:", X_test_scaled)