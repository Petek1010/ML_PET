import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.base import clone
from sklearn.svm import SVC

# === your columns ===
numerical_cols = [
    "ADRP_NY_gis002","ADRP_SLO__gis001","CBDRP1010_NY_gis001","CBDRP_SLOV_gis001",
    "CJDRP_gis001_004","DLBRP_SLO_spm5_gis001","DMN_NY__gis001","HDRP_gis001",
    "MSARP_SLO_gis001","MSA_NY_gis001","PDCP_NY_gis002","PDRP_NY_gis001",
    "PDRP_SLO_gis001","PSPRP_NY_gis001","PSPRP_SLO_gis001","bFDRP_NY_gis001","bFDRP_SLO_gis001"
]
categorical_cols = ["FirstThreeChars"]

# === pipelines ===
numerical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="mean")),
    ('scaler', StandardScaler())
])
categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="most_frequent")),
    # set sparse_output=False to get dense arrays (sometimes helps compatibility)
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer([
    ('num', numerical_pipeline, numerical_cols),
    ('cat', categorical_pipeline, categorical_cols)
])

# === classifier: make sure to set random_state for reproducibility if supported ===
# Example: SVC
base_clf = SVC(kernel='rbf', C=1.0, probability=True, random_state=0)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', base_clf)
])

# === X, y must be your full dataset (pandas DataFrame / numpy arrays) ===
# Example placeholders (replace with real X, y):
# X = your_dataframe_or_array
# y = your_array_or_series

# === 1) cross_val_predict with LeaveOneOut ===
loo = LeaveOneOut()
pred_cvp = cross_val_predict(pipeline, X, y, cv=loo, method='predict', n_jobs=1)

# === 2) manual LOO (labels) ===
def manual_loo_predict(pipe, X, y):
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    n = X_arr.shape[0]
    preds = np.empty(n, dtype=object)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_arr):
        p = clone(pipe)
        p.fit(X_arr[train_idx], y_arr[train_idx])
        preds[test_idx[0]] = p.predict(X_arr[test_idx])[0]
    return preds

pred_manual = manual_loo_predict(pipeline, X, y)

# === 3) Compare ===
print("Labels equal (array_equal):", np.array_equal(pred_cvp, pred_manual))
mismatches = np.where(pred_cvp != pred_manual)[0]
print("Mismatch count:", len(mismatches))
if len(mismatches) > 0:
    print("First 10 mismatch indices:", mismatches[:10])
    for i in mismatches[:10]:
        print(f"idx={i}: cvp={pred_cvp[i]}, manual={pred_manual[i]}, true={y.iloc[i] if hasattr(y,'iloc') else y[i]}")

# === 4) If you want probabilities, compare with allclose ===
proba_cvp = cross_val_predict(pipeline, X, y, cv=loo, method='predict_proba', n_jobs=1)
def manual_loo_proba(pipe, X, y):
    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = y.values if hasattr(y, "values") else np.asarray(y)
    n = X_arr.shape[0]
    # number of classes from a fit on full data
    tmp = clone(pipe).fit(X_arr, y_arr)
    n_classes = len(tmp.named_steps['model'].classes_)
    probs = np.empty((n, n_classes), dtype=float)
    loo = LeaveOneOut()
    for train_idx, test_idx in loo.split(X_arr):
        p = clone(pipe)
        p.fit(X_arr[train_idx], y_arr[train_idx])
        probs[test_idx[0], :] = p.predict_proba(X_arr[test_idx])[0]
    return probs

proba_manual = manual_loo_proba(pipeline, X, y)
print("Probabilities close (allclose):", np.allclose(proba_cvp, proba_manual))
