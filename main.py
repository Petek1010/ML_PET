import sys

import pandas as pd
import xlsxwriter
import numpy as np
from scipy.stats import shapiro

from data_process import DataProcess, GeneralPreprocess
from classifier import Classifier

def normality_score(X, alpha=0.05):
    """Return fraction of features that pass Shapiro test."""
    passed = 0
    for col in X.columns:
        p = shapiro(X[col].dropna())[1]
        if p > alpha:
            passed += 1
    return passed / len(X.columns)

def normality_score_W(X, alpha=None):
    """Return mean Shapiro-W score across features."""
    W_values = []
    for col in X.columns:
        data = X[col].dropna()
        if len(data) < 3:
            continue
        if len(data) > 5000:
            data = data.sample(5000, random_state=0)
        W, p = shapiro(data)
        W_values.append(W if not np.isnan(W) else 0)
    return np.mean(W_values) if W_values else np.nan

def correlation_score(X_train):
    df_features = X_train
    pear_matrix = df_features.corr(method='pearson').abs()
    upper_tri = pear_matrix.where(np.triu(np.ones(pear_matrix.shape), k=1).astype(bool))
    # mean correlation
    mean_corr_score = upper_tri.stack().mean()
    return mean_corr_score

def imbalance_score(y_train):
    # Add class count (class distribution) (closer to 0 -> more balanced data, less penalty)
    # Class count score
    c = y_train.value_counts().to_numpy()
    p = c / c.sum()
    # entropy score (normalized)
    entropy = -np.sum(p * np.log2(p))
    max_entropy = np.log2(len(c))
    balance_entropy = 1 - (entropy / max_entropy)
    return balance_entropy

def get_most_significant_warning(model, mean_corr_score, norm_score, balance_entropy):
    """
    Returns the most significant warning for a given model based on model-specific weights.
    """
    # Define thresholds
    CORRELATION_LIMIT = 0.70
    NORMALITY_LIMIT = 0.90
    BALANCE_LIMIT = 0.25

    # Model-specific weights (higher = more important)
    model_weights = {
        "logistic_regression": {"correlation": 2.0, "normality": 1.0, "balance": 1.0},
        "svm": {"correlation": 1.0, "normality": 1.0, "balance": 2.0},
        "naive_bayes": {"correlation": 1.0, "normality": 2.0, "balance": 1.0},
    }

    weights = model_weights.get(model, {"correlation": 1.0, "normality": 1.0, "balance": 1.0})

    warning_candidates = []

    # Check each condition and store weighted magnitude
    if mean_corr_score > CORRELATION_LIMIT:
        weighted_score = (mean_corr_score - CORRELATION_LIMIT) * weights["correlation"]
        warning_candidates.append((weighted_score, "multikolinearnost"))

    if norm_score < NORMALITY_LIMIT:
        weighted_score = (NORMALITY_LIMIT - norm_score) * weights["normality"]
        warning_candidates.append((weighted_score, "Nenormalna porazdelitev"))

    if balance_entropy > BALANCE_LIMIT:
        weighted_score = (balance_entropy - BALANCE_LIMIT) * weights["balance"]
        warning_candidates.append((weighted_score, "Neuravnotežena porazdelitev"))

    # Pick the most significant warning after weighting
    if warning_candidates:
        warning_candidates.sort(reverse=True, key=lambda x: x[0])
        return warning_candidates[0][1]
    else:
        return "/"


if __name__ == '__main__':
    clinical_question_roster = [
        # Default test - Orange
        #["AD", "CJD", "FTD", "NC"], ["ADRP", "CJDRP", "BFDRP"],
        # Demence
        ["AD", "DLB", "FTD"],["ADRP", "DLBRP","BFDRP"],
        ["AD", "DLB", "FTD"], ["ADRP", "DLBRP", "BFDRP", "DMN"],
        ["AD", "DLB"], ["ADRP", "DLBRP"],
        ["AD", "DLB"], ["ADRP", "DLBRP", "DMN"],
        ["AD", "DLB"], ["ADRP","PDRP", "DLBRP"],
        ["AD", "DLB"], ["ADRP","PDRP", "DLBRP", "DMN"],
        ["AD", "DLB"], ["ADRP", "PDRP","PDCP", "DLBRP"],
        ["AD", "DLB"], ["ADRP", "PDRP", "PDCP", "DLBRP", "DMN"],
        ["AD", "DLB", "FTD", "CJD"], ["ADRP", "DLBRP", "BFDRP", "CJDRP"],
        ["AD", "DLB", "FTD", "CJD"], ["ADRP", "DLBRP","BFDRP", "CJDRP", "DMN"],
        ["AD", "FTD", "CJD"], ["ADRP", "DLBRP", "BFDRP", "CJDRP"],
        ["AD", "FTD", "CJD"], ["ADRP", "DLBRP", "BFDRP", "CJDRP", "DMN"],

        # Parkinsonizmi
        ["PD", "MSA", "PSP"], ["PDRP", "MSARP", "PSPRP"],
        ["PD", "MSA", "PSP"], ["PDRP", "MSARP", "PSPRP", "DMN"],
        ["PD", "MSA", "PSP"], ["PDRP","PDCP" ,"MSARP", "PSPRP"],
        ["PD", "MSA", "PSP"], ["PDRP", "PDCP", "MSARP", "PSPRP", "DMN"],
        ["PD", "MSA", "PSP", "CBD"], ["PDRP", "MSARP", "PSPRP", "CBDRP"],
        ["PD", "MSA", "PSP", "CBD"], ["PDRP", "MSARP", "PSPRP", "CBDRP", "DMN"],
        ["MSA", "PSP"], ["MSARP", "PSPRP"],
        ["MSA", "PSP"], ["PDRP", "MSARP", "PSPRP"],
        ["MSA", "PSP"], ["PDRP", "PDCP","MSARP","PSPRP"],
        ["MSA", "PSP"], ["PDRP", "PDCP", "MSARP", "PSPRP", "DMN"],
        ["PD", "CBD"], ["PDRP", "PDCP", "CBDRP"],
        ["PD", "CBD"], ["PDRP", "PDCP", "CBDRP", "DMN"],

        # Parkinsonizem z demenco
        ["DLB", "PSP"], ["DLBRP", "PSPRP"],
        ["DLB", "PSP"], ["DLBRP", "PSPRP", "DMN"],
        ["DLB", "PSP"], ["ADRP", "PDRP", "DLBRP","PSPRP"],
        ["DLB", "PSP"], ["ADRP", "PDRP", "DLBRP", "PSPRP", "DMN"],
        ["PD","DLB", "PSP"], ["ADRP", "PDRP", "DLBRP", "PSPRP"],
        ["PD", "DLB", "PSP"], ["ADRP", "PDRP", "DLBRP", "PSPRP", "DMN"],
    ]

    results = []
    model_suggestions = []
    plot_results = {}
    #data = GeneralPreprocess(["AD", "CJD", "FTD", "NC"], ["ADRP", "CJDRP", "BFDRP"], "AllScores.xlsx", "ZScores")


    for i in range(0, len(clinical_question_roster), 2):
        print(f'------------ New set ({i+1}) -------------')
        target_set = clinical_question_roster[i]
        feature_set = clinical_question_roster[i+1]

        data = GeneralPreprocess(target_set,feature_set,"AllScores.xlsx", "ZScores", country='SI')
        X_train, y_train, X_test, y_test = data.pipeline()



        target_arr = data.get_clinical_targets()
        bayes_priors = data.get_priors()
        population_priors = data.get_population_priors()

        # LOOP THROUGH ALL MODELS - MACHINE LEARNING
        # Loop through all models "svm", "logistic_regression", "naive_bayes", "naive_bayes_priors"
        for model_name in ["svm", "logistic_regression", "naive_bayes", "naive_bayes_priors"]:
            model = Classifier(X_train, y_train, X_test, y_test, population_priors, bayes_priors, model_name)
            model.evaluate(report=False, use_bayes_correction=True)
            spec = model.specificity_score()
            spec_corr = model.get_spec_corr()


            report_dic = model.get_report_dictionary()
            report_dic_cor = model.get_corrected_report_dictionary()



            print("-------- Collecting metrics for Excel export -----------")

            class_counts = y_train.value_counts().to_dict()
            class_dist_str = ' | '.join(f"{label}:{count}" for label, count in class_counts.items())

            norm_score = normality_score_W(X_train)
            mean_corr_score = correlation_score(X_train)
            balance_entropy = imbalance_score(y_train)

            # Get macro AUC
            auc_results = model.auc_result

            auc_results_corr = model.auc_result_corr

            f1 = report_dic['macro avg']['f1-score']
            f1_corr = report_dic_cor['macro avg']['f1-score']
            recall = report_dic['macro avg']['recall']
            recall_corr = report_dic_cor['macro avg']['recall']
            acc = report_dic['accuracy']
            acc_corr = report_dic_cor['accuracy']
            #auc = model.auc_result['AUC macro']
            auc = model.auc_result
            #auc_corr = model.auc_result_corr['AUC macro']
            auc_corr = model.auc_result

            # You can tune it depending on whether false negatives or false positives hurt more
            performance_score = 0.4*f1 + 0.2*acc + 0.2*recall + 0.2*auc
            performance_score_corr = 0.4*f1_corr + 0.2*acc_corr + 0.2*recall_corr + 0.2*auc_corr
            penalty = 0

            if model == "logistic_regression":
                penalty += mean_corr_score * 0.3
                penalty += balance_entropy * 0.2
                penalty += norm_score * 0.05

            elif model == "naive_bayes":
                penalty += mean_corr_score * 0.4
                penalty += norm_score * 0.3
                penalty += balance_entropy * 0.1

            elif model == "svm":
                penalty += balance_entropy * 0.4
                penalty += mean_corr_score * 0.05

            final_score = performance_score - penalty
            final_score_corr = performance_score_corr - penalty

            ############### Result analysis ###########################

            # Tresholds
            # --- Performance thresholds ---
            RECALL_HIGH = 0.90  # “high” sensitivity
            SPECIFICITY_HIGH = 0.90  # “high” specificity
            METRIC_HIGH = 0.75  # overall high performance (avg. of all metrics)


            most_significant = get_most_significant_warning(model,mean_corr_score,norm_score,balance_entropy)
            role = None

            # --- 1. Determine model role ---
            if recall >= RECALL_HIGH and spec < SPECIFICITY_HIGH:
                role = "Presejalni test"
            elif spec >= SPECIFICITY_HIGH and recall < RECALL_HIGH:
                role = "Potrditveni test"
            elif np.mean([f1, acc, recall, spec]) >= METRIC_HIGH:
                role = "Splošni diagnostični test"
            else:
                role = "Neopredeljen (inconsistent metrics)"




            model_suggestions.append({
                "target_set": ",".join(target_set),
                "feature_set": ",".join(feature_set),
                "class_distribution": class_dist_str,
                "model": model_name,
                "f1_macro": f1,
                "f1_macro_C": f1_corr,
                "recall": recall,
                "recall_C": recall_corr,
                "specificity": spec,
                "specificity_C": spec_corr,
                "accuracy": acc,
                "accuracy_C": acc_corr,
                "auc_macro": auc,
                "mean_corr": mean_corr_score,
                "norm_score": norm_score,
                "balance_score": balance_entropy,
                "priporocilo": role,
                "opozorilo": most_significant,
                "final_score": final_score,
                "final_score_corr": final_score_corr
            })


    suggestions_df = pd.DataFrame(model_suggestions)
    # Save to Pickle for plotting in analysis.py
    suggestions_df.to_pickle(" suggestions_df.pkl")


    # Create Excel table from DataFrame

    # Pick best per target/feature set
    best_models = (
        suggestions_df
        .sort_values(["target_set", "feature_set", "final_score", "final_score_corr"],
                     ascending=[True, True, False, False])
        .groupby(["target_set", "feature_set"])
        .first()
        .reset_index()
    )

    # Columns that define the groups on the left of the table
    group_cols = ['target_set', 'class_distribution', 'feature_set']

    # (Recommended) ensure rows are grouped together so merging makes sense
    suggestions_df = suggestions_df.sort_values(group_cols, kind="stable").reset_index(drop=True)

    # Identify numeric model columns to compare/highlight (exclude group cols)
    numeric_model_cols = [
        c for c in suggestions_df.columns
        if c not in group_cols and pd.api.types.is_numeric_dtype(suggestions_df[c])
    ]
    numeric_model_col_idxs = [suggestions_df.columns.get_loc(c) for c in numeric_model_cols]

    with pd.ExcelWriter(path="PET_ML_Results_AUC.xlsx", engine='xlsxwriter') as writer:
        # Write the full table
        suggestions_df.to_excel(writer, sheet_name='Model Suggestion', index=False)

        wb = writer.book
        ws = writer.sheets['Model Suggestion']

        # Formats
        highlight_fmt = wb.add_format({'bold': True, 'bg_color': '#C6EFCE', 'font_color': '#000000'})
        highlight_corr_fmt = wb.add_format({'bold': True,"bg_color": "#FFD966", "font_color": "#000000"})
        merge_fmt = wb.add_format({'align': 'center', 'valign': 'vcenter', 'border': 1})



        # ---- Highlight max per row across numeric model columns ----
        n_rows = len(suggestions_df)
        header_offset = 1  # because row 0 is the header in Excel
        index_written = False
        excel_col_offset = 1 if index_written else 0

        for r in range(n_rows):
            excel_row = r + header_offset

            # Row numeric values only for the model columns
            row_vals = suggestions_df.iloc[r][numeric_model_cols].to_numpy(dtype=float)
            if row_vals.size == 0:
                continue

            # Use nanmax so rows with NaNs still work
            max_val = np.nanmax(row_vals)
            if not np.isfinite(max_val):
                continue



        # TESTING COLORING
        # Column names
        feature_set_col = "feature_set"
        final_score_col = "final_score"
        final_score_corr_col = "final_score_corr"
        model_name_col = "model"

        # Column indexes in Excel
        final_col_idx = suggestions_df.columns.get_loc(final_score_col)
        final_corr_col_idx = suggestions_df.columns.get_loc(final_score_corr_col)
        model_col_idx = suggestions_df.columns.get_loc(model_name_col)

        for r, row in suggestions_df.iterrows():
            excel_row = r + header_offset  # always apply offset here
            feature_set = row["feature_set"] # adjust if column name is different

            # --- Max values per feature set ---
            max_final = suggestions_df.loc[suggestions_df["feature_set"] == feature_set, final_score_col].max()
            max_final_corr = suggestions_df.loc[suggestions_df["feature_set"] == feature_set, final_score_corr_col].max()

            # Current row values
            val_final = row[final_score_col]
            val_final_corr = row[final_score_corr_col]
            model_name_val = row[model_name_col]
            precision = 8

            # --- Highlight best final_score ---
            if round(val_final, precision) == round(max_final, precision):
                ws.write_number(excel_row, final_col_idx, float(val_final), highlight_fmt)
                ws.write_string(excel_row, model_col_idx, str(model_name_val), highlight_fmt)

            # --- Highlight best final_score_corr ---
            if np.isclose(val_final_corr, max_final_corr, rtol=1e-3, atol=1e-8):
                ws.write_number(excel_row, final_corr_col_idx, float(val_final_corr), highlight_corr_fmt)
                ws.write_string(excel_row, model_col_idx, str(model_name_val), highlight_corr_fmt)

        # ---- Merge identical contiguous cells in each group column ----
        for col_name in group_cols:
            c_idx = suggestions_df.columns.get_loc(col_name)
            values = suggestions_df[col_name].tolist()

            block_start = header_offset
            for i in range(1, n_rows):
                # If value changes, close previous block if its length > 1
                if values[i] != values[i - 1]:
                    if (i - 1 + header_offset) > block_start:
                        ws.merge_range(block_start, c_idx, i - 1 + header_offset, c_idx, values[i - 1], merge_fmt)
                    block_start = i + header_offset

            # Close the last block
            if (n_rows - 1 + header_offset) > block_start:
                ws.merge_range(block_start, c_idx, n_rows - 1 + header_offset, c_idx, values[-1], merge_fmt)

        # (Optional) nicer widths
        for idx, col in enumerate(suggestions_df.columns):
            width = max(12, min(40, suggestions_df[col].astype(str).map(len).max() + 2))
            ws.set_column(idx, idx, width)

    print(suggestions_df.describe())
    print(best_models.describe())






    
























