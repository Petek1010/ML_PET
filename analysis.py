import sys
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from plotly.colors import qualitative
from sklearn.metrics import precision_recall_curve, average_precision_score

def bar_plot_model_vs_metric(df, model="naive_bayes", metric="f1_macro", use_features=False, n_lowest=4):
    # Filter for the chosen model
    nb_results = df[df["model"] == model].copy()

    # Create labels
    if use_features:
        nb_results["label"] = nb_results["target_set"] + " | " + nb_results["feature_set"]
    else:
        nb_results["label"] = nb_results["target_set"]

    # Sort values and pick the n lowest
    lowest_indices = nb_results.nsmallest(n_lowest, metric).index

    # Create a gradient of colors from light red to dark red
    reds = plt.cm.Reds(np.linspace(0.4, 1, n_lowest))

    # Default color list
    colors = ["skyblue"] * len(nb_results)

    # Apply gradient colors to the lowest bars
    for i, idx in enumerate(lowest_indices):
        colors[nb_results.index.get_loc(idx)] = reds[i]

    # Plot
    plt.figure(figsize=(12, 7))
    bars = plt.bar(nb_results["label"], nb_results[metric], color=colors)
    plt.ylabel(str(metric))
    plt.xlabel("Target / Feature Set" if use_features else "Target Set")
    plt.title(f"{model} - {metric}")
    plt.ylim(0, 1)

    # Set x-ticks with positions and labels
    tick_positions = np.arange(len(nb_results))
    tick_labels = nb_results["label"]
    fontweights = ["bold" if idx in lowest_indices else "normal" for idx in nb_results.index]
    for pos, label, weight in zip(tick_positions, tick_labels, fontweights):
        plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")  # set all labels
        plt.gca().get_xticklabels()[pos].set_fontweight(weight)  # make specific ones bold

    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()


def barplot_with_background(df, model="naive_bayes", metric="f1_macro",
                             final_corr=False, use_features=False, n_lowest=4):

    # --- Larger fonts for readability ---
    plt.rcParams.update({
        "font.size": 14,         # base
        "axes.titlesize": 18,    # title
        "axes.labelsize": 16,    # axis labels
        "xtick.labelsize": 12,   # x tick font
        "ytick.labelsize": 12,   # y tick font
        "legend.fontsize": 14    # legend
    })

    metric_map = {
        "f1_macro": "F1 ocena",
        "recall": "Občutljivost",
        "accuracy": "Točnost",
        "auc_macro": "AUC",
        "specificity": "Specifičnost"
    }

    model_name_map = {
        "svm": "SVM",
        "logistic_regression": "Logistična regresija",
        "naive_bayes": "Naivni Bayes",
        "naive_bayes_priors": "Naivni Bayes (prioirs)"
    }

    nb_results = df[df["model"] == model].copy()

    if use_features:
        nb_results["label"] = nb_results["target_set"] + " | " + nb_results["feature_set"]
    else:
        nb_results["label"] = nb_results["target_set"]

    # Lowest bars for coloring
    lowest_indices2 = nb_results.nsmallest(4, metric).index
    lowest_indices = nb_results.nsmallest(n_lowest, metric).index
    reds = plt.cm.Reds(np.linspace(1, 0.4, n_lowest))
    colors = ["skyblue"] * len(nb_results)
    for i, idx in enumerate(lowest_indices):
        colors[nb_results.index.get_loc(idx)] = reds[i]

    # Plot
    plt.figure(figsize=(13, 8))
    tick_positions = np.arange(len(nb_results))

    bars = plt.bar(tick_positions, nb_results[metric], color=colors)

    # Background overlay curves
    if final_corr:
        plt.plot(tick_positions, nb_results["final_score_corr"], color='black', alpha=0.5,
                 linestyle='--', marker='o')
    else:
        plt.plot(tick_positions, nb_results["mean_corr"], color='black', alpha=0.5,
                 linestyle='--', marker='o')
        plt.plot(tick_positions, nb_results["balance_score"], color='purple', alpha=0.5,
                 linestyle='--', marker='o')

    # Labels
    plt.ylabel("Vrednost metrike")
    plt.xlabel("Tarče / Značilnosti" if use_features else "Tarčni set")

    plt.title(
        f"[{model_name_map[model]} - {metric_map[metric]}]\n"
        f"s faktorjem korelacije in neenakomerne porazdelitve preiskovancev po razredu"
    )

    plt.ylim(0, 1)

    # Bold x labels for lowest ones
    tick_labels = nb_results["label"]
    plt.xticks(tick_positions, tick_labels, rotation=45, ha="right")

    for pos, idx in enumerate(nb_results.index):
        if idx in lowest_indices2:
            plt.gca().get_xticklabels()[pos].set_fontweight("bold")

    plt.grid(axis="y", linestyle="--", alpha=0.7)

    # Legend
    if final_corr:
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   alpha=0.5, markersize=9),
        ]
        plt.legend(custom_lines, ['Bayesov\npopravek'])
    else:
        custom_lines = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='black',
                   alpha=0.5, markersize=9),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='purple',
                   alpha=0.5, markersize=9)
        ]
        plt.legend(custom_lines, ['$K_{kor}$', '$K_{nnp}$'])

    plt.tight_layout()

    # Save
    file_path = f"ModelVsMetricPlots/{model_name_map[model]}_{metric_map[metric]}.png"
    plt.savefig(file_path, dpi=600, bbox_inches='tight')

    print("Plot saved:", file_path)


def all_models_scatter(df, metric="final_score", use_features=True):
    metric_map = {
        "f1_macro": "F1 ocena",
        "recall": "Občutljivost",
        "accuracy": "Točnost",
        "auc_macro": "AUC",
        "final_score": "Rezultat"
    }

    model_name_map = {
        "svm": "SVM",
        "logistic_regression": "Logistična regresija",
        "naive_bayes": "Naivni Bayes",
        "naive_bayes_priors": "Naivni Bayes (priors)"
    }

    # Create label for x-axis
    if use_features:
        df["label"] = df["target_set"] + " | " + df["feature_set"]
    else:
        df["label"] = df["target_set"]

    labels = df["label"].unique()
    x_positions = np.arange(len(labels))

    plt.figure(figsize=(14, 7))

    markers = ["o", "s", "^", "D"]  # one marker per model
    colors = ["blue", "green", "orange", "red"]

    for i, (model_key, marker, color) in enumerate(zip(model_name_map.keys(), markers, colors)):
        model_df = df[df["model"] == model_key]
        # Align x positions
        y_values = [model_df[model_df["label"] == lbl][metric].values[0] if lbl in model_df["label"].values else np.nan
                    for lbl in labels]
        plt.scatter(x_positions, y_values, label=model_name_map[model_key], color=color, marker=marker, s=100)
        # Optional: connect points with a line
        plt.plot(x_positions, y_values, color=color, alpha=0.5, linestyle="--")

    # Optional background metrics (like in your original)
    if "mean_corr" in df.columns:
        mean_corr = [df[df["label"] == lbl]["mean_corr"].values[0] if lbl in df["label"].values else np.nan
                     for lbl in labels]
        plt.plot(x_positions, mean_corr, color='black', alpha=0.5, linestyle=':', marker='o', label="$K_{kor}$")
    if "balance_score" in df.columns:
        balance = [df[df["label"] == lbl]["balance_score"].values[0] if lbl in df["label"].values else np.nan
                   for lbl in labels]
        plt.plot(x_positions, balance, color='purple', alpha=0.5, linestyle=':', marker='o', label="$K_{nnp}$")

    plt.xticks(x_positions, labels, rotation=45, ha="right")
    plt.ylabel(metric_map.get(metric, metric))
    plt.xlabel("Tarče / Značilnosti" if use_features else "Tarčni set")
    plt.ylim(0, 1)
    plt.title(f"Final score per model for each target/feature set", fontsize=14)
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()


def all_models_scatter_split(df, metric="final_score", use_features=True, chunk_sizes=[12, 12, 6]):
    metric_map = {
        "f1_macro": "F1 ocena",
        "recall": "Občutljivost",
        "accuracy": "Točnost",
        "auc_macro": "AUC",
        "final_score": "Rezultat"
    }

    model_name_map = {
        "svm": "SVM",
        "logistic_regression": "Logistična regresija",
        "naive_bayes": "Naivni Bayes",
        "naive_bayes_priors": "Naivni Bayes (priors)"
    }

    # Create label for x-axis
    if use_features:
        df["label"] = df["target_set"] + " | " + df["feature_set"]
    else:
        df["label"] = df["target_set"]

    labels = df["label"].unique()

    # Split labels into chunks
    start = 0
    for i, chunk_size in enumerate(chunk_sizes):
        end = start + chunk_size
        chunk_labels = labels[start:end]
        x_positions = np.arange(len(chunk_labels))

        plt.figure(figsize=(10, 7))
        markers = ["o", "s", "^", "D"]
        colors = ["blue", "green", "orange", "red"]

        for model_key, marker, color in zip(model_name_map.keys(), markers, colors):
            model_df = df[df["model"] == model_key]
            y_values = [
                model_df[model_df["label"] == lbl][metric].values[0] if lbl in model_df["label"].values else np.nan
                for lbl in chunk_labels]
            plt.scatter(x_positions, y_values, label=model_name_map[model_key], color=color, marker=marker, s=100)
            plt.plot(x_positions, y_values, color=color, alpha=0.5, linestyle="--")

        # Optional background metrics
        if "mean_corr" in df.columns:
            mean_corr = [df[df["label"] == lbl]["mean_corr"].values[0] if lbl in df["label"].values else np.nan
                         for lbl in chunk_labels]
            plt.plot(x_positions, mean_corr, color='black', alpha=0.5, linestyle=':', marker='o', label="$K_{kor}$")
        if "balance_score" in df.columns:
            balance = [df[df["label"] == lbl]["balance_score"].values[0] if lbl in df["label"].values else np.nan
                       for lbl in chunk_labels]
            plt.plot(x_positions, balance, color='purple', alpha=0.5, linestyle=':', marker='o', label="$K_{nnp}$")

        plt.xticks(x_positions, chunk_labels, rotation=45, ha="right")
        plt.ylabel(metric_map.get(metric, metric))
        plt.xlabel("Tarče / Značilnosti" if use_features else "Tarčni set")
        plt.ylim(0, 1)
        plt.title(f"Final score per model (chunk {i + 1})", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

        start = end

def all_models_bars_split(df, metric="final_score", use_features=True, chunk_sizes=[12, 12, 6]):
    metric_map = {
        "f1_macro": "F1 ocena",
        "recall": "Občutljivost",
        "accuracy": "Točnost",
        "auc_macro": "AUC",
        "final_score": "Rezultat"
    }

    model_name_map = {
        "svm": "SVM",
        "logistic_regression": "Logistična regresija",
        "naive_bayes": "Naivni Bayes",
        "naive_bayes_priors": "Naivni Bayes (priors)"
    }

    if use_features:
        df["label"] = df["target_set"] + " | " + df["feature_set"]
    else:
        df["label"] = df["target_set"]

    labels = df["label"].unique()
    start = 0

    for i, chunk_size in enumerate(chunk_sizes):
        end = start + chunk_size
        chunk_labels = labels[start:end]
        x = np.arange(len(chunk_labels))
        width = 0.2  # width of each model bar

        plt.figure(figsize=(14, 7))
        colors = ["#0072B2",  # strong blue
          "#E69F00",  # bright orange
          "#009E73",  # teal green
          "#D55E00"] # colors for each model

        # Plot bars for each model
        for j, (model_key, color) in enumerate(zip(model_name_map.keys(), colors)):
            model_df = df[df["model"] == model_key]
            y_values = [model_df[model_df["label"] == lbl][metric].values[0] if lbl in model_df["label"].values else 0
                        for lbl in chunk_labels]
            plt.bar(x + j * width, y_values, width=width, color=color, label=model_name_map[model_key], alpha=1)

        # Background metrics
        if "mean_corr" in df.columns:
            mean_corr = [df[df["label"] == lbl]["mean_corr"].values[0] if lbl in df["label"].values else np.nan
                         for lbl in chunk_labels]
            plt.plot(x + width * 1.5, mean_corr, color='black', alpha=0.5, linestyle='--', marker='o',
                     label="$K_{kor}$")

        if "balance_score" in df.columns:
            balance = [df[df["label"] == lbl]["balance_score"].values[0] if lbl in df["label"].values else np.nan
                       for lbl in chunk_labels]
            plt.plot(x + width * 1.5, balance, color='purple', alpha=0.5, linestyle='--', marker='o', label="$K_{nnp}$")

        plt.xticks(x + width * 1.5, chunk_labels, rotation=45, ha="right")
        plt.ylabel(metric_map.get(metric, metric))
        plt.xlabel("Tarče / Značilnosti" if use_features else "Tarčni set")
        plt.ylim(0, 1)
        plt.title(f"Final score per model with background metrics (chunk {i + 1})", fontsize=14)
        plt.grid(axis="y", linestyle="--", alpha=0.6)
        plt.legend()
        plt.tight_layout()
        plt.show()

        start = end


def all_models_heatmap_simple(df, metric="final_score", use_features=True):
    metric_map = {
        "f1_macro": "F1 ocena",
        "recall": "Občutljivost",
        "accuracy": "Točnost",
        "auc_macro": "AUC",
        "final_score": "Rezultat"
    }

    model_name_map = {
        "svm": "SVM",
        "logistic_regression": "Logistična regresija",
        "naive_bayes": "Naivni Bayes",
        "naive_bayes_priors": "Naivni Bayes (priors)"
    }

    # Create label for rows
    if use_features:
        df["label"] = df["target_set"] + " | " + df["feature_set"]
    else:
        df["label"] = df["target_set"]

    # Pivot table: rows=labels, columns=models, values=metric
    df_pivot = df.pivot(index="label", columns="model", values=metric)
    df_pivot = df_pivot[list(model_name_map.keys())]  # ensure consistent column order
    df_pivot.rename(columns=model_name_map, inplace=True)

    plt.figure(figsize=(14, max(5, 0.3*len(df_pivot))))  # height scales with number of targets
    sns.heatmap(df_pivot, annot=True, fmt=".3f", cmap="cividis", vmin=0, vmax=1,
                cbar_kws={'label': metric_map.get(metric, metric)})

    plt.xlabel("Model")
    plt.ylabel("Target / Feature Set" if use_features else "Target Set")
    plt.title(f"Heatmap of {metric_map.get(metric, metric)} per model and target/feature set", fontsize=14)
    plt.tight_layout()
    plt.show()

def all_models_lollipop(df, metric="final_score_corr", use_features=True):
    metric_map = {
        "f1_macro": "F1 ocena",
        "recall": "Občutljivost",
        "accuracy": "Točnost",
        "auc_macro": "AUC",
        "final_score": "Rezultat"
    }

    model_name_map = {
        "svm": "SVM",
        "logistic_regression": "Logistična regresija",
        "naive_bayes": "Naivni Bayes",
        "naive_bayes_priors": "Naivni Bayes (priors)"
    }

    # Create label for rows
    if use_features:
        df["label"] = df["target_set"] + " | " + df["feature_set"]
    else:
        df["label"] = df["target_set"]

    # Pivot table: rows=labels, columns=models, values=metric
    df_pivot = df.pivot(index="label", columns="model", values=metric)
    df_pivot = df_pivot[list(model_name_map.keys())]  # ensure consistent order
    df_pivot.rename(columns=model_name_map, inplace=True)

    # Plot
    plt.figure(figsize=(14, max(6, 0.3*len(df_pivot))))
    x = np.arange(len(df_pivot.index))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # nice distinct colors

    for i, model in enumerate(df_pivot.columns):
        plt.vlines(x=x, ymin=0, ymax=df_pivot[model], color=colors[i], alpha=0.4, linewidth=1)  # stick
        plt.scatter(x, df_pivot[model], color=colors[i], label=model, s=60)  # dot

    plt.xticks(x, df_pivot.index, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel(metric_map.get(metric, metric))
    plt.xlabel("Tarče / Značilnosti" if use_features else "Target Set")
    plt.title(f"Primerjava rezultata (Bayesov popravek) med modeli strojnega učenja ", fontsize=14)
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.savefig(
        f"ModelVsModelPlots/LoliRezultati_corr.png",  # File name and format
        dpi=400,  # Resolution in dots per inch
        bbox_inches='tight',  # Remove extra whitespace
    )
    plt.show()

def all_models_lollipop_zones(df, metric="final_score", use_features=True):

    # --- Larger fonts ---
    plt.rcParams.update({
        "font.size": 16,
        "axes.titlesize": 20,
        "axes.labelsize": 18,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 14
    })

    metric_map = {
        "f1_macro": "F1 ocena",
        "recall": "Občutljivost",
        "accuracy": "Točnost",
        "auc_macro": "AUC",
        "final_score": "Rezultat",
        "specificity": "Specifičnost"
    }

    model_name_map = {
        "svm": "SVM",
        "logistic_regression": "Logistična regresija",
        "naive_bayes": "Naivni Bayes",
        "naive_bayes_priors": "Naivni Bayes (priors)"
    }

    metric_label = metric_map.get(metric, metric)

    if use_features:
        df["label"] = df["target_set"] + " | " + df["feature_set"]
    else:
        df["label"] = df["target_set"]

    df_pivot = df.pivot(index="label", columns="model", values=metric)
    df_pivot = df_pivot[list(model_name_map.keys())]
    df_pivot.rename(columns=model_name_map, inplace=True)

    plt.figure(figsize=(14, max(6, 0.3 * len(df_pivot))))
    x = np.arange(len(df_pivot.index))
    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    zone_colors = ["#cce5ff", "#ffe0b3", "#e6e6e6"]
    zone_splits = [12, 24, len(df_pivot)]
    start = 0
    for zc, end in zip(zone_colors, zone_splits):
        plt.axvspan(start - 0.5, end - 0.5, facecolor=zc, alpha=0.5, zorder=0)
        start = end

    for i, model in enumerate(df_pivot.columns):
        plt.vlines(x=x, ymin=0, ymax=df_pivot[model], color=colors[i], alpha=0.4, linewidth=1)
        plt.scatter(x, df_pivot[model], color=colors[i], label=model, s=75)

    plt.xticks(x, df_pivot.index, rotation=45, ha="right")
    plt.ylim(0, 1.05)
    plt.ylabel(metric_label)
    plt.xlabel("Tarče / Značilnosti" if use_features else "Tarčni set")
    plt.title(f"Primerjava metrike: {metric_label} med modeli strojnega učenja")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()

    safe_metric = metric.replace("/", "_")
    plt.savefig(
        f"ModelVsModelPlots/Loli_{safe_metric}.png",
        dpi=400,
        bbox_inches="tight",
    )

    plt.show()




def make_all_barplots_with_background():
    for metric in ["f1_macro", "recall", "specificity"]:
        for model in ["svm", "logistic_regression", "naive_bayes", "naive_bayes_priors"]:
            barplot_with_background(results_df, model=model, metric=metric, n_lowest=8, use_features=True)

def precision_recall_curves_from_results(df, target_label):
    """
    Nariše macro-average Precision-Recall krivulje iz results_df
    """
    plt.figure(figsize=(8, 6))

    for model in df["model"].unique():
        # Predpostavljam, da imaš shranjene y_test in y_score (ali vsaj probability outputs) v results_df
        y_true = np.vstack(df[df["model"] == model]["Y_test"].values)
        y_score = np.vstack(df[df["model"] == model]["y_score"].values)

        precision, recall, _ = precision_recall_curve(y_true.ravel(), y_score.ravel())
        ap = average_precision_score(y_true, y_score, average="macro")

        plt.plot(recall, precision, lw=2, label=f"{model} (AP={ap:.2f})")

    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Macro-average Precision-Recall curves for {target_label}")
    plt.legend()
    plt.grid(alpha=0.6, linestyle="--")
    plt.tight_layout()
    plt.show()


def delta_plot(df, base="f1_macro", corrected="f1_macro_C", use_features=True):
    """
    Creates a delta plot showing (corrected - uncorrected) metric for all models.
    Rows = target|feature combinations (label)
    Columns = models.
    """

    # --- Larger fonts (same style as other plots) ---
    plt.rcParams.update({
        "font.size": 14,         # base size
        "axes.titlesize": 18,    # title
        "axes.labelsize": 16,    # x/y labels
        "xtick.labelsize": 13,   # x tick text
        "ytick.labelsize": 13,   # y tick text
        "legend.fontsize": 14    # legend
    })

    metric_title_map = {
        "f1_macro": "F1 ocena",
        "recall": "Občutljivost",
        "specificity": "Specifičnost",
        "accuracy": "Točnost",
        "auc_macro": "AUC makro"
    }

    model_name_map = {
        "svm": "SVM",
        "logistic_regression": "Logistična regresija",
        "naive_bayes": "Naivni Bayes",
        "naive_bayes_priors": "Naivni Bayes (priors)"
    }

    # Label rows
    if use_features:
        df["label"] = df["target_set"] + " | " + df["feature_set"]
    else:
        df["label"] = df["target_set"]

    # Compute deltas
    df["delta"] = df[corrected] - df[base]

    # Pivot (rows = labels, columns = models)
    df_pivot = df.pivot(index="label", columns="model", values="delta")
    df_pivot = df_pivot[list(model_name_map.keys())]  # enforce order
    df_pivot.rename(columns=model_name_map, inplace=True)

    # Plot
    plt.figure(figsize=(14, max(6, 0.3 * len(df_pivot))))
    x = np.arange(len(df_pivot.index))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

    # Baseline once (not inside loop)
    plt.axhline(0, color="gray", linewidth=1, alpha=0.5)

    for i, model in enumerate(df_pivot.columns):

        # Sticks from 0 to delta
        plt.vlines(x, 0, df_pivot[model], color=colors[i], alpha=0.5, linewidth=1.5)

        # Dot showing direction & magnitude
        plt.scatter(x, df_pivot[model], color=colors[i], label=model, s=75)

    plt.xticks(x, df_pivot.index, rotation=45, ha="right")
    plt.ylabel("Δ " + metric_title_map.get(base, base))
    plt.xlabel("Tarče / Značilnosti" if use_features else "Target Set")

    plt.title(
        f"Delta upoštevanja Bayesovega popravka: {metric_title_map.get(base, base)}"
    )

    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.legend()
    plt.tight_layout()

    out_name = f"ModelVsModelPlots/Delta_{base}.png"
    plt.savefig(out_name, dpi=400, bbox_inches="tight")
    plt.show()



    print(f"Saved: {out_name}")
def plot_delta_bar(df, metric_base, metric_corr, title=None, epsilon=0.005):
    """
    Delta bar plot: corrected - uncorrected metric
    Green  = improvement (positive)
    Red    = decline (negative)
    Gray   = negligible change
    """

    df = df.copy()

    # Compute delta
    df["delta"] = df[metric_corr] - df[metric_base]

    # Label for rows
    df["label"] = df["model"] + " | " + df["target_set"] + " | " + df["feature_set"]

    # Sort by delta magnitude (largest change first)
    df = df.sort_values("delta", ascending=False)

    plt.figure(figsize=(12, max(6, 0.35 * len(df))))

    # Determine colors
    colors = df["delta"].apply(
        lambda x: "green" if x > epsilon else ("red" if x < -epsilon else "gray")
    )

    # Horizontal bar plot
    plt.barh(df["label"], df["delta"], color=colors)

    plt.axvline(0, color="black", linewidth=1)

    plt.xlabel(f"Δ {metric_base} (corrected - uncorrected)")
    plt.ylabel("Model | Target | Features")
    plt.title(title if title else f"Delta Plot for {metric_base.upper()}")
    plt.grid(axis="x", linestyle="--", alpha=0.4)

    plt.tight_layout()
    plt.show()

#######################################################
results_df = pd.read_pickle(" suggestions_df.pkl")


delta_plot(results_df, base="f1_macro", corrected="f1_macro_C")
delta_plot(results_df, base="recall", corrected="recall_C")
delta_plot(results_df, base="specificity", corrected="specificity_C")












#"f1_macro": "F1 ocena",
#"recall": "Občutljivost",
#"specificity": "Specifičnost"

#all_models_lollipop_zones(results_df, metric="specificity")



#make_all_barplots_with_background()

sys.exit()
for model in ["svm", "logistic_regression", "naive_bayes", "naive_bayes_priors"]:
    barplot_with_background(results_df, model=model, metric="final_score", final_corr=True, n_lowest=8, use_features=True)