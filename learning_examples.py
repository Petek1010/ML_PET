import sys

import matplotlib.pyplot as plt
import numpy as np
from sklearn import svm
from sklearn.inspection import DecisionBoundaryDisplay

import matplotlib.pyplot as plt

population_priors_raw = np.array([1500, 150, 15])     # counts per 100k
train_priors_raw       = np.array([50, 5, 0.5]  )     # counts in train

population_priors_norm = population_priors_raw / sum(population_priors_raw)
train_priors_norm       = train_priors_raw / sum(train_priors_raw)

print(population_priors_norm)
print(train_priors_norm)
print(population_priors_raw / train_priors_raw)
print(population_priors_norm / train_priors_norm)

sys.exit()

# Imena modelov in F1-macro rezultati
models = ["Logistic Regression", "Naive Bayes", "NB + priors", "SVM"]
f1_scores = [0.839, 0.789, 0.636, 0.820]

# Barve: označimo najboljši model
colors = ['green' if score == max(f1_scores) else 'skyblue' for score in f1_scores]

# Narišemo stolpčni graf
plt.figure(figsize=(8, 5))
plt.bar(models, f1_scores, color=colors)
plt.ylim(0, 1)
plt.ylabel("F1-macro")
plt.title("Primerjava F1-macro med modeli")
plt.xticks(rotation=15)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Dodamo številke na stolpce
for i, score in enumerate(f1_scores):
    plt.text(i, score + 0.02, f"{score:.3f}", ha='center', fontsize=10)

plt.tight_layout()
plt.show()


''' Sample Data '''
X = np.array(
    [
        [0.4, -0.7],
        [-1.5, -1.0],
        [-1.4, -0.9],
        [-1.3, -1.2],
        [-1.1, -0.2],
        [-1.2, -0.4],
        [-0.5, 1.2],
        [-1.5, 2.1],
        [1.0, 1.0],
        [1.3, 0.8],
        [1.2, 0.5],
        [0.2, -2.0],
        [0.5, -2.4],
        [0.2, -2.3],
        [0.0, -2.7],
        [1.3, 2.1],
    ]
)

y = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


''' Plot Sample Data '''
# Plotting settings
fig, ax = plt.subplots(figsize=(4, 3))
x_min, x_max, y_min, y_max = -3, 3, -3, 3
ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

# Plot samples by color and add legend
scatter = ax.scatter(X[:, 0], X[:, 1], s=60, c=y, label=y, edgecolors="k")
ax.legend(*scatter.legend_elements(), loc="upper right", title="Razred")
ax.set_title("Vzorci v dvodimenzionalnem prostoru značilnosti")
plt.savefig("SVM1.png", dpi=600, bbox_inches="tight")
_ = plt.show()



def plot_training_data_with_decision_boundary(kernel, ax=None, long_title=True, support_vectors=True):
    # Train the SVC
    clf = svm.SVC(kernel=kernel, gamma=2).fit(X, y)

    # Settings for plotting
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    x_min, x_max, y_min, y_max = -3, 3, -3, 3
    ax.set(xlim=(x_min, x_max), ylim=(y_min, y_max))

    # Plot decision boundary and margins
    DecisionBoundaryDisplay.from_estimator(
        estimator=clf,
        X=X,
        ax=ax,
        response_method="predict",
        plot_method="pcolormesh",
        alpha=0.3,
    )
    DecisionBoundaryDisplay.from_estimator(
        estimator=clf,
        X=X,
        ax=ax,
        response_method="decision_function",
        plot_method="contour",
        levels=[-1, 0, 1],
        colors=["k", "k", "k"],
        linestyles=["--", "-", "--"],
    )

    if support_vectors:
        # Plot bigger circles around samples that serve as support vectors
        ax.scatter(
            clf.support_vectors_[:, 0],
            clf.support_vectors_[:, 1],
            s=150,
            facecolors="none",
            edgecolors="k",
        )

    # Plot samples by color and add legend
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, s=30, edgecolors="k")
    ax.legend(*scatter.legend_elements(), loc="upper right", title="Razred")

    if long_title:
        ax.set_title(f"SVM pri polinomskem jedru")
    else:
        ax.set_title(kernel)
    plt.savefig("SVM_poly.png", dpi=600, bbox_inches="tight")
    plt.show()  # Always show the plot



plot_training_data_with_decision_boundary("poly")

