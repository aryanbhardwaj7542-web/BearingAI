
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
)
from sklearn.model_selection import cross_val_predict

DATA_FILE = "bearing_features.csv"

FEATURES = [
    "rms",
    "std",
    "kurtosis",
    "skewness",
    "peak_to_peak",
    "crest_factor",
]

df = pd.read_csv(DATA_FILE)
X = df[FEATURES].replace([np.inf, -np.inf], np.nan)
y = df["fault_type"]

valid = X.notna().all(axis=1)
X = X.loc[valid].reset_index(drop=True)
y = y.loc[valid].reset_index(drop=True)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

models = {
    "Random Forest": RandomForestClassifier(
        n_estimators=300,
        class_weight="balanced",
        random_state=42,
    ),
    "SVM": make_pipeline(
        StandardScaler(),
        SVC(kernel="rbf", class_weight="balanced"),
    ),
}

results = []

for name, model in models.items():
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"],
    )

    results.append({
        "Model": name,
        "Accuracy": scores["test_accuracy"].mean(),
        "Precision": scores["test_precision_macro"].mean(),
        "Recall": scores["test_recall_macro"].mean(),
        "F1-score": scores["test_f1_macro"].mean(),
    })

    print(f"\n{'=' * 45}")
    print(name)
    print(f"{'=' * 45}")

    for metric in ["accuracy", "precision_macro", "recall_macro", "f1_macro"]:
        print(f"{metric}: {scores['test_' + metric].mean():.4f}")

    predictions = cross_val_predict(model, X, y, cv=cv)

    print("\nClassification Report:")
    print(classification_report(y, predictions, zero_division=0))

    ConfusionMatrixDisplay.from_predictions(
        y,
        predictions,
        xticks_rotation=30,
        cmap="Blues",
    )
    plt.title(f"{name} - 5-Fold Cross-Validation")
    plt.tight_layout()
    plt.savefig(
        name.lower().replace(" ", "_") + "_confusion_matrix.png",
        dpi=150,
    )
    plt.close()

results_df = pd.DataFrame(results)
results_df.to_csv("model_comparison.csv", index=False)

print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))
print("\nSaved: model_comparison.csv")
print("Saved: random_forest_confusion_matrix.png")
print("Saved: svm_confusion_matrix.png")

# Train final Random Forest model on all available samples.
final_model = RandomForestClassifier(
    n_estimators=300,
    class_weight="balanced",
    random_state=42,
)
final_model.fit(X, y)

import joblib

joblib.dump(
    {"model": final_model, "features": FEATURES},
    "bearing_model.joblib",
)

print("\nFinal model saved: bearing_model.joblib")
