import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc
)


def evaluate_model(model, X_test, y_test):
    """
    Evaluates the IDS model:
    - Multiclass accuracy
    - Multiclass confusion matrix
    - Classification report heatmap
    - Binary confusion matrix (TP, TN, FP, FN)
    - ROC curve for binary detection
    """

    # -------------------------------------------------
    # Create results directory
    # -------------------------------------------------
    os.makedirs("results", exist_ok=True)

    # -------------------------------------------------
    # Predictions
    # -------------------------------------------------
    y_pred = model.predict(X_test)

    # -------------------------------------------------
    # Multiclass Accuracy & Report
    # -------------------------------------------------
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0)

    print("\n================ IDS MODEL EVALUATION ================")
    print(f"Multiclass Accuracy : {accuracy:.4f}\n")
    print("Classification Report:")
    print(report)

    # Save evaluation results
    metrics = {
        "accuracy": accuracy,
        "classification_report": report
    }

    with open("results/evaluation_results.pkl", "wb") as f:
        pickle.dump(metrics, f)

    # -------------------------------------------------
    # Multiclass Confusion Matrix
    # -------------------------------------------------
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, cmap="Blues", cbar=True)
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Multiclass Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/multiclass_confusion_matrix.png")
    plt.close()

    # -------------------------------------------------
    # Classification Report Heatmap
    # -------------------------------------------------
    report_dict = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0
    )

    report_df = pd.DataFrame(report_dict).transpose()

    plt.figure(figsize=(10, 6))
    sns.heatmap(
        report_df.iloc[:-1, :-1],
        annot=True,
        cmap="YlGnBu"
    )
    plt.title("Classification Report Heatmap")
    plt.tight_layout()
    plt.savefig("results/classification_report_heatmap.png")
    plt.close()

    # -------------------------------------------------
    # Binary Conversion (Normal vs Attack)
    # -------------------------------------------------
    y_test_binary = (y_test != "normal").astype(int)
    y_pred_binary = (y_pred != "normal").astype(int)

    # -------------------------------------------------
    # Binary Confusion Matrix
    # -------------------------------------------------
    cm_binary = confusion_matrix(y_test_binary, y_pred_binary)

    tn, fp, fn, tp = cm_binary.ravel()

    print("\nBinary Confusion Matrix Values:")
    print(f"True Negatives (TN): {tn}")
    print(f"False Positives (FP): {fp}")
    print(f"False Negatives (FN): {fn}")
    print(f"True Positives (TP): {tp}")

    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm_binary,
        annot=True,
        fmt='d',
        cmap="Blues",
        xticklabels=["Normal", "Attack"],
        yticklabels=["Normal", "Attack"]
    )
    plt.xlabel("Predicted Label")
    plt.ylabel("Actual Label")
    plt.title("Binary Confusion Matrix")
    plt.tight_layout()
    plt.savefig("results/binary_confusion_matrix.png")
    plt.close()

    # -------------------------------------------------
    # ROC Curve (Binary)
    # -------------------------------------------------
    classes = list(model.classes_)
    normal_index = classes.index("normal")

    y_prob_attack = 1 - model.predict_proba(X_test)[:, normal_index]

    fpr, tpr, _ = roc_curve(y_test_binary, y_prob_attack)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(6, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.4f}")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve (Binary Attack Detection)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("results/roc_curve.png")
    plt.close()

    print(f"\nROC-AUC Score (Binary): {roc_auc:.4f}")

    print("======================================================")
    print("Evaluation completed successfully.")
    print("All result plots saved in the 'results/' folder.\n")