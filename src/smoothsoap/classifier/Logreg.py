import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    f1_score, accuracy_score, precision_score,
    recall_score, confusion_matrix, classification_report
)

def run_logistic_regression(X, y, outfile_prefix, random_state=42, solver='lbfgs', max_iter=500):
    """
    Perform Logistic Regression on data of shape (T, N, 4)
    and compute classification metrics on the training data.
    Saves metrics to a text file with suffix '_metrics.txt'.

    Parameters
    ----------
    X : np.ndarray
        Input data of shape (T, N, 4)
    y : np.ndarray
        Target labels of shape (T,) or (T, N)
    outfile_prefix : str
        Base filename to save metrics (e.g. 'results/run1')
    random_state : int
        Random seed for reproducibility
    solver : str
        Solver for LogisticRegression (e.g. 'lbfgs', 'liblinear')
    max_iter : int
        Maximum number of iterations

    Returns
    -------
    model : sklearn.linear_model.LogisticRegression
        The trained logistic regression model
    metrics : dict
        A dictionary containing F1 score, accuracy, precision, recall, and confusion matrix
    """

    # --- Validate input shape ---
    if X.ndim != 3 or X.shape[2] != 4:
        raise ValueError(f"Expected X of shape (T, N, 4), got {X.shape}")

    # Flatten data
    T, N, F = X.shape
    X_flat = X.reshape(T * N, F)
    y_flat = y.reshape(-1) if y.ndim > 1 else y

    # --- Train model ---
    model = LogisticRegression(random_state=random_state, solver=solver, max_iter=max_iter)
    model.fit(X_flat, y_flat)

    # --- Predictions ---
    y_pred = model.predict(X_flat)

    # --- Compute metrics ---
    metrics = {
        "accuracy": accuracy_score(y_flat, y_pred),
        "precision": precision_score(y_flat, y_pred, average="weighted", zero_division=0),
        "recall": recall_score(y_flat, y_pred, average="weighted", zero_division=0),
        "f1_score": f1_score(y_flat, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_flat, y_pred)
    }

    report = classification_report(y_flat, y_pred, zero_division=0)

    # --- Save to file ---
    outfile = f"{outfile_prefix}_metrics.txt"
    with open(outfile, "w") as f:
        f.write("=== Logistic Regression Results ===\n")
        f.write(f"Accuracy:  {metrics['accuracy']:.4f}\n")
        f.write(f"Precision: {metrics['precision']:.4f}\n")
        f.write(f"Recall:    {metrics['recall']:.4f}\n")
        f.write(f"F1 Score:  {metrics['f1_score']:.4f}\n\n")
        f.write("Confusion Matrix:\n")
        f.write(np.array2string(metrics['confusion_matrix']))
        f.write("\n\nDetailed Classification Report:\n")
        f.write(report)

    print(f"Metrics written to {outfile}")

    return model, metrics
