"""
Phishing Email Classifier

Trains and evaluates two models on the feature matrix produced by
feature_extractor.py:
  1. Logistic Regression  — fast, interpretable baseline
  2. Linear SVM           — typically stronger on high-dimensional text data

Both are linear models, meaning they assign a weight to every feature.
That weight tells us exactly how much each word or signal pushes the
prediction toward phishing or legitimate — which directly answers the
"which features predict phishing?" question from the proposal.

Run:
    python classifier.py
"""

from __future__ import annotations
import logging
import time
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

from data_loader import get_splits
from feature_extractor import PhishingFeatureExtractor, HAND_CRAFTED_NAMES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate(model, X, y, label: str = "") -> dict:
    """
    Print a full evaluation report and return a dict of key metrics.
    Works with any fitted sklearn classifier.
    """
    preds = model.predict(X)

    # ROC-AUC needs probability scores or decision scores
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.decision_function(X)

    auc = roc_auc_score(y, scores)
    cm  = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel()

    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    print(classification_report(y, preds, target_names=["Legitimate", "Phishing"]))
    print(f"ROC-AUC:          {auc:.4f}")
    print(f"Confusion matrix:")
    print(f"                 Predicted Legit  Predicted Phishing")
    print(f"  Actual Legit        {tn:>6,}            {fp:>6,}")
    print(f"  Actual Phishing     {fn:>6,}            {tp:>6,}")

    return {"auc": auc, "tp": tp, "fp": fp, "tn": tn, "fn": fn}


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def top_features(
    model,
    feature_names: list[str],
    n: int = 20,
) -> None:
    """
    Print the top N features most predictive of phishing and legitimate email,
    using the model's learned coefficients.

    Only works with linear models (LogisticRegression, LinearSVC).
    """
    if not hasattr(model, "coef_"):
        print("Feature importance not available for this model type.")
        return

    coef = np.asarray(model.coef_).ravel()

    # Highest positive coefficients → most predictive of phishing (label=1)
    top_phishing_idx = np.argsort(coef)[-n:][::-1]
    # Most negative coefficients → most predictive of legitimate (label=0)
    top_legit_idx = np.argsort(coef)[:n]

    print(f"\n{'='*55}")
    print(f"  Top {n} features predicting PHISHING")
    print(f"{'='*55}")
    for i in top_phishing_idx:
        print(f"  {coef[i]:+.4f}  {feature_names[i]}")

    print(f"\n{'='*55}")
    print(f"  Top {n} features predicting LEGITIMATE")
    print(f"{'='*55}")
    for i in top_legit_idx:
        print(f"  {coef[i]:+.4f}  {feature_names[i]}")


def hand_crafted_importance(
    model,
    feature_names: list[str],
) -> None:
    """
    Print importance for just the 22 hand-crafted features so we can see
    which structural signals matter most — separate from the TF-IDF vocab.
    """
    if not hasattr(model, "coef_"):
        return

    coef = np.asarray(model.coef_).ravel()
    n_tfidf = len(feature_names) - len(HAND_CRAFTED_NAMES)
    hc_coef = coef[n_tfidf:]

    print(f"\n{'='*55}")
    print("  Hand-crafted feature weights (phishing = positive)")
    print(f"{'='*55}")
    order = np.argsort(hc_coef)[::-1]
    for i in order:
        print(f"  {hc_coef[i]:+.4f}  {HAND_CRAFTED_NAMES[i]}")


# ---------------------------------------------------------------------------
# Edge case analysis
# ---------------------------------------------------------------------------

def show_edge_cases(model, X_test, df_test: pd.DataFrame, n: int = 5) -> None:
    """
    Show the most confident mistakes the model makes:
      - False positives: legitimate emails predicted as phishing
      - False negatives: phishing emails the model missed
    """
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
    else:
        raw = model.decision_function(X_test)
        scores = (raw - raw.min()) / (raw.max() - raw.min())  # normalize to [0,1]

    preds = model.predict(X_test)
    y = df_test["label"].values

    fp_mask = (preds == 1) & (y == 0)
    fn_mask = (preds == 0) & (y == 1)

    # Most confident false positives (high phishing score but actually legit)
    fp_scores = np.where(fp_mask, scores, -1)
    fp_idx = np.argsort(fp_scores)[-n:][::-1]

    # Most confident false negatives (low phishing score but actually phishing)
    fn_scores = np.where(fn_mask, scores, 2)
    fn_idx = np.argsort(fn_scores)[:n]

    print(f"\n{'='*55}")
    print(f"  Edge cases — False Positives (legit misclassified as phishing)")
    print(f"{'='*55}")
    for i in fp_idx:
        if fp_mask[i]:
            row = df_test.iloc[i]
            print(f"  Score: {scores[i]:.3f} | From: {row['dataset_name']}")
            print(f"  Subject: {str(row['subject'])[:80]}")
            print(f"  Sender:  {str(row['sender_email'])[:60]}")
            print(f"  Text preview: {str(row['text'])[:120]}\n")

    print(f"\n{'='*55}")
    print(f"  Edge cases — False Negatives (phishing the model missed)")
    print(f"{'='*55}")
    for i in fn_idx:
        if fn_mask[i]:
            row = df_test.iloc[i]
            print(f"  Score: {scores[i]:.3f} | From: {row['dataset_name']}")
            print(f"  Subject: {str(row['subject'])[:80]}")
            print(f"  Sender:  {str(row['sender_email'])[:60]}")
            print(f"  Text preview: {str(row['text'])[:120]}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    # --- Load data ---
    print("Loading data...")
    train, test = get_splits()
    print(f"Train: {len(train):,}  |  Test: {len(test):,}")
    print(f"Phishing rate — Train: {train['label'].mean():.1%}  "
          f"Test: {test['label'].mean():.1%}")

    # --- Extract features ---
    print("\nExtracting features...")
    t0 = time.time()
    extractor = PhishingFeatureExtractor()
    X_train = extractor.fit_transform(train)
    X_test  = extractor.transform(test)
    y_train = train["label"].values
    y_test  = test["label"].values
    print(f"Feature matrix: {X_train.shape}  ({time.time()-t0:.1f}s)")

    feature_names = extractor.feature_names

    # -------------------------------------------------------------------
    # Model 1: Logistic Regression
    # class_weight='balanced' compensates if phishing/legit counts differ
    # C=1.0 is standard regularization strength (lower = more regularized)
    # -------------------------------------------------------------------
    print("\nTraining Logistic Regression...")
    t0 = time.time()
    lr = LogisticRegression(
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        n_jobs=-1,
    )
    lr.fit(X_train, y_train)
    print(f"Trained in {time.time()-t0:.1f}s")

    evaluate(lr, X_test, y_test, label="Logistic Regression — Test Set")
    top_features(lr, feature_names, n=15)
    hand_crafted_importance(lr, feature_names)

    # -------------------------------------------------------------------
    # Model 2: Linear SVM
    # Generally stronger than LR on high-dimensional text features.
    # Wrapped in CalibratedClassifierCV to get probability scores for AUC.
    # -------------------------------------------------------------------
    print("\nTraining Linear SVM...")
    t0 = time.time()
    svm = CalibratedClassifierCV(
        LinearSVC(C=0.1, max_iter=2000, class_weight="balanced"),
        cv=3,
    )
    svm.fit(X_train, y_train)
    print(f"Trained in {time.time()-t0:.1f}s")

    evaluate(svm, X_test, y_test, label="Linear SVM — Test Set")

    # Feature weights come from the inner LinearSVC
    top_features(svm.estimator, feature_names, n=15)

    # --- Edge case analysis (using the better model — SVM) ---
    show_edge_cases(svm, X_test, test, n=3)


if __name__ == "__main__":
    main()
