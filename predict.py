"""
Phishing Email Prediction CLI

Loads a cached phishing classifier and feature extractor to run inference on
new, unseen emails. If cache files do not exist, it trains the model once on
the training split and saves both artifacts under models/.

Usage:
    python predict.py
    python predict.py --file path/to/email.eml
    python predict.py --file path/to/email.txt
    python predict.py --retrain
"""

from __future__ import annotations

import argparse
import logging
import re
from email import policy
from email.parser import BytesParser
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.svm import LinearSVC

from data_loader import _parse_sender, get_splits
from feature_extractor import HAND_CRAFTED_NAMES, PhishingFeatureExtractor

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent
MODELS_DIR = REPO_ROOT / "models"
MODEL_PATH = MODELS_DIR / "svm_classifier.joblib"
EXTRACTOR_PATH = MODELS_DIR / "feature_extractor.joblib"

URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)

HC_EXPLANATION_LABELS: dict[str, str] = {
    "text_urgency_kw": "high urgency keyword count",
    "text_financial_kw": "financial-account language",
    "text_action_kw": "strong call-to-action language",
    "text_threat_kw": "threat or account-lock wording",
    "text_prize_kw": "prize/winner language",
    "subject_urgency_kw": "urgent subject wording",
    "subject_action_kw": "action-oriented subject wording",
    "subject_all_caps": "all-caps subject style",
    "subject_exclamation": "excessive subject punctuation",
    "sender_free_provider": "free email provider sender",
    "sender_domain_has_digits": "sender domain with numeric pattern",
    "url_count": "multiple URLs",
    "url_density": "elevated URL density",
    "text_has_html": "HTML-heavy message body",
    "text_caps_ratio": "unusually high all-caps ratio",
}


def _extract_body_from_message(message) -> str:
    """Extract text body from an EmailMessage."""
    if message.is_multipart():
        parts: list[str] = []
        for part in message.walk():
            ctype = part.get_content_type()
            if ctype == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")).lower():
                payload = part.get_content()
                if payload:
                    parts.append(str(payload))
        return "\n".join(parts).strip()
    payload = message.get_content()
    return str(payload).strip() if payload else ""


def _parse_text_file(content: str) -> tuple[str, str, str]:
    """Best-effort parse of plain text files with optional headers."""
    lines = content.splitlines()
    subject = ""
    sender = ""
    body_start = 0

    for idx, line in enumerate(lines[:20]):
        lower = line.lower().strip()
        if lower.startswith("subject:") and not subject:
            subject = line.split(":", 1)[1].strip()
            body_start = max(body_start, idx + 1)
        elif lower.startswith("from:") and not sender:
            sender = line.split(":", 1)[1].strip()
            body_start = max(body_start, idx + 1)
        elif lower == "":
            body_start = max(body_start, idx + 1)
            if subject or sender:
                break

    body = "\n".join(lines[body_start:]).strip() if (subject or sender) else content.strip()
    return body, subject, sender


def _load_email_input(file_path: Path | None) -> tuple[str, str, str]:
    """Return (body, subject, sender_raw) from interactive input or file."""
    if file_path is None:
        print("Paste/type email body. End with a single line containing only: END")
        chunks: list[str] = []
        while True:
            line = input()
            if line.strip() == "END":
                break
            chunks.append(line)
        body = "\n".join(chunks).strip()
        subject = input("Subject: ").strip()
        sender = input("Sender (e.g. Name <email@domain.com>): ").strip()
        return body, subject, sender

    if not file_path.exists():
        raise FileNotFoundError(f"Input file not found: {file_path}")

    suffix = file_path.suffix.lower()
    if suffix == ".eml":
        with file_path.open("rb") as f:
            msg = BytesParser(policy=policy.default).parse(f)
        body = _extract_body_from_message(msg)
        subject = str(msg.get("subject", "") or "").strip()
        sender = str(msg.get("from", "") or "").strip()
        return body, subject, sender

    content = file_path.read_text(encoding="utf-8", errors="replace")
    return _parse_text_file(content)


def _to_inference_dataframe(body: str, subject: str, sender_raw: str) -> pd.DataFrame:
    """Coerce a single message into extractor-compatible schema."""
    sender_name, sender_email, sender_domain = _parse_sender(sender_raw)
    url_count = len(URL_RE.findall(body or ""))

    row = {
        "text": body or "",
        "subject": subject or "",
        "sender": sender_raw or None,
        "sender_name": sender_name,
        "sender_email": sender_email,
        "sender_domain": sender_domain,
        "receiver": None,
        "receiver_email": None,
        "receiver_domain": None,
        "urls": float(url_count),
        "date": pd.NaT,
        "dataset_name": "inference",
    }
    return pd.DataFrame([row])


def _train_and_cache() -> tuple[CalibratedClassifierCV, PhishingFeatureExtractor]:
    """Train the SVM model on full training split and persist artifacts."""
    logger.info("Training model from scratch...")
    train, _ = get_splits()

    extractor = PhishingFeatureExtractor()
    X_train = extractor.fit_transform(train)
    y_train = train["label"].values

    model = CalibratedClassifierCV(
        LinearSVC(C=0.1, max_iter=2000, class_weight="balanced"),
        cv=3,
    )
    model.fit(X_train, y_train)

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(extractor, EXTRACTOR_PATH)
    logger.info("Saved model to %s", MODEL_PATH)
    logger.info("Saved extractor to %s", EXTRACTOR_PATH)
    return model, extractor


def _load_or_train(retrain: bool) -> tuple[CalibratedClassifierCV, PhishingFeatureExtractor]:
    """Load cached artifacts or train if missing/forced."""
    if retrain:
        if MODEL_PATH.exists():
            MODEL_PATH.unlink()
            logger.info("Deleted cached model: %s", MODEL_PATH)
        if EXTRACTOR_PATH.exists():
            EXTRACTOR_PATH.unlink()
            logger.info("Deleted cached extractor: %s", EXTRACTOR_PATH)

    if MODEL_PATH.exists() and EXTRACTOR_PATH.exists():
        logger.info("Loading cached model and extractor...")
        model = joblib.load(MODEL_PATH)
        extractor = joblib.load(EXTRACTOR_PATH)
        return model, extractor

    return _train_and_cache()


def _top_handcrafted_signals(
    model: CalibratedClassifierCV,
    extractor: PhishingFeatureExtractor,
    X_row,
    top_k: int = 5,
) -> list[tuple[str, float]]:
    """Return top hand-crafted features by absolute contribution."""
    # The fitted SVMs live inside the calibrated wrappers (one per CV fold).
    # Average their coefficient vectors for a stable importance estimate.
    fitted_coefs = []
    for cc in model.calibrated_classifiers_:
        # sklearn ≥1.1 uses .estimator; older versions use .base_estimator
        inner = getattr(cc, "estimator", None) or cc.base_estimator
        fitted_coefs.append(np.asarray(inner.coef_).ravel())
    coef = np.mean(fitted_coefs, axis=0)
    offset = len(extractor.feature_names) - len(HAND_CRAFTED_NAMES)
    hc_coef = coef[offset:]
    hc_vals = X_row[:, offset:].toarray().ravel()
    contributions = hc_coef * hc_vals

    order = np.argsort(np.abs(contributions))[::-1][:top_k]
    return [(HAND_CRAFTED_NAMES[i], float(contributions[i])) for i in order]


def _explanation_from_signals(signals: list[tuple[str, float]], is_phishing: bool) -> str:
    """Build one-line explanation from top hand-crafted contributions."""
    if not signals:
        return "The prediction relied mostly on text patterns learned from training data."

    direction = 1 if is_phishing else -1
    selected: list[str] = []
    for name, score in signals:
        if score * direction <= 0:
            continue
        selected.append(HC_EXPLANATION_LABELS.get(name, name.replace("_", " ")))
        if len(selected) == 3:
            break

    if not selected:
        selected = [HC_EXPLANATION_LABELS.get(signals[0][0], signals[0][0].replace("_", " "))]

    if len(selected) == 1:
        return f"The strongest signal was {selected[0]}."
    if len(selected) == 2:
        return f"The strongest signals were {selected[0]} and {selected[1]}."
    return f"The strongest signals were {selected[0]}, {selected[1]}, and {selected[2]}."


_MIN_WORD_COUNT = 50
_MIN_TFIDF_HITS = 10


def _ood_warnings(email_df: pd.DataFrame, X) -> list[str]:
    """Return warnings if the input looks out-of-distribution."""
    warnings: list[str] = []
    body = str(email_df.iloc[0]["text"])
    word_count = len(body.split())

    if word_count < _MIN_WORD_COUNT:
        warnings.append(
            f"Very short email ({word_count} words). The model was trained on "
            f"full-length emails; predictions on short messages are less reliable."
        )

    # Count how many TF-IDF features fired (non-zero entries excluding hand-crafted)
    n_hc = len(HAND_CRAFTED_NAMES)
    tfidf_nnz = X[0, :-n_hc].nnz if n_hc else X[0].nnz
    if tfidf_nnz < _MIN_TFIDF_HITS:
        warnings.append(
            f"Only {tfidf_nnz} vocabulary terms matched the training data. "
            f"The email text may be too different from the training corpus."
        )

    return warnings


def _print_prediction(
    model: CalibratedClassifierCV,
    extractor: PhishingFeatureExtractor,
    email_df: pd.DataFrame,
) -> None:
    """Run prediction and print verdict with supporting signals."""
    X = extractor.transform(email_df)
    probs = model.predict_proba(X)[0]
    pred = int(model.predict(X)[0])

    verdict = "PHISHING" if pred == 1 else "LEGITIMATE"
    confidence = probs[pred] * 100.0

    signals = _top_handcrafted_signals(model, extractor, X, top_k=5)
    explanation = _explanation_from_signals(signals, is_phishing=(pred == 1))
    warnings = _ood_warnings(email_df, X)

    print("\n=== Prediction ===")
    print(f"Verdict: {verdict}")
    print(f"Confidence: {confidence:.2f}%")

    if warnings:
        print("\n⚠️  Confidence warnings:")
        for w in warnings:
            print(f"  • {w}")

    print("\nTop 5 hand-crafted feature signals:")
    for idx, (name, score) in enumerate(signals, start=1):
        direction = "phishing" if score >= 0 else "legitimate"
        print(f"  {idx}. {name:<26} contribution={score:+.4f} ({direction})")
    print(f"\nExplanation: {explanation}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict phishing vs legitimate email.")
    parser.add_argument(
        "--file",
        type=Path,
        help="Path to input email file (.eml or .txt). If omitted, interactive mode is used.",
    )
    parser.add_argument(
        "--retrain",
        action="store_true",
        help="Delete cached model/extractor and force retraining before prediction.",
    )
    return parser.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    args = parse_args()

    model, extractor = _load_or_train(retrain=args.retrain)
    body, subject, sender = _load_email_input(args.file)
    email_df = _to_inference_dataframe(body=body, subject=subject, sender_raw=sender)
    _print_prediction(model, extractor, email_df)


if __name__ == "__main__":
    main()
