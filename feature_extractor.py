"""
Feature Extractor for Phishing Email Detection

Transforms the cleaned DataFrame (from data_loader.py) into a feature matrix
ready for classification. Combines three feature groups into one sparse matrix:
  - TF-IDF on email body        (up to max_text_features columns)
  - TF-IDF on subject line      (up to max_subject_features columns)
  - 22 hand-crafted features    (structural, keyword, sender, URL signals)

Usage:
    from data_loader import get_splits
    from feature_extractor import PhishingFeatureExtractor

    train, test = get_splits()
    extractor = PhishingFeatureExtractor()
    X_train = extractor.fit_transform(train)
    X_test  = extractor.transform(test)
    y_train = train["label"].values
    y_test  = test["label"].values
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import MaxAbsScaler


# ---------------------------------------------------------------------------
# Keyword sets
# ---------------------------------------------------------------------------

_URGENCY_WORDS = {
    "urgent", "immediately", "asap", "hurry", "expires", "deadline",
    "limited time", "act now", "last chance", "today only", "expiring",
    "respond now",
}
_FINANCIAL_WORDS = {
    "bank", "account", "credit", "debit", "paypal", "wire transfer",
    "payment", "invoice", "billing", "refund", "transaction",
    "social security", "routing number",
}
_ACTION_WORDS = {
    "click here", "click the link", "verify", "confirm", "login", "log in",
    "sign in", "update your", "validate", "submit", "download", "install",
}
_THREAT_WORDS = {
    "suspended", "locked", "unauthorized", "security alert", "hacked",
    "compromised", "breach", "violation", "illegal", "lawsuit",
}
_PRIZE_WORDS = {
    "winner", "you have won", "prize", "lottery", "congratulations",
    "you have been selected", "free gift", "reward", "million dollars",
}

_FREE_PROVIDERS = {
    "gmail.com", "yahoo.com", "hotmail.com", "outlook.com", "aol.com",
    "mail.com", "protonmail.com", "icloud.com", "live.com", "msn.com",
    "ymail.com", "inbox.com",
}


# ---------------------------------------------------------------------------
# Vectorized hand-crafted feature builder
# ---------------------------------------------------------------------------

def _kw_count(series: pd.Series, keywords: set[str]) -> np.ndarray:
    """Count how many keywords from the set appear in each string (case-insensitive)."""
    lower = series.str.lower()
    counts = np.zeros(len(series), dtype=np.float32)
    for kw in keywords:
        counts += lower.str.contains(kw, regex=False, na=False).astype(np.float32).values
    return counts


def _hand_crafted_features(df: pd.DataFrame) -> np.ndarray:
    """Return (n_samples, 22) float32 array of hand-crafted features."""
    text = df["text"].fillna("").astype(str)
    subject = df["subject"].fillna("").astype(str)
    sender_domain = df["sender_domain"].fillna("").astype(str).str.lower()
    urls = df["urls"].fillna(0).astype(float)

    n_words = text.str.split().str.len().clip(lower=1).astype(float)
    n_chars = text.str.len().astype(float)

    X = np.column_stack([
        # --- text body ---
        n_chars.values,                                                          # text_char_count
        n_words.values,                                                          # text_word_count
        (n_chars / n_words).values,                                              # text_avg_word_len
        text.str.count("!").fillna(0).values,                                    # text_exclamation_count
        text.str.count(r"\?").fillna(0).values,                                  # text_question_count
        (text.str.count(r"\b[A-Z]{2,}\b").fillna(0) / n_words).values,          # text_caps_ratio
        text.str.contains(r"<[a-zA-Z/][^>]*>", regex=True, na=False)            # text_has_html
            .astype(np.float32).values,
        _kw_count(text, _URGENCY_WORDS),                                         # text_urgency_kw
        _kw_count(text, _FINANCIAL_WORDS),                                       # text_financial_kw
        _kw_count(text, _ACTION_WORDS),                                          # text_action_kw
        _kw_count(text, _THREAT_WORDS),                                          # text_threat_kw
        _kw_count(text, _PRIZE_WORDS),                                           # text_prize_kw

        # --- subject ---
        subject.str.len().fillna(0).values,                                      # subject_len
        subject.str.count("!").fillna(0).values,                                 # subject_exclamation
        ((subject.str.upper() == subject) & (subject.str.len() > 3))            # subject_all_caps
            .astype(np.float32).values,
        _kw_count(subject, _URGENCY_WORDS),                                      # subject_urgency_kw
        _kw_count(subject, _ACTION_WORDS),                                       # subject_action_kw

        # --- sender ---
        sender_domain.isin(_FREE_PROVIDERS).astype(np.float32).values,          # sender_free_provider
        sender_domain.str.contains(r"\d", regex=True, na=False)                 # sender_domain_has_digits
            .astype(np.float32).values,
        df["sender_name"].notna().astype(np.float32).values,                    # sender_has_display_name

        # --- URLs ---
        urls.values,                                                             # url_count
        (urls / n_words).values,                                                 # url_density
    ])

    return X.astype(np.float32)


HAND_CRAFTED_NAMES: list[str] = [
    "text_char_count",
    "text_word_count",
    "text_avg_word_len",
    "text_exclamation_count",
    "text_question_count",
    "text_caps_ratio",
    "text_has_html",
    "text_urgency_kw",
    "text_financial_kw",
    "text_action_kw",
    "text_threat_kw",
    "text_prize_kw",
    "subject_len",
    "subject_exclamation",
    "subject_all_caps",
    "subject_urgency_kw",
    "subject_action_kw",
    "sender_free_provider",
    "sender_domain_has_digits",
    "sender_has_display_name",
    "url_count",
    "url_density",
]


# ---------------------------------------------------------------------------
# Main extractor class
# ---------------------------------------------------------------------------

class PhishingFeatureExtractor:
    """
    Fit on training data, then transform any split without leakage.

    Output is a scipy CSR sparse matrix:
        [ TF-IDF(body) | TF-IDF(subject) | hand-crafted ]

    The hand-crafted block is MaxAbs-scaled so it plays nicely alongside
    the already-normalized TF-IDF values, making linear models work well
    without any additional preprocessing.
    """

    def __init__(
        self,
        max_text_features: int = 10_000,
        max_subject_features: int = 2_000,
    ):
        self._text_tfidf = TfidfVectorizer(
            max_features=max_text_features,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=5,
        )
        self._subject_tfidf = TfidfVectorizer(
            max_features=max_subject_features,
            sublinear_tf=True,
            strip_accents="unicode",
            analyzer="word",
            ngram_range=(1, 2),
            min_df=3,
        )
        self._scaler = MaxAbsScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "PhishingFeatureExtractor":
        self._text_tfidf.fit(df["text"].fillna("").astype(str))
        self._subject_tfidf.fit(df["subject"].fillna("").astype(str))
        self._scaler.fit(_hand_crafted_features(df))
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        if not self._fitted:
            raise RuntimeError("Call fit() before transform().")
        X_text = self._text_tfidf.transform(df["text"].fillna("").astype(str))
        X_subj = self._subject_tfidf.transform(df["subject"].fillna("").astype(str))
        X_hc = sp.csr_matrix(self._scaler.transform(_hand_crafted_features(df)))
        return sp.hstack([X_text, X_subj, X_hc], format="csr")

    def fit_transform(self, df: pd.DataFrame) -> sp.csr_matrix:
        return self.fit(df).transform(df)

    @property
    def feature_names(self) -> list[str]:
        if not self._fitted:
            raise RuntimeError("Call fit() before accessing feature_names.")
        body_names = [f"body_{n}" for n in self._text_tfidf.get_feature_names_out()]
        subj_names = [f"subj_{n}" for n in self._subject_tfidf.get_feature_names_out()]
        return body_names + subj_names + HAND_CRAFTED_NAMES

    @property
    def n_features(self) -> int:
        return len(self.feature_names)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")

    from data_loader import get_splits

    train, test = get_splits()
    print(f"Train rows: {len(train):,}  |  Test rows: {len(test):,}")

    extractor = PhishingFeatureExtractor()
    X_train = extractor.fit_transform(train)
    X_test  = extractor.transform(test)

    print(f"\nX_train: {X_train.shape}  (dtype={X_train.dtype})")
    print(f"X_test:  {X_test.shape}")
    print(f"\nFeature breakdown:")
    print(f"  Body TF-IDF:    {len(extractor._text_tfidf.get_feature_names_out()):>6,}")
    print(f"  Subject TF-IDF: {len(extractor._subject_tfidf.get_feature_names_out()):>6,}")
    print(f"  Hand-crafted:   {len(HAND_CRAFTED_NAMES):>6,}")
    print(f"  Total:          {extractor.n_features:>6,}")

    # Sanity check: phishing emails should score higher on urgency keywords
    train_copy = train.copy()
    hc = _hand_crafted_features(train_copy)
    phishing_mask = train["label"].values == 1
    legit_mask = ~phishing_mask
    print(f"\nHand-crafted feature sanity check (phishing vs legit means):")
    for i, name in enumerate(HAND_CRAFTED_NAMES):
        p_mean = hc[phishing_mask, i].mean()
        l_mean = hc[legit_mask, i].mean()
        if abs(p_mean - l_mean) > 0.01:
            print(f"  {name:<30}  phishing={p_mean:.3f}  legit={l_mean:.3f}")


if __name__ == "__main__":
    main()
