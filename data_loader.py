"""
Dataset Loader and Parser

Loads `puyang2025/seven-phishing-email-datasets` from HuggingFace
- cleans it, parses sender/receiver into structured fields below

CLEANED DATAFRAME TABLE 

Produced by _clean() in this module. 
Saved to data/processed/phishing_emails.parquet.

--- From upstream (normalized) ---------------------------------------------
Column        Type      Source
------------  --------  --------------------------------------------------
text          string    Email body, whitespace-stripped
subject       string    Email subject, whitespace-stripped
label         int       0 = legitimate, 1 = phishing/spam
sender        string    Raw sender field, e.g. "John Smith <john@x.com>"
receiver      string    Raw recipient field

--- Derived by parsing `sender` --------------------------------------------
Column          Type    How it's built
--------------  ------  --------------------------------------------------
sender_name     string  Display name only ("John Smith"). Null if absent.
sender_email    string  Just the address ("john@x.com"). Lowercased.
sender_domain   string  Just the domain ("x.com"). Regex-extracted.


--- Derived by parsing `receiver` ------------------------------------------
Column            Type    How it's built
----------------  ------  ------------------------------------------------
receiver_email    string  Same parsing logic as sender_email
receiver_domain   string  Same parsing logic as sender_domain
(No receiver_name -- recipient display names rarely useful for detection.)


--- Pass-through metadata --------------------------------------------------
Column         Type        Source
-------------  ----------  -----------------------------------------------
date           timestamp   When the email was sent (may be null)
urls           int         URL count in body, pre-extracted upstream
dataset_name   string      One of: Assassin, CEAS-08, Enron, Ling,
                             TREC-05, TREC-06, TREC-07
split          string      "train" or "test" (preserves upstream split)


PARSING EXAMPLES
============================================================================
Raw `sender` value                              -> sender_name, sender_email, sender_domain
----------------------------------------------     ----------------------------------------
"Lissette Patterson <jstepanekov@eve-team.com>" -> "Lissette Patterson",
                                                   "jstepanekov@eve-team.com",
                                                   "eve-team.com"
"alert@broadcast.shareholder.com"               -> None,
                                                   "alert@broadcast.shareholder.com",
                                                   "broadcast.shareholder.com"
"\"Shah, Rajen\" <rshah@enron.com>"             -> "Shah, Rajen",
                                                   "rshah@enron.com",
                                                   "enron.com"
"null" or None                                  -> None, None, None
"""

from __future__ import annotations
import logging
import re
from dataclasses import dataclass
from email.utils import parseaddr
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
 
logger = logging.getLogger(__name__)

"""
CONFIG==============================================================================
"""

HF_DATASET_NAME = "puyang2025/seven-phishing-email-datasets"
 
# Resolve repo paths relative to this file: src/phishing_detector/data/loader.py
# -> repo root is three parents up.
REPO_ROOT = Path(__file__).resolve().parents[0]
DATA_DIR = REPO_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
PROCESSED_PARQUET = PROCESSED_DIR / "phishing_emails.parquet"
 
RANDOM_SEED = 42  # fixed for Identical Pasrsing
 
 
@dataclass(frozen=True)
class DatasetStats:
    n_total: int
    n_phishing: int
    n_legit: int
    pct_phishing: float
    avg_text_length: float
    n_with_sender: int
    by_source: dict[str, int]
 
    def __str__(self) -> str:
        sources = "\n".join(f"    {k}: {v:,}" for k, v in self.by_source.items())
        return (
            f"Total: {self.n_total:,} emails\n"
            f"  Phishing/spam: {self.n_phishing:,} ({self.pct_phishing:.1%})\n"
            f"  Legitimate:    {self.n_legit:,} ({1 - self.pct_phishing:.1%})\n"
            f"  Avg text length: {self.avg_text_length:,.0f} chars\n"
            f"  With non-null sender: {self.n_with_sender:,}\n"
            f"  By source:\n{sources}"
        )

"""
FIELD PARSING HELPER ==== ==============================================================================
  Matches a domain after the @ in an email address. 
"""
_DOMAIN_RE = re.compile(r"@([A-Za-z0-9.\-]+\.[A-Za-z]{2,})")
 
 
"""
INPUT: Parse a sender string of the form 'Display Name <user@domain.com>'.
OUTPUT: Returns (display_name, email, domain), May be NULL
"""
def _parse_sender(raw: str | None) -> tuple[str | None, str | None, str | None]:

    if raw is None:
        return None, None, None
    s = str(raw).strip()
    if not s or s.lower() in {"null", "none", "nan"}:
        return None, None, None
 
    name, email = parseaddr(s)  # stdlib RFC 2822 parser
    name = name.strip() or None
    email = email.strip().lower() or None
 
    domain = None
    if email:
        match = _DOMAIN_RE.search(email)
        if match:
            domain = match.group(1).lower()
 
    return name, email, domain
 
"""
LOADING===============================================================================================
"""
def _download_from_hf() -> pd.DataFrame:
    """Download the raw dataset (both splits) from HuggingFace Hub."""
    from datasets import load_dataset  
 
    logger.info("Downloading %s from HuggingFace...", HF_DATASET_NAME)
    ds = load_dataset(HF_DATASET_NAME)

    """
    Combine Splits into 1 Frame, Save original assignment for recovery later
    """
    frames = []
    for split_name in ds.keys():
        df_split = ds[split_name].to_pandas()
        df_split["split"] = split_name
        frames.append(df_split)
 
    df = pd.concat(frames, ignore_index=True)
    logger.info("Downloaded %d total rows across splits %s", len(df), list(ds.keys()))
    return df
 
 
def _clean(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean up the Data
    """
    n_before = len(df)
 
    """ Make sure data exits and is as exoected """
    expected = {"text", "subject", "label", "sender", "receiver", "date",
                "urls", "dataset_name", "split"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {missing}")
 
    """ Remove Whitespace """
    for col in ["text", "subject", "sender", "receiver", "dataset_name"]:
        df[col] = df[col].astype("string").str.strip()
 
    """ Remove Unusable Rows """
    bad = {"", "null", "empty", "nan", "none"}
    text_norm = df["text"].fillna("").str.lower()
    df = df[~text_norm.isin(bad)].copy()
 
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)
 
    """ Parse Sender into [Name, Email, Domain] """
    sender_parts = df["sender"].apply(_parse_sender)
    df["sender_name"] = [t[0] for t in sender_parts]
    df["sender_email"] = [t[1] for t in sender_parts]
    df["sender_domain"] = [t[2] for t in sender_parts]
 
    receiver_parts = df["receiver"].apply(_parse_sender)
    df["receiver_email"] = [t[1] for t in receiver_parts]
    df["receiver_domain"] = [t[2] for t in receiver_parts]
 
    """ Drop Duplicates"""
    df = df.drop_duplicates(subset=["text", "subject", "sender"]).reset_index(drop=True)
 
    n_after = len(df)
    logger.info("Cleaned: kept %d / %d rows (%d dropped)",
                n_after, n_before, n_before - n_after)
 
    """ Re-order data """
    cols = [
        "text", "subject", "label",
        "sender", "sender_name", "sender_email", "sender_domain",
        "receiver", "receiver_email", "receiver_domain",
        "date", "urls", "dataset_name", "split",
    ]
    return df[cols]
 
 
def load_emails(force_refresh: bool = False) -> pd.DataFrame:
    """
    Load Clean Data Set and put into cache
 
    Args:
        force_refresh: if True, re-download and overwrite the cache.
 
    Returns:
        DataFrame with columns:
            text, subject, label,
            sender, sender_name, sender_email, sender_domain,
            receiver, receiver_email, receiver_domain,
            date, urls, dataset_name, split
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
 
    if PROCESSED_PARQUET.exists() and not force_refresh:
        logger.info("Loading cached dataset from %s", PROCESSED_PARQUET)
        return pd.read_parquet(PROCESSED_PARQUET)
 
    df = _download_from_hf()
    df = _clean(df)
    df.to_parquet(PROCESSED_PARQUET, index=False)
    logger.info("Saved cleaned dataset to %s", PROCESSED_PARQUET)
    return df
 
 
def get_splits(
    val_size: float = 0.0,
    seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, ...]:
    """
    Return train/test splits.
 
    By default uses the dataset's pre-defined train/test split (162k / 40.6k).
    If val_size > 0, carves a stratified validation set out of the train split.
 
    Args:
        val_size: fraction of the *training* portion to use as validation.
                  0 -> returns (train, test). Otherwise (train, val, test).
        seed:     random seed for the val-from-train split (only used if val_size > 0).
    """
    df = load_emails()
 
    train = df[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    test = df[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
 
    if val_size <= 0:
        logger.info("Splits: train=%d, test=%d", len(train), len(test))
        return train, test
 
    train, val = train_test_split(
        train,
        test_size=val_size,
        stratify=train["label"],
        random_state=seed,
    )
    train = train.reset_index(drop=True)
    val = val.reset_index(drop=True)
    logger.info("Splits: train=%d, val=%d, test=%d", len(train), len(val), len(test))
    return train, val, test
 
"""
STATS=============================================================================================== 
"""
def compute_stats(df: pd.DataFrame) -> DatasetStats:
    n_total = len(df)
    n_phishing = int((df["label"] == 1).sum())
    n_legit = n_total - n_phishing
    by_source = df["dataset_name"].value_counts().to_dict()
    return DatasetStats(
        n_total=n_total,
        n_phishing=n_phishing,
        n_legit=n_legit,
        pct_phishing=n_phishing / n_total if n_total else 0.0,
        avg_text_length=float(df["text"].str.len().mean()) if n_total else 0.0,
        n_with_sender=int(df["sender_email"].notna().sum()),
        by_source=by_source,
    )
 
 
def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
 
    df = load_emails()
    print("\n=== Full dataset ===")
    print(compute_stats(df))
 
    train, val, test = get_splits(val_size=0.15)
    print("\n=== Train ===")
    print(compute_stats(train))
    print("\n=== Val ===")
    print(compute_stats(val))
    print("\n=== Test ===")
    print(compute_stats(test))
 
    # Show one parsed row so the team can see what derived fields look like.
    print("\n=== Sample parsed phishing row ===")
    sample = df[df["label"] == 1].sample(1, random_state=0).iloc[0]
    for col in ["dataset_name", "sender", "sender_name", "sender_email",
                "sender_domain", "receiver_domain", "subject", "urls"]:
        print(f"  {col}: {sample[col]}")
    print(f"  text (first 200 chars): {sample['text'][:200]}...")
 
 
if __name__ == "__main__":
    main()