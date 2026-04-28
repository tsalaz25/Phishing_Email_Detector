# Phishing Email Detector
CS 444 Cybersecurity Project | Oscar M. | Isaac T. | Tomas S.

---

## Project Structure

```
Cyber_Project/
├── data_loader.py          # Downloads + cleans the email dataset
├── feature_extractor.py    # Converts emails into numeric features for ML
├── classifier.py           # Trains models and prints evaluation results
└── data/
    └── processed/
        └── phishing_emails.parquet   # Cached dataset (auto-generated)
```

---

## First-Time Setup

You only need to do this once.

**1. Create the virtual environment**
```bash
cd ~/Desktop/CS\ 444/Cyber_Project
python -m venv .venv
```

**2. Activate it**
```bash
source .venv/bin/activate
```
You should see `(.venv)` at the start of your terminal prompt.

**3. Install dependencies**
```bash
pip install --upgrade pip
pip install datasets pandas pyarrow scikit-learn scipy
```

---

## Running the Project

Every time you open a new terminal, activate the virtual environment first:
```bash
source .venv/bin/activate
```

Then run each script in order:

**Step 1 — Load and clean the dataset**
```bash
python data_loader.py
```
Downloads ~200k labeled emails from HuggingFace and saves a cleaned cache
to `data/processed/phishing_emails.parquet`. Takes 2–5 minutes on the first
run (requires internet). Subsequent runs load from cache and finish in seconds.

**Step 2 — Test feature extraction**
```bash
python feature_extractor.py
```
Transforms the emails into a numeric feature matrix and prints a sanity check
showing which hand-crafted features differ most between phishing and legit emails.

**Step 3 — Train models and see results**
```bash
python classifier.py
```
Trains a Logistic Regression and a Linear SVM, then prints:
- Precision, Recall, F1, and ROC-AUC for each model
- Top words/phrases most predictive of phishing
- Weights for structural features (exclamation counts, sender signals, etc.)
- Edge cases: the most confidently wrong predictions in each direction

When done, run `deactivate` to exit the virtual environment.
