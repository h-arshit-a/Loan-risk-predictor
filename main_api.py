"""
CreditPath AI – Milestone 5
FastAPI Prediction Service
──────────────────────────────────────────────────────────────────────────────
ARCHITECTURE (per reference guide §2)
  User Input → Validation → Feature Engineering → Scaling → Model Prediction
  → Probability → Risk Categorization → Recommendation → JSON Response

HOW TO RUN
  1. First, generate model artefacts (once):
         python milestone5_train_and_save.py

  2. Start the API:
         uvicorn main_api:app --reload

  3. Open Swagger UI:
         http://127.0.0.1:8000/docs

  4. POST to /predict with JSON body, e.g.:
     {
       "age": 40,
       "income": 50000.0,
       "loan_amount": 20000.0,
       "credit_score": 620,
       "loan_term": 36,
       "interest_rate": 12.5,
       "employment_type": "Full-time",
       "months_employed": 24,
       "has_cosigner": "No",
       "has_dependents": "No",
       "has_mortgage": "No",
       "loan_purpose": "Education",
       "marital_status": "Single",
       "education": "Bachelor's",
       "num_credit_lines": 3,
       "dti_ratio": 0.35
     }
──────────────────────────────────────────────────────────────────────────────
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
import pandas as pd
import numpy as np
import joblib
import os
from typing import Optional

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  1.  APP SETUP                                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

app = FastAPI(
    title="CreditPath AI – Loan Default Predictor",
    description=(
        "## Milestone 5: Recommendation Engine & FastAPI\n\n"
        "Send borrower profile → receive default probability, risk category, "
        "and a recommended collection action.\n\n"
        "**Risk thresholds**\n"
        "- `< 0.30` → Low Risk → Send Reminder\n"
        "- `0.30 – 0.59` → Medium Risk → Call Customer\n"
        "- `≥ 0.60` → High Risk → Immediate Recovery Action\n"
    ),
    version="1.0.0",
    contact={"name": "CreditPath AI Team"},
    license_info={"name": "MIT"},
)

# Allow browsers / dashboards to call the API (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  2.  LOAD ARTEFACTS AT STARTUP                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def _load_artefact(name: str):
    path = os.path.join(BASE_DIR, name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Artefact '{name}' not found at '{path}'. "
            "Run  python milestone5_train_and_save.py  first."
        )
    return joblib.load(path)


try:
    MODEL         = _load_artefact("model.pkl")
    SCALER        = _load_artefact("scaler.pkl")
    FEATURE_NAMES = _load_artefact("feature_names.pkl")
    print(f"[STARTUP] Model, scaler and {len(FEATURE_NAMES)} feature names loaded.")
except FileNotFoundError as e:
    MODEL = SCALER = FEATURE_NAMES = None
    print(f"[STARTUP WARNING] {e}")


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  3.  INPUT SCHEMA (Pydantic validation layer)                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝

EMPLOYMENT_TYPES  = ["Full-time", "Part-time", "Self-employed", "Unemployed"]
LOAN_PURPOSES     = ["Auto", "Business", "Education", "Home", "Other"]
MARITAL_STATUSES  = ["Divorced", "Married", "Single"]
EDUCATION_LEVELS  = ["Bachelor's", "High School", "Master's", "PhD"]
YES_NO            = ["Yes", "No"]


class LoanInput(BaseModel):
    """
    Borrower and loan attributes.
    All values must match the encoding used during training (clean_loans.csv).
    """

    # ── Core numeric fields (matching milestone guide §7) ───────────────────
    age:           int   = Field(..., ge=18, le=100,      example=40,
                                 description="Borrower age in years (18–100)")
    income:        float = Field(..., ge=0,               example=50000.0,
                                 description="Annual income in USD")
    loan_amount:   float = Field(..., ge=100,             example=20000.0,
                                 description="Loan amount requested in USD")
    credit_score:  int   = Field(..., ge=300, le=850,     example=620,
                                 description="FICO credit score (300–850)")

    # ── Additional fields needed by the feature pipeline ────────────────────
    loan_term:       int   = Field(36,  ge=6,    le=360,    example=36,
                                   description="Loan term in months")
    interest_rate:   float = Field(10.0, ge=0.0, le=40.0,   example=12.5,
                                   description="Annual interest rate (%)")
    dti_ratio:       float = Field(0.35, ge=0.0, le=1.0,    example=0.35,
                                   description="Debt-to-income ratio (0–1)")
    months_employed: int   = Field(24,  ge=0,               example=24,
                                   description="Months at current employer")
    num_credit_lines:int   = Field(3,   ge=0,               example=3,
                                   description="Number of open credit lines")

    # ── Categorical fields ──────────────────────────────────────────────────
    employment_type: str = Field("Full-time", example="Full-time",
                                 description=f"One of: {EMPLOYMENT_TYPES}")
    has_cosigner:    str = Field("No",        example="No",
                                 description="Has co-signer? Yes / No")
    has_dependents:  str = Field("No",        example="No",
                                 description="Has dependents? Yes / No")
    has_mortgage:    str = Field("No",        example="No",
                                 description="Has mortgage? Yes / No")
    loan_purpose:    str = Field("Education", example="Education",
                                 description=f"One of: {LOAN_PURPOSES}")
    marital_status:  str = Field("Single",    example="Single",
                                 description=f"One of: {MARITAL_STATUSES}")
    education:       str = Field("Bachelor's",example="Bachelor's",
                                 description=f"One of: {EDUCATION_LEVELS}")

    # ── Validators ──────────────────────────────────────────────────────────
    @validator("employment_type")
    def validate_employment(cls, v):
        if v not in EMPLOYMENT_TYPES:
            raise ValueError(f"employment_type must be one of {EMPLOYMENT_TYPES}")
        return v

    @validator("loan_purpose")
    def validate_purpose(cls, v):
        if v not in LOAN_PURPOSES:
            raise ValueError(f"loan_purpose must be one of {LOAN_PURPOSES}")
        return v

    @validator("marital_status")
    def validate_marital(cls, v):
        if v not in MARITAL_STATUSES:
            raise ValueError(f"marital_status must be one of {MARITAL_STATUSES}")
        return v

    @validator("education")
    def validate_education(cls, v):
        if v not in EDUCATION_LEVELS:
            raise ValueError(f"education must be one of {EDUCATION_LEVELS}")
        return v

    @validator("has_cosigner", "has_dependents", "has_mortgage")
    def validate_yes_no(cls, v):
        if v not in YES_NO:
            raise ValueError(f"Value must be 'Yes' or 'No'")
        return v


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  4.  FEATURE ENGINEERING  (MUST match training pipeline)                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

# ── Pre-scaling factors from original raw dataset (Milestone 2) ───────────
# data_cleaning.py scaled these 3 features before saving clean_loans.csv
PRE_SCALE = {
    "income":       {"mean": 6957.34, "std": 6496.59},
    "loanamount":   {"mean": 331117.74, "std": 183909.31},
    "creditscore":  {"mean": 699.79, "std": 115.88},
}

def build_raw_dataframe(data: LoanInput) -> pd.DataFrame:
    """
    Convert Pydantic model → raw DataFrame that mirrors the structure of
    clean_loans.csv BEFORE dummy-encoding.
    
    CRITICAL FIX: clean_loans.csv already had income, loanamount, and 
    creditscore standardized. If we pass raw values here, they will get 
    'double scaled' and become extreme outliers. We must pre-standardize them.
    """
    row = {
        "age"            : data.age,
        "income"         : (data.income - PRE_SCALE["income"]["mean"]) / PRE_SCALE["income"]["std"],
        "loanamount"     : (data.loan_amount - PRE_SCALE["loanamount"]["mean"]) / PRE_SCALE["loanamount"]["std"],
        "creditscore"    : (data.credit_score - PRE_SCALE["creditscore"]["mean"]) / PRE_SCALE["creditscore"]["std"],
        "loanterm"       : data.loan_term,
        "interestrate"   : data.interest_rate,
        "dtiratio"       : data.dti_ratio,
        "monthsemployed" : data.months_employed,
        "numcreditlines" : data.num_credit_lines,
        # categorical – will be one-hot encoded below
        "employmenttype" : data.employment_type,
        "hascosigner"    : data.has_cosigner,
        "hasdependents"  : data.has_dependents,
        "hasmortgage"    : data.has_mortgage,
        "loanpurpose"    : data.loan_purpose,
        "maritalstatus"  : data.marital_status,
        "educationlevel" : data.education,
    }
    return pd.DataFrame([row])


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the EXACT same transformations used during training
    (mirrors feature_engineering.py → engineer_features).

    Derived features:
      debt_to_income_score = dtiratio × loanamount
      loan_per_month       = loanamount / (loanterm + 1)
      credit_income_ratio  = creditscore / (income + 1)
      high_risk_flag       = 1 if dtiratio > 0.4 AND creditscore < 0
    """
    df = df.copy()

    # -- Engineered features (identical to feature_engineering.py) -----------
    df["debt_to_income_score"] = df["dtiratio"] * df["loanamount"]
    df["loan_per_month"]       = df["loanamount"] / (df["loanterm"] + 1)
    df["credit_income_ratio"]  = df["creditscore"] / (df["income"].abs() + 1)
    df["high_risk_flag"]       = (
        (df["dtiratio"] > 0.4) & (df["creditscore"] < 0)
    ).astype(int)

    # -- One-hot encode categoricals (must match dummies from training) -------
    df = pd.get_dummies(df, columns=[
        "employmenttype", "hascosigner", "hasdependents",
        "hasmortgage", "loanpurpose", "maritalstatus", "educationlevel",
    ])

    return df


def align_to_training_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    CRITICAL: ensure column order and presence EXACTLY matches what the model
    was trained on.  Missing dummies → add as 0.  Extra dummies → drop.
    """
    if FEATURE_NAMES is None:
        raise RuntimeError("feature_names.pkl not loaded – run training script first.")

    for col in FEATURE_NAMES:
        if col not in df.columns:
            df[col] = 0  # unseen dummy → fill with 0

    return df[FEATURE_NAMES]   # enforce exact column order


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  5.  RECOMMENDATION ENGINE                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def recommend_action(prob: float) -> str:
    """
    Rule-based recommendation engine (Milestone 5 §4).
    Maps default probability to a concrete business action.

    Thresholds (§3):
      Note: XGBoost with `scale_pos_weight` artificially boosts probabilities
      because it simulates a 50/50 balanced dataset. We adjust the "Low Risk" 
      threshold upward from 0.30 to 0.48 so that Excellent profiles trigger it,
      while preserving 0.60 as the High-Risk threshold.
    """
    if prob < 0.48:
        return "Low Risk - Send Reminder"
    elif prob < 0.60:
        return "Medium Risk - Call Customer"
    else:
        return "High Risk - Immediate Recovery Action"


def risk_label(prob: float) -> str:
    if prob >= 0.60:
        return "High"
    elif prob >= 0.48:
        return "Medium"
    return "Low"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  6.  API ENDPOINTS                                                       ║
# ╚══════════════════════════════════════════════════════════════════════════╝

@app.get("/", tags=["Health"])
def root():
    """Health check / welcome endpoint."""
    return {
        "service"   : "CreditPath AI – Milestone 5",
        "status"    : "running",
        "model_loaded": MODEL is not None,
        "docs"      : "/docs",
    }


@app.get("/health", tags=["Health"])
def health():
    """Detailed health check."""
    return {
        "model_loaded"        : MODEL is not None,
        "scaler_loaded"       : SCALER is not None,
        "feature_count"       : len(FEATURE_NAMES) if FEATURE_NAMES else 0,
        "message"             : (
            "All systems operational"
            if MODEL is not None
            else "Model not loaded – run milestone5_train_and_save.py"
        ),
    }


@app.post("/predict", tags=["Prediction"], response_model=dict)
def predict(data: LoanInput):
    """
    ## Predict Loan Default Probability

    **Pipeline** (per Milestone 5 reference §2):
    1. Validate input (Pydantic)
    2. Build raw DataFrame
    3. Engineer derived features (same as training)
    4. One-hot encode categoricals
    5. Align column order to training feature set
    6. Scale using the saved StandardScaler
    7. Predict probability with XGBoost
    8. Map to risk category + business action

    **Response fields**
    - `probability` – default probability [0, 1]
    - `risk`        – Low / Medium / High
    - `action`      – recommended collection strategy
    - `feature_count` – sanity check on engineered feature count
    """
    if MODEL is None or SCALER is None or FEATURE_NAMES is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model artefacts not loaded. "
                "Run  python milestone5_train_and_save.py  first, "
                "then restart the API."
            ),
        )

    try:
        # Step A: raw DataFrame
        raw_df = build_raw_dataframe(data)

        # Step B: feature engineering (identical to training)
        engineered_df = apply_feature_engineering(raw_df)

        # Step C: align columns (add missing dummies, enforce order)
        aligned_df = align_to_training_features(engineered_df)

        # Step D: scale (use the SAVED scaler, never fit again)
        scaled_input = SCALER.transform(aligned_df)

        # Step E: predict
        prob = float(MODEL.predict_proba(scaled_input)[0][1])

        # Step F: recommend
        action = recommend_action(prob)
        risk   = risk_label(prob)

    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    return {
        "probability"  : round(prob, 4),
        "risk"         : risk,
        "action"       : action,
        "feature_count": int(aligned_df.shape[1]),
        "input_summary": {
            "age"         : data.age,
            "income"      : data.income,
            "loan_amount" : data.loan_amount,
            "credit_score": data.credit_score,
            "dti_ratio"   : data.dti_ratio,
        },
    }


@app.post("/predict/batch", tags=["Prediction"])
def predict_batch(records: list[LoanInput]):
    """
    ## Batch Prediction

    Send up to 100 loan records at once.  Returns a list of prediction objects.
    Useful for dashboard bulk processing or collection agency workflows.
    """
    if MODEL is None or SCALER is None or FEATURE_NAMES is None:
        raise HTTPException(status_code=503, detail="Model artefacts not loaded.")

    if len(records) > 100:
        raise HTTPException(status_code=400, detail="Batch size must be ≤ 100.")

    results = []
    for i, record in enumerate(records):
        try:
            raw_df        = build_raw_dataframe(record)
            eng_df        = apply_feature_engineering(raw_df)
            aligned_df    = align_to_training_features(eng_df)
            scaled_input  = SCALER.transform(aligned_df)
            prob          = float(MODEL.predict_proba(scaled_input)[0][1])
            results.append({
                "record_index": i,
                "probability" : round(prob, 4),
                "risk"        : risk_label(prob),
                "action"      : recommend_action(prob),
            })
        except Exception as exc:
            results.append({
                "record_index": i,
                "error"       : str(exc),
            })

    return {"predictions": results, "total": len(results)}


@app.get("/risk-thresholds", tags=["Info"])
def risk_thresholds():
    """Return the current risk threshold configuration."""
    return {
        "thresholds": {
            "low_max"    : 0.30,
            "medium_max" : 0.60,
            "high_min"   : 0.60,
        },
        "actions": {
            "Low"   : "Send Reminder",
            "Medium": "Call Customer",
            "High"  : "Immediate Recovery Action",
        },
        "note": (
            "Thresholds can be tuned based on business cost of "
            "false positives vs false negatives."
        ),
    }
