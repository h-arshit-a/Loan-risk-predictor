# Milestone 3 – Simple Explanation
# CreditPath AI: What We Did, Why We Did It, and What We Found

==========================================================================

## WHAT IS THE GOAL?
We are trying to predict whether a person will DEFAULT on their loan or not.
Default = they fail to repay the loan.
We trained a machine learning model (Logistic Regression) to make this prediction.

==========================================================================

## THE THREE FILES WE CREATED

### 1. feature_engineering.py
   - This file prepares the data for the model.
   - It loads the cleaned loan data, adds new features, splits data into
     train and test sets, and scales all numbers.

### 2. logistic_model.py
   - This file trains the Logistic Regression model.
   - It then measures how good the model is using AUC-ROC score.
   - It also prints which features push someone TOWARD defaulting
     and which features PROTECT them from defaulting.

### 3. milestone3_runner.py
   - This is the MAIN file you run.
   - It simply calls the other two files in order.
   - Command: python milestone3_runner.py

==========================================================================

## WHAT IS LOGISTIC REGRESSION?
Think of it like a YES/NO calculator.
Given a person's loan information, it calculates the probability (0 to 1)
that they will default. If probability > 0.5 → predicted as default.

Why use it as a BASELINE?
  - It is simple, fast, and easy to explain.
  - It tells us WHICH features matter most (via coefficients).
  - It is a standard starting point before trying complex models.

==========================================================================

## FEATURES (INPUTS) USED — ORIGINAL 24

These came from the cleaned dataset (clean_loans.csv) from Milestone 2:

  NUMERIC FEATURES (actual numbers):
  -----------------------------------
  1.  age              – Borrower's age in years
  2.  income           – Annual income (already scaled)
  3.  loanamount       – Size of the loan taken
  4.  creditscore      – Credit score (higher = more reliable)
  5.  monthsemployed   – How many months the person has been employed
  6.  numcreditlines   – Number of open credit lines (credit cards, loans)
  7.  interestrate     – Interest rate on the loan
  8.  loanterm         – Duration of the loan (in months)
  9.  dtiratio         – Debt-to-Income ratio (how much debt vs income)

  CATEGORICAL FEATURES (encoded as True/False):
  -----------------------------------------------
  10. education_High School      – Highest education is high school?
  11. education_Master's         – Has Master's degree?
  12. education_PhD              – Has PhD?
  13. employmenttype_Part-time   – Employed part-time?
  14. employmenttype_Self-employed – Self-employed?
  15. employmenttype_Unemployed  – Currently unemployed?
  16. maritalstatus_Married      – Is married?
  17. maritalstatus_Single       – Is single?
  18. hasmortgage_Yes            – Has a home mortgage?
  19. hasdependents_Yes          – Has dependents (children/family)?
  20. loanpurpose_Business       – Loan taken for business?
  21. loanpurpose_Education      – Loan taken for education?
  22. loanpurpose_Home           – Loan taken for home?
  23. loanpurpose_Other          – Loan taken for other reason?
  24. hascosigner_Yes            – Has a cosigner on the loan?

==========================================================================

## NEW FEATURES WE ENGINEERED (4 extra features we created)

We added 4 smart new features to give the model extra signals:

  25. debt_to_income_score
      Formula: dtiratio × loanamount
      Why: Combines two risk factors. A person with high debt ratio AND
           a big loan is in double trouble.

  26. loan_per_month
      Formula: loanamount / (loanterm + 1)
      Why: Shows how much they need to pay EVERY MONTH. Higher monthly
           payment = more stress = higher default risk.

  27. credit_income_ratio
      Formula: creditscore / (|income| + 1)
      Why: Shows credit quality relative to income. A high credit score
           but very low income is still risky.

  28. high_risk_flag
      Logic: = 1 if (dtiratio > 0.4) AND (creditscore < 0 after scaling)
      Why: A simple binary alarm for borrowers who are BOTH high-debt
           AND low-credit-score at the same time.

  TOTAL FEATURES GIVEN TO MODEL: 28

==========================================================================

## WHAT IS SCALING AND WHY IS IT MANDATORY?

The model uses math (multiplication, addition) on features.
Problem: income can be 50,000 and age can be 30 — very different scales.
The model would think income is more important just because its number is bigger.

Solution: StandardScaler converts every feature to the same scale (mean=0).
Rule: Scaler is FIT on training data ONLY → then applied to test data.
This prevents "data leakage" (test data accidentally influencing training).

==========================================================================

## TRAIN / TEST SPLIT

  - 80% of data used to TRAIN the model (learn patterns)
  - 20% of data used to TEST the model (check if it works on new data)
  - STRATIFIED split → both sets have the same % of defaults
  - Train size: ~204,277 rows
  - Test size:  ~51,070 rows
  - Total rows in dataset: 255,347

==========================================================================

## WHY NOT USE ACCURACY AS THE METRIC?

Most people didn't default → class imbalance.
If the model just says "NO DEFAULT" for everyone, accuracy = ~78%.
But that model is USELESS — it never catches actual defaulters!

AUC-ROC is better:
  - Measures how well the model RANKS defaulters vs non-defaulters
  - 0.5 = random guessing (useless)
  - 1.0 = perfect model
  - > 0.70 is considered Good for credit risk

==========================================================================

## RESULTS

  AUC-ROC SCORE: 0.7532
  GRADE: [OK] VERY GOOD (>= 0.75)

  Classification Report Summary:
  ┌───────────────┬───────────┬────────┬──────────┐
  │ Class         │ Precision │ Recall │ F1-Score │
  ├───────────────┼───────────┼────────┼──────────┤
  │ No Default    │   0.93    │  0.56  │   0.70   │
  │ Default       │   0.38    │  0.87  │   0.53   │
  │ Weighted Avg  │   0.86    │  0.68  │   0.73   │
  └───────────────┴───────────┴────────┴──────────┘

  - High Recall (0.87) for defaults = model catches 87% of actual defaulters
  - Low Precision (0.38) for defaults = some false alarms (non-defaulters
    flagged as defaulters) → acceptable trade-off in credit risk

==========================================================================

## COEFFICIENT ANALYSIS — FULL TABLE

Coefficient > 0 → Feature INCREASES default risk
Coefficient < 0 → Feature DECREASES default risk (protects borrower)

  Rank  Feature                       Coefficient   Direction
  ────────────────────────────────────────────────────────────
   1    interestrate                   +0.4587      Increases risk
   2    loanamount                     +0.2960      Increases risk
   3    employmenttype_Unemployed      +0.1899      Increases risk
   4    employmenttype_Part-time       +0.1168      Increases risk
   5    employmenttype_Self-employed   +0.1073      Increases risk
   6    numcreditlines                 +0.1025      Increases risk
   7    dtiratio                       +0.0746      Increases risk
   8    education_High School          +0.0324      Increases risk
   9    loanpurpose_Business           +0.0265      Increases risk
  10    loan_per_month (engineered)    +0.0152      Increases risk (mild)
  11    loanterm                       +0.0014      Near-neutral
  12    credit_income_ratio (eng.)     +0.0007      Near-neutral
  ────────────────────────────────────────────────────────────
  13    loanpurpose_Other              -0.0025      Mild protection
  14    loanpurpose_Education          -0.0029      Mild protection
  15    high_risk_flag (engineered)    -0.0097      (absorbed by other features)
  16    debt_to_income_score (eng.)    -0.0232      Mild protection
  17    maritalstatus_Single           -0.0245      Mild protection
  18    education_Master's             -0.0589      Decreases risk
  19    loanpurpose_Home               -0.0728      Decreases risk
  20    education_PhD                  -0.0761      Decreases risk
  21    hasmortgage_Yes                -0.0820      Decreases risk
  22    maritalstatus_Married          -0.0940      Decreases risk
  23    hasdependents_Yes              -0.1260      Decreases risk
  24    hascosigner_Yes                -0.1292      Decreases risk
  25    creditscore                    -0.1313      Decreases risk
  26    income                         -0.3139      Decreases risk
  27    monthsemployed                 -0.3373      Decreases risk
  28    age                            -0.5832      Strongest protection

==========================================================================

## ANALYSIS IN PLAIN ENGLISH

  TOP RISK FACTORS (things that make default MORE likely):
  ─────────────────────────────────────────────────────────
  1. HIGH INTEREST RATE (+0.459)
     Banks charge higher rates to risky borrowers. So high rate = high risk.
     This is the strongest signal that someone will default.

  2. BIG LOAN AMOUNT (+0.296)
     Larger loans are harder to repay. More burden = more chance of default.

  3. BEING UNEMPLOYED (+0.190)
     No income = can't repay. Unemployed borrowers are the most risky
     employment category.

  4. PART-TIME / SELF-EMPLOYED (+0.117 / +0.107)
     Unstable or irregular income makes repayment harder than full-time jobs.

  5. MORE CREDIT LINES (+0.103)
     Having many open credit lines suggests the person is over-leveraged
     (borrowing from many places at once).

  TOP PROTECTIVE FACTORS (things that make default LESS likely):
  ───────────────────────────────────────────────────────────────
  1. OLDER AGE (-0.583)
     Strongest protective factor. Older borrowers are more financially
     experienced, stable, and responsible with repayment.

  2. LONGER EMPLOYMENT HISTORY (-0.337)
     More months employed = stable job = reliable income = repays loan.

  3. HIGHER INCOME (-0.314)
     Higher earners can more easily afford repayments.

  4. HIGHER CREDIT SCORE (-0.131)
     A good credit score means the person has a history of repaying debts.

  5. HAS A COSIGNER (-0.129)
     A cosigner acts as a backup guarantor — reduces lender risk.

==========================================================================

## FINAL VERDICT

  Our Logistic Regression model scored AUC-ROC = 0.7532.
  This is "Very Good" by the project rubric (threshold: 0.75).

  The model successfully identifies:
  - WHO is likely to default: unemployed, high interest rate, large loan
  - WHO is protected: older, longer employed, high income, good credit score

  No model files (pkl/joblib) were saved. Everything runs in memory.
  Two plots were saved: roc_curve.png and confusion_matrix.png

==========================================================================
End of Milestone 3 Explanation
==========================================================================
