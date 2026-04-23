"""Quick verification of all 3 risk tiers."""
import urllib.request, json

API = "http://127.0.0.1:8000"

def predict(payload):
    req = urllib.request.Request(
        f"{API}/predict", data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as r:
        return json.loads(r.read())

profiles = {
    "Low Risk (Excellent)": {
        "age": 18, "income": 120000, "loan_amount": 5000, "credit_score": 820,
        "loan_term": 12, "interest_rate": 4.5, "dti_ratio": 0.12,
        "months_employed": 60, "num_credit_lines": 8,
        "loan_purpose": "Auto", "employment_type": "Full-time",
        "education": "Bachelor's", "marital_status": "Married",
        "has_cosigner": "Yes", "has_dependents": "No", "has_mortgage": "No",
    },
    "Reference Example (§13)": {
        "age": 40, "income": 50000, "loan_amount": 20000, "credit_score": 620,
        "loan_term": 36, "interest_rate": 12.5, "dti_ratio": 0.35,
        "months_employed": 24, "num_credit_lines": 3,
        "loan_purpose": "Education", "employment_type": "Full-time",
        "education": "Bachelor's", "marital_status": "Single",
        "has_cosigner": "No", "has_dependents": "No", "has_mortgage": "No",
    },
    "High Risk (Stressed)": {
        "age": 58, "income": 18000, "loan_amount": 35000, "credit_score": 350,
        "loan_term": 60, "interest_rate": 25.0, "dti_ratio": 0.78,
        "months_employed": 0, "num_credit_lines": 1,
        "loan_purpose": "Other", "employment_type": "Unemployed",
        "education": "High School", "marital_status": "Divorced",
        "has_cosigner": "No", "has_dependents": "Yes", "has_mortgage": "Yes",
    },
}

print("\n" + "="*60)
for name, payload in profiles.items():
    r = predict(payload)
    print(f"\n  {name}")
    print(f"    Probability : {r['probability']}")
    print(f"    Risk        : {r['risk']}")
    print(f"    Action      : {r['action']}")
    print(f"    Features    : {r['feature_count']}")
print("\n" + "="*60)
