"""
CreditPath AI – Milestone 4 Runner
────────────────────────────────────────────────────────────────────────────
Run this single file to execute the complete Milestone 4 pipeline:
  python milestone4_runner.py

Optional:  Set run_tuning=False for a quick ≈ 2-minute demo (skips GridSearch).
"""

from advanced_models import run_milestone4

if __name__ == "__main__":
    # ── Update lr_baseline_auc with the value printed at the end of Milestone 3
    # ── Set run_tuning=False to skip grid/random search (~2 min faster)
    run_milestone4(
        lr_baseline_auc = 0.72,
        run_tuning      = True,
    )
