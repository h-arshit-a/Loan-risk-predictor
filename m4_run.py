"""
Milestone 4 – clean execution wrapper.
Redirects all output to m4_results.txt and the console.
"""
import sys, io, os

# Force UTF-8 for console output
os.environ["PYTHONIOENCODING"] = "utf-8"

# Capture real numbers to a results file too
log = open("m4_results.txt", "w", encoding="utf-8")

class Tee:
    def __init__(self, *streams):
        self.streams = streams
    def write(self, data):
        for s in self.streams:
            try:
                s.write(data)
            except Exception:
                pass
    def flush(self):
        for s in self.streams:
            try:
                s.flush()
            except Exception:
                pass

sys.stdout = Tee(sys.__stdout__, log)
sys.stderr = Tee(sys.__stderr__, log)

# ── Run the actual pipeline ──────────────────────────────────────────────────
from advanced_models import run_milestone4

results = run_milestone4(
    lr_baseline_auc = 0.72,
    run_tuning      = True,
)

log.close()
print("\n[DONE] Results written to m4_results.txt", file=sys.__stdout__)
