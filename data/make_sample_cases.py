"""Create a small SYNTHETIC dataset of brain-morphometry results.

Each "case" is one fictional patient. For each brain region we store:

  - z_score:       what the software shows: how far the region is from a normative
                   (healthy, same-age) reference, in standard deviations.
                   Negative = smaller than expected, positive = larger.
                   It includes measurement noise, like a real scan would.
  - ground_truth:  1 if the region is truly abnormal, else 0. It comes from the
                   hidden "true" state of the region, NOT from the shown z-score,
                   so a reader cannot get it right just by applying a threshold.
  - ai_flag:       what a simulated AI assistant suggests (1 = "abnormal").
  - ai_confidence: "high" or "low". The simulated AI is less sure near the
                   decision boundary, and most of its mistakes are low-confidence
                   (but not all of them), like a reasonably calibrated real model.

No real patient data is used anywhere in this project.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CASES = 8
THRESHOLD = 2.0          # |true z| >= 2 counts as truly abnormal
MEASUREMENT_NOISE = 0.5  # noise between the true value and the shown z-score

REGIONS = [
    "Hippocampus",
    "Entorhinal cortex",
    "Amygdala",
    "Precuneus",
    "Superior frontal gyrus",
    "Superior temporal gyrus",
    "Insula",
    "Lateral ventricle",
]
HEMISPHERES = ["Left", "Right"]


def make_cases(seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for case_number in range(1, N_CASES + 1):
        case_id = f"CASE-{case_number:02d}"
        # Half of the cases get an "atrophy pattern" in memory-related regions.
        has_pattern = case_number % 2 == 0
        for region in REGIONS:
            for hemi in HEMISPHERES:
                true_z = rng.normal(0, 0.9)
                if has_pattern and region in {"Hippocampus", "Entorhinal cortex", "Amygdala"}:
                    true_z -= rng.uniform(1.5, 3.2)   # smaller than expected
                if has_pattern and region == "Lateral ventricle":
                    true_z += rng.uniform(1.5, 3.0)   # ventricles enlarge with atrophy
                truth = int(abs(true_z) >= THRESHOLD)
                shown_z = round(float(true_z + rng.normal(0, MEASUREMENT_NOISE)), 2)

                # The AI sees the noisy value too. Close to the threshold it is unsure.
                margin = abs(abs(shown_z) - THRESHOLD)
                confidence = "high" if margin > 0.6 else "low"
                p_correct = 0.95 if confidence == "high" else 0.60
                ai_correct = rng.random() < p_correct
                ai_flag = truth if ai_correct else 1 - truth

                rows.append({
                    "case_id": case_id,
                    "region": f"{hemi} {region}",
                    "z_score": shown_z,
                    "ground_truth": truth,
                    "ai_flag": ai_flag,
                    "ai_confidence": confidence,
                })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_cases.csv"
    df = make_cases()
    df.to_csv(out, index=False)
    ai_ok = df.ai_flag == df.ground_truth
    print(f"Wrote {len(df)} rows ({df.case_id.nunique()} cases) to {out}")
    print(f"Truly abnormal regions: {df.ground_truth.sum()}")
    print(f"AI accuracy overall: {ai_ok.mean():.0%}")
    print(df.assign(ai_ok=ai_ok).groupby("ai_confidence").ai_ok.agg(["size", "mean"]).round(2))
