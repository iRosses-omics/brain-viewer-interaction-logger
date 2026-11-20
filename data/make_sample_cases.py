"""Create a small SYNTHETIC dataset of brain-morphometry results.

Each "case" is one fictional patient. For each brain region we store:
  - z_score:      how far the region's volume is from a normative (healthy, same-age)
                  reference, in standard deviations. Negative = smaller than expected
                  (atrophy); positive = larger than expected (e.g. enlarged ventricles).
  - ground_truth: 1 if the region is truly abnormal (|z| >= 2), else 0.
  - ai_flag:      what a simulated AI assistant suggests (1 = "abnormal").
                  The simulated AI is right most of the time but makes some mistakes
                  on purpose, so we can study how users react to wrong AI advice.

No real patient data is used anywhere in this project.
"""

from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
N_CASES = 8
AI_ACCURACY = 0.80          # the simulated AI is wrong ~20% of the time
ABNORMAL_THRESHOLD = 2.0    # |z| >= 2 counts as abnormal

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
                z = rng.normal(0, 0.9)
                if has_pattern and region in {"Hippocampus", "Entorhinal cortex", "Amygdala"}:
                    z -= rng.uniform(1.8, 3.2)      # smaller than expected
                if has_pattern and region == "Lateral ventricle":
                    z += rng.uniform(1.8, 3.0)      # ventricles enlarge with atrophy
                z = round(float(z), 2)
                truth = int(abs(z) >= ABNORMAL_THRESHOLD)
                ai_correct = rng.random() < AI_ACCURACY
                ai_flag = truth if ai_correct else 1 - truth
                rows.append(
                    {
                        "case_id": case_id,
                        "region": f"{hemi} {region}",
                        "z_score": z,
                        "ground_truth": truth,
                        "ai_flag": ai_flag,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    out = Path(__file__).parent / "sample_cases.csv"
    df = make_cases()
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} rows ({df.case_id.nunique()} cases) to {out}")
    print(f"Truly abnormal regions: {df.ground_truth.sum()}")
    print(f"Simulated AI accuracy: {(df.ai_flag == df.ground_truth).mean():.0%}")
