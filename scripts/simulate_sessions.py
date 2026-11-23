"""Generate SIMULATED session logs, so the analysis can be tested before real users exist.

Simple behaviour model for each simulated reader:
  - Own decision: correct with probability `skill` (lower for regions close to
    the decision boundary, which are harder).
  - ai_on cases: after the own decision, the AI advice appears. If the AI agrees,
    the reader keeps the decision. If it disagrees, the reader switches with
    probability `trust_high` (high-confidence advice) or `trust_low`
    (low-confidence advice). A reader with trust_low < trust_high is
    "calibrated": they rely less on uncertain advice.
  - Reviewing the advice takes extra time.

The output has exactly the same columns as the real logger, so the same
analysis notebook works on simulated and real data.

These are NOT real study results. They only demonstrate the analysis pipeline.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from logger import FIELDS, SCHEMA_VERSION  # noqa: E402

SEED = 7
N_PARTICIPANTS = 12
THRESHOLD = 2.0


def simulate(cases: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    case_ids = sorted(cases.case_id.unique())
    rows = []
    for p in range(1, N_PARTICIPANTS + 1):
        pid, session = f"SIM{p:02d}", f"sim{p:04d}"
        skill = rng.uniform(0.80, 0.93)
        trust_high = rng.uniform(0.55, 0.90)
        trust_low = trust_high * rng.uniform(0.35, 1.0)
        speed = rng.uniform(0.75, 1.35)  # some readers are slower than others
        # Same random split as the real viewer.
        ai_cases = set(random.Random(pid).sample(case_ids, k=len(case_ids) // 2))

        for case_id in case_ids:
            cond = "ai_on" if case_id in ai_cases else "ai_off"
            base = dict(participant_id=pid, session_id=session, case_id=case_id, ai_condition=cond)
            t = 0.0
            rows.append({**base, "event": "case_open", "ms_since_case_open": 0})

            for rec in cases[cases.case_id == case_id].itertuples():
                region = dict(region=rec.region, z_score=rec.z_score)
                truth = rec.ground_truth
                hard = abs(abs(rec.z_score) - THRESHOLD) <= 0.6

                t += rng.lognormal(np.log(700 * speed), 0.3)
                rows.append({**base, **region, "event": "region_select", "ms_since_case_open": round(t)})

                p_own = skill - (0.25 if hard else 0.0)
                initial = truth if rng.random() < p_own else 1 - truth
                t += rng.lognormal(np.log(2200 * speed), 0.4)
                rows.append({**base, **region, "event": "rating_initial", "rating": initial,
                             "ground_truth": truth, "ms_since_case_open": round(t)})

                final, ai = initial, {}
                if cond == "ai_on":
                    ai = dict(ai_flag=rec.ai_flag, ai_confidence=rec.ai_confidence)
                    rows.append({**base, **region, **ai, "event": "ai_suggestion_shown",
                                 "ms_since_case_open": round(t + 50)})
                    if rec.ai_flag != initial:
                        trust = trust_high if rec.ai_confidence == "high" else trust_low
                        if rng.random() < trust:
                            final = rec.ai_flag
                    t += rng.lognormal(np.log(1300 * speed), 0.4)
                rows.append({**base, **region, **ai, "event": "rating_final", "rating": final,
                             "ground_truth": truth, "ms_since_case_open": round(t)})

            rows.append({**base, "event": "case_complete", "ms_since_case_open": round(t + 800)})

    df = pd.DataFrame(rows)
    df["schema_version"] = SCHEMA_VERSION
    df["timestamp_utc"] = None
    return df.reindex(columns=FIELDS)


if __name__ == "__main__":
    cases = pd.read_csv(ROOT / "data" / "sample_cases.csv")
    out = ROOT / "logs" / "simulated_sessions.csv"
    df = simulate(cases)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} simulated events for {df.participant_id.nunique()} participants to {out}")
