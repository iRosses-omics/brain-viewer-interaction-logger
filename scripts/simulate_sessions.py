"""Generate SIMULATED session logs so the analysis can be tested before real users exist.

The simulated participants follow a simple behaviour model:
  - Without AI, they judge each region correctly with probability `skill`.
  - With AI, they follow the AI suggestion with probability `trust`
    (this models "automation bias"); otherwise they use their own judgement.
  - With AI they decide a bit faster.

The output has exactly the same columns as the real logger, so the same
analysis notebook works on simulated and real data.

These are NOT real study results - they only demonstrate the analysis pipeline.
"""

import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
from logger import FIELDS  # noqa: E402

SEED = 7
N_PARTICIPANTS = 12


def simulate(cases: pd.DataFrame, seed: int = SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    case_ids = sorted(cases.case_id.unique())
    rows = []
    for p in range(1, N_PARTICIPANTS + 1):
        pid = f"SIM{p:02d}"
        session = f"sim{p:04d}"
        skill = rng.uniform(0.75, 0.92)   # own accuracy
        trust = rng.uniform(0.55, 0.90)   # tendency to follow the AI
        ai_cases = set(random.Random(pid).sample(case_ids, k=len(case_ids) // 2))
        for case_id in case_ids:
            cond = "ai_on" if case_id in ai_cases else "ai_off"
            t = 0.0
            base = dict(participant_id=pid, session_id=session, case_id=case_id, ai_condition=cond)
            rows.append({**base, "event": "case_open", "ms_since_case_open": 0})
            for rec in cases[cases.case_id == case_id].itertuples():
                t += rng.lognormal(mean=np.log(2500 if cond == "ai_off" else 2000), sigma=0.4)
                own = rec.ground_truth if rng.random() < skill else 1 - rec.ground_truth
                if cond == "ai_on":
                    rating = rec.ai_flag if rng.random() < trust else own
                    ai_flag = rec.ai_flag
                else:
                    rating, ai_flag = own, None
                rows.append(
                    {
                        **base,
                        "event": "rating",
                        "region": rec.region,
                        "z_score": rec.z_score,
                        "ai_flag": ai_flag,
                        "rating": int(rating),
                        "ground_truth": rec.ground_truth,
                        "ms_since_case_open": round(t),
                    }
                )
            rows.append({**base, "event": "case_complete", "ms_since_case_open": round(t + 800)})
    df = pd.DataFrame(rows)
    df["timestamp_utc"] = None
    return df.reindex(columns=FIELDS)


if __name__ == "__main__":
    cases = pd.read_csv(ROOT / "data" / "sample_cases.csv")
    out = ROOT / "logs" / "simulated_sessions.csv"
    df = simulate(cases)
    df.to_csv(out, index=False)
    print(f"Wrote {len(df)} simulated events for {df.participant_id.nunique()} participants to {out}")
