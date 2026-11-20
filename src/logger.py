"""A small, reusable interaction logger.

Every user action becomes one row in a CSV file. Keeping one flat, well-defined
schema makes the logs easy to analyse later with pandas.

Design choices:
  - Append-only CSV, one file per session -> nothing is ever overwritten.
  - Timing is measured with a monotonic clock (time.perf_counter), so it is not
    affected if the computer's clock changes during a session.
  - No personal data: participants are identified only by a pseudonymous ID.
"""

import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

FIELDS = [
    "timestamp_utc",       # wall-clock time of the event (ISO 8601)
    "session_id",          # random ID for this session
    "participant_id",      # pseudonymous ID typed in by the participant
    "case_id",             # which case was on screen
    "ai_condition",        # "ai_on" or "ai_off" for the current case
    "event",               # what happened (see EVENTS)
    "region",              # brain region involved, if any
    "z_score",             # z-score of that region, if any
    "ai_flag",             # what the AI suggested for that region (1/0), if shown
    "rating",              # the participant's rating (1 = abnormal, 0 = normal)
    "ground_truth",        # true label, stored for later analysis only (never shown)
    "ms_since_case_open",  # milliseconds since the case was opened
]

EVENTS = {
    "session_start",
    "case_open",
    "region_select",
    "ai_suggestion_shown",
    "rating",
    "case_complete",
    "session_end",
}


class InteractionLogger:
    def __init__(self, participant_id: str, log_dir: str | Path = "logs"):
        self.participant_id = participant_id
        self.session_id = uuid.uuid4().hex[:8]
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = self.log_dir / f"session_{stamp}_{participant_id}_{self.session_id}.csv"
        self._case_t0 = None
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def start_case_timer(self) -> None:
        self._case_t0 = time.perf_counter()

    def log(self, event: str, **fields) -> None:
        if event not in EVENTS:
            raise ValueError(f"Unknown event type: {event}")
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise ValueError(f"Unknown fields: {unknown}")
        elapsed = None
        if self._case_t0 is not None:
            elapsed = round((time.perf_counter() - self._case_t0) * 1000)
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "event": event,
            "ms_since_case_open": elapsed,
            **fields,
        }
        self._writer.writerow(row)
        self._file.flush()  # write immediately, so nothing is lost if the app crashes

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
