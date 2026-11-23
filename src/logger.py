"""A small, reusable interaction logger.

Idea: every user action becomes ONE row in a CSV file, always with the same
columns. One flat, fixed table is easy to analyse later with pandas.

Design choices (each one answers a practical question):
  - What is stored?   Only the columns in FIELDS. Anything else is refused.
  - Is data lost if the app crashes?   No: each row is written to disk at once.
  - Is timing reliable?   Yes: durations use a monotonic clock
    (time.perf_counter), which is not affected if the computer clock changes.
  - Is it private?   Participants appear only as a pseudonymous ID.
  - Can old logs still be read after the schema changes?   Yes: every row
    carries SCHEMA_VERSION.
"""

import csv
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = "2.0"

# The columns of every log file, in order.
FIELDS = [
    "schema_version",      # version of this column layout
    "timestamp_utc",       # wall-clock time of the event (ISO 8601, UTC)
    "session_id",          # random ID for this session
    "participant_id",      # pseudonymous ID, e.g. "P01" (no names)
    "case_id",             # which case is on screen
    "ai_condition",        # "ai_on" or "ai_off" for this case
    "event",               # what happened (see EVENTS)
    "region",              # brain region involved, if any
    "z_score",             # z-score shown for that region
    "rating",              # the reader's decision: 1 = abnormal, 0 = normal
    "ai_flag",             # the AI's suggestion: 1 = abnormal, 0 = normal
    "ai_confidence",       # the AI's confidence: "high" or "low"
    "ground_truth",        # true label, for analysis only (never shown)
    "ms_since_case_open",  # milliseconds since the case was opened (monotonic)
]

# The only events that may be logged.
EVENTS = [
    "session_start",        # participant starts
    "case_open",            # a new case is shown; the case timer starts
    "region_select",        # reader clicks a region
    "rating_initial",       # reader's OWN decision, before any AI advice
    "ai_suggestion_shown",  # AI advice appears (ai_on cases only)
    "rating_final",         # reader's final decision (may differ after AI)
    "case_complete",        # reader moves to the next case
    "session_end",          # all cases done
]


class InteractionLogger:
    """Writes one CSV file per session. Usage: logger.log("case_open", case_id=...)."""

    def __init__(self, participant_id: str, log_dir="logs"):
        self.participant_id = participant_id
        self.session_id = uuid.uuid4().hex[:8]
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        self.path = log_dir / f"session_{stamp}_{participant_id}_{self.session_id}.csv"

        self._case_start = None  # monotonic time when the current case opened
        self._file = open(self.path, "w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._file, fieldnames=FIELDS)
        self._writer.writeheader()

    def start_case_timer(self) -> None:
        """Call when a new case is shown. Later events store the time since then."""
        self._case_start = time.perf_counter()

    def log(self, event: str, **fields) -> None:
        """Write one row. `fields` must be names from FIELDS, e.g. region="..."."""
        if event not in EVENTS:
            raise ValueError(f"Unknown event: {event}")
        unknown = set(fields) - set(FIELDS)
        if unknown:
            raise ValueError(f"Unknown fields: {unknown}")

        elapsed_ms = None
        if self._case_start is not None:
            elapsed_ms = round((time.perf_counter() - self._case_start) * 1000)

        row = {
            "schema_version": SCHEMA_VERSION,
            "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "session_id": self.session_id,
            "participant_id": self.participant_id,
            "event": event,
            "ms_since_case_open": elapsed_ms,
            **fields,
        }
        self._writer.writerow(row)
        self._file.flush()  # write to disk now, so nothing is lost if the app crashes

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()
