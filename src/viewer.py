"""Minimal PyQt5 brain-morphometry viewer with interaction logging.

Study protocol ("decide first, then see the AI"):
  1. The participant enters a pseudonymous ID (e.g. P01).
  2. For each case, they see the regional z-scores and rate every region.
  3. For each region they first give their OWN decision (Normal / Abnormal).
  4. In "ai_on" cases, the AI suggestion then appears, with its confidence
     (high / low), and they give a FINAL decision: keep or change.
     In "ai_off" cases, the first decision is final.
  5. Half of the cases are ai_on, chosen at random (reproducible per participant).

Why this design? Because we record the decision BEFORE and AFTER the advice,
we can see exactly when the AI changed someone's mind, whether that change
helped or hurt, and whether low-confidence advice is followed less.

Everything is written to logs/ by InteractionLogger.
Run:  python src/viewer.py
"""

import random
import sys
from pathlib import Path

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QMainWindow,
    QMessageBox, QPushButton, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from logger import InteractionLogger

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sample_cases.csv"
LOG_DIR = ROOT / "logs"


def z_to_color(z: float) -> QColor:
    """Blue = smaller than normal, red = larger than normal, white near 0."""
    strength = min(abs(z) / 3.0, 1.0)
    level = int(255 * (1 - 0.65 * strength))
    return QColor(level, level, 255) if z < 0 else QColor(255, level, level)


def label(rating: int) -> str:
    return "Abnormal" if rating == 1 else "Normal"


class ViewerWindow(QMainWindow):
    def __init__(self, cases: pd.DataFrame, participant_id: str, log_dir: Path = LOG_DIR):
        super().__init__()
        self.cases = cases
        self.case_ids = sorted(cases.case_id.unique())
        self.logger = InteractionLogger(participant_id, log_dir)

        # Which cases show the AI: random, but the same every time for this participant.
        rng = random.Random(participant_id)
        ai_cases = set(rng.sample(self.case_ids, k=len(self.case_ids) // 2))
        self.condition = {c: ("ai_on" if c in ai_cases else "ai_off") for c in self.case_ids}

        self.case_index = 0
        self.final_ratings = {}   # region -> final rating, for the current case
        self.initial_rating = None  # set while waiting for the final decision
        self.pending_row = None     # row that still needs its final decision

        self.setWindowTitle("Brain Morphometry Viewer - interaction logging prototype")
        self.resize(1050, 620)
        self._build_ui()
        self.logger.log("session_start")
        self._open_case()

    # ------------------------------------------------------------------ UI
    def _build_ui(self):
        self.case_label = QLabel()
        self.case_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        # Left: table of regions and z-scores
        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Region", "z-score", "Final rating"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_region_selected)

        # Right: the decision panel
        self.region_label = QLabel("Select a region on the left.")
        self.region_label.setWordWrap(True)

        self.step_label = QLabel()
        self.step_label.setStyleSheet("color:#444;")

        self.ai_label = QLabel()
        self.ai_label.setWordWrap(True)
        self.ai_label.hide()

        self.normal_btn = QPushButton("Normal")
        self.abnormal_btn = QPushButton("Abnormal")
        self.normal_btn.clicked.connect(lambda: self._on_rating(0))
        self.abnormal_btn.clicked.connect(lambda: self._on_rating(1))

        self.next_btn = QPushButton("Next case  ->")
        self.next_btn.setEnabled(False)
        self.next_btn.clicked.connect(self._next_case)
        self.progress_label = QLabel()

        buttons = QHBoxLayout()
        buttons.addWidget(self.normal_btn)
        buttons.addWidget(self.abnormal_btn)

        side = QVBoxLayout()
        side.addWidget(QLabel("<b>Selected region</b>"))
        side.addWidget(self.region_label)
        side.addWidget(self.step_label)
        side.addWidget(self.ai_label)
        side.addLayout(buttons)
        side.addStretch()
        side.addWidget(self.progress_label)
        side.addWidget(self.next_btn)

        body = QHBoxLayout()
        body.addWidget(self.table, 3)
        body.addLayout(side, 2)

        root = QVBoxLayout()
        root.addWidget(self.case_label)
        root.addLayout(body)
        container = QWidget()
        container.setLayout(root)
        self.setCentralWidget(container)
        self._set_buttons_enabled(False)

    def _set_buttons_enabled(self, enabled: bool):
        self.normal_btn.setEnabled(enabled)
        self.abnormal_btn.setEnabled(enabled)

    # ------------------------------------------------------------------ cases
    def current_case(self) -> pd.DataFrame:
        case_id = self.case_ids[self.case_index]
        return self.cases[self.cases.case_id == case_id].reset_index(drop=True)

    def _open_case(self):
        case = self.current_case()
        case_id = self.case_ids[self.case_index]
        self.final_ratings = {}
        self.initial_rating = None
        self.case_label.setText(f"{case_id}   ({self.case_index + 1} of {len(self.case_ids)})")

        self.table.blockSignals(True)  # filling the table should not count as clicks
        self.table.setRowCount(len(case))
        for row, rec in case.iterrows():
            z_item = QTableWidgetItem(f"{rec.z_score:+.2f}")
            z_item.setBackground(z_to_color(rec.z_score))
            z_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, QTableWidgetItem(rec.region))
            self.table.setItem(row, 1, z_item)
            self.table.setItem(row, 2, QTableWidgetItem("-"))
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.pending_row = None

        self.region_label.setText("Select a region on the left.")
        self.step_label.setText("")
        self.ai_label.hide()
        self._set_buttons_enabled(False)
        self._update_progress()

        self.logger.start_case_timer()
        self.logger.log("case_open", case_id=case_id, ai_condition=self.condition[case_id])

    def _selected(self):
        """Return (row number, record) of the selected region, or (None, None)."""
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None, None
        row = rows[0].row()
        return row, self.current_case().iloc[row]

    def _common(self, rec) -> dict:
        """Fields that every region-level log row shares."""
        return dict(case_id=rec.case_id, ai_condition=self.condition[rec.case_id],
                    region=rec.region, z_score=rec.z_score)

    # ------------------------------------------------------------------ step 0: select
    def _on_region_selected(self):
        row, rec = self._selected()
        if rec is None:
            return
        if self.pending_row is not None and row != self.pending_row:
            # The current region still needs its final decision: stay on it.
            self.table.blockSignals(True)
            self.table.selectRow(self.pending_row)
            self.table.blockSignals(False)
            return
        self.initial_rating = None
        self.ai_label.hide()
        self.region_label.setText(f"{rec.region}\nz-score: {rec.z_score:+.2f}")
        self.step_label.setText("Step 1: your own decision")
        self._set_buttons_enabled(True)
        self.logger.log("region_select", **self._common(rec))

    # ------------------------------------------------------------------ steps 1 and 2: rate
    def _on_rating(self, rating: int):
        row, rec = self._selected()
        if rec is None:
            return
        ai_on = self.condition[rec.case_id] == "ai_on"

        if self.initial_rating is None:
            # Step 1: the reader's own decision, before any advice.
            self.initial_rating = rating
            self.logger.log("rating_initial", rating=rating,
                            ground_truth=int(rec.ground_truth), **self._common(rec))
            if ai_on:
                self.pending_row = row
                self._show_ai(rec)
                return  # wait for the final decision
            self._finish_region(row, rec, rating)  # ai_off: the first decision is final
        else:
            # Step 2: the final decision, after seeing the AI.
            self._finish_region(row, rec, rating)

    def _show_ai(self, rec):
        verdict = label(int(rec.ai_flag)).upper()
        conf = rec.ai_confidence
        style = ("background:#fff4d6; border:1px solid #e0b84a;" if conf == "high"
                 else "background:#f1f1f1; border:1px dashed #999;")
        self.ai_label.setStyleSheet(style + " padding:8px; border-radius:4px;")
        self.ai_label.setText(
            f"AI suggestion: <b>{verdict}</b><br>AI confidence: <b>{conf}</b><br>"
            f"<small>Your decision: {label(self.initial_rating)}</small>")
        self.ai_label.show()
        self.step_label.setText("Step 2: your final decision (keep or change)")
        self.logger.log("ai_suggestion_shown", ai_flag=int(rec.ai_flag),
                        ai_confidence=conf, **self._common(rec))

    def _finish_region(self, row: int, rec, final: int):
        ai_on = self.condition[rec.case_id] == "ai_on"
        self.logger.log(
            "rating_final", rating=final, ground_truth=int(rec.ground_truth),
            ai_flag=int(rec.ai_flag) if ai_on else None,
            ai_confidence=rec.ai_confidence if ai_on else None,
            **self._common(rec))
        self.final_ratings[rec.region] = final
        self.table.item(row, 2).setText(label(final))
        self.initial_rating = None
        self.pending_row = None
        self._update_progress()
        if row + 1 < self.table.rowCount():
            self.table.selectRow(row + 1)  # go to the next region automatically
        else:
            self._set_buttons_enabled(False)
            self.ai_label.hide()
            self.step_label.setText("")

    def _update_progress(self):
        n = len(self.current_case())
        done = len(self.final_ratings)
        self.progress_label.setText(f"Rated {done} of {n} regions")
        self.next_btn.setEnabled(done == n)

    def _next_case(self):
        case_id = self.case_ids[self.case_index]
        self.logger.log("case_complete", case_id=case_id, ai_condition=self.condition[case_id])
        if self.case_index + 1 < len(self.case_ids):
            self.case_index += 1
            self._open_case()
        else:
            self.logger.log("session_end")
            self.logger.close()
            QMessageBox.information(self, "Done", f"Thank you! Saved to:\n{self.logger.path}")
            self.close()

    def closeEvent(self, event):
        self.logger.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    pid, ok = QInputDialog.getText(None, "Participant", "Pseudonymous participant ID (e.g. P01):")
    if not ok or not pid.strip():
        return
    window = ViewerWindow(pd.read_csv(DATA_PATH), pid.strip())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
