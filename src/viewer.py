"""Minimal PyQt5 brain-morphometry viewer with interaction logging.

What the participant does:
  1. Enters a pseudonymous participant ID.
  2. For each case, looks at the regional z-scores and rates every region as
     "Normal" or "Abnormal".
  3. For half of the cases (chosen at random per participant) an AI suggestion is
     shown next to the selected region ("ai_on"); for the other half it is hidden
     ("ai_off"). This within-subject design lets us compare decisions with and
     without AI support.

Everything the participant does is written to logs/ by InteractionLogger.

Run:  python src/viewer.py
"""

import random
import sys
from pathlib import Path

import pandas as pd
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QApplication,
    QHBoxLayout,
    QHeaderView,
    QInputDialog,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from logger import InteractionLogger

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "sample_cases.csv"
LOG_DIR = ROOT / "logs"


def z_to_color(z: float) -> QColor:
    """Blue for smaller-than-normal, red for larger-than-normal, white near 0."""
    strength = min(abs(z) / 3.0, 1.0)
    level = int(255 * (1 - 0.65 * strength))
    return QColor(level, level, 255) if z < 0 else QColor(255, level, level)


class ViewerWindow(QMainWindow):
    def __init__(self, cases: pd.DataFrame, participant_id: str, log_dir: Path = LOG_DIR):
        super().__init__()
        self.cases = cases
        self.case_ids = sorted(cases.case_id.unique())
        self.logger = InteractionLogger(participant_id, log_dir)

        # Randomly (but reproducibly per participant) choose which cases show the AI.
        rng = random.Random(participant_id)
        ai_cases = set(rng.sample(self.case_ids, k=len(self.case_ids) // 2))
        self.condition = {c: ("ai_on" if c in ai_cases else "ai_off") for c in self.case_ids}

        self.case_index = 0
        self.ratings = {}  # region -> rating for the current case

        self.setWindowTitle("Brain Morphometry Viewer - interaction logging prototype")
        self.resize(1000, 600)
        self._build_ui()
        self.logger.log("session_start")
        self._open_case()

    # ---------- UI ----------
    def _build_ui(self):
        self.case_label = QLabel()
        self.case_label.setStyleSheet("font-size: 16px; font-weight: bold;")

        self.table = QTableWidget(0, 3)
        self.table.setHorizontalHeaderLabels(["Region", "z-score", "Your rating"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setSelectionMode(QTableWidget.SingleSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.itemSelectionChanged.connect(self._on_region_selected)

        self.region_label = QLabel("Select a region on the left.")
        self.region_label.setWordWrap(True)
        self.ai_label = QLabel()
        self.ai_label.setWordWrap(True)
        self.ai_label.setStyleSheet(
            "background:#fff4d6; border:1px solid #e0b84a; padding:8px; border-radius:4px;"
        )
        self.ai_label.hide()

        self.normal_btn = QPushButton("Normal")
        self.abnormal_btn = QPushButton("Abnormal")
        self.normal_btn.clicked.connect(lambda: self._rate(0))
        self.abnormal_btn.clicked.connect(lambda: self._rate(1))
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

    # ---------- case handling ----------
    def current_case(self) -> pd.DataFrame:
        return self.cases[self.cases.case_id == self.case_ids[self.case_index]].reset_index(drop=True)

    def _open_case(self):
        case = self.current_case()
        case_id = self.case_ids[self.case_index]
        self.ratings = {}
        self.case_label.setText(
            f"{case_id}   ({self.case_index + 1} of {len(self.case_ids)})"
        )
        self.table.blockSignals(True)
        self.table.setRowCount(len(case))
        for row, rec in case.iterrows():
            region_item = QTableWidgetItem(rec.region)
            z_item = QTableWidgetItem(f"{rec.z_score:+.2f}")
            z_item.setBackground(z_to_color(rec.z_score))
            z_item.setTextAlignment(Qt.AlignCenter)
            self.table.setItem(row, 0, region_item)
            self.table.setItem(row, 1, z_item)
            self.table.setItem(row, 2, QTableWidgetItem("-"))
        self.table.clearSelection()
        self.table.blockSignals(False)
        self.region_label.setText("Select a region on the left.")
        self.ai_label.hide()
        self.next_btn.setEnabled(False)
        self._update_progress()
        self.logger.start_case_timer()
        self.logger.log("case_open", case_id=case_id, ai_condition=self.condition[case_id])

    def _selected_record(self):
        rows = self.table.selectionModel().selectedRows()
        if not rows:
            return None
        return self.current_case().iloc[rows[0].row()]

    def _on_region_selected(self):
        rec = self._selected_record()
        if rec is None:
            return
        case_id = rec.case_id
        cond = self.condition[case_id]
        self.region_label.setText(f"{rec.region}\nz-score: {rec.z_score:+.2f}")
        common = dict(case_id=case_id, ai_condition=cond, region=rec.region, z_score=rec.z_score)
        self.logger.log("region_select", **common)
        if cond == "ai_on":
            verdict = "ABNORMAL" if rec.ai_flag == 1 else "normal"
            self.ai_label.setText(f"AI suggestion: this region looks <b>{verdict}</b>.")
            self.ai_label.show()
            self.logger.log("ai_suggestion_shown", ai_flag=int(rec.ai_flag), **common)
        else:
            self.ai_label.hide()

    def _rate(self, rating: int):
        rec = self._selected_record()
        if rec is None:
            QMessageBox.information(self, "No region", "Please select a region first.")
            return
        cond = self.condition[rec.case_id]
        self.ratings[rec.region] = rating
        row = self.table.selectionModel().selectedRows()[0].row()
        self.table.item(row, 2).setText("Abnormal" if rating else "Normal")
        self.logger.log(
            "rating",
            case_id=rec.case_id,
            ai_condition=cond,
            region=rec.region,
            z_score=rec.z_score,
            ai_flag=int(rec.ai_flag) if cond == "ai_on" else None,
            rating=rating,
            ground_truth=int(rec.ground_truth),
        )
        # Move to the next region automatically to speed things up.
        if row + 1 < self.table.rowCount():
            self.table.selectRow(row + 1)
        self._update_progress()

    def _update_progress(self):
        n = len(self.current_case())
        done = len(self.ratings)
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
            QMessageBox.information(
                self, "Done", f"Thank you! Your session was saved to:\n{self.logger.path}"
            )
            self.close()

    def closeEvent(self, event):
        self.logger.close()
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    pid, ok = QInputDialog.getText(None, "Participant", "Pseudonymous participant ID (e.g. P01):")
    if not ok or not pid.strip():
        return
    cases = pd.read_csv(DATA_PATH)
    window = ViewerWindow(cases, pid.strip())
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
