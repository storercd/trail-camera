"""Windowed dashboard for launching trail camera processes with macOS Dock-tile progress."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass

from PySide6.QtCore import QProcess, QProcessEnvironment, QTimer
from PySide6.QtWidgets import (
    QApplication,
    QFrame,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

PROGRESS_PATTERNS = (
    re.compile(r"\[(\d+)/(\d+)\]"),
    re.compile(r"copied (\d+)/(\d+) files"),
)
ANSI_ESCAPE = re.compile(r"\033\[[0-9;]*m")
MAX_LOG_CHARS = 200_000


@dataclass(frozen=True)
class ProcessDefinition:
    """A command the dashboard can run."""

    id: str
    name: str
    description: str
    command: tuple[str, ...]
    confirm: str = ""


PROCESSES: tuple[ProcessDefinition, ...] = (
    ProcessDefinition(
        "ingest-card",
        "Ingest memory card",
        "Copy media from the mounted memory card into input_dir, then eject.",
        ("uv", "run", "trail-camera-ingest-card"),
    ),
    ProcessDefinition(
        "process-new",
        "Process new media",
        "Run MegaDetector and SpeciesNet on newly discovered media.",
        ("uv", "run", "trail-camera-process", "--mode", "new-only"),
    ),
    ProcessDefinition(
        "reprocess-input",
        "Reprocess input",
        "Reset catalog rows for files in input and re-ingest them as new.",
        ("uv", "run", "trail-camera-process", "--mode", "reprocess-input"),
    ),
    ProcessDefinition(
        "reprocess-existing",
        "Reprocess existing",
        "Reprocess outdated or failed catalog entries from stored originals.",
        ("uv", "run", "trail-camera-process", "--mode", "reprocess-existing"),
    ),
    ProcessDefinition(
        "cleanup-dry-run",
        "Cleanup uninteresting (dry run)",
        "List uninteresting matches without modifying anything.",
        ("uv", "run", "scripts/cleanup_uninteresting.py", "--dry-run"),
    ),
    ProcessDefinition(
        "cleanup",
        "Cleanup uninteresting",
        "Delete files and catalog rows for media matching uninteresting filters.",
        ("uv", "run", "scripts/cleanup_uninteresting.py"),
        confirm="This permanently deletes files. Continue?",
    ),
    ProcessDefinition(
        "report-app",
        "Reporting app",
        "Start the local reporting web app at http://127.0.0.1:8000 (runs until stopped).",
        ("uv", "run", "trail-camera-report"),
    ),
)


def parse_progress(line: str) -> tuple[int, int] | None:
    """Extract ``(done, total)`` from a log line.

    Returns:
        tuple[int, int] | None: The counts, or None if the line has no progress.
    """
    for pattern in PROGRESS_PATTERNS:
        match = pattern.search(line)
        if match:
            done, total = int(match.group(1)), int(match.group(2))
            if total > 0:
                return min(done, total), total
    return None


class DockProgress:
    """Shows progress on the macOS Dock tile (no-op on other platforms)."""

    def __init__(self) -> None:
        """Defer Dock tile setup until the first update, after Qt has initialized NSApp."""
        self._indicator = None
        self._tile = None
        self._ready = sys.platform != "darwin"

    def _setup(self) -> None:
        self._ready = True
        try:
            from AppKit import (  # noqa: PLC0415
                NSApplication,
                NSImageView,
                NSMakeRect,
                NSProgressIndicator,
                NSView,
            )
        except ImportError:
            return
        app = NSApplication.sharedApplication()
        self._tile = app.dockTile()
        container = NSView.alloc().initWithFrame_(NSMakeRect(0, 0, 128, 128))
        icon = NSImageView.alloc().initWithFrame_(NSMakeRect(0, 0, 128, 128))
        icon.setImage_(app.applicationIconImage())
        container.addSubview_(icon)
        indicator = NSProgressIndicator.alloc().initWithFrame_(NSMakeRect(8, 6, 112, 20))
        indicator.setStyle_(0)  # NSProgressIndicatorStyleBar
        indicator.setIndeterminate_(False)
        indicator.setUsesThreadedAnimation_(False)
        indicator.setMinValue_(0.0)
        indicator.setMaxValue_(1.0)
        container.addSubview_(indicator)
        self._tile.setContentView_(container)
        self._indicator = indicator

    def update(self, fraction: float | None, *, busy: bool) -> None:
        """Show a determinate fraction, an indeterminate bar, or hide the bar."""
        if not self._ready:
            self._setup()
        if self._indicator is None:
            return
        if not busy:
            self._indicator.stopAnimation_(None)
            self._indicator.setHidden_(True)
        elif fraction is None:
            self._indicator.setHidden_(False)
            self._indicator.setIndeterminate_(True)
            self._indicator.startAnimation_(None)
        else:
            self._indicator.stopAnimation_(None)
            self._indicator.setHidden_(False)
            self._indicator.setIndeterminate_(False)
            self._indicator.setDoubleValue_(fraction)
        self._tile.display()


class ProcessRow(QFrame):
    """One process: description, Run/Stop button, and status."""

    def __init__(self, definition: ProcessDefinition, window: DashboardWindow) -> None:
        """Build the row widgets."""
        super().__init__()
        self.definition = definition
        self.window_ref = window
        self.setFrameShape(QFrame.Shape.StyledPanel)
        layout = QHBoxLayout(self)
        text = QVBoxLayout()
        title = QLabel(f"<b>{definition.name}</b>")
        description = QLabel(definition.description)
        description.setWordWrap(True)
        text.addWidget(title)
        text.addWidget(description)
        layout.addLayout(text, 1)
        self.status = QLabel("Idle")
        self.status.setMinimumWidth(90)
        layout.addWidget(self.status)
        self.button = QPushButton("Run")
        self.button.setAccessibleName(f"Run {definition.name}")
        self.button.clicked.connect(lambda: window.run_process(self))
        layout.addWidget(self.button)


class DashboardWindow(QMainWindow):
    """Main window with a button per process, a progress bar, and a log."""

    def __init__(self, definitions: tuple[ProcessDefinition, ...] = PROCESSES) -> None:
        """Build the window."""
        super().__init__()
        self.setWindowTitle("Trail Camera Dashboard")
        self.resize(760, 640)
        self.dock = DockProgress()
        self.process: QProcess | None = None
        self.current_row: ProcessRow | None = None
        self.stopped = False
        self.line_buffer = ""

        central = QWidget()
        layout = QVBoxLayout(central)
        self.rows = [ProcessRow(d, self) for d in definitions]
        for row in self.rows:
            layout.addWidget(row)

        self.progress_label = QLabel("No process running")
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setAccessibleName("Process progress")
        self.progress.setTextVisible(True)
        self.progress.setVisible(False)
        layout.addWidget(self.progress_label)
        layout.addWidget(self.progress)

        self.log = QPlainTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumBlockCount(5000)
        self.log.setAccessibleName("Process output")
        layout.addWidget(self.log, 1)
        self.setCentralWidget(central)

    def run_process(self, row: ProcessRow) -> None:
        """Start the row's process, or stop it if it is the running one."""
        if self.process is not None:
            if row is self.current_row:
                self.stopped = True
                self.process.terminate()
                QTimer.singleShot(5000, self._kill_if_running)
            return
        if row.definition.confirm:
            answer = QMessageBox.question(self, row.definition.name, row.definition.confirm)
            if answer != QMessageBox.StandardButton.Yes:
                return
        self.current_row = row
        self.stopped = False
        self.line_buffer = ""
        self.log.clear()
        process = QProcess(self)
        process.setProcessChannelMode(QProcess.ProcessChannelMode.MergedChannels)
        environment = QProcessEnvironment.systemEnvironment()
        environment.insert("PYTHONUNBUFFERED", "1")
        process.setProcessEnvironment(environment)
        process.readyReadStandardOutput.connect(self._read_output)
        process.finished.connect(self._finished)
        process.errorOccurred.connect(self._error)
        self.process = process
        command, *args = row.definition.command
        process.start(command, args)
        self._set_busy(row, started=True)

    def _kill_if_running(self) -> None:
        if self.process is not None:
            self.process.kill()

    def _set_busy(self, row: ProcessRow, *, started: bool) -> None:
        for other in self.rows:
            other.button.setEnabled(other is row or not started)
        row.button.setText("Stop" if started else "Run")
        row.button.setAccessibleName(f"{row.button.text()} {row.definition.name}")
        if started:
            row.status.setText("Running")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
            self.progress_label.setText(f"{row.definition.name}: working…")
            self.dock.update(None, busy=True)

    def _read_output(self) -> None:
        data = bytes(self.process.readAllStandardOutput()).decode("utf-8", errors="replace")
        text = ANSI_ESCAPE.sub("", self.line_buffer + data)
        *lines, self.line_buffer = text.split("\n")
        for line in lines:
            self.log.appendPlainText(line.rstrip("\r"))
            progress = parse_progress(line)
            if progress:
                self._show_progress(*progress)

    def _show_progress(self, done: int, total: int) -> None:
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.progress.setFormat("%v / %m")
        self.progress_label.setText(f"{self.current_row.definition.name}: {done} of {total}")
        self.dock.update(done / total, busy=True)

    def _error(self, error: QProcess.ProcessError) -> None:
        if error == QProcess.ProcessError.FailedToStart:
            self.log.appendPlainText(f"Failed to start: {self.process.errorString()}")
            self._finish("Failed")

    def _finished(self, exit_code: int, _status: QProcess.ExitStatus) -> None:
        self._read_output()
        if self.line_buffer:
            self.log.appendPlainText(self.line_buffer)
        if self.stopped:
            self._finish("Stopped")
        elif exit_code == 0:
            self._finish("Succeeded")
        else:
            self._finish(f"Failed ({exit_code})")

    def _finish(self, status: str) -> None:
        row = self.current_row
        if self.process is None or row is None:
            return
        self.process.deleteLater()
        self.process = None
        self.current_row = None
        row.status.setText(status)
        self._set_busy(row, started=False)
        self.progress.setVisible(False)
        self.progress_label.setText(f"{row.definition.name}: {status}")
        self.dock.update(None, busy=False)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt override
        """Terminate any running process before closing."""
        if self.process is not None:
            self.process.kill()
            self.process.waitForFinished(2000)
        super().closeEvent(event)


def main() -> int:
    """Run the dashboard window.

    Returns:
        int: Exit code.
    """
    app = QApplication(sys.argv)
    app.setApplicationName("Trail Camera Dashboard")
    window = DashboardWindow()
    window.show()
    window.raise_()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
