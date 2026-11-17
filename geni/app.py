from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QThread, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .engine import LocalStore, data_directory
from .model import ByteTransformer, generate_stream, load_checkpoint, train_model


class TaskWorker(QThread):
    progress = Signal(int, str)
    text = Signal(str)
    result = Signal(object)
    error = Signal(str)

    def __init__(self, task: Callable[["TaskWorker"], object]) -> None:
        super().__init__()
        self.cancel_event = threading.Event()
        self.task = task

    def run(self) -> None:
        try:
            self.result.emit(self.task(self))
        except Exception as exc:
            self.error.emit(str(exc))

    def cancel(self) -> None:
        self.cancel_event.set()


class ChatPanel(QWidget):
    def __init__(self, store: LocalStore, checkpoint: Path, parent: "MainWindow") -> None:
        super().__init__()
        self.store = store
        self.checkpoint = checkpoint
        self.parent_window = parent
        self.active_id = self.store.conversations()[0]["id"]

        self.search = QLineEdit()
        self.search.setPlaceholderText("Sök i konversationer …")
        self.search.textChanged.connect(self.refresh_conversations)
        self.conversation_list = QListWidget()
        self.conversation_list.currentItemChanged.connect(self.select_conversation)
        self.new_button = QPushButton("Ny konversation")
        self.new_button.clicked.connect(self.new_conversation)
        self.rename_button = QPushButton("Byt namn")
        self.rename_button.clicked.connect(self.rename_conversation)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.addWidget(self.search)
        left_layout.addWidget(self.conversation_list)
        left_layout.addWidget(self.new_button)
        left_layout.addWidget(self.rename_button)

        self.history = QPlainTextEdit()
        self.history.setReadOnly(True)
        self.prompt = QLineEdit()
        self.prompt.setPlaceholderText("Skriv ett meddelande …")
        self.prompt.returnPressed.connect(self.send)
        self.send_button = QPushButton("Skicka")
        self.send_button.clicked.connect(self.send)
        send_row = QHBoxLayout()
        send_row.addWidget(self.prompt)
        send_row.addWidget(self.send_button)
        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.addWidget(self.history)
        right_layout.addLayout(send_row)

        splitter = QSplitter()
        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(1, 1)
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Modellen körs lokalt. Träna den på godkänt material i Träningsstudion."))
        layout.addWidget(splitter)
        self.refresh_conversations()
        self.refresh_history()

    def refresh_conversations(self) -> None:
        self.conversation_list.blockSignals(True)
        self.conversation_list.clear()
        for row in self.store.conversations(self.search.text().strip()):
            item = QListWidgetItem(row["title"])
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            self.conversation_list.addItem(item)
            if row["id"] == self.active_id:
                self.conversation_list.setCurrentItem(item)
        self.conversation_list.blockSignals(False)

    def select_conversation(self, current: QListWidgetItem | None, _: QListWidgetItem | None) -> None:
        if current:
            self.active_id = current.data(Qt.ItemDataRole.UserRole)
            self.refresh_history()

    def refresh_history(self) -> None:
        lines = []
        for message in self.store.messages(self.active_id):
            name = "Du" if message["role"] == "user" else "GENI"
            lines.append(f"{name}:\n{message['content']}")
        self.history.setPlainText("\n\n".join(lines))
        self.history.verticalScrollBar().setValue(self.history.verticalScrollBar().maximum())

    def new_conversation(self) -> None:
        self.active_id = self.store.new_conversation()
        self.search.clear()
        self.refresh_conversations()
        self.refresh_history()

    def rename_conversation(self) -> None:
        item = self.conversation_list.currentItem()
        if not item:
            return
        title, accepted = QInputDialog.getText(self, "Byt namn", "Konversationens namn:", text=item.text())
        if accepted and title.strip():
            self.store.rename_conversation(self.active_id, title.strip())
            self.refresh_conversations()

    def send(self) -> None:
        prompt = self.prompt.text().strip()
        if not prompt or self.parent_window.busy:
            return
        self.prompt.clear()
        self.store.add_message(self.active_id, "user", prompt)
        self.refresh_history()
        self.send_button.setEnabled(False)
        self.parent_window.run_task(
            self._generate(prompt),
            self._generated,
            failed=lambda: self.send_button.setEnabled(True),
        )

    def _generate(self, prompt: str) -> Callable[[TaskWorker], object]:
        conversation_id = self.active_id

        def task(worker: TaskWorker) -> str:
            model = load_checkpoint(self.checkpoint) if self.checkpoint.exists() else ByteTransformer()
            if __import__("torch").cuda.is_available():
                model = model.to("cuda")
            previous = self.store.messages(conversation_id)[-6:]
            context = "\n".join(f"{row['role']}: {row['content']}" for row in previous)
            answer = ""
            for answer in generate_stream(
                model, context + "\nassistant: ", cancel=worker.cancel_event
            ):
                worker.text.emit(answer)
            return conversation_id, answer

        return task

    def _generated(self, answer: object) -> None:
        if isinstance(answer, tuple) and len(answer) == 2:
            conversation_id, text = answer
            if text:
                self.store.add_message(conversation_id, "assistant", text)
        elif isinstance(answer, str) and answer:
            self.store.add_message(self.active_id, "assistant", answer)
        self.send_button.setEnabled(True)
        self.refresh_history()


class ApprenticePanel(QWidget):
    def __init__(self, store: LocalStore, checkpoint: Path, parent: "MainWindow") -> None:
        super().__init__()
        self.store = store
        layout = QVBoxLayout(self)
        tabs = QTabWidget()
        tabs.addTab(ChatPanel(store, checkpoint, parent), "Lärlingens chatt")
        review = QWidget()
        review_layout = QVBoxLayout(review)
        review_layout.addWidget(QLabel("Rättelser sparas som väntande exempel och ändrar inte modellvikterna."))
        form = QFormLayout()
        self.prompt = QPlainTextEdit()
        self.answer = QPlainTextEdit()
        self.prompt.setPlaceholderText("Fråga eller uppgift")
        self.answer.setPlaceholderText("Önskat svar eller rättelse")
        form.addRow("Uppgift", self.prompt)
        form.addRow("Godkänt målsvar", self.answer)
        review_layout.addLayout(form)
        self.save_button = QPushButton("Spara som väntande exempel")
        self.save_button.clicked.connect(self.save_example)
        review_layout.addWidget(self.save_button)
        self.examples = QListWidget()
        review_layout.addWidget(self.examples)
        self.approve_button = QPushButton("Godkänn för framtida träning")
        self.approve_button.clicked.connect(self.approve_example)
        review_layout.addWidget(self.approve_button)
        tabs.addTab(review, "Granska exempel")
        layout.addWidget(tabs)
        self.refresh()

    def refresh(self) -> None:
        self.examples.clear()
        for row in self.store.examples():
            status = "Godkänt" if row["approved"] else "Väntar på granskning"
            item = QListWidgetItem(f"{status}: {row['prompt'][:80]}")
            item.setData(Qt.ItemDataRole.UserRole, row["id"])
            item.setData(Qt.ItemDataRole.UserRole + 1, row["approved"])
            self.examples.addItem(item)

    def save_example(self) -> None:
        try:
            self.store.add_example(self.prompt.toPlainText(), self.answer.toPlainText())
        except ValueError as exc:
            QMessageBox.warning(self, "Exemplet kunde inte sparas", str(exc))
            return
        self.prompt.clear()
        self.answer.clear()
        self.refresh()

    def approve_example(self) -> None:
        item = self.examples.currentItem()
        if item:
            self.store.approve_example(item.data(Qt.ItemDataRole.UserRole))
            self.refresh()


class TrainingPanel(QWidget):
    def __init__(self, store: LocalStore, checkpoint: Path, parent: "MainWindow") -> None:
        super().__init__()
        self.store = store
        self.checkpoint = checkpoint
        self.parent_window = parent
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Importerade dokument och godkända lärlingsexempel används; väntande exempel ignoreras."))
        self.document_list = QListWidget()
        layout.addWidget(self.document_list)
        row = QHBoxLayout()
        import_button = QPushButton("Importera svensk textfil")
        import_button.clicked.connect(self.import_document)
        sample_button = QPushButton("Lägg till exempeldata")
        sample_button.clicked.connect(self.import_sample)
        row.addWidget(import_button)
        row.addWidget(sample_button)
        layout.addLayout(row)
        settings = QFormLayout()
        self.steps = QSpinBox()
        self.steps.setRange(1, 10000)
        self.steps.setValue(100)
        self.batch = QSpinBox()
        self.batch.setRange(1, 128)
        self.batch.setValue(8)
        settings.addRow("Träningssteg", self.steps)
        settings.addRow("Batchstorlek", self.batch)
        layout.addLayout(settings)
        self.status = QLabel("Ingen aktiv träning.")
        layout.addWidget(self.status)
        self.progress = QProgressBar()
        layout.addWidget(self.progress)
        controls = QHBoxLayout()
        self.start_button = QPushButton("Starta träning")
        self.start_button.clicked.connect(self.start_training)
        self.stop_button = QPushButton("Stoppa säkert")
        self.stop_button.clicked.connect(parent.cancel_task)
        controls.addWidget(self.start_button)
        controls.addWidget(self.stop_button)
        layout.addLayout(controls)
        self.refresh()

    def refresh(self) -> None:
        self.document_list.clear()
        for document in self.store.documents():
            self.document_list.addItem(document["name"])
        docs = len(self.store.documents())
        approved = sum(bool(row["approved"]) for row in self.store.examples())
        self.status.setText(f"{docs} dokument · {approved} godkända exempel · "
                            f"{'checkpoint sparad' if self.checkpoint.exists() else 'ingen tränad modell'}")

    def import_document(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Importera text", "", "Textfiler (*.txt);;Alla filer (*)")
        if not path:
            return
        try:
            if Path(path).stat().st_size > 20_000_000:
                raise ValueError("Textfilen får vara högst 20 MB i den här versionen.")
            text = Path(path).read_text(encoding="utf-8-sig")
            self.store.add_document(Path(path).name, text)
            self.refresh()
        except (OSError, UnicodeError, ValueError) as exc:
            QMessageBox.warning(self, "Importen misslyckades", str(exc))

    def import_sample(self) -> None:
        sample = Path(__file__).resolve().parent.parent / "data" / "example_swedish.txt"
        try:
            self.store.add_document(sample.name, sample.read_text(encoding="utf-8"))
            self.refresh()
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Exempeldata saknas", str(exc))

    def start_training(self) -> None:
        texts = list(self.store.training_texts())
        if not texts:
            QMessageBox.information(self, "Träningsdata saknas", "Importera en textfil eller godkänn lärlingsexempel först.")
            return
        steps, batch_size = self.steps.value(), self.batch.value()
        self.progress.setValue(0)
        self.parent_window.run_task(
            lambda worker: train_model(
                texts,
                self.checkpoint,
                steps=steps,
                batch_size=batch_size,
                cancel=worker.cancel_event,
                progress=lambda step, loss: worker.progress.emit(
                    int(step * 100 / steps), f"Steg {step}/{steps} · förlust {loss:.3f}"
                ),
            )[1],
            self.training_finished,
        )

    def training_finished(self, result: object) -> None:
        count = len(result) if isinstance(result, list) else 0
        if count:
            self.progress.setValue(100)
        self.status.setText(
            f"{'Träningen avslutad' if count else 'Träningen stoppades'} · {count} steg"
            f"{' · checkpoint sparad' if self.checkpoint.exists() else ''}"
        )
        self.refresh()


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("GENI – lokal AI-arbetsstation")
        self.resize(1100, 720)
        self.store = LocalStore()
        self.checkpoint = data_directory() / "checkpoints" / "geni.pt"
        self.worker: TaskWorker | None = None
        self.worker_finished: Callable[[object], None] | None = None
        self.worker_failed: Callable[[], None] | None = None
        self.tabs = QTabWidget()
        self.chat = ChatPanel(self.store, self.checkpoint, self)
        self.apprentice = ApprenticePanel(self.store, self.checkpoint, self)
        self.training = TrainingPanel(self.store, self.checkpoint, self)
        self.tabs.addTab(self._dashboard(), "Översikt")
        self.tabs.addTab(self.chat, "AI-chatt")
        self.tabs.addTab(self.apprentice, "Lärlingen")
        self.tabs.addTab(self.training, "Träningsstudio")
        self.setCentralWidget(self.tabs)
        self.statusBar().showMessage(f"Lokal lagring: {data_directory()}")

    def _dashboard(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.addWidget(QLabel("GENI · lokal språkmodell från slumpmässiga vikter"))
        layout.addWidget(QLabel(
            "All chatt, träning och lagring sker på den här datorn. Ingen modell är förtränad; "
            "importera text och träna innan du förväntar dig meningsfulla svar."
        ))
        self.model_status = QLabel()
        self.update_status()
        layout.addWidget(self.model_status)
        theme = QComboBox()
        theme.addItems(["Mörkt tema", "Ljust tema"])
        theme.currentIndexChanged.connect(self.set_theme)
        self.set_theme(0)
        layout.addWidget(theme)
        layout.addStretch(1)
        return widget

    def update_status(self) -> None:
        self.model_status.setText(
            f"Modell: {'tränad checkpoint finns' if self.checkpoint.exists() else 'slumpmässigt initierad'}"
            f" · Beräkningsenhet: {'CUDA' if __import__('torch').cuda.is_available() else 'CPU'}"
        )

    def set_theme(self, index: int) -> None:
        if index == 0:
            self.setStyleSheet("QWidget { background: #20242b; color: #e8eaf0; } "
                               "QLineEdit, QPlainTextEdit, QListWidget { background: #15181d; color: #e8eaf0; }")
        else:
            self.setStyleSheet("")

    @property
    def busy(self) -> bool:
        return self.worker is not None

    def run_task(
        self,
        task: Callable[[TaskWorker], object],
        done: Callable[[object], None],
        *,
        failed: Callable[[], None] | None = None,
    ) -> None:
        if self.busy:
            return
        worker = TaskWorker(task)
        self.worker = worker
        self.worker_finished = done
        self.worker_failed = failed
        worker.text.connect(self.chat.history.setPlainText)
        worker.progress.connect(self.on_progress)
        worker.error.connect(self.on_error)
        worker.result.connect(self.on_result)
        worker.finished.connect(self.on_worker_finished)
        worker.start()
        self.statusBar().showMessage("Arbetar lokalt …")

    def on_progress(self, value: int, message: str) -> None:
        self.training.progress.setValue(value)
        self.training.status.setText(message)

    def on_result(self, value: object) -> None:
        if self.worker_finished:
            self.worker_finished(value)

    def on_error(self, message: str) -> None:
        self.statusBar().showMessage("Åtgärden misslyckades.")
        QMessageBox.warning(self, "GENI", message)
        if self.worker_failed:
            self.worker_failed()

    def on_worker_finished(self) -> None:
        self.worker = None
        self.worker_finished = None
        self.worker_failed = None
        self.statusBar().showMessage(f"Lokal lagring: {data_directory()}")
        self.chat.refresh_history()
        self.training.refresh()
        self.apprentice.refresh()
        self.update_status()

    def cancel_task(self) -> None:
        if self.worker:
            self.worker.cancel()
            self.statusBar().showMessage("Avbryter säkert efter aktuellt träningssteg …")

    def closeEvent(self, event: object) -> None:
        if self.worker:
            self.worker.cancel()
            self.worker.wait()
        event.accept()


def main() -> int:
    app = QApplication(sys.argv)
    app.setApplicationName("GENI")
    window = MainWindow()
    window.show()
    return app.exec()
