# GENI

GENI is an early, local-first Windows desktop application for experimenting with a small Swedish language model. It includes a PySide6 UI, local SQLite conversation and corpus storage, and a byte-level causal Transformer implemented in PyTorch. The model starts with random weights; no pretrained model or external AI service is used.

## Run from source

Requires 64-bit Python 3.10+ and an internet connection only to install the Python packages:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m geni
```

In the app, import a UTF-8 text file or the included sample corpus in **Träningsstudio**, then start a short training run. In **Lärlingen**, enter a prompt and a corrected target answer, save it for review, and explicitly approve it before it can be used in training. Approval never changes model weights by itself. Chats, documents, reviewed examples, and checkpoints are stored under `%LOCALAPPDATA%\GENI`.

The app can use CUDA when PyTorch detects a supported device; otherwise it runs on CPU. Keep the first training runs small, particularly on CPU. Training and generation run on a background worker so the UI stays responsive. Stopping is cooperative and occurs between training steps.

## Windows portable build

On Windows, run `.\build_windows.ps1` from the repository root. This installs the pinned dependencies and creates a portable application folder in `dist\GENI`. It does **not** create an installer. A Windows build has not been verified outside Windows; do not treat a successful build on another operating system as Windows verification.

Run the automated tests with:

```powershell
python -m unittest discover -s tests -v
```

## Current scope and limitations

This is a small working foundation, not the complete workstation described in the original specification. The transformer uses UTF-8 bytes and a fixed context window; meaningful Swedish generation requires substantially more curated training text than the included sample. Training currently uses a simple next-byte objective and does not provide validation splits, pause/resume, growth operations, forgetting/stability evaluation, architecture visualization, model comparisons, or an installer. CPU/GPU resource estimates and live hardware telemetry are also not implemented. The project makes no claim of human-like understanding or consciousness.

## Project layout

```text
geni/       Desktop UI, local storage, and model/training implementation
tests/      Storage, training, checkpoint, and generation tests
data/       Small, original Swedish example corpus
```
