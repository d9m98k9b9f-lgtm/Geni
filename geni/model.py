from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Callable, Iterator

import torch
from torch import nn


class ByteTransformer(nn.Module):
    """Small causal byte-level transformer initialized from random weights."""

    def __init__(
        self,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        context: int = 128,
    ) -> None:
        super().__init__()
        if width < 8 or heads < 1 or width % heads or layers < 1 or context < 8:
            raise ValueError("Ogiltiga inställningar för modellens storlek.")
        self.config = {"width": width, "heads": heads, "layers": layers, "context": context}
        self.token_embedding = nn.Embedding(256, width)
        self.position_embedding = nn.Embedding(context, width)
        layer = nn.TransformerEncoderLayer(
            d_model=width,
            nhead=heads,
            dim_feedforward=width * 4,
            batch_first=True,
            norm_first=True,
            activation="gelu",
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.output = nn.Linear(width, 256)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        _, length = tokens.shape
        if length > self.config["context"]:
            raise ValueError("Sekvensen är längre än modellens kontextfönster.")
        positions = torch.arange(length, device=tokens.device)
        hidden = self.token_embedding(tokens) + self.position_embedding(positions)
        causal_mask = torch.triu(
            torch.ones(length, length, dtype=torch.bool, device=tokens.device), diagonal=1
        )
        return self.output(self.norm(self.blocks(hidden, mask=causal_mask)))


def train_model(
    texts: list[str],
    checkpoint: Path,
    *,
    steps: int = 100,
    batch_size: int = 8,
    learning_rate: float = 0.001,
    context: int = 64,
    cancel: threading.Event | None = None,
    progress: Callable[[int, float], None] | None = None,
) -> tuple[ByteTransformer, list[float]]:
    data = b"\n".join(text.encode("utf-8") for text in texts)
    if len(data) < context + 1:
        raise ValueError(f"Träningsmaterialet måste innehålla minst {context + 1} byte.")
    if not 1 <= steps <= 100_000 or not 1 <= batch_size <= 256:
        raise ValueError("Träningsinställningarna ligger utanför tillåtna gränser.")
    if not 0 < learning_rate <= 0.1:
        raise ValueError("Inlärningshastigheten måste vara mellan 0 och 0,1.")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ByteTransformer(context=context).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    tokens = torch.tensor(list(data), dtype=torch.long, device=device)
    history: list[float] = []
    model.train()
    for step in range(steps):
        if cancel and cancel.is_set():
            break
        starts = torch.randint(0, len(tokens) - context, (batch_size, 1), device=device)
        offsets = torch.arange(context, device=device)
        x = tokens[starts + offsets]
        y = tokens[starts + offsets + 1]
        optimizer.zero_grad(set_to_none=True)
        loss = nn.functional.cross_entropy(model(x).reshape(-1, 256), y.reshape(-1))
        if not torch.isfinite(loss):
            raise RuntimeError("Träningen avbröts eftersom förlustvärdet blev instabilt.")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        value = float(loss.detach().cpu())
        history.append(value)
        if progress:
            progress(step + 1, value)
    if history:
        save_checkpoint(model, checkpoint)
    return model, history


def save_checkpoint(model: ByteTransformer, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(model.state_dict(), temporary)
    temporary.replace(path)
    metadata = path.with_suffix(".json")
    metadata_tmp = metadata.with_suffix(".json.tmp")
    metadata_tmp.write_text(json.dumps(model.config), encoding="utf-8")
    metadata_tmp.replace(metadata)


def load_checkpoint(path: Path) -> ByteTransformer:
    config = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    model = ByteTransformer(**config)
    state = torch.load(path, map_location="cpu", weights_only=True)
    model.load_state_dict(state, strict=True)
    model.eval()
    return model


@torch.inference_mode()
def generate_stream(
    model: ByteTransformer,
    prompt: str,
    *,
    max_new_tokens: int = 300,
    temperature: float = 0.8,
    cancel: threading.Event | None = None,
) -> Iterator[str]:
    if not 0.1 <= temperature <= 2.0 or not 1 <= max_new_tokens <= 2_000:
        raise ValueError("Genereringsinställningarna ligger utanför tillåtna gränser.")
    device = next(model.parameters()).device
    tokens = list(prompt.encode("utf-8"))
    generated = bytearray()
    model.eval()
    for _ in range(max_new_tokens):
        if cancel and cancel.is_set():
            break
        context = tokens[-model.config["context"] :]
        inputs = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(inputs)[0, -1] / temperature
        next_token = int(torch.multinomial(torch.softmax(logits, dim=-1), 1).item())
        tokens.append(next_token)
        generated.append(next_token)
        yield generated.decode("utf-8", errors="replace")
