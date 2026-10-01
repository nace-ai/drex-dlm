"""Load a merged Drex DLM checkpoint.

The backbone shards live next to ``head.pt``. There is no adapter to merge.
"""
from pathlib import Path

import torch

from .model import DecisionModel, load_tokenizer


def read_meta(path):
    path = Path(path)
    return torch.load(path / "head.pt", map_location="cpu", weights_only=False)


def load(path, device=None, dtype=None):
    """Return ``(tokenizer, model)`` in eval mode.

    ``device`` defaults to CUDA, then Apple MPS, then CPU. ``dtype`` defaults
    to bfloat16 on CUDA and MPS, and float32 on CPU.
    """
    path = Path(path)
    meta = read_meta(path)
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    if dtype is None:
        dtype = torch.bfloat16 if device in ("cuda", "mps") else torch.float32
    tok = load_tokenizer(str(path))
    model = DecisionModel(
        str(path),
        tok,
        device,
        lora=None,
        head_dim=int(meta.get("head_dim", 256)),
        option_isolation=bool(meta.get("option_isolation", False)),
        dtype=dtype,
        state_bidir=bool(meta.get("state_bidir", True)),
    )
    model.head.load_state_dict(meta["head"])
    model.head.temperature = float(meta.get("temperature", 1.0))
    model.eval()
    return tok, model
