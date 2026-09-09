from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np

@dataclass
class ModelInfo:
    name: str
    version: str
    device: str
    target_gsd_m: float

class SREngine:
    """Adapter boundary for the official ESAOpenSR/SEN2SR implementation.

    Antigravity should replace the placeholder methods after verifying the current
    upstream API rather than guessing imports or model-loading calls.
    """

    def __init__(self, device: str = "cuda") -> None:
        self.device = device
        self.model: Any = None

    def load(self) -> None:
        raise NotImplementedError("Wire this adapter to the currently documented SEN2SRLite API.")

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("SR model is not loaded")
        raise NotImplementedError

    def model_info(self) -> ModelInfo:
        return ModelInfo(
            name="SEN2SRLite",
            version="verified-at-install-time",
            device=self.device,
            target_gsd_m=2.5,
        )
