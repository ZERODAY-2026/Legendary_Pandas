"""
Local Gemma runtime engine.
Strictly encapsulates local inference (e.g. Gemma 3n E2B / E4B or edge quantized GGUF via llama-cpp-python or PyTorch).
Hard constraint: No cloud LLM calls, hosted APIs, or remote fallbacks under any condition.
"""

from typing import Any
from backend.app.config import settings


class GemmaRuntimeError(Exception):
    """Raised when local model inference fails."""

    pass


class GemmaRuntime:
    """Manages local Gemma model lifecycle, memory allocation, and generation."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path or str(settings.GEMMA_MODEL_PATH)
        self.model_name = settings.GEMMA_MODEL_NAME
        self.is_loaded = False
        self._model = None

    def load_model(self) -> None:
        """Load local Gemma weights into memory.

        Document exact model, quantization, runtime, and resource usage.
        """
        # Full loading and benchmarking implemented in Phase 5
        self.is_loaded = True

    def generate(self, prompt: str, max_tokens: int = 1024) -> str:
        """Run local inference on prompt.

        Prompt injection defense: prompt contains untrusted evidence block.
        """
        if not self.is_loaded:
            self.load_model()

        # Skeleton fallback for Phase 1
        return ""

    def get_resource_usage(self) -> dict[str, Any]:
        """Return memory footprint and runtime statistics."""
        return {
            "model_name": self.model_name,
            "quantization": "Q4_K_M (planned)",
            "context_window": settings.GEMMA_CONTEXT_WINDOW,
            "device": settings.DEVICE,
            "is_loaded": self.is_loaded,
        }
