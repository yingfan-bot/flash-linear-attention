from __future__ import annotations

"""
Harness entry for lm-eval.
- Robustly imports your sibling repo `repo/custom_models/...`
- Optionally imports specific submodules (sba, lact_model)
- Registers a lightweight --model fla wrapper (HuggingFace HFLM-based)
Run example:
  accelerate launch -m evals.harness --tasks ... --model fla --model_args pretrained=...,dtype=bfloat16,max_length=32768,trust_remote_code=True
"""

# -------------------------- custom models bootstrap ---------------------------
import os
import sys
import importlib
import pkgutil

def _maybe_import_custom_models() -> None:
    """
    Import custom_models from the local repository.
    """
    base_dir        = os.path.dirname(os.path.abspath(__file__))      # .../flash-linear-attention/evals
    repo_root       = os.path.abspath(os.path.join(base_dir, ".."))   # .../flash-linear-attention
    custom_models_dir = os.path.join(repo_root, "custom_models")

    # Add custom_models to sys.path so we can import lact_model and sba directly
    if custom_models_dir not in sys.path:
        sys.path.insert(0, custom_models_dir)

    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)

    # Explicit imports for debugging
    try:
        import lact_model
        import sba
        print(f"[harness] custom_models (lact_model, sba) loaded explicitly", flush=True)
    except ImportError as e:
        print(f"[harness] warn: failed to import custom models: {e!r}", flush=True)

try:
    _maybe_import_custom_models()
except Exception as e:
    print(f"[harness] warn: _maybe_import_custom_models failed: {e!r}", flush=True)
# -----------------------------------------------------------------------------


# Ensure FLA kernels are registered
import fla  # noqa: F401

from lm_eval.__main__ import cli_evaluate
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM


@register_model("fla")
class FlashLinearAttentionLMWrapper(HFLM):
    """
    Minimal wrapper so we can use --model fla in lm_eval.
    Heavy lifting is done by HFLM; kernels come from `fla`.
    Any custom architectures are imported above via custom_models.
    """
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)


if __name__ == "__main__":
    cli_evaluate()
