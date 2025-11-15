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
    Ensure we can import your custom models located at:
        <parent_dir>/repo/custom_models/...

    We try both import paths:
      1)  import custom_models               (requires sys.path contains /.../repo)
      2)  import repo.custom_models     (requires sys.path contains /.../)

    Precedence of search roots:
      - FLAME_PATH env var (should point to /abs/path/to/repo)
      - sibling repo guess: <repo_root>/../repo
      - fallback guesses if you move things around

    On success we also alias to 'custom_models' in sys.modules so that
    `import custom_models.xxx` works uniformly.
    """
    base_dir        = os.path.dirname(os.path.abspath(__file__))      # .../flash-linear-attention/evals
    repo_root       = os.path.abspath(os.path.join(base_dir, ".."))   # .../flash-linear-attention
    parent_of_repo  = os.path.abspath(os.path.join(repo_root, ".."))  # parent that contains both repos

    candidates = []
    # 1) explicit env var
    if os.environ.get("FLAME_PATH"):
        candidates.append(os.environ["FLAME_PATH"])  # expected to be .../repo

    # 2) sibling repo guess: flash-linear-attention and repo are siblings
    candidates.append(os.path.join(parent_of_repo, "repo"))

    # 3) repo-internal guess (if someday you vendor it inside)
    candidates.append(os.path.join(repo_root, "repo"))

    # Add candidates to sys.path (if they exist)
    for p in candidates:
        if os.path.isdir(p) and p not in sys.path:
            sys.path.insert(0, p)

    cm_module = None
    errs = []

    # Try plain 'custom_models' (works if sys.path contains /.../repo)
    try:
        cm_module = importlib.import_module("custom_models")
    except Exception as e:
        errs.append(f"import custom_models -> {e!r}")

    # Fallback: 'repo.custom_models' (works if sys.path contains the parent)
    if cm_module is None:
        try:
            cm_module = importlib.import_module("repo.custom_models")
            # alias so that 'import custom_models' also works
            sys.modules.setdefault("custom_models", cm_module)
        except Exception as e:
            errs.append(f"import repo.custom_models -> {e!r}")

    if cm_module is None:
        print(f"[harness] warn: cannot load custom_models: {' | '.join(errs)}", flush=True)
        return

    # Optional explicit submodules you asked for (ignore if missing)
    for mod in ("custom_models.sba", "custom_models.lact_model",
                "repo.custom_models.sba", "repo.custom_models.lact_model"):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError:
            pass
        except Exception as e:
            print(f"[harness] warn: failed to import {mod}: {e!r}", flush=True)

    # Auto-import all first-level subpackages under custom_models
    try:
        pkg_path = os.path.dirname(cm_module.__file__)
        for _, name, _ in pkgutil.iter_modules([pkg_path]):
            full = f"{cm_module.__name__}.{name}"
            try:
                importlib.import_module(full)
            except Exception as e:
                print(f"[harness] warn: failed to import {full}: {e!r}", flush=True)
        print(f"[harness] custom_models loaded from: {pkg_path}", flush=True)
    except Exception as e:
        print(f"[harness] warn: iter modules error: {e!r}", flush=True)

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
