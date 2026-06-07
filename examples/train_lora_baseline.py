"""LoRA baseline entry — DEPRECATED shim. The LoRA baseline is now ``method: lora`` in the run-config.

The training entry is unified: ``examples/train.py`` selects the adapter from the run-config's
``method:`` field (gadra | gadra-mono | gadra-soft | lora), dispatched by ``examples/methods.py``.
This file is kept only so existing invocations and SLURM file-path wrappers keep working::

    python -m examples.train_lora_baseline --config examples/config/train_lora.yaml

It forces ``--override method=lora`` (unless the argv already sets ``method``) and delegates to the
single entry, so there is exactly ONE trainer code path and the LoRA baseline can never drift from
the GaDRA recipe again. Prefer the unified form directly::

    python -m examples.train --config examples/config/train_lora.yaml
    python -m examples.train --config examples/config/train.yaml --override method=lora

Will be removed in a future release.
"""

from __future__ import annotations

import pathlib
import sys
import warnings

# Repo-only tooling: support both ``python -m examples.train_lora_baseline`` and the file-path form.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))


def main() -> None:
    warnings.warn(
        "examples.train_lora_baseline is deprecated; use "
        "`examples.train --config <cfg> --override method=lora` (or set `method: lora` in the config).",
        DeprecationWarning,
        stacklevel=2,
    )
    from examples.train import main as _train_main

    # Force method=lora unless the argv already pins method (so train_lora.yaml works whether or not
    # it carries the `method:` line). Appended last, so it wins over any earlier --override method=X.
    already_sets_method = any(
        arg == "method" or arg.startswith("method=")
        for i, tok in enumerate(sys.argv)
        for arg in ([sys.argv[i + 1]] if tok == "--override" and i + 1 < len(sys.argv) else [])
    )
    if not already_sets_method:
        sys.argv += ["--override", "method=lora"]
    _train_main()


if __name__ == "__main__":
    main()
