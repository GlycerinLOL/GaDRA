"""Independence + purity gate — the wheel is the METHOD ONLY, importing nothing from the research repo.

After the reproduction tooling (``data`` / ``eval`` / ``compat``) moved out of ``src/gadra`` into the
repo-only ``examples/`` tree, ``import gadra`` must:

1. succeed and resolve from the ``src``-layout package (not a stray top-level ``gadra/``);
2. NOT make ``gadra.data`` / ``gadra.eval`` / ``gadra.compat`` importable — they are no longer shipped;
3. pull in none of the research-repo modules (its tokenizer/inference/adapter stack);
4. pull in none of the heavy reproduction-only deps (``datasets`` / ``evaluate``) — proving the pure
   method needs zero data/eval dependencies.

Checked in a clean subprocess: pytest shares ``sys.modules`` across the session, so an in-process check
could be polluted. The subprocess gets only the package's own ``src`` on ``PYTHONPATH`` (never the
research-repo root and never the repo root), so the moved/banned modules are not importable — and the
assertions prove ``gadra`` itself stays method-only.
"""

import os
import pathlib
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Bare top-level modules that exist ONLY in the research repo (its tokenizer/inference/adapter stack).
# Distinct from any vendored submodule (peft has ``peft.peft_model``, not bare ``peft_model``;
# transformers has ``transformers.utils``, not bare ``utils``).
BANNED = [
    "inference_common",
    "inference_cc2024",
    "pretraining_inference",
    "inference_mc",
    "forgetting_loss_inference",
    "autoregressive_generation",
    "peft_model",
    "peft_module",
    "peft_config",
    "peft_losses",
    "peft_gamma_generation",
    "Preprocessing",
    "utils",
]

# Submodules that USED to live in the package and now live in the repo-only ``examples/`` tree — they
# must NOT be importable as ``gadra.*`` after the split.
EVICTED = ["gadra.data", "gadra.eval", "gadra.compat"]

# Reproduction-only deps the pure method must never pull in on ``import gadra``.
REPRO_DEPS = ["datasets", "evaluate"]


def test_import_gadra_is_method_only(tmp_path):
    code = (
        "import importlib.util, sys\n"
        "import gadra\n"
        # (1) src-layout: gadra must be imported from src/, never a shadowing top-level gadra/.
        f"src = {str(SRC)!r}\n"
        "assert gadra.__file__ and gadra.__file__.startswith(src), 'gadra not from src-layout: ' + str(gadra.__file__)\n"
        # (2) evicted submodules are no longer importable.
        f"evicted = {EVICTED!r}\n"
        "present = sorted(m for m in evicted if importlib.util.find_spec(m) is not None)\n"
        # (3) no research-repo module leaked.
        f"banned = {BANNED!r}\n"
        "leaked = sorted(m for m in banned if m in sys.modules)\n"
        # (4) no heavy reproduction dep pulled in.
        f"repro = {REPRO_DEPS!r}\n"
        "heavy = sorted(m for m in repro if m in sys.modules)\n"
        "print('PRESENT:' + ','.join(present))\n"
        "print('LEAKED:' + ','.join(leaked))\n"
        "print('HEAVY:' + ','.join(heavy))\n"
        "sys.exit(1 if (present or leaked or heavy) else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    # Run from a neutral cwd so a stray local module can't shadow the installed package.
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), env=env
    )
    assert proc.returncode == 0, (
        "`import gadra` is not method-only (evicted submodule importable, research module leaked, "
        "or a heavy data/eval dep was pulled in):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
