"""Independence gate: ``import gadra`` is method-only, with no research-repo modules or heavy deps."""

import os
import pathlib
import subprocess
import sys

SRC = pathlib.Path(__file__).resolve().parent.parent / "src"

# Bare top-level modules that exist only in the research repo.
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

# Submodules moved to examples/ — must not be importable as gadra.*.
EVICTED = ["gadra.data", "gadra.eval", "gadra.compat"]

# Reproduction-only deps that must not be pulled in by import gadra.
REPRO_DEPS = ["datasets", "evaluate"]


def test_import_gadra_is_method_only(tmp_path):
    code = (
        "import importlib.util, sys\n"
        "import gadra\n"
        # (1) src-layout: gadra must be imported from src/, never a shadowing top-level gadra/.
        f"src = {str(SRC)!r}\n"
        "assert gadra.__file__ and gadra.__file__.startswith(src), 'gadra not from src-layout: ' + str(gadra.__file__)\n"
        f"evicted = {EVICTED!r}\n"
        "present = sorted(m for m in evicted if importlib.util.find_spec(m) is not None)\n"
        f"banned = {BANNED!r}\n"
        "leaked = sorted(m for m in banned if m in sys.modules)\n"
        f"repro = {REPRO_DEPS!r}\n"
        "heavy = sorted(m for m in repro if m in sys.modules)\n"
        "print('PRESENT:' + ','.join(present))\n"
        "print('LEAKED:' + ','.join(leaked))\n"
        "print('HEAVY:' + ','.join(heavy))\n"
        "sys.exit(1 if (present or leaked or heavy) else 0)\n"
    )
    env = dict(os.environ)
    env["PYTHONPATH"] = str(SRC) + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, cwd=str(tmp_path), env=env
    )
    assert proc.returncode == 0, (
        "`import gadra` is not method-only (evicted submodule importable, research module leaked, "
        "or a heavy data/eval dep was pulled in):\n"
        f"stdout: {proc.stdout}\nstderr: {proc.stderr}"
    )
