"""Wheel-purity gate — the built ``gadra`` wheel ships the METHOD ONLY.

After the reproduction tooling (``data`` / ``eval`` / ``compat``) moved out of ``src/`` into the repo-only
``examples/`` tree, the wheel must contain exactly the method modules and zero data/eval/compat/jinja.

Two checks, both in the package suite so they gate every PR in the deps-absent ``[dev]``-only CI job:
(1) a fast, build-free assertion on setuptools package discovery; (2) the definitive build of the wheel +
namelist AND metadata inspection (skips cleanly if a wheel build is unavailable in the environment).

The metadata check guards the uv migration's core invariant: heavy reproduction deps (datasets / evaluate /
accelerate / flash-attn / deepspeed) live in the PEP 735 ``[dependency-groups]`` (uv-local, never in wheel
metadata), NOT in ``[project.optional-dependencies]`` (extras, which DO become ``Requires-Dist`` /
``Provides-Extra``). Moving one into an extra by mistake would re-couple the published wheel — this fails it.
"""

import glob
import pathlib
import subprocess
import sys
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

# The 7 method modules that MUST ship.
CORE = [
    "gadra/__init__.py",
    "gadra/_enum_shim.py",
    "gadra/_register.py",
    "gadra/config.py",
    "gadra/gate.py",
    "gadra/layer.py",
    "gadra/model.py",
]
# Reproduction tooling that must NEVER ship in the wheel.
FORBIDDEN_PREFIXES = ("gadra/data/", "gadra/eval/", "gadra/compat/")
# Heavy reproduction deps that must NEVER appear in the wheel's Requires-Dist (they live in the uv
# `gpu` dependency-group, not in extras). Matched case-insensitively against the dist name.
FORBIDDEN_DIST = ("datasets", "evaluate", "accelerate", "flash-attn", "flash_attn", "deepspeed")


def test_setuptools_discovers_method_package_only():
    """Fast, build-free: package discovery under src/ yields exactly the ``gadra`` method package."""
    from setuptools import find_packages

    pkgs = set(find_packages(where=str(REPO / "src")))
    assert pkgs == {"gadra"}, f"expected only the 'gadra' method package under src/, got: {sorted(pkgs)}"


def test_built_wheel_ships_method_only(tmp_path):
    """Definitive: build the wheel and assert it carries the method only (no data/eval/compat/jinja)."""
    proc = subprocess.run(
        [sys.executable, "-m", "pip", "wheel", "--no-deps", "--no-build-isolation", "-w", str(tmp_path), str(REPO)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        pytest.skip(f"wheel build unavailable in this env:\n{proc.stdout}\n{proc.stderr}")

    wheels = glob.glob(str(tmp_path / "gadra-*.whl"))
    assert wheels, f"no gadra wheel built in {tmp_path}"
    names = zipfile.ZipFile(wheels[0]).namelist()

    leaked = [n for n in names if n.startswith(FORBIDDEN_PREFIXES)]
    assert not leaked, f"reproduction code leaked into the wheel: {leaked}"
    assert not [n for n in names if n.endswith(".jinja")], "a .jinja template leaked into the wheel"

    for top in {n.split("/", 1)[0] for n in names}:
        assert top == "gadra" or top.endswith(".dist-info"), f"unexpected top-level wheel entry: {top}"
    for core in CORE:
        assert core in names, f"core method module missing from wheel: {core}"

    # Metadata purity: no heavy reproduction dep may appear as a Requires-Dist (they belong to the uv
    # `gpu` dependency-group, never an extra). Catches a dep accidentally moved into optional-dependencies.
    with zipfile.ZipFile(wheels[0]) as zf:
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata = zf.read(meta_name).decode("utf-8")
    requires = [
        line[len("Requires-Dist:"):].strip().lower()
        for line in metadata.splitlines()
        if line.startswith("Requires-Dist:")
    ]
    leaked_dist = [r for r in requires if any(bad in r for bad in FORBIDDEN_DIST)]
    assert not leaked_dist, f"reproduction dep leaked into wheel Requires-Dist (use a dependency-group): {leaked_dist}"
