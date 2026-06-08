"""Wheel-purity gate: the built ``gadra`` wheel ships the method only, with no reproduction tooling or heavy deps."""

import glob
import pathlib
import subprocess
import sys
import zipfile

import pytest

REPO = pathlib.Path(__file__).resolve().parent.parent

CORE = [
    "gadra/__init__.py",
    "gadra/_enum_shim.py",
    "gadra/_register.py",
    "gadra/config.py",
    "gadra/gate.py",
    "gadra/layer.py",
    "gadra/model.py",
]
# Reproduction tooling that must never ship in the wheel.
FORBIDDEN_PREFIXES = ("gadra/data/", "gadra/eval/", "gadra/compat/")
# Heavy reproduction deps that must never appear in Requires-Dist (belong in uv dependency-groups).
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
