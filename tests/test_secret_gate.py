"""Secret gate — no API-key-shaped literal may live in the package tree.

The research repo carries live OpenAI keys (``run_inference.slurm``, ``evaluate_model_output_args.py``);
this gate ensures none are ever copied into the package during a port. It matches realistic key shapes
(``sk-...`` / ``sk-proj-...`` followed by a long run of key chars), NOT the bare ``sk-`` / ``sk-proj-``
prefixes that legitimately appear in prose (e.g. docs describing this very gate).
"""

import pathlib
import re

ROOT = pathlib.Path(__file__).resolve().parent.parent  # GaDRA/
SCAN_DIRS = ["src", "tests", "examples", "docs", ".github"]
SCAN_FILES = ["pyproject.toml", "README.md"]
TEXT_SUFFIXES = {".py", ".md", ".toml", ".jinja", ".txt", ".cfg", ".yaml", ".yml", ".json", ".sh"}

# OpenAI-style secret: sk- or sk-proj- followed by a long key body. The {30,} run excludes the bare
# "sk-proj-" mentioned in documentation.
KEY_RE = re.compile(r"sk-(?:proj-)?[A-Za-z0-9_-]{30,}")
# Generic high-entropy assignments to obvious secret names (defence in depth).
ASSIGN_RE = re.compile(r"""(?i)(api[_-]?key|secret|token|password)\s*[:=]\s*['"][^'"\s]{24,}['"]""")


def _iter_files():
    for d in SCAN_DIRS:
        base = ROOT / d
        if not base.is_dir():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in TEXT_SUFFIXES:
                yield p
    for f in SCAN_FILES:
        p = ROOT / f
        if p.is_file():
            yield p


def test_no_api_key_literal_in_package():
    offenders = []
    for p in _iter_files():
        try:
            text = p.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for m in KEY_RE.finditer(text):
            offenders.append(f"{p.relative_to(ROOT)}: key-shaped literal {m.group(0)[:10]}…")
        for m in ASSIGN_RE.finditer(text):
            offenders.append(f"{p.relative_to(ROOT)}: secret-looking assignment {m.group(0)[:24]}…")
    assert not offenders, "secret-shaped literal(s) found in the package tree:\n" + "\n".join(offenders)
