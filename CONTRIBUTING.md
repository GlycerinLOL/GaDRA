# Contributing to GaDRA

Thanks for your interest. GaDRA is the official implementation of the GaDRA paper; contributions that fix
bugs, improve docs, or extend model coverage are welcome.

## Development setup

```bash
git clone https://github.com/GlycerinLOL/GaDRA && cd GaDRA
uv sync --group gpu              # reproduction env (cu124 torch + prebuilt flash-attn + deepspeed)
# method-only, no GPU deps:  pip install -e ".[dev]"
```

## Before opening a PR

Run exactly what CI runs and keep it green:

```bash
ruff check src tests examples            # lint
pytest -q -m "not gpu" tests/            # method suite (27 tests; no data/GPU needed)
pytest -q -m "not gpu" examples/tests/   # reproduction tooling (CI-safe subset)
uv lock --check                          # lockfile in sync with pyproject.toml
```

## Invariants (please don't break these)

- **The wheel is method-only.** `pip install gadra` must pull zero data / eval / inference deps, and
  `src/gadra` must never import `examples/`. `tests/test_wheel_purity.py` and `tests/test_independence.py`
  enforce this — keep them passing.
- **Parity is sacred.** The golden baselines (`tests/_golden/`, `examples/tests/_golden/`) encode bit-exact
  parity with the original research implementation; don't regenerate them casually.
- **GaDRA is non-mergeable.** The gate is per-token and input-dependent, so `merge*` raises by design.

## Bugs and security

Open an issue at <https://github.com/GlycerinLOL/GaDRA/issues>. For security reports, see
[`SECURITY.md`](SECURITY.md). Common pitfalls are covered in the [FAQ](docs/FAQ.md).
