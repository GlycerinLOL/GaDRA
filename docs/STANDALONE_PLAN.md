# GaDRA Standalone Package — Architecture & Execution Plan (P6.x)

> Status: **planning complete, decisions locked 2026-06-03** (not yet authorized to start coding —
> package frozen until job 225224 lands + P5 sign-off).
> Authored from a 13-agent design workflow (Standards-First winner) + 3 adversarial critiques, then
> re-verified against the live repo. One workflow claim (PREREQ-A) was **debunked** (see §0).

---

## 0. Verification corrections & locked decisions

### 0.1 Corrections after re-checking the live repo
| Workflow claim | Verdict | Consequence |
|---|---|---|
| **PREREQ-A:** "peft 0.18.1; `set_adapter` crashes `get_peft_model`" | **FALSE.** `condaenv py312` = **peft 0.16.0 / transformers 4.53.3 / torch 2.6.0+cu124 / py3.12.11** — exactly the pin. peft 0.16.0 calls `self.set_adapter(self.active_adapters)` (single positional). Live proof: job 225224 trains through `get_peft_model`+`save_pretrained` on this env. | **No blocking bug.** `set_adapter` widening → *optional forward-compat* only, gates nothing. |
| "5 tests fail on this interpreter" | Tied to the false 0.18.1 premise; suite was 27-green at P4 on 0.16.0. | Re-confirm green once (batched); not a blocker. |
| **PREREQ-B:** job 225224 RUNNING | **TRUE.** | Freeze package until it lands + P5 sign-off. |
| `README.md:104`/`DESIGN.md` lock diagnostics/data/eval OUT of scope | **TRUE.** | See §0.2 — user reverses **data + eval only**, keeps **diagnostics out**. |
| Live `sk-proj-` literal in `evaluate_model_output_args.py` | **TRUE** (one file in this tree). | Flag-only + package grep gate; user rotates. |
| "bit-exact at 8B" | Observed `max|Δ|=0`, top-1 100%, NLL Δ=0 — bit-exact **under eager TF** (verify uses `attn_implementation="eager"`, single sequence). | FA2-packing path validated separately (G5/HITL). |

### 0.2 Locked decisions (user, 2026-06-03)
1. **Scope:** approve **`gadra.data` + `gadra.eval`** as first-class. **NO `gadra.diagnostics`** — the
   per-token γ/CR extraction tooling **stays in the research repo, remains out of scope** for the OSS
   package. (The README/DESIGN scope-lock is reversed for data+eval only; diagnostics stays locked.)
2. **Eval:** **`paper_repro` first** (verbatim paper scorers + generation runner reproduce the
   canonical GaDRA numbers). lm-eval adapter is **kept as a future extension stub**, not the default.
3. **Training-script independence:** **deferred to P7** (rewriting `train_gadra_cpt.py` onto
   `gadra.data` + a fresh HITL CPT re-validation). Library work (P6.x) does not wait on it.
4. **Security:** **flag-only** — note the key, add the package grep gate; user rotates + scrubs the
   account/history separately. No edits to the parent file from this plan.

**Net effect of dropping diagnostics:** the riskiest ports vanish (dtype-faithful CR/R², the
651-line `_sample` generation-capture fork, the gate-logits channel) and **`layer.py` is left
untouched** (no D0 stash, no RNG-neutrality proof needed). Parity surface shrinks to the data path +
the paper_repro scorers.

### 0.3 Package/repo split (P6.6, 2026-06-04 — supersedes §2 in-package placement)

**Decision (user):** the pip-installable `gadra` package is the **method only**. The data + eval +
inference tooling, while still living in this repo so a clone reproduces the paper, is **moved out of the
wheel** into a repo-only `examples/` tree:

- `src/gadra/data` → `examples/data`, `src/gadra/eval` → `examples/eval`, `src/gadra/compat` →
  `examples/compat`, `src/gadra/data/templates` → `examples/templates`.
- Import surface: `gadra.data`/`gadra.eval`/`gadra.compat` → `examples.data`/`examples.eval`/
  `examples.compat`. Reproduction runs as `python -m examples.<script>` from the repo root; deps via
  `requirements-repro.txt` (`-e .` + datasets/evaluate/accelerate).
- Wheel purity is **structural** (`where=["src"]` ⇒ only `src/gadra` ships); the `[data]/[eval]/[train]`
  extras and the package-data jinja are dropped. New/strengthened gates: `tests/test_wheel_purity.py`
  (build-namelist + `find_packages` ⇒ method-only) and `tests/test_independence.py` (asserts
  `gadra.data`/`gadra.eval` are **not** importable and `datasets`/`evaluate` are not pulled by
  `import gadra`). CI splits into a `package` job (`.[dev]` only, `pytest tests/`) and a `reproduction`
  job (`-r requirements-repro.txt`, `pytest examples/tests/`).
- Net: `pip install gadra` = clean peft tuner; clone + repro tooling = full paper reproduction. The §2
  layout below predates this split (data/eval shown under `src/gadra`); the authoritative layout is now
  in README / DESIGN §10.

### 0.4 uv environment + `examples/` flattening (P6.7, 2026-06-05)

**Decision (user):** manage the repo's training + inference environment with **uv** (replacing conda),
and flatten the repo-only `examples/` tree so it is easy to clone-and-run.

- **uv env:** one PEP 735 dependency-group `gpu` (datasets/evaluate/accelerate/deepspeed/pyyaml +
  flash-attn). torch pinned to the **cu124** index; flash-attn installed from a **prebuilt wheel URL**
  (`cu12torch2.6cxx11abiFALSE-cp312`, no source build / nvcc); torch↔FA2 ABI coupled via
  `constraint-dependencies = ["torch==2.6.0"]`. `.python-version = 3.12`; commit `uv.lock`. Groups (not
  extras) keep the published wheel method-only — guarded by a new METADATA assertion in
  `tests/test_wheel_purity.py`. Single-GPU and multi-GPU install the SAME packages; multi-GPU only changes
  the launch (`accelerate launch` + `examples/config/deepspeed_zero2.yaml`, ZeRO-2 — copied verbatim from
  the parent's PT recipe). HPC: login-node pre-sync, offline compute-node `uv run --no-sync --offline`,
  cache + python-install-dir + `.venv` on the shared FS, conda isolated per-job. SLURM template:
  `examples/slurm/train.slurm`. Honest caveat (README prereqs): uv cannot install the NVIDIA driver
  (cu124 needs ≥ 525.60.13). Conda is kept account-wide for the parent repo (coexist).
- **Flattening:** `examples/{data,eval,compat}/*` (13 nested modules) consolidated into 3 flat modules —
  `examples/processing.py` (tokenizer+EOS+packing), `examples/evaluation.py` (parsers+scorers+runner+PPL),
  `examples/convert.py` (checkpoint converter) — function bodies VERBATIM (golden tests unchanged). Entry
  points are now **config-driven single files**: `examples/train.py` + `examples/inference.py` (modes
  generate|eval), reading `examples/config/{train,inference}.yaml` via `examples.load_run_config`
  (`--override key=value` supported). Tests renamed: `test_processing.py` / `test_evaluation.py` /
  `test_convert.py`. The `pip install -r requirements-repro.txt` path remains as a portable pip fallback.

---

## 1. Executive Summary

A **two-ring** standalone `gadra` (diagnostics excluded per §0.2):

- **Ring 1 — idiomatic core (already shipped, parent-free, working on pinned peft 0.16.0).** The
  peft-native tuner; `import gadra` + `get_peft_model` + `PeftModel.from_pretrained` + stock
  `transformers.Trainer` is the whole train/load path.
- **Ring 2 — what no library provides, made first-class:**
  - `gadra.data` — the parity-exact FA2 `PackingCollator` + EOS-append + tokenizer build, **plus** a
    generic `group_texts` packing strategy for broad reuse.
  - `gadra.eval` — **`paper_repro`** (generation runner + verbatim char-F1/EM, `answer_list` EM,
    Llama CoT parsers, PPL) that reproduces the paper's canonical GaDRA rows independently; **a
    designed lm-eval adapter stub** for future generic-benchmark delegation.

**Reconciling "fully independent" with "general + idiomatic":** independence is **proven** by a CI
`sys.modules` assertion (`inference_common`/`peft_module`/`peft_config`/`Preprocessing` absent after
`import gadra`) + fresh-venv install + a secret-literal grep gate. Generality comes from
model-agnostic `target_modules` name-matching, the stock peft/Trainer API, and dual packing
strategies. The research harness (4238-line `inference_common`, 13 evaluators, GPT/NLI judges, the
γ/CR diagnostics, the live OpenAI key) is **never absorbed**.

---

## 2. Target Package Layout

```
GaDRA/                                     # standalone repo (dist name: gadra)
├── pyproject.toml                         # src-layout, PEP 621, recursive extras
├── README.md                              # Scope: +data +eval, diagnostics still OUT (see §9.0)
├── LICENSE
├── PARITY.md                              # NEW: 2-tier parity ledger + provenance
├── src/gadra/
│   ├── __init__.py                        # registers method on import (unchanged)
│   ├── config.py                          # Ring 1 — GaDRAConfig (+ from_legacy_peft_config)
│   ├── layer.py                           # Ring 1 — GaDRALinear (UNCHANGED — no diagnostics stash)
│   ├── model.py                           # Ring 1 — GaDRAModel (optional set_adapter widen, non-blocking)
│   ├── gate.py                            # Ring 1 — gumbel_sigmoid + compute_gamma (unchanged)
│   ├── _register.py · _enum_shim.py · _compat.py(NEW, tiny)
│   ├── compat/                            # Ring 1 — convert_checkpoint (legacy→peft-native)
│   ├── data/                              # Ring 2(a) — FIRST-CLASS
│   │   ├── tokenizer.py                   # build_tokenizer reimpl (~40 lines, NO inference_common)
│   │   ├── eos.py                         # _ensure_eos + check_tokenizer_auto_adds_eos + first-sample probe
│   │   ├── packing.py                     # PackingCollator (VERBATIM) + PT tokenize + pack_text_dataset(strategy=)
│   │   └── templates/                     # package data: Jinja chat templates (shipped, not referenced)
│   └── eval/                              # Ring 2(b)
│       ├── runner.py                      # greedy generation loop (verbatim gen params) + PPL (CE→exp)
│       ├── paper_repro/                   # PRIMARY (per decision)
│       │   ├── tasks.py                   # BBC-QA / GSM8K / MBPP / TiEBe prompt formatting + answer_list
│       │   ├── qa_metrics.py              # char-level F1/EM + answer_list EM (VERBATIM)
│       │   ├── llama_parsers.py           # llama_gsm8k/mbpp/humaneval parsers (VERBATIM)
│       │   └── code_exec.py               # MBPP/HumanEval pass@k via HF evaluate("code_eval") backend
│       └── lm_eval_adapter.py             # FUTURE stub — register_lm_eval_model -> HFLM subclass
├── examples/  train_gadra.py · generate.py · eval_paper_repro.py(NEW)
└── tests/
    ├── _golden/                           # frozen .pt (layer) + NEW frozen data + scorer goldens
    ├── test_parity.py                     # 7a/7b layer parity (UNCHANGED — layer untouched)
    ├── test_data_parity.py                # NEW — token stream + packed batch parity
    ├── test_paper_repro.py                # NEW — char-F1/EM + parser goldens
    ├── test_independence.py               # NEW — sys.modules assertion (constraint 5)
    ├── test_secret_gate.py                # NEW — grep gate: no sk- literal in tree
    └── conftest.py                        # importorskip for [eval] extras
```

---

## 3. Module-by-Module Disposition Table

| Parent subsystem / capability | Disposition | Target / Replacement | Rationale |
|---|---|---|---|
| Preprocessing — PT `tokenize_function_pt` (packing) | **MIGRATE (verbatim)** | `gadra.data.packing` | Parity-critical: `add_special_tokens=True`, `truncation@max_len`, `labels=copy`. |
| Preprocessing — EOS-append (`_ensure_eos`+`auto_add_eos`) | **MIGRATE (verbatim)** | `gadra.data.eos` | Bool polarity footgun (`not False→append` for Llama-3.1); golden on the *id stream*. |
| utils — `check_tokenizer_auto_adds_eos`, `get_first_sample_text` | **MIGRATE** | `gadra.data.eos` | Pure-python probe driving the EOS branch. |
| inference_common — `build_tokenizer`/`get_tokenizer_for_preprocess` | **REPLACE (reimpl ~40 LOC)** | `gadra.data.tokenizer` | Original drags openai/evaluate/accelerate + legacy `PeftPatcher`; keep only load-bearing flags. |
| Preprocessing — `PackingCollator` (FA2 varlen) | **MIGRATE (verbatim)** | `gadra.data.packing` | No `attention_mask`, per-doc `range()` position_ids; trl can't reproduce this layout. |
| Preprocessing — `grouping_concate_all` (group_texts) | **REPLACE (generic strategy)** | `pack_text_dataset(strategy="group")` | Standard run_clm recipe; broad-reuse path, NOT canonical parity path. |
| Chat templates `configs/*.jinja` | **MIGRATE (package data)** | `gadra/data/templates/` | Ship in-package or accept a path; never reference parent `configs/`. |
| inference_common — `load_base_model` adapter branch (`PeftPatcher`) | **REPLACE** | `PeftModel.from_pretrained` | gadra loads peft-native; drop `peft_module`. |
| inference_common — `get_generation_params` | **MIGRATE (verbatim params)** | `eval.runner` | Greedy determinism + EOS-id set are parity-critical. |
| inference_common — `compute_sequence_log_likelihood`/`greedy_match`/`normalize_answer` | **MIGRATE (verbatim)** | `eval.paper_repro` | Encode lm-eval conventions; parity-critical for MC/LL/F1. |
| inference_common — char-level F1/EM + `answer_list` EM | **MIGRATE (verbatim)** | `eval.paper_repro.qa_metrics` | Char-level Counter-over-`list(str)` — most error-prone port; char-level golden. |
| utils — `llama_gsm8k/mbpp/humaneval` parsers | **MIGRATE (verbatim)** | `eval.paper_repro.llama_parsers` | Needed for paper-number repro (now the primary eval path). |
| MBPP/HumanEval pass@k engine | **DELEGATE (keep backend)** | HF `evaluate("code_eval")` | Security-sensitive solved problem; never hand-roll a sandbox. |
| BBC-QA/GSM8K/MBPP/TiEBe prompt formatting + task config | **MIGRATE (minimal)** | `eval.paper_repro.tasks` | The 4 canonical GaDRA tasks only; drop the 19-task `eval_config` dispatch. |
| **peft_model.py — γ/CR compute, jsonl writers; peft_gamma_generation.py `_sample` fork; CR/R² metrics** | **LEAVE-IN-RESEARCH-REPO** | — | **Diagnostics excluded per §0.2** — not ported. |
| inference_common — `BaseEvaluator` + 13 evaluators; GPT/NLI/MTBench judges | **LEAVE-IN-RESEARCH-REPO** | (future lm-eval stub) | Dataset/parser/judge-coupled research harness. |
| inference scripts (cc2024/pretraining/forgetting/mc/autoregressive) | **LEAVE-IN-RESEARCH-REPO** | — | Hardcoded MODEL_PATHS/TASK_CONFIG harness. |
| evaluate_*.py standalone re-scorers | **LEAVE-IN-RESEARCH-REPO** | — | Legacy duplicates; `evaluate_model_output_args.py` holds the live key — never a copy source. |
| GSM8K/MMLU/TriviaQA/NQ/WebQS/commonsense (generic) | **FUTURE (lm-eval stub)** | `eval.lm_eval_adapter` | Deferred per decision 2; designed, not default. |

---

## 4. Diagnostics — explicitly OUT of scope (decision §0.2)

The per-token **γ / CR / R²** extraction (`peft_gamma_generation.py`, `peft_model.py` γ-threading,
the `_sample` fork, the `gamma_values.jsonl`/`cr_values.jsonl` writers, the analysis notebooks)
**remains in the research repo and is NOT ported.** Consequences locked in by this decision:
- `GaDRALinear.forward` is **left byte-for-byte unchanged** — no `_last_gamma`/`_last_z` stash, no
  capture flag. The existing layer-parity golden (G1) stands as-is; no ON/OFF RNG-neutrality proof
  is needed.
- The package ships **no `gadra.diagnostics`** module, **no `[diagnostics]` extra**, and the README
  Scope line keeping diagnostics out of scope **stays** (only data + eval are added).
- Anyone needing γ analyses uses the research repo against a converted checkpoint. If this is ever
  reconsidered, it returns as a fresh scoped proposal — it is not a hidden TODO here.

---

## 5. Data + Eval Strategy

| Concern | Ported (owned) | Delegated | Parity-critical bits |
|---|---|---|---|
| Tokenization / packing | `PackingCollator` (verbatim), PT tokenize, EOS-append, tokenizer build | `load_dataset("json")`; generic `group_texts` for non-FA2 reuse | `add_special_tokens=True`, `truncation@max_len`, per-doc `range()` position_ids, **no `attention_mask`**, EOS-bool polarity |
| Generation (eval) | greedy `GenerationConfig` builder; generation loop | `generate`/`PeftModel.from_pretrained` | `do_sample=False`, EOS-id set, `pad=eos` |
| PPL / retention | `eval.runner` CE→exp | — | CPT retention is a PPL story; owned |
| **Paper-exact QA/code (PRIMARY)** | char-F1/EM + `answer_list` EM + Llama parsers + 4-task prompts (verbatim) | HF `code_eval` for pass@k | char-level Counter over `list(str)`; `normalize_answer` char set |
| Generic benchmarks (FUTURE) | — | lm-eval HFLM subclass (stub) | n/a — future extension only |

**FA2 hard-assert:** `PackingCollator` emits no `attention_mask`; under SDPA/eager that silently
becomes full causal over the concatenated sequence (cross-document leakage) → wrong numbers, no
error. The collator **hard-asserts `attn_implementation == "flash_attention_2"`** when emitting
position_ids-without-mask.

**Two packing strategies behind one API:** `pack_text_dataset(strategy="fa2_collator" | "group")` —
canonical reproduction (default) + broad reuse (~15 LOC), same `[packing]`-class deps, zero core
weight. This is the concrete answer to "general + idiomatic."

**paper_repro is self-contained:** load model → `eval.runner` greedy generate → `paper_repro` parse +
score, for the 4 canonical GaDRA tasks (BBC-QA, GSM8K, MBPP, TiEBe) only. The 19-task `eval_config`
dispatch, GPT/NLI judging, and MT-Bench are **not** ported.

---

## 6. Dependencies & extras_require

```toml
[project]
requires-python = ">=3.10"
dependencies = ["torch>=2.1", "transformers>=4.53,<4.56", "peft>=0.16,<0.19", "safetensors>=0.4"]

[project.optional-dependencies]
train       = ["accelerate>=0.30", "datasets>=2.18"]
packing     = ["datasets>=2.18", "trl>=0.15,<0.17"]      # generic group strategy
eval        = ["datasets>=2.18", "evaluate>=0.4"]        # paper_repro (PRIMARY): scorers + code_eval
eval-lmeval = ["lm-eval>=0.4", "transformers>=4.53,<4.56"]  # FUTURE generic-benchmark delegation
all         = ["gadra[train,packing,eval]"]              # recursive; EXCLUDES eval-lmeval (future)
dev         = ["pytest>=8", "ruff>=0.5"]
```
- **No `[diagnostics]` extra** (dropped per §0.2). No `openai` anywhere (judges not ported).
- Pin widened to `peft>=0.16,<0.19` for upstream-ability via `_compat.py` feature-detect — **not**
  because the env is broken (0.16.0 works). Forward-compat only.
- `eval-lmeval` is unvalidatable on this host (`lm-eval` absent); its tests are `importorskip`-gated
  and only exercised in CI when implemented. Default `eval` (paper_repro) needs only `datasets` +
  `evaluate`, both present-class.

---

## 7. Phased Roadmap

**Status (2026-06-04):** P5 ✅ reproduction validated · P6.2 ✅ `gadra.data` (`c688e93` docs(scope) +
`ee52754` feat(data)) · P6.3 ✅ independence + secret gates + CI (`4c771c0` + `18f3ca8`) · P6.4 ✅
`gadra.eval` paper_repro (`9b92cf6`) · **P6.6 ✅ package/repo split** (data/eval/compat → repo-only
`examples/`; method-only wheel; see §0.3) · **P6.7 ✅ uv env + examples flattening** (single `gpu` group,
cu124 + prebuilt FA2 wheel, config-driven `train.py`/`inference.py`, `uv.lock`; see §0.4). Code roadmap
complete. **Remaining: P6.5 (HITL GPU end-to-end eval validation) and P7 (training-script independence, deferred).**

Commit-per-phase (conventional commits, no AI trailer) in the `GaDRA/` subrepo; each committed before
the next. Login-node load limits → **batch all checks into a single pytest invocation per phase**.

| Phase | Scope | Deliverable | Commit | GPU/SLURM? | Offline validation |
|---|---|---|---|---|---|
| **P6.0 — Gate (wait + sign-off)** | Wait for job **225224**; user confirms P5 (logit parity + canonical metrics in tolerance). Cron-monitor, then STOP. | go/no-go in PARITY.md | — | **N/A — wait** | n/a |
| **P6.1 — (OPTIONAL) forward-compat** | *Only if supporting peft ≥0.17:* widen `set_adapter(..., inference_mode=False)`, add `_compat.py`. **Not a blocker** (0.16.0 works). May be skipped/deferred. | range-compat | `chore(compat): peft range support` | No | full suite green; fresh-venv `pip install .` |
| **P6.2 — `gadra.data` (Ring 2a)** | `PackingCollator` verbatim + PT tokenize + EOS-append + `build_tokenizer` reimpl + `pack_text_dataset(strategy=)`; ship Jinja templates; FA2 hard-assert; rewrite `examples/train_gadra.py`. **First commit = `docs(scope)`** updating README/DESIGN (+data +eval, diagnostics still out). | data API + data golden | `docs(scope)` then `feat(data): first-class FA2 packing + tokenizer` | No | **G2 data golden:** ids/labels/position_ids + EOS-bool == frozen parent stream (CPU, no model) |
| **P6.3 — independence + security gates** | `test_independence.py` (`sys.modules`); `test_secret_gate.py` (no `sk-` literal); CI workflow. | constraint-5 proof + secret gate | `test: import-isolation + secret-literal gate` | No | `pytest`; fresh-venv install with only PyPI deps |
| **P6.4 — `gadra.eval` paper_repro (Ring 2b)** | `eval.runner` (greedy gen + PPL); `paper_repro/` (4-task prompts + char-F1/EM + `answer_list` EM + Llama parsers + `code_eval` pass@k, verbatim); `[eval]` extra; `lm_eval_adapter.py` as a documented future stub. | PPL + paper_repro eval | `feat(eval): paper_repro runner + scorers` | No (offline scorers); generation validated at HITL | char-F1/EM + parser goldens (CPU, frozen parent outputs); importorskip for eval-lmeval |
| **P6.5 — HITL host gate** | **One inference-class SLURM job** (no retrain): (1) re-run `verify_gadra_parity.py` with **FA2 on both paths**; (2) run `gadra.eval` paper_repro on the GaDRA checkpoint, confirm BBC-QA/GSM8K/MBPP/TiEBe vs `distribution_stats.json` in tolerance. Submit via existing scripts + Cron, then STOP. | host-validated parity in PARITY.md | `docs(parity): host gate results` | **Yes (1 inf job)** | host only; everything before is CPU |
| **P7 — (DEFERRED) training-script independence** | Rewrite `train_gadra_cpt.py` onto `gadra.data`; fresh full CPT to re-validate parent-free training end-to-end. Separate roadmap; does not gate P6.x. | parent-free training run | (P7) | **Yes (1 CPT run)** | HITL |

**Sequencing:** P6.0 freezes the package against the in-flight run so the parity *story* references
the build actually validated. Data before eval (paper_repro generation needs the load path; its
scorers are independent). Independence/secret gates land before eval so the secret gate guards the
eval port. Parent scripts (`train_gadra_cpt.py`, `verify_gadra_parity.py`, `.slurm`) are **left
as-is** (read-only) until P6.5/P7.

---

## 8. Parity & Verification Plan (two frozen CPU goldens + scorer goldens + one HITL diff)

**PARITY.md two-tier ledger:** Tier 1 verbatim-copy (PackingCollator, PT tokenize, EOS, gate fns,
char-F1/EM, Llama parsers) — identical-source diff + frozen golden. Tier 2 one HITL end-to-end
spot-check (P6.5). (No diagnostics tier — diagnostics not ported.)

| Gate | Proves | Runs | Ground truth | Tolerance |
|---|---|---|---|---|
| **G1 — Layer** | γ math bit-exact (toy) — UNCHANGED | CPU/login | `tests/_golden/*.pt` (4 variants) | `atol 1e-6` fp32 |
| **G2 — Data-path** | token stream + packed batch identical | CPU/login (no model) | frozen parent stream (maintainer capture, `GADRA_SOURCE_REPO`) | exact ids/labels/position_ids + EOS-bool |
| **G3 — paper_repro scorers** | char-F1/EM + parsers reproduce | CPU/login | frozen parent prediction→score pairs | exact metric value |
| **G4 — Independence** | constraint 5 | CPU + fresh venv | `sys.modules` absence of parent modules | hard assert |
| **G5 — Secret gate** | no key literal | CPU/login | grep `sk-`/`sk-proj-` | hard fail |
| **G6 — HITL host** | 8B end-to-end + FA2 + canonical metrics | SLURM (1 inf job) | `verify_gadra_parity.py` (FA2) + `distribution_stats.json` | top-1>99.9% / NLL<1e-2 under FA2; metrics in tolerance |

**Parity-claim restatement:** proven facts are (a) layer forward **fp32-bit-exact at toy scale**
(G1); (b) 8B old-vs-new **bit-exact under eager TF** (observed `max|Δ|=0`). G6 re-runs **with FA2 on
both paths** before claiming the FA2-packing canonical numbers reproduce. **Scope = GaDRA-family rows
only** (converter hard-rejects MiLoRA/svd_minor/STE/attention-targets/`output_scalar≠1.0`).

---

## 9. Risks & Open Items

### 9.0 Scope contract (resolved — first P6.2 commit)
README/DESIGN are updated to add **data + eval** to scope and **explicitly keep diagnostics out**, as
a dedicated `docs(scope)` commit before any feature code, so contract and code never disagree.

### 9.1 Security
This package ships **no secrets**: there is no API-key literal anywhere in the tree (working copy or git
history), and no code path calls an external paid API. The GPT-judged "Correct %" is intentionally not
ported, so no OpenAI key is needed; `examples.evaluation` computes only the deterministic metrics. The
absence of key-shaped literals is enforced on every PR by `tests/test_secret_gate.py`.

### 9.2 Residual decisions (defaults taken, reversible)
- **Chat templates:** default = ship Jinja as package data **and** accept an override path
  (self-contained). Reversible to path-only.
- **paper_repro task set:** BBC-QA / GSM8K / MBPP / TiEBe only (the canonical GaDRA rows). MMLU/
  TriviaQA/etc. arrive with the future lm-eval stub.

### 9.3 Overruled critique items
- `get_generation_params` arity "drift" — non-issue (deliberate 1-arg local wrapper over the 2-arg fn). Freeze as-is.
- "Defer the generic packing strategy" — kept: ~15 LOC behind `[packing]`, the explicit OSS-reuse promise, near-zero core weight. Parity collator stays default.

---

**Bottom line:** a tiny idiomatic core (already parent-free, working on pinned peft 0.16.0) + a
first-class `gadra.data` (FA2 parity packing + generic strategy) + a paper-first `gadra.eval`
(paper_repro now, lm-eval later). **Diagnostics stays in the research repo** (user decision) — which
removes the riskiest ports and leaves `layer.py` untouched. Fully independent (proven by `sys.modules`
+ fresh-venv + secret gate), broadly reusable (model-agnostic name-matching, stock peft/Trainer API,
dual packing), parity-safe (two frozen CPU goldens + scorer goldens + one HITL FA2 diff),
login-node-feasible (every gate except G6 runs on CPU). Training-run independence is **P7 (deferred)**.
**Gates: wait for 225224 + P5 sign-off (P6.0).** No blocking peft bug — that workflow claim was debunked.
