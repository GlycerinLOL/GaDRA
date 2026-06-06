"""Paper-repro inference + eval with a trained GaDRA adapter (config-driven).

Reproduces the research harness's eval path: it derives the prompt / reference / judge-document from each
sample the same way the parent does (``examples.inference`` mirrors ``inference_common``), generates greedily
with the same parameters, and scores with the same verbatim scorers (``examples.evaluation``). Because the
same tokenizer + chat template + greedy params produce identical ``input_ids``, the deterministic metrics
match the original bit-for-bit.

Two modes (mirrors the research repo's ``task_types`` list + per-task ``configs/eval_config/*.json``):

* **Suite** — set ``adapter`` once and run several benchmarks in one job. The config lists ``tasks:`` (paths
  to per-task configs, or inline dicts); a score summary is printed at the end::

      uv run python -m examples.inference --config examples/config/inference.yaml

* **Single task** — the config carries ``task:`` / ``eval_file:`` / ... directly (no ``tasks:`` key)::

      uv run python -m examples.inference --config examples/config/inference_gsm8k.yaml

Checkpoint iteration (mirrors the research repo's ``get_checkpoints_to_process``): when ``adapter`` points at
an experiment ROOT, EVERY ``checkpoint-N/`` plus the final/root adapter is evaluated in order (model reloaded
per checkpoint, all tasks run for each). Point ``adapter`` at a single ``checkpoint-N/`` (or a root with no
sub-checkpoints) to evaluate just one. Each (checkpoint, task) writes ``distribution_stats.json`` (aggregate
metrics) + ``inference_result.json`` (per-sample) under ``<checkpoint>/inference_results/<task>/``.

Speed: greedy decoding is cache-implementation-independent, so the default uses the fast dynamic KV cache.
Set ``cache_implementation: static`` (+ ``attn_implementation``) per task only to reproduce the research
repo's verbatim generation knobs (static-cache-without-compile is markedly slower).

Tasks: ``qa`` -> char-F1/EM, ``gsm8k`` -> exact-match, ``mbpp`` -> pass@1 (all deterministic / bit-exact);
``bbcqa`` / ``tiebe`` -> GPT-judged "Correct %" (verbatim judge, needs ``OPENAI_API_KEY`` in the environment —
the key is never stored in the repo; these are judge-dependent, NOT bit-exact).

Eval JSONL — one object per line; the prompt is built like the parent (``prompt`` verbatim > ``text`` when
``use_raw_text`` > ``messages`` via chat template > ``question`` via chat template). The reference / judge
inputs are derived faithfully (``messages[-1]`` = gold answer, ``messages[-2]`` = question, BBC-QA judge
document from ``documents_file`` keyed by the sample id with the ``-QA``/``-1turnQA`` suffix stripped), so
the paper's raw eval files (e.g. ``id`` + ``messages``) work directly. Explicit fields override the
derivation: ``answer``/``answers``/``answer_list`` (QA/GSM8K reference), ``test`` (MBPP), ``document`` (BBC-QA).
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import pathlib
import socket
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

# Repo-only tooling: support both ``python -m examples.inference`` and the file-path form.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gadra  # noqa: E402, F401  (registers the "gadra" method with peft)
from examples import load_run_config  # noqa: E402

VALID_TASKS = ("qa", "gsm8k", "mbpp", "bbcqa", "tiebe")
DEFAULT_BASE = "meta-llama/Llama-3.1-8B-Instruct"


def _openai_reachable(timeout: float = 5.0) -> bool:
    """Quick TCP probe to the OpenAI endpoint, so GPT-judged tasks fail FAST on an offline compute node
    instead of hanging on per-request retries/timeouts. Skipped when a custom OPENAI_BASE_URL is set."""
    if os.environ.get("OPENAI_BASE_URL"):
        return True  # custom endpoint (Azure/proxy) — don't second-guess reachability
    try:
        with socket.create_connection(("api.openai.com", 443), timeout=timeout):
            return True
    except OSError:
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Paper-repro inference/eval with a trained GaDRA adapter (config-driven).")
    p.add_argument("--config", required=True, help="Path to a YAML run-config (see examples/config/inference.yaml).")
    p.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                   help="Override a config field (YAML-typed value); repeatable.")
    return p.parse_args()


# --- Faithful per-sample extraction (verbatim from inference_common: question/doc-id/reference) ---
def _question_text(row: dict) -> str:
    """Verbatim ``inference_common._determine_question_text``: explicit question, else messages[-2]/[-1]."""
    if row.get("question"):
        return row["question"]
    msgs = row.get("messages") or []
    if isinstance(msgs, list) and len(msgs) >= 2 and isinstance(msgs[-2], dict):
        return msgs[-2].get("content", "")
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        return msgs[-1].get("content", "")
    return ""


def _reference_text(row: dict) -> str:
    """The gold answer string (QA reference / TiEBe expected): messages[-1], else answer/reference_answer."""
    msgs = row.get("messages") or []
    if isinstance(msgs, list) and msgs and isinstance(msgs[-1], dict):
        return msgs[-1].get("content", "")
    return row.get("reference_answer") or row.get("answer") or ""


def _answer_list(row: dict) -> list[str]:
    """Reference answers as a list (handles a real list, or a Python-repr stringified list, or a scalar)."""
    refs = row.get("answers") or row.get("answer_list") or row.get("answer")
    if isinstance(refs, str):
        try:
            parsed = ast.literal_eval(refs)
            if isinstance(parsed, (list, tuple)):
                return [str(x) for x in parsed]
        except (ValueError, SyntaxError):
            pass
        return [refs]
    if isinstance(refs, (list, tuple)):
        return [str(x) for x in refs]
    return []


def _resolve_doc_id(sample_id):
    """Verbatim ``inference_common._resolve_doc_id``: strip the ``-1turnQA`` / ``-QA`` QA suffix."""
    if not isinstance(sample_id, str):
        return sample_id
    for suffix in ("-1turnQA", "-QA"):
        if sample_id.endswith(suffix):
            return sample_id[: -len(suffix)]
    return sample_id


def _build_prompt(row: dict, tokenizer, use_raw_text: bool, system_prompt: str | None = None) -> str:
    """Build the model prompt, matching the research harness (``DocumentQAEvaluator.prepare_inputs``).

    For chat rows WITHOUT an explicit ``answer_list``, the trailing message is the gold answer (it is the
    reference), so it is DROPPED — the prompt is built from ``messages[:-1]`` — otherwise the gold answer
    would leak into the prompt and the model could just echo it. With an ``answer_list`` (e.g. TriviaQA /
    NQ-open) the messages are the prompt and are used whole. Order: explicit ``prompt`` > raw ``text`` (when
    ``use_raw_text``) > ``messages`` > ``question``.
    """
    if "prompt" in row:
        return row["prompt"]
    if use_raw_text and row.get("text"):
        return row["text"]
    msgs = row.get("messages")
    if msgs:
        messages = [dict(m) for m in msgs if isinstance(m, dict)]
        if not messages and row.get("question"):
            messages = [{"role": "user", "content": row["question"]}]
        has_answer_list = isinstance(row.get("answer_list"), list) and len(row["answer_list"]) > 0
        dialogue = messages if has_answer_list else (messages[:-1] if len(messages) > 1 else messages)
        if system_prompt:
            dialogue = [{"role": "system", "content": system_prompt}, *dialogue]
        return tokenizer.apply_chat_template(dialogue, tokenize=False, add_generation_prompt=True)
    question = _question_text(row)
    if question:
        msgs2 = [{"role": "user", "content": question}]
        if system_prompt:
            msgs2 = [{"role": "system", "content": system_prompt}, *msgs2]
        return tokenizer.apply_chat_template(msgs2, tokenize=False, add_generation_prompt=True)
    raise KeyError(f"row has no prompt source (need 'prompt' / 'text' / 'messages' / 'question'): {sorted(row)}")


def _load_documents(documents_file: str | None) -> dict:
    """Load an original-document JSONL (BBC-QA judge document source) into {doc_id: text}."""
    docs: dict = {}
    if not documents_file:
        return docs
    with open(documents_file, encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            d = json.loads(line)
            doc_id = d.get("id") or d.get("doc_id")
            docs[doc_id] = d.get("text") or d.get("document") or d.get("content") or ""
    return docs


def _load_model(base: str, adapter: str, attn_implementation: str = "sdpa"):
    """Load base + GaDRA adapter (gate stays attached — non-mergeable). Reused across all tasks for one
    checkpoint; only the tokenizer is rebuilt per task (its chat template differs), which is cheap.

    ``attn_implementation`` defaults to ``sdpa`` — matching the research repo's inference
    (GENERATION_CONFIG ``attn_implementation='sdpa'``); ``device_map='auto'`` as in the original.
    """
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    base_model = AutoModelForCausalLM.from_pretrained(
        base, torch_dtype=torch.bfloat16, device_map="auto", attn_implementation=attn_implementation
    )
    return PeftModel.from_pretrained(base_model, adapter).eval()


def _discover_checkpoints(adapter_dir: str) -> list[tuple[str, str]]:
    """Mirror the research repo's ``get_checkpoints_to_process``: evaluate EVERY checkpoint under an
    experiment root, not just one. Returns ``[(label, path), ...]`` sorted ``checkpoint-N`` ascending then
    the final (root) adapter last.

    * a ``checkpoint-N`` dir (or any single HF-peft adapter dir)  -> just that one
    * an experiment root with ``checkpoint-N/`` subdirs           -> each checkpoint + the root final
    * a non-local id (HF hub) or a dir with no adapter_config     -> treated as a single adapter
    """
    import glob

    norm = os.path.normpath(adapter_dir)
    base = os.path.basename(norm)
    has_cfg = os.path.exists(os.path.join(norm, "adapter_config.json"))
    if base.startswith("checkpoint-") and has_cfg:
        return [(base, norm)]

    ckpts: list[tuple[int, str]] = []
    for sub in glob.glob(os.path.join(norm, "checkpoint-*")):
        if os.path.isdir(sub) and os.path.exists(os.path.join(sub, "adapter_config.json")):
            try:
                ckpts.append((int(os.path.basename(sub).split("-", 1)[1]), sub))
            except ValueError:
                pass
    ckpts.sort(key=lambda x: x[0])
    out: list[tuple[str, str]] = [(f"checkpoint-{n}", p) for n, p in ckpts]
    if has_cfg:
        out.append(("final", norm))  # root save_pretrained = final adapter (the research repo's "LoRA/")
    return out or [(base or norm, norm)]  # fall back to a single adapter (e.g. an HF hub id)


def _save_task_results(ckpt_path: str, task: str, result: dict) -> None:
    """Write per-checkpoint, per-task results like the research repo: a ``distribution_stats.json`` (aggregate
    metrics — the first place to look) and an ``inference_result.json`` (per-sample) under
    ``<ckpt_path>/inference_results/<task>/``. Skipped if the adapter isn't a writable local dir."""
    if not os.path.isdir(ckpt_path):
        return
    out_dir = os.path.join(ckpt_path, "inference_results", task)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "distribution_stats.json"), "w", encoding="utf-8") as fh:
        json.dump(result["stats"], fh, ensure_ascii=False, indent=2)
    with open(os.path.join(out_dir, "inference_result.json"), "w", encoding="utf-8") as fh:
        json.dump(result["samples"], fh, ensure_ascii=False, indent=2)
    print(f"[{task}] saved -> {out_dir}/{{distribution_stats,inference_result}}.json", flush=True)


def _tokenizer_for(base: str, chat_template_path: str | None):
    """Inference tokenizer (verbatim ``inference_common.setup_tokenizer``): left-padding + the task's chat
    template, BOS/EOS left at the model defaults, pad_token defaults to EOS. Crucially does NOT force
    ``add_eos_token=True`` (that is the TRAINING-preprocess tokenizer; appending EOS to a generation prompt
    is wrong — benign on the Llama-3 fast tokenizer, which ignores the flag, but breaks other backbones)."""
    from examples.processing import build_tokenizer

    return build_tokenizer(base, padding_side="left", chat_template_path=chat_template_path)


def run_task(model, task_cfg: dict) -> dict:
    """Run one task against an already-loaded ``model``; print live progress and return a result dict
    ``{label, line, stats, samples}`` (stats -> distribution_stats.json, samples -> inference_result.json)."""
    from examples.evaluation import (
        GPTJudge,
        compute_gsm8k_metrics,
        compute_pass_at_k,
        compute_qa_metrics,
        compute_qa_metrics_answer_list,
        generate_greedy,
        llama_mbpp_parse,
    )

    task = task_cfg.get("task", "gsm8k")
    if task not in VALID_TASKS:
        raise ValueError(f"Unknown task {task!r}; expected one of {VALID_TASKS}.")
    # GPT-judged tasks need a key AND outbound internet — check BOTH before the (slow) generation, so on an
    # offline compute node bbcqa/tiebe fail in seconds instead of hanging on per-request retries.
    if task in ("bbcqa", "tiebe"):
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError(f"task {task!r} is GPT-judged but OPENAI_API_KEY is not set in the environment.")
        if not _openai_reachable():
            raise RuntimeError(
                f"task {task!r} is GPT-judged but api.openai.com is unreachable from this node. SLURM compute "
                "nodes are usually offline — run bbcqa/tiebe on a node with internet, or drop them from the "
                "suite's `tasks:` list. The deterministic tasks (gsm8k / mbpp) run fully offline."
            )

    tokenizer = _tokenizer_for(task_cfg["base"], task_cfg.get("chat_template_path"))
    use_raw_text = bool(task_cfg.get("use_raw_text", False))
    system_prompt = task_cfg.get("system_prompt") if task_cfg.get("use_system_prompt") else None

    with open(task_cfg["eval_file"], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    prompts = [_build_prompt(r, tokenizer, use_raw_text, system_prompt) for r in rows]

    batch_size = int(task_cfg.get("batch_size", 16))
    max_new_tokens = int(task_cfg.get("max_new_tokens", 256))
    cache_impl = task_cfg.get("cache_implementation")  # None = dynamic cache (fast); "static" = verbatim repo
    total = len(prompts)

    # GPT-judged tasks: judge each batch's predictions in a thread pool the moment the batch is generated,
    # while the NEXT batch keeps generating on the GPU — so judging (network) overlaps generation (GPU).
    # This mirrors the research repo's per-batch generate->judge pipeline with its 32-way parallel judging,
    # replacing the previous one-call-at-a-time loop (which made ~N sequential API calls).
    is_judged = task in ("bbcqa", "tiebe")
    judge = pool = None
    gpt_workers = 0
    futures: dict = {}
    samples: list[dict] = []
    if is_judged:
        judge = GPTJudge(
            language="en" if task == "bbcqa" else "tiebe",
            model=task_cfg.get("gpt_model", "gpt-4.1-mini"),
            evaluation_repeats=int(task_cfg.get("gpt_repeats", 1)),
            vote_rate=int(task_cfg.get("gpt_vote_rate", 50)),
        )
        docs = _load_documents(task_cfg.get("documents_file")) if task == "bbcqa" else {}
        gpt_workers = max(1, int(task_cfg.get("gpt_workers", 32)))
        pool = ThreadPoolExecutor(max_workers=gpt_workers)

    print(f"[{task}] generating {total} prompts (batch_size={batch_size}, max_new_tokens={max_new_tokens})", flush=True)
    preds: list[str] = []
    for i in range(0, total, batch_size):
        batch_rows = rows[i : i + batch_size]
        batch_preds = generate_greedy(model, tokenizer, prompts[i : i + batch_size], max_new_tokens=max_new_tokens, cache_implementation=cache_impl)
        preds.extend(batch_preds)
        generated = min(i + batch_size, total)
        if is_judged:  # build each per-sample record + dispatch its judge call now (judging overlaps the next batch)
            for j, (p, r) in enumerate(zip(batch_preds, batch_rows)):
                gidx = i + j
                reference = _reference_text(r)  # gold = messages[-1] (the turn dropped from the prompt)
                if task == "bbcqa":
                    judge_doc = docs.get(_resolve_doc_id(r.get("id"))) or r.get("document", "")
                    if not judge_doc:  # original skips samples whose source article is missing (not judged/counted)
                        continue
                else:  # tiebe — the expected answer doubles as the judge "document"
                    judge_doc = reference
                sc = compute_qa_metrics(p, reference)  # char-level F1/EM, exactly as DocumentQAEvaluator records
                futures[pool.submit(judge.evaluate_detailed, judge_doc, _question_text(r), p)] = len(samples)
                samples.append({"id": r.get("id") or f"sample_{gidx}", "input": prompts[gidx],
                                "reference": reference, "prediction": p, "f1": sc["f1"], "em": sc["em"]})
            # Make the gen<->judge overlap visible: how many of the dispatched judge calls have ALREADY finished on
            # the 32-worker pool *while the GPU kept generating*. A "judged" count that climbs across these lines
            # (rather than staying 0 until generation ends) is the overlap at work. f.done() is non-blocking.
            judged_done = sum(1 for f in futures if f.done())
            print(f"[{task}] generated {generated}/{total} | judged {judged_done}/{len(futures)} done (overlapping)", flush=True)
        else:
            print(f"[{task}] generated {generated}/{total}", flush=True)

    if task == "qa":
        samples, f1s, ems = [], [], []
        for p, r in zip(preds, rows):
            refs = _answer_list(r)
            sc = compute_qa_metrics_answer_list(p, refs) if len(refs) > 1 else compute_qa_metrics(p, refs[0] if refs else _reference_text(r))
            f1s.append(sc["f1"])
            ems.append(sc["em"])
            samples.append({"prediction": p, "reference": refs or _reference_text(r), "f1": sc["f1"], "em": sc["em"]})
        f1, em = sum(f1s) / total, sum(ems) / total
        line, label = f"F1={f1:.2f} EM={em:.2f} over {total} samples", "QA"
        stats = {"task": task, "f1": f1, "em": em, "num_samples": total}
    elif task == "gsm8k":
        samples, n_correct = [], 0
        for gidx, (p, r) in enumerate(zip(preds, rows)):
            x = compute_gsm8k_metrics(p, _answer_list(r))  # raw membership, verbatim GSM8KEvaluator
            n_correct += int(x["is_correct"])
            samples.append({"id": r.get("id") or f"sample_{gidx}", "question": _question_text(r),
                            "input": prompts[gidx], "prediction": p, "parsed_prediction": x["parsed_prediction"],
                            "answers": x["answers"], "is_correct": x["is_correct"]})
        exact_match = (n_correct / total * 100) if total else 0.0
        line, label = f"EM={exact_match:.2f} over {total} samples", "GSM8K"
        stats = {
            "gsm8k_results": {"correct": n_correct, "incorrect": total - n_correct, "total": total, "exact_match": exact_match},
            "final_scores": {"exact_match": exact_match, "correct": n_correct, "total": total},
        }
    elif task == "mbpp":
        tests = [r["test"] for r in rows]
        # SECURITY: pass@1 EXECUTES the model-generated code against the reference tests
        # (HF_ALLOW_CODE_EVAL=1). This runs untrusted code — only do so in an isolated /
        # containerized environment, never on a host with credentials or a network you care about.
        print(
            "WARNING: MBPP eval executes model-generated code (HF_ALLOW_CODE_EVAL=1); "
            "run only in an isolated/sandboxed environment.",
            file=sys.stderr, flush=True,
        )
        metrics, passed = compute_pass_at_k(
            preds,
            tests,
            k=task_cfg.get("code_eval_k", [1]),
            num_workers=int(task_cfg.get("code_eval_num_workers", 8)),  # research-repo default
            timeout=float(task_cfg.get("code_eval_timeout", 30.0)),
            return_passed=True,
        )
        samples = [
            {"id": r.get("id") or f"sample_{gidx}", "input": prompts[gidx], "prediction": p,
             "candidates": [llama_mbpp_parse(p)], "reference": t,
             "passed": bool(passed[gidx]) if gidx < len(passed) else False}
            for gidx, (p, r, t) in enumerate(zip(preds, rows, tests))
        ]
        n_correct = sum(1 for s in samples if s["passed"])
        pass1 = float(metrics.get("pass@1", 0.0))
        line, label = f"pass@1={pass1 * 100:.2f}% over {total} samples", "MBPP"
        stats = {
            "code_eval": metrics,  # pass@k as a fraction in [0,1] (verbatim the research repo's code_eval block)
            "code_eval_stats": {"total": total, "correct": n_correct, "wrong": total - n_correct},
            "final_scores": metrics,
        }
    else:  # bbcqa | tiebe — drain the judge futures dispatched during generation (overlapped with the GPU)
        n_correct = done = 0
        total_j = len(futures)
        for fut in as_completed(futures):
            idx = futures[fut]
            try:
                gpt = fut.result()
            except Exception:  # a failed judge future counts as Incorrect — mirror the parent's per-future fallback
                reps = max(1, int(getattr(judge, "evaluation_repeats", 1)))
                gpt = {"final_decision": judge.incorrect_label, "all_results": [judge.incorrect_label] * reps,
                       "correct_count": 0, "total_count": reps, "correct_rate": 0.0}
            samples[idx]["gpt_evaluation"] = gpt
            if gpt["final_decision"] == judge.correct_label:
                n_correct += 1
            done += 1
            if done % 25 == 0 or done == total_j:  # collecting verdicts dispatched during generation (the tail past gen)
                print(f"[judge] collected {done}/{total_j} ({judge.model}, {gpt_workers} workers)", flush=True)
        pool.shutdown()
        n_incorrect = total_j - n_correct
        skipped = total - total_j  # bbcqa samples whose source document was missing (matches the original)
        f1_mean = sum(s["f1"] for s in samples) / total_j if total_j else 0.0
        em_mean = sum(s["em"] for s in samples) / total_j if total_j else 0.0
        correct = n_correct / total_j * 100 if total_j else 0.0
        suffix = f" ({skipped} skipped: missing doc)" if skipped else ""
        line = f"Correct={correct:.2f}% over {total_j} samples{suffix} (GPT judge: {judge.model})"
        label = task.upper()
        stats = {
            "gpt_evaluation": {
                judge.correct_label: {"count": n_correct, "percentage": correct},
                judge.incorrect_label: {"count": n_incorrect, "percentage": (n_incorrect / total_j * 100) if total_j else 0.0},
            },
            "final_scores": {"f1": f1_mean, "em": em_mean},
        }

    print(f"{label}: {line}", flush=True)
    return {"label": label, "line": line, "stats": stats, "samples": samples}


def _resolve_task_configs(cfg: dict) -> tuple[str, str, list[dict]]:
    """Return ``(base, adapter, [task_cfg, ...])`` for suite (``tasks:`` list) or single-task (``task:``) mode.

    ``adapter`` lives ONLY at the top level (set once); the per-task configs carry no model path (like the
    research repo's ``configs/eval_config/*.json``). ``base`` defaults to ``DEFAULT_BASE``. In suite mode the
    shared ``base`` / ``adapter`` are stamped onto every task.
    """
    base = cfg.get("base", DEFAULT_BASE)
    adapter = cfg.get("adapter")
    tasks = cfg.get("tasks")
    if isinstance(tasks, list) and tasks:
        if not adapter:
            raise ValueError("suite config needs `adapter:` (set it ONCE; it is applied to every task).")
        resolved = []
        for entry in tasks:
            tc = dict(load_run_config(entry)) if isinstance(entry, str) else dict(entry)
            tc["base"], tc["adapter"] = base, adapter  # set-adapter-once: top-level values win
            resolved.append(tc)
        return base, adapter, resolved
    # single-task: the config itself is the task config
    if not adapter:
        raise ValueError(
            "config needs `adapter:` (the trained GaDRA dir). Set it once in the suite "
            "(examples/inference.yaml), or pass `--override adapter=save/.../CPT/<EXP_NAME>`."
        )
    cfg = {**cfg, "base": base}
    return base, adapter, [cfg]


def main() -> None:
    args = parse_args()
    cfg = load_run_config(args.config, args.override)

    base, adapter, task_cfgs = _resolve_task_configs(cfg)
    attn = cfg.get("attn_implementation", "sdpa")

    # Fail fast on a bad task name BEFORE the expensive model load.
    for tc in task_cfgs:
        t = tc.get("task", "gsm8k")
        if t not in VALID_TASKS:
            raise ValueError(f"Unknown task {t!r}; expected one of {VALID_TASKS}.")

    # Mirror the research repo: evaluate EVERY checkpoint under the experiment root, in order
    # (checkpoint-N ascending, then the final/root adapter), reloading the model per checkpoint.
    checkpoints = _discover_checkpoints(adapter)
    print(f"base={base}", flush=True)
    print(f"tasks: {[tc.get('task') for tc in task_cfgs]}", flush=True)
    print(f"checkpoints to process ({len(checkpoints)}): {[c[0] for c in checkpoints]}", flush=True)

    failures = 0
    for ckpt_label, ckpt_path in checkpoints:
        print(f"\n########## checkpoint {ckpt_label}: {ckpt_path} ##########", flush=True)
        model = _load_model(base, ckpt_path, attn)
        summary: list[tuple[str, str]] = []
        for tc in task_cfgs:
            t = tc.get("task", "gsm8k")
            try:
                res = run_task(model, tc)
                summary.append((res["label"], res["line"]))
                _save_task_results(ckpt_path, t, res)
            except Exception as exc:  # one task's failure must not abort the rest of this checkpoint
                failures += 1
                summary.append((t.upper(), f"FAILED: {exc}"))
                print(f"[{t}] FAILED: {exc}", file=sys.stderr, flush=True)

        print(f"\n===== summary (checkpoint={ckpt_label}, adapter={ckpt_path}) =====", flush=True)
        for label, line in summary:
            print(f"  {label:8s} {line}", flush=True)

        del model  # free the GPU before the next checkpoint's base+adapter load
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except Exception:
            pass

    if failures:
        print(f"{failures} task(s) failed across all checkpoints", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
