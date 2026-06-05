"""Paper-repro inference + eval with a trained GaDRA adapter (config-driven).

Reproduces the research harness's eval path: it derives the prompt / reference / judge-document from each
sample the same way the parent does (``examples.inference`` mirrors ``inference_common``), generates greedily
with the same parameters, and scores with the same verbatim scorers (``examples.evaluation``). Because the
same tokenizer + chat template + greedy params produce identical ``input_ids``, the deterministic metrics
match the original bit-for-bit.

    uv run python -m examples.inference --config examples/config/inference.yaml [--override key=value ...]

Tasks: ``qa`` → char-F1/EM, ``gsm8k`` → exact-match, ``mbpp`` → pass@1 (all deterministic / bit-exact);
``bbcqa`` / ``tiebe`` → GPT-judged "Correct %" (verbatim judge, needs ``OPENAI_API_KEY`` in the environment —
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
import pathlib
import sys

# Repo-only tooling: support both ``python -m examples.inference`` and the file-path form.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import gadra  # noqa: E402, F401  (registers the "gadra" method with peft)
from examples import load_run_config  # noqa: E402


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


def _build_prompt(row: dict, tokenizer, use_raw_text: bool) -> str:
    """Build the model prompt, matching the research harness (raw text > messages/question chat template)."""
    if "prompt" in row:
        return row["prompt"]
    if use_raw_text and row.get("text"):
        return row["text"]
    if "messages" in row:
        return tokenizer.apply_chat_template(row["messages"], tokenize=False, add_generation_prompt=True)
    question = _question_text(row)
    if question:
        msgs = [{"role": "user", "content": question}]
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
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


def _load_model(base: str, adapter: str, chat_template_path: str | None):
    """Load base + GaDRA adapter (gate stays attached — non-mergeable); left-pad + chat template for parity."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    from examples.processing import get_tokenizer_for_preprocess

    tokenizer = get_tokenizer_for_preprocess(base, chat_template_path=chat_template_path, padding_side="left")
    base_model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.bfloat16, device_map="auto")
    model = PeftModel.from_pretrained(base_model, adapter).eval()
    return model, tokenizer


def main() -> None:
    from examples.evaluation import (
        GPTJudge,
        compute_gsm8k_metrics,
        compute_pass_at_k,
        compute_qa_metrics,
        compute_qa_metrics_answer_list,
        generate_greedy,
    )

    args = parse_args()
    cfg = load_run_config(args.config, args.override)

    task = cfg.get("task", "gsm8k")
    if task not in ("qa", "gsm8k", "mbpp", "bbcqa", "tiebe"):
        raise ValueError(f"Unknown task {task!r}; expected 'qa', 'gsm8k', 'mbpp', 'bbcqa', or 'tiebe'.")

    model, tokenizer = _load_model(cfg["base"], cfg["adapter"], cfg.get("chat_template_path"))
    use_raw_text = bool(cfg.get("use_raw_text", False))

    with open(cfg["eval_file"], encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    prompts = [_build_prompt(r, tokenizer, use_raw_text) for r in rows]

    batch_size = int(cfg.get("batch_size", 16))
    max_new_tokens = int(cfg.get("max_new_tokens", 256))
    preds: list[str] = []
    for i in range(0, len(prompts), batch_size):
        preds.extend(generate_greedy(model, tokenizer, prompts[i : i + batch_size], max_new_tokens=max_new_tokens))

    if task == "qa":
        scores = []
        for p, r in zip(preds, rows):
            refs = _answer_list(r)
            scores.append(compute_qa_metrics_answer_list(p, refs) if len(refs) > 1 else compute_qa_metrics(p, refs[0] if refs else _reference_text(r)))
        f1 = sum(s["f1"] for s in scores) / len(scores)
        em = sum(s["em"] for s in scores) / len(scores)
        print(f"QA: F1={f1:.2f} EM={em:.2f} over {len(scores)} samples")
    elif task == "gsm8k":
        results = [compute_gsm8k_metrics(p, _answer_list(r)) for p, r in zip(preds, rows)]
        em = sum(x["em"] for x in results) / len(results)
        print(f"GSM8K: EM={em:.2f} over {len(results)} samples")
    elif task == "mbpp":
        tests = [r["test"] for r in rows]
        # SECURITY: pass@1 EXECUTES the model-generated code against the reference tests
        # (HF_ALLOW_CODE_EVAL=1). This runs untrusted code — only do so in an isolated /
        # containerized environment, never on a host with credentials or a network you care about.
        print(
            "WARNING: MBPP eval executes model-generated code (HF_ALLOW_CODE_EVAL=1); "
            "run only in an isolated/sandboxed environment.",
            file=sys.stderr,
        )
        result = compute_pass_at_k(preds, tests, k=[1])
        print(f"MBPP: pass@1={result['pass@1'] * 100:.2f} over {len(rows)} samples")
    else:  # bbcqa | tiebe — GPT-judged "Correct %" (needs OPENAI_API_KEY; NOT bit-exact)
        judge = GPTJudge(
            language="en" if task == "bbcqa" else "tiebe",
            model=cfg.get("gpt_model", "gpt-4.1-mini"),
            evaluation_repeats=int(cfg.get("gpt_repeats", 1)),
            vote_rate=int(cfg.get("gpt_vote_rate", 50)),
        )
        docs = _load_documents(cfg.get("documents_file")) if task == "bbcqa" else {}
        items = []
        for p, r in zip(preds, rows):
            if task == "bbcqa":
                # judge document = the original source article (by id, suffix-stripped), or an explicit field.
                document = docs.get(_resolve_doc_id(r.get("id"))) or r.get("document", "")
            else:
                # tiebe judges against the expected (gold) answer.
                document = _reference_text(r)
            items.append({"document": document, "question": _question_text(r), "answer": p})
        correct = judge.correct_percent(items)
        print(f"{task.upper()}: Correct={correct:.2f}% over {len(rows)} samples (GPT judge: {judge.model})")


if __name__ == "__main__":
    main()
