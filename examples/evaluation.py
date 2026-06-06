"""Evaluation for GaDRA paper reproduction — answer parsers, deterministic scorers, greedy runner, PPL.

Consolidated (and ported VERBATIM) from the research repo so the **deterministic** metrics in the paper's
``distribution_stats`` reproduce bit-for-bit (golden-tested, gate G3): QA char-F1/EM, GSM8K exact-match,
MBPP pass@1. The GPT-judged "Correct %" for BBC/TiEBe needs an external judge (OpenAI) and is deliberately
NOT computed here — see ``docs/STANDALONE_PLAN.md`` §5.

Five concerns, one module:
  * parsers  — :func:`normalize_answer`, :func:`normalize_numeric_answer`, :func:`llama_gsm8k_parse`,
               :func:`extract_gsm8k`, :func:`llama_mbpp_parse`, :func:`llama_humaneval_reconstruct`
  * scorers  — :func:`compute_qa_metrics`, :func:`compute_qa_metrics_answer_list`, :func:`compute_gsm8k_metrics`
  * runner   — :func:`get_generation_params`, :func:`generate_greedy`, :func:`sequence_perplexity` (need a model/GPU)
  * code     — :func:`compute_pass_at_k` (MBPP/HumanEval pass@k via ``evaluate('code_eval')``, opt-in)
  * judge    — :class:`GPTJudge` (BBC-QA / TiEBe "Correct %" via OpenAI; key from ``OPENAI_API_KEY``, NOT bit-exact)

The parsers + scorers are pure CPU functions. The runner + ``compute_pass_at_k`` need a model / the
``evaluate`` library; :class:`GPTJudge` needs ``openai`` + a key (lazy-imported). All are exercised at the
HITL gate (P6.5), not in CPU unit tests.
"""

from __future__ import annotations

import os
import re
import string
from collections import Counter
from typing import Any, Dict, List, Optional, Sequence

import torch


# --------------------------------------------------------------------------------------------------
# Answer parsers / normalizers (verbatim — any change shifts the reproduced numbers)
#   normalize_answer         <- inference_common.normalize_answer
#   normalize_numeric_answer <- evaluate_gsm8k.normalize_numeric_answer
#   llama_*                  <- utils
# --------------------------------------------------------------------------------------------------
def normalize_answer(text: str, strip_articles: bool = False) -> str:
    """Lowercase, drop punctuation, collapse whitespace (verbatim ``inference_common.normalize_answer``).

    ``strip_articles=True`` additionally removes leading a/an/the to match lm-eval-harness NQ-open.
    """
    text = text.lower()
    if strip_articles:
        text = re.sub(r"\b(a|an|the)\b", " ", text)
    remove_punc = str.maketrans("", "", string.punctuation)
    return " ".join(text.translate(remove_punc).split())


def normalize_numeric_answer(s: str) -> str:
    """Normalize a numeric answer — strip currency/commas/space, canonicalize int vs float (verbatim)."""
    if not s:
        return ""
    s = re.sub(r"[\$€£¥\s,]", "", s.strip()).rstrip(".")
    try:
        num = float(s)
        return str(int(num)) if num == int(num) else f"{num:g}"
    except ValueError:
        return s


def llama_gsm8k_parse(text: str) -> str:
    """Extract the final numeric answer from a Llama GSM8K generation (verbatim ``utils.llama_gsm8k_parse``)."""
    text = text.lower()
    t = text.strip()

    m = re.search(
        r"(?:the\s+final\s+answer\s+is|final\s+answer\s+is)\s*"
        r"([$€£¥]?\s*-?\d[\d,]*(?:\.\d+)?\.?)",
        t,
        flags=re.IGNORECASE,
    )

    if m:
        ans = m.group(1)
        ans = re.sub(r"^[^0-9\-]+", "", ans)
        return ans

    return ""


def extract_gsm8k(text: Optional[str]) -> str:
    """Extract the numeric answer after the final ``####`` (verbatim ``utils.extract_gsm8k``)."""
    if not text:
        return ""

    snippet = text.split("####")[-1].strip() if "####" in text else text.strip()
    if not snippet:
        return ""

    snippet = snippet.replace(",", "").strip()
    match = re.search(r"-?\d+(?:\.\d+)?", snippet)
    if match:
        return match.group(0)
    return snippet


def llama_mbpp_parse(text: str) -> str:
    """Extract the python code block from a Llama MBPP generation (verbatim ``utils.llama_mbpp_parse``)."""
    BLOCK_RE = re.compile(r"```(?:python)?\s*(.*?)```", re.S | re.I)
    SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")
    s = text.replace("\r\n", "\n")
    s = SPECIAL_TOKEN_RE.sub("", s)

    m = BLOCK_RE.search(s)
    if m:
        return m.group(1).strip()

    # Fallback when there is no matched fence pair (e.g. only a closing ```).
    s = s.replace("```python", "").replace("```", "")
    return s.strip()


def llama_humaneval_reconstruct(prediction: str, prompt: str) -> str:
    """Reconstruct a full function from prompt preamble + generated body (verbatim ``utils``).

    Meta's Llama HumanEval format puts the signature + docstring in the assistant preamble (an unclosed
    ```python block); the model generates only the body. This prepends the preamble to the body.
    """
    SPECIAL_TOKEN_RE = re.compile(r"<\|.*?\|>")

    fence_marker = "```python"
    fence_idx = prompt.rfind(fence_marker)
    if fence_idx == -1:
        return llama_mbpp_parse(prediction)

    preamble = prompt[fence_idx + len(fence_marker):]
    preamble = SPECIAL_TOKEN_RE.sub("", preamble)
    preamble = preamble.rstrip()

    body = prediction.replace("\r\n", "\n")
    body = SPECIAL_TOKEN_RE.sub("", body)
    fence_close_idx = body.find("```")
    if fence_close_idx != -1:
        body = body[:fence_close_idx]
    body = body.rstrip()

    return preamble + "\n" + body


# --------------------------------------------------------------------------------------------------
# Deterministic scorers (verbatim ``evaluate_document_qa.compute_metrics`` + ``evaluate_gsm8k``)
# --------------------------------------------------------------------------------------------------
def compute_qa_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """Char-level F1 + EM in ``[0, 100]`` (verbatim ``evaluate_document_qa.compute_metrics``)."""
    pred_chars = list(normalize_answer(prediction))
    ref_chars = list(normalize_answer(reference))
    if not pred_chars or not ref_chars:
        return {"f1": 0.0, "em": 0.0}
    common = Counter(pred_chars) & Counter(ref_chars)
    num_same = sum(common.values())
    precision, recall = num_same / len(pred_chars), num_same / len(ref_chars)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    em = float(normalize_answer(prediction) == normalize_answer(reference))
    return {"f1": f1 * 100, "em": em * 100}


def compute_qa_metrics_answer_list(prediction: str, references: List[str]) -> Dict[str, float]:
    """Best F1/EM over a list of acceptable references (``answer_list`` datasets)."""
    if not references:
        return {"f1": 0.0, "em": 0.0}
    scored = [compute_qa_metrics(prediction, r) for r in references]
    return {"f1": max(s["f1"] for s in scored), "em": max(s["em"] for s in scored)}


def compute_gsm8k_metrics(prediction: str, answers: List[str]) -> Dict[str, Any]:
    """GSM8K exact-match via the Llama parser + numeric normalization (verbatim ``evaluate_gsm8k``)."""
    parsed = llama_gsm8k_parse(prediction or "")
    norm_pred = normalize_numeric_answer(parsed)
    norm_ans = list(dict.fromkeys([normalize_numeric_answer(str(a)) for a in answers if a]))
    is_correct = bool(norm_pred) and norm_pred in norm_ans
    return {
        "parsed_prediction": parsed,
        "normalized_prediction": norm_pred,
        "normalized_answers": norm_ans,
        "is_correct": is_correct,
        "em": 100.0 if is_correct else 0.0,
    }


# --------------------------------------------------------------------------------------------------
# Greedy generation + perplexity (verbatim ``inference_common.get_generation_params``; need a model/GPU)
# --------------------------------------------------------------------------------------------------
def get_generation_params(tokenizer: Any, max_new_tokens: int) -> Dict[str, Any]:
    """Deterministic greedy params + robust multi-family EOS-id set (verbatim research repo).

    The EOS-id list always contains the tokenizer's ``eos_token_id`` plus any of ``{<|eot_id|>, <|im_end|>}``
    that resolve to a real vocab id (Llama-3 has ``<|eot_id|>``; Qwen3 has ``<|im_end|>``), never ``None``.
    """
    eos_ids = [tokenizer.eos_token_id]
    unk_id = getattr(tokenizer, "unk_token_id", None)
    for tok in ("<|eot_id|>", "<|im_end|>"):
        tid = tokenizer.convert_tokens_to_ids(tok)
        if isinstance(tid, int) and tid >= 0 and tid != unk_id:
            eos_ids.append(tid)
    eos_ids = sorted(set(eos_ids))
    return {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
        "top_p": None,
        "temperature": None,
        "cache_implementation": "static",
        "disable_compile": True,
        "pad_token_id": tokenizer.eos_token_id,
        "eos_token_id": eos_ids,
    }


@torch.no_grad()
def generate_greedy(
    model,
    tokenizer,
    prompts: Sequence[str],
    *,
    max_new_tokens: int = 256,
    device: Optional[Any] = None,
) -> List[str]:
    """Batched greedy generation; returns the decoded continuation (prompt stripped) per prompt.

    Use a left-padding tokenizer (``tokenizer.padding_side = 'left'``) for correct batched decoding.
    """
    device = device or next(model.parameters()).device
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True).to(device)
    gen = model.generate(**enc, **get_generation_params(tokenizer, max_new_tokens))
    prompt_len = enc["input_ids"].shape[1]
    return [tokenizer.decode(gen[i, prompt_len:], skip_special_tokens=True) for i in range(gen.shape[0])]


@torch.no_grad()
def sequence_perplexity(model, input_ids, *, attention_mask: Optional[Any] = None) -> float:
    """Teacher-forcing perplexity ``exp(mean CE)`` over a single sequence (CPT retention metric)."""
    if not torch.is_tensor(input_ids):
        input_ids = torch.tensor(input_ids)
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    input_ids = input_ids.to(next(model.parameters()).device)
    out = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
    return float(torch.exp(out.loss))


# --------------------------------------------------------------------------------------------------
# MBPP / HumanEval pass@k (delegated to the sandboxed ``evaluate('code_eval')``; explicit opt-in)
# --------------------------------------------------------------------------------------------------
def compute_pass_at_k(
    predictions: Sequence[str],
    test_cases: Sequence[str],
    *,
    k: Optional[List[int]] = None,
    num_workers: int = 4,
    timeout: float = 30.0,
    extract: bool = True,
) -> Dict[str, float]:
    """pass@k with one candidate per problem (the GaDRA paper's MBPP/HumanEval setting).

    Code execution is a security-sensitive, already-solved problem — we delegate to the sandboxed
    ``evaluate('code_eval')`` rather than hand-roll a runner. Candidate code is extracted from Llama
    generations via :func:`llama_mbpp_parse`. Requires the reproduction deps (``evaluate``) and the
    explicit opt-in ``HF_ALLOW_CODE_EVAL=1`` (running model-generated code executes arbitrary Python —
    do so only in a sandbox).

    Args:
        predictions: model generation for each problem (code extracted via ``llama_mbpp_parse`` when ``extract``).
        test_cases:  the test program (asserts) appended after the candidate for each problem.
        k:           pass@k values (default ``[1]``).
    """
    if os.environ.get("HF_ALLOW_CODE_EVAL") != "1":
        raise RuntimeError(
            "Refusing to execute model-generated code: set HF_ALLOW_CODE_EVAL=1 (and run in a sandbox) "
            "to enable evaluate('code_eval')."
        )
    try:
        import evaluate
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError("install the reproduction deps (provides the `evaluate` library) for code execution.") from e

    k = k or [1]
    code_eval = evaluate.load("code_eval")
    candidates = [[llama_mbpp_parse(p) if extract else p] for p in predictions]
    references = list(test_cases)
    results, _ = code_eval.compute(
        predictions=candidates, references=references, k=k, num_workers=num_workers, timeout=timeout
    )
    return {f"pass@{kk}": results[f"pass@{kk}"] for kk in k}


# --------------------------------------------------------------------------------------------------
# lm-evaluation-harness adapter — FUTURE stub (paper_repro is the supported path)
# --------------------------------------------------------------------------------------------------
def register_lm_eval_model() -> None:
    """Register a ``gadra`` model type with lm-evaluation-harness (future).

    Intended implementation: subclass ``lm_eval.models.huggingface.HFLM`` so its ``_create_model`` wraps the
    loaded base with ``PeftModel.from_pretrained`` (GaDRA is non-mergeable — the gate stays attached), then
    call ``lm_eval.api.registry.register_model('gadra')(GaDRAHFLM)``. Deferred per the standalone plan.
    """
    raise NotImplementedError(
        "examples.evaluation.register_lm_eval_model is a future stub; the supported eval path is the "
        "paper_repro scorers in this module. lm-eval integration is deferred — see docs/STANDALONE_PLAN.md."
    )


# --------------------------------------------------------------------------------------------------
# GPT-as-judge — BBC-QA / TiEBe "Correct %" (verbatim ``inference_common.GPTEvaluator``)
#
# Unlike the deterministic scorers above, this metric calls OpenAI, so it is NOT bit-exact (it depends on
# the served model version) — only method-equivalent. The API key is read from ``OPENAI_API_KEY`` (never
# stored in the repo); ``openai`` is imported lazily, so this section costs nothing unless a GPT-judged task
# (``bbcqa`` / ``tiebe``) is actually run.
# --------------------------------------------------------------------------------------------------
# Language-specific judge prompts + labels — VERBATIM from inference_common.GPTEvaluator.PROMPTS.
_JUDGE_PROMPTS = {
    "zh": {
        "system": (
            '請根據以下 "文章內容" 的資訊，判斷 "答案"是否有包含能夠正確回答"問題"的內容。\n'
            "評分規則：\n"
            '只要"答案"中 有出現足以回答"問題"的相關資訊，即使同時包含錯誤或多餘的描述，也一律判斷為 "正確"。\n'
            '只有在"答案"完全沒有包含能回答"問題"的內容時，才判斷為 "錯誤"。\n'
            '請嚴格只回答 "正確" 或 "錯誤"，不要輸出其他內容。'
        ),
        "user_template": '{{"文章內容": "{document}", "問題": "{question}", "答案": "{answer}"}}',
        "correct": "正確",
        "incorrect": "錯誤",
    },
    "en": {
        "system": (
            'Based on the information in the "Document", determine whether the "Answer" contains content that correctly answers the "Question".\n'
            "Scoring rules:\n"
            'As long as the "Answer" contains relevant information sufficient to answer the "Question", even if it also includes incorrect or extraneous descriptions, it should be judged as "Correct".\n'
            'Only when the "Answer" contains absolutely no content that answers the "Question" should it be judged as "Incorrect".\n'
            'Please strictly respond only with "Correct" or "Incorrect", and do not output any other content.'
        ),
        "user_template": '{{"Document": "{document}", "Question": "{question}", "Answer": "{answer}"}}',
        "correct": "Correct",
        "incorrect": "Incorrect",
    },
    "tiebe": {
        "system": (
            "I will provide a question, an expected answer, and the candidate's answer. "
            "Your task is to verify if the candidate's answer is correct. The expected answer "
            "is the ground truth, so if the candidate's answer contradicts the expected answer "
            "or refuses to answer, it is incorrect.\n"
            'Please strictly respond only with "Correct" or "Incorrect", and do not output any other content.'
        ),
        "user_template": 'Question: "{question}"\nExpected answer: "{document}"\nCandidate answer: "{answer}"',
        "correct": "Correct",
        "incorrect": "Incorrect",
    },
}


class GPTJudge:
    """GPT answer-correctness judge (verbatim protocol). Key from ``OPENAI_API_KEY`` (never stored).

    ``language``: ``"en"`` for BBC-QA (judges the answer against the source document), ``"tiebe"`` for TiEBe
    (judges against the expected answer), ``"zh"`` for the Chinese document-QA variant.
    """

    def __init__(
        self,
        language: str = "en",
        *,
        model: str = "gpt-4.1-mini",
        evaluation_repeats: int = 1,
        vote_rate: int = 50,
    ):
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "GPT-judged tasks (bbcqa / tiebe) need an OpenAI key: set OPENAI_API_KEY in your environment. "
                "The key is read from the environment only — it is never written into the repo."
            )
        try:
            import openai
        except ImportError as exc:  # pragma: no cover - optional dep
            raise ImportError("install the GPU/repro deps (`uv sync --group gpu`) — they include `openai`.") from exc

        self.model = model
        self.evaluation_repeats = evaluation_repeats
        self.vote_rate = vote_rate
        self.language = language if language in _JUDGE_PROMPTS else "zh"
        cfg = _JUDGE_PROMPTS[self.language]
        self.system_message = cfg["system"]
        self.user_template = cfg["user_template"]
        self.correct_label = cfg["correct"]
        self.incorrect_label = cfg["incorrect"]
        # Bounded retries + a per-request timeout so a degraded/offline network surfaces quickly instead of
        # hanging on the SDK's default (600s timeout, long backoff). Inference also runs a fast reachability
        # preflight before this point, so this is a second line of defense.
        self._client = openai.OpenAI(api_key=api_key, max_retries=2, timeout=60.0)

    def evaluate_single(self, document: str, question: str, answer: str) -> str:
        user_message = self.user_template.format(document=document, question=question, answer=answer)
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": self.system_message}, {"role": "user", "content": user_message}],
            temperature=0,
        )
        result = response.choices[0].message.content
        if result and result.strip() == self.correct_label:
            return self.correct_label
        return self.incorrect_label

    def evaluate(self, document: str, question: str, answer: str) -> str:
        """Majority-vote verdict over ``evaluation_repeats`` calls (verbatim ``evaluate_multiple_times``)."""
        results: List[str] = []
        for _ in range(self.evaluation_repeats):
            try:
                results.append(self.evaluate_single(document, question, answer))
            except Exception:
                results.append(self.incorrect_label)
        correct_count = results.count(self.correct_label)
        total = len(results)
        correct_rate = (correct_count / total) * 100 if total > 0 else 0
        return self.correct_label if correct_rate >= self.vote_rate else self.incorrect_label

    def correct_percent(self, items: List[Dict[str, str]]) -> float:
        """Correct % over items, each a dict with keys ``document`` / ``question`` / ``answer`` (prediction)."""
        if not items:
            return 0.0
        total = len(items)
        n_correct = 0
        for idx, it in enumerate(items, 1):
            if self.evaluate(it["document"], it["question"], it["answer"]) == self.correct_label:
                n_correct += 1
            if idx % 25 == 0 or idx == total:  # flushed progress so the judge phase isn't a silent black box
                print(f"[judge] {idx}/{total} ({self.model})", flush=True)
        return n_correct / total * 100
