"""Unit tests for per-sample extraction and run-config loading in the inference entry."""

from __future__ import annotations

import json

import pytest

from examples import load_run_config
from examples.inference import (
    _answer_list,
    _build_prompt,
    _load_documents,
    _question_text,
    _reference_text,
    _resolve_doc_id,
)


class _FakeTokenizer:
    """Minimal tokenizer stub that records the chat-template call."""

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        return "<CHAT>" + "|".join(m.get("content", "") for m in messages) + "</CHAT>"


def test_bbcqa_raw_shape_extraction():
    """BBC-QA raw file: ``{id: '...-QA', messages: [user, assistant]}`` (no explicit question/answer)."""
    row = {
        "id": "bbc_news_alltime-933-QA",
        "messages": [
            {"role": "user", "content": "What did the report find?"},
            {"role": "assistant", "content": "It found a 3% rise."},
        ],
    }
    assert _question_text(row) == "What did the report find?"  # messages[-2]
    assert _reference_text(row) == "It found a 3% rise."        # messages[-1]
    assert _resolve_doc_id(row["id"]) == "bbc_news_alltime-933"  # -QA stripped


def test_gsm8k_raw_shape_extraction():
    """GSM8K raw file: explicit ``question`` + ``answers`` list."""
    row = {"question": "2+2?", "answers": ["4"]}
    assert _question_text(row) == "2+2?"
    assert _answer_list(row) == ["4"]


def test_tiebe_messages_only_extraction():
    """TiEBe raw file: messages only — question = messages[-2], expected = messages[-1]."""
    row = {"messages": [{"role": "user", "content": "Who won?"}, {"role": "assistant", "content": "Team A"}]}
    assert _question_text(row) == "Who won?"
    assert _reference_text(row) == "Team A"


def test_question_text_single_message_fallback():
    """A lone user turn falls back to messages[-1] (matches _determine_question_text)."""
    row = {"messages": [{"role": "user", "content": "Solo question"}]}
    assert _question_text(row) == "Solo question"


def test_build_prompt_routing():
    """_build_prompt picks the source in priority order: explicit > raw text > messages > question."""
    tok = _FakeTokenizer()
    # explicit prompt wins, verbatim (tokenizer not consulted)
    assert _build_prompt({"prompt": "RAW PROMPT", "text": "x"}, tok, use_raw_text=True) == "RAW PROMPT"
    # use_raw_text -> the pre-formatted text field (e.g. GSM8K 8-shot)
    assert _build_prompt({"text": "8-shot block"}, tok, use_raw_text=True) == "8-shot block"
    # messages -> chat template
    msgs_prompt = _build_prompt({"messages": [{"role": "user", "content": "Q"}]}, tok, use_raw_text=False)
    assert msgs_prompt == "<CHAT>Q</CHAT>"
    # bare question -> wrapped as a user turn through the chat template
    q_prompt = _build_prompt({"question": "Just a Q"}, tok, use_raw_text=False)
    assert q_prompt == "<CHAT>Just a Q</CHAT>"


def test_load_documents_roundtrip(tmp_path):
    """BBC-QA judge documents load into {id: text}; an absent path is an empty map."""
    docs_file = tmp_path / "docs.jsonl"
    docs_file.write_text(
        json.dumps({"id": "bbc_news_alltime-933", "text": "Full article body."}) + "\n"
        + json.dumps({"id": "bbc_news_alltime-934", "text": "Another article."}) + "\n",
        encoding="utf-8",
    )
    docs = _load_documents(str(docs_file))
    assert docs["bbc_news_alltime-933"] == "Full article body."
    assert docs["bbc_news_alltime-934"] == "Another article."
    assert _load_documents(None) == {}


def test_build_prompt_raises_without_any_source():
    with pytest.raises(KeyError):
        _build_prompt({"unrelated": 1}, _FakeTokenizer(), use_raw_text=False)


def test_load_run_config_rejects_malformed_override(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("task: gsm8k\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_run_config(str(cfg), ["this_has_no_equals"])


def test_load_run_config_rejects_non_mapping(tmp_path):
    cfg = tmp_path / "list.yaml"
    cfg.write_text("- a\n- b\n", encoding="utf-8")  # a YAML sequence, not a mapping
    with pytest.raises(ValueError):
        load_run_config(str(cfg), [])


def test_answer_list_handles_list_stringified_scalar_and_missing():
    assert _answer_list({"answers": ["3", "4"]}) == ["3", "4"]          # real list
    assert _answer_list({"answers": "['a', 'b']"}) == ["a", "b"]        # Python-repr stringified list
    assert _answer_list({"answer": "42"}) == ["42"]                     # scalar string
    assert _answer_list({"answers": "not-a-list"}) == ["not-a-list"]    # unparseable string -> singleton
    assert _answer_list({}) == []                                       # missing -> empty


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("doc-1turnQA", "doc"),   # multi-turn QA suffix stripped first
        ("doc-QA", "doc"),        # single-turn QA suffix stripped
        ("doc-plain", "doc-plain"),  # unrelated suffix preserved
        (12345, 12345),           # non-str passthrough
    ],
)
def test_resolve_doc_id_suffix_handling(raw, expected):
    assert _resolve_doc_id(raw) == expected


def test_load_run_config_overrides_are_yaml_typed(tmp_path):
    cfg = tmp_path / "run.yaml"
    cfg.write_text("task: gsm8k\nbatch_size: 16\n", encoding="utf-8")
    out = load_run_config(str(cfg), ["batch_size=8", "task=bbcqa", "use_raw_text=true"])
    assert out["batch_size"] == 8 and isinstance(out["batch_size"], int)  # int, not "8"
    assert out["task"] == "bbcqa"                                          # str override
    assert out["use_raw_text"] is True                                    # bool, not "true"
