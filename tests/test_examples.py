"""Smoke test: GaDRA trains with a plain ``Trainer``, saves, reloads, and generates."""

from __future__ import annotations

import torch

import gadra  # noqa: F401  (registers the "gadra" method)
from gadra import GaDRAConfig
from gadra.layer import GaDRALinear


def _tiny_llama():
    from transformers import LlamaConfig, LlamaForCausalLM

    cfg = LlamaConfig(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=4,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    torch.manual_seed(0)
    return LlamaForCausalLM(cfg)


class _RandomLM(torch.utils.data.Dataset):
    def __init__(self, n=8, seq=16, vocab=64):
        self.n, self.seq, self.vocab = n, seq, vocab

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        gen = torch.Generator().manual_seed(idx)
        ids = torch.randint(0, self.vocab, (self.seq,), generator=gen)
        return {"input_ids": ids, "labels": ids.clone()}


def test_standard_trainer_trains_gadra_then_generates(tmp_path):
    from peft import PeftModel, get_peft_model
    from transformers import Trainer, TrainingArguments, default_data_collator

    base = _tiny_llama()
    model = get_peft_model(base, GaDRAConfig(r=8, gate="soft", task_type="CAUSAL_LM"))

    gate_before = None
    for m in model.modules():
        if isinstance(m, GaDRALinear):
            gate_before = m.gadra_gate["default"].weight.detach().clone()
            break

    args = TrainingArguments(
        output_dir=str(tmp_path / "run"),
        max_steps=3,
        per_device_train_batch_size=2,
        learning_rate=1e-2,
        report_to=[],
        logging_steps=1,
        save_strategy="no",
        use_cpu=True,
        bf16=False,
        fp16=False,
        seed=42,
    )
    trainer = Trainer(model=model, args=args, train_dataset=_RandomLM(), data_collator=default_data_collator)
    result = trainer.train()
    assert result.training_loss is not None and result.training_loss > 0

    gate_after = None
    for m in model.modules():
        if isinstance(m, GaDRALinear):
            gate_after = m.gadra_gate["default"].weight.detach()
            break
    assert not torch.allclose(gate_before, gate_after)

    model.save_pretrained(str(tmp_path / "adapter"))
    reloaded = PeftModel.from_pretrained(_tiny_llama(), str(tmp_path / "adapter"))
    reloaded.eval()
    prompt = torch.randint(0, 64, (1, 4))
    with torch.no_grad():
        out = reloaded.generate(input_ids=prompt, max_new_tokens=3, do_sample=False)
    assert out.shape[1] == 4 + 3
