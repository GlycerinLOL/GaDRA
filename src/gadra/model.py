"""``GaDRAModel`` — the GaDRA tuner (``BaseTuner``) that injects ``GaDRALinear`` by name-matching.

Dispatched via ``config.peft_type == PeftType.GADRA`` once :func:`gadra._register.register_gadra`
has run. No merging, no quantization dispatch, no weighted-adapter combination.
"""

from __future__ import annotations

import warnings

import torch
from torch import nn

from peft.tuners.tuners_utils import BaseTuner, BaseTunerLayer, _ExcludedModule, check_target_module_exists
from peft.utils.other import AuxiliaryTrainingWrapper, get_pattern_key

from .config import DEFAULT_TARGET_MODULES
from .layer import GaDRALinear, GaDRAParallelBlock


class GaDRAModel(BaseTuner):
    """Creates a GaDRA model from a pretrained transformers model.

    Attributes:
        model (`torch.nn.Module`): the model to be adapted.
        peft_config (`GaDRAConfig`): the GaDRA configuration.
    """

    prefix: str = "gadra_"

    def __init__(self, model, config, adapter_name, low_cpu_mem_usage: bool = False) -> None:
        super().__init__(model, config, adapter_name, low_cpu_mem_usage=low_cpu_mem_usage)

    @staticmethod
    def _check_target_module_exists(gadra_config, key):
        matched = check_target_module_exists(gadra_config, key)
        if matched or isinstance(matched, _ExcludedModule):
            return matched
        parallel = getattr(gadra_config, "parallel_modules", None)
        if parallel:
            names = [parallel] if isinstance(parallel, str) else parallel
            if key in names or any(key.endswith(f".{name}") for name in names):
                return True
        return matched

    @staticmethod
    def _prepare_adapter_config(peft_config, model_config):
        # GaDRA applies its MLP-projection defaults by name; it does not consult model_config.
        if peft_config.target_modules is None:
            peft_config.target_modules = list(DEFAULT_TARGET_MODULES)
        return peft_config

    def _create_and_replace(
        self,
        gadra_config,
        adapter_name,
        target,
        target_name,
        parent,
        current_key,
    ) -> None:
        if current_key is None:
            raise ValueError("Current Key shouldn't be `None`")

        # optional peft-standard per-module heterogeneity
        r_key = get_pattern_key(gadra_config.rank_pattern.keys(), current_key)
        alpha_key = get_pattern_key(gadra_config.alpha_pattern.keys(), current_key)
        r = gadra_config.rank_pattern.get(r_key, gadra_config.r)
        alpha = gadra_config.alpha_pattern.get(alpha_key, gadra_config.lora_alpha)

        kwargs = {
            "r": r,
            "lora_alpha": alpha,
            "lora_dropout": gadra_config.lora_dropout,
            "router_conditioning": gadra_config.router_conditioning,
            "gate": gadra_config.gate,
            "gamma_threshold": gadra_config.gamma_threshold,
            "tau": gadra_config.tau,
            "gate_bias_init": gadra_config.gate_bias_init,
            "init_lora_weights": gadra_config.init_lora_weights,
        }

        if isinstance(target, (GaDRALinear, GaDRAParallelBlock)):
            target.update_layer(adapter_name, **kwargs)
        else:
            new_module = self._create_new_module(gadra_config, adapter_name, target, current_key, **kwargs)
            if adapter_name not in self.active_adapters:
                new_module.requires_grad_(False)
            self._replace_module(parent, target_name, new_module, target)

    def _create_new_module(self, gadra_config, adapter_name, target, current_key, **kwargs):
        base_layer = target.get_base_layer() if isinstance(target, BaseTunerLayer) else target
        if isinstance(base_layer, nn.Linear):
            return GaDRALinear(target, adapter_name, **kwargs)
        # non-Linear match -> whole-block parallel adapter (e.g. an MLP/MoE block)
        parallel = getattr(gadra_config, "parallel_modules", None)
        if parallel is not None:
            names = [parallel] if isinstance(parallel, str) else parallel
            if not (current_key in names or any(current_key.endswith(f".{name}") for name in names)):
                raise ValueError(
                    f"GaDRA matched non-Linear module {current_key!r} ({type(base_layer).__name__}); add it to "
                    "`parallel_modules` to wrap it as a block-parallel adapter, or restrict `target_modules` to "
                    "linear projections."
                )
        return GaDRAParallelBlock(target, adapter_name, hidden_size=self._resolve_hidden_size(base_layer), **kwargs)

    def _resolve_hidden_size(self, base_layer) -> int:
        model_config = getattr(getattr(self, "model", None), "config", None)
        h = getattr(model_config, "hidden_size", None)
        if isinstance(h, int) and h > 0:
            return h
        for module in base_layer.modules():
            if isinstance(module, nn.Linear):
                return module.in_features
        raise ValueError(
            "could not resolve hidden_size for a block-parallel target: the base model exposes no "
            "config.hidden_size and the block has no nn.Linear to probe."
        )

    def _replace_module(self, parent, child_name, new_module, child):
        setattr(parent, child_name, new_module)
        if hasattr(child, "base_layer"):
            child = child.base_layer

        # only the new gadra_* submodules need device placement (the base layer is reused)
        meta = torch.device("meta")
        for name, module in new_module.named_modules():
            if self.prefix in name:
                if hasattr(child, "weight"):
                    weight = child.weight
                else:
                    weight = next(child.parameters())
                if not any(p.device == meta for p in module.parameters()):
                    module.to(weight.device)

    def _mark_only_adapters_as_trainable(self, model: nn.Module) -> None:
        for n, p in model.named_parameters():
            if self.prefix not in n:
                p.requires_grad = False

    def _set_adapter_layers(self, enabled: bool = True) -> None:
        for module in self.model.modules():
            if isinstance(module, (BaseTunerLayer, AuxiliaryTrainingWrapper)):
                module.enable_adapters(enabled)

    def enable_adapter_layers(self) -> None:
        """Enable all GaDRA adapters."""
        self._set_adapter_layers(enabled=True)

    def disable_adapter_layers(self) -> None:
        """Disable all GaDRA adapters (model output reverts to the base model)."""
        self._set_adapter_layers(enabled=False)

    def merge_and_unload(self, *args, **kwargs):
        raise NotImplementedError(
            "GaDRA is non-mergeable (gamma is a per-token, input-dependent gate); merge_and_unload() "
            "is not supported. Keep the adapter attached and run inference with PeftModel."
        )

    def merge_adapter(self, *args, **kwargs):
        raise NotImplementedError("GaDRA is non-mergeable; merge_adapter() is not supported.")

    def unload(self, *args, **kwargs):
        raise NotImplementedError(
            "GaDRA.unload() is not implemented; discard the PeftModel wrapper to recover the base model."
        )

    def delete_adapter(self, *args, **kwargs):
        raise NotImplementedError("GaDRA.delete_adapter() is not implemented.")

    def add_weighted_adapter(self, *args, **kwargs):
        raise NotImplementedError("GaDRA is non-mergeable; add_weighted_adapter() is not supported.")

    def set_adapter(self, adapter_name: str | list[str], inference_mode: bool = False) -> None:
        """Set the active adapter(s) and mark them trainable."""
        for module in self.model.modules():
            if isinstance(module, (GaDRALinear, GaDRAParallelBlock)):
                if module.merged:
                    warnings.warn("Adapter cannot be set when the model is merged. Unmerging the model first.")
                    module.unmerge()
                module.set_adapter(adapter_name, inference_mode=inference_mode)
        self.active_adapter = adapter_name

    def __getattr__(self, name: str):
        """Forward missing attributes to the wrapped model."""
        try:
            return super().__getattr__(name)
        except AttributeError:
            if name == "model":  # avoid recursion before init
                raise
            return getattr(self.model, name)
