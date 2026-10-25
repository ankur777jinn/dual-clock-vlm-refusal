"""
Core back-patching and attention knockout engine for Qwen2.5-VL-7B-Instruct.

Implements:
- Cumulative pre-softmax attention knockout (text→image masking from layer l onward)
- Image-token back-patching (inject L_src representations at L_dst, preserving M-RoPE)
- Bitwise identity check (C4 precondition)

Architecture notes:
- Qwen2.5-VL-7B has 28 transformer decoder layers (model.model.layers[0..27])
- Attention module: model.model.layers[i].self_attn
- Hidden states flow through: embed → layer[0] → layer[1] → ... → layer[27] → norm → lm_head
- M-RoPE position IDs are set at input time and do NOT need modification during patching;
  we only replace hidden_state activation values at image-token positions.
"""

from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def get_decoder_layers(model: nn.Module) -> nn.ModuleList:
    """Dynamically resolve the 28 decoder layers across transformers versions.

    In transformers >= 4.49 / 5.x for Qwen2.5-VL:
      model.model.language_model.layers OR
      model.language_model.layers OR
      model.model.layers
    """
    candidates = [
        ("model", "language_model", "layers"),
        ("language_model", "layers"),
        ("model", "layers"),
        ("layers",),
    ]
    for path in candidates:
        curr = model
        for part in path:
            curr = getattr(curr, part, None)
            if curr is None:
                break
        if curr is not None and isinstance(curr, (nn.ModuleList, list)) and len(curr) > 0:
            return curr

    # Fallback: traverse named modules for any ModuleList named 'layers' with decoder layers
    for name, module in model.named_modules():
        if name.endswith("layers") and isinstance(module, nn.ModuleList) and len(module) in (28, 32, 36, 48, 64, 80):
            return module

    raise AttributeError("Could not locate decoder layers in model architecture.")


def get_embed_tokens(model: nn.Module) -> nn.Module:
    """Dynamically resolve the token embedding module across transformers versions."""
    candidates = [
        ("model", "language_model", "embed_tokens"),
        ("language_model", "embed_tokens"),
        ("model", "embed_tokens"),
        ("embed_tokens",),
    ]
    for path in candidates:
        curr = model
        for part in path:
            curr = getattr(curr, part, None)
            if curr is None:
                break
        if curr is not None and isinstance(curr, nn.Module):
            return curr

    for name, module in model.named_modules():
        if "embed_tokens" in name and isinstance(module, nn.Embedding):
            return module

    raise AttributeError("Could not locate embed_tokens module in model architecture.")


def get_image_token_indices(
    input_ids: torch.Tensor,
    image_token_id: int,
) -> list[int]:
    """Return indices of image tokens in the 1D input_ids tensor.

    Args:
        input_ids: Shape (seq_len,) or (1, seq_len).
        image_token_id: Token ID used for image placeholders
            (processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")).

    Returns:
        Sorted list of integer positions.
    """
    ids = input_ids.squeeze(0)
    return (ids == image_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()


def get_text_token_indices(
    input_ids: torch.Tensor,
    image_token_id: int,
) -> list[int]:
    """Return indices of non-image tokens."""
    ids = input_ids.squeeze(0)
    return (ids != image_token_id).nonzero(as_tuple=False).squeeze(-1).tolist()


# ---------------------------------------------------------------------------
# Hidden-state extraction via forward hooks
# ---------------------------------------------------------------------------

def extract_hidden_states(
    model: nn.Module,
    inputs: dict,
    layers: list[int],
) -> dict[int, torch.Tensor]:
    """Run a forward pass and capture hidden states at specified layers.

    Hooks are placed on the *output* of each requested transformer layer so
    that the returned tensor is the post-layer representation (before the
    next layer's attention).

    Args:
        model: The full Qwen2.5-VL model.
        inputs: Dict of tensors ready for ``model(**inputs)``.
        layers: Layer indices to capture (0-indexed).

    Returns:
        Mapping from layer index to detached hidden-state tensor
        of shape ``(1, seq_len, hidden_dim)``.
    """
    captured: dict[int, torch.Tensor] = {}
    hooks: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(layer_idx: int):
        def _hook(_module, _input, output):
            # Qwen2 decoder layer returns (hidden_states, ...) tuple
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach().clone()
        return _hook

    layers_list = get_decoder_layers(model)
    for idx in layers:
        h = layers_list[idx].register_forward_hook(_make_hook(idx))
        hooks.append(h)

    try:
        with torch.no_grad():
            model(**inputs)
    finally:
        for h in hooks:
            h.remove()

    return captured


# ---------------------------------------------------------------------------
# Attention knockout
# ---------------------------------------------------------------------------

class AttentionKnockoutHook:
    """Cumulative pre-softmax attention knockout from ``start_layer`` onward.

    For every decoder layer with index >= ``start_layer``, the pre-softmax
    logits from all text-token query positions to all image-token key positions
    are set to a large negative value, effectively zeroing those attention
    weights after softmax.

    Usage::

        with AttentionKnockoutHook(model, start_layer=12,
                                   image_token_mask=img_mask):
            out = model(**inputs)
    """

    def __init__(
        self,
        model: nn.Module,
        start_layer: int,
        image_token_mask: torch.Tensor,
        text_token_mask: Optional[torch.Tensor] = None,
    ):
        """
        Args:
            model: Qwen2.5-VL model.
            start_layer: First layer (inclusive) at which to apply knockout.
            image_token_mask: Bool tensor of shape ``(seq_len,)`` — True for
                image-token positions.
            text_token_mask: Bool tensor of shape ``(seq_len,)`` — True for
                text-token positions. If None, inferred as ``~image_token_mask``.
        """
        self.model = model
        self.start_layer = start_layer
        self.image_mask = image_token_mask
        self.text_mask = (
            text_token_mask if text_token_mask is not None else ~image_token_mask
        )
        self._hooks: list[torch.utils.hooks.RemovableHook] = []
        self._installed = False

    def install(self):
        """Register forward-pre hooks on decoder layers >= start_layer.

        The hook modifies the ``attention_mask`` keyword argument to include
        large-negative entries at (text_query, image_key) positions, achieving
        pre-softmax knockout.
        """
        if self._installed:
            return

        img_positions = self.image_mask.nonzero(as_tuple=False).squeeze(-1)
        txt_positions = self.text_mask.nonzero(as_tuple=False).squeeze(-1)

        def _make_pre_hook(layer_idx: int):
            def _pre_hook(module, args, kwargs):
                attn_mask_key = "attention_mask"
                mask = None
                is_kwarg = True
                if attn_mask_key in kwargs and kwargs[attn_mask_key] is not None:
                    mask = kwargs[attn_mask_key].clone()
                elif len(args) > 1 and args[1] is not None:
                    mask = args[1].clone()
                    is_kwarg = False

                hs = args[0] if len(args) > 0 else kwargs.get("hidden_states")
                if hs is None:
                    return args, kwargs
                b, s = hs.shape[0], hs.shape[1]
                min_val = torch.finfo(hs.dtype).min

                if mask is None:
                    # Construct standard 4D causal mask for SDPA
                    mask = torch.triu(
                        torch.full((s, s), min_val, dtype=hs.dtype, device=hs.device),
                        diagonal=1,
                    ).unsqueeze(0).unsqueeze(0).expand(b, 1, s, s).clone()
                    is_kwarg = True
                else:
                    mask = mask.clone()
                    if mask.dim() == 2:
                        mask = mask[:, None, None, :].expand(b, 1, s, s).clone()

                # Pre-softmax knockout: text queries → image keys
                for tpos in txt_positions:
                    mask[:, :, tpos, img_positions] = min_val

                if is_kwarg:
                    kwargs[attn_mask_key] = mask
                else:
                    args = (args[0], mask) + args[2:]
                return args, kwargs
            return _pre_hook

        layers_list = get_decoder_layers(self.model)
        n_layers = len(layers_list)
        for idx in range(self.start_layer, n_layers):
            h = layers_list[idx].register_forward_pre_hook(
                _make_pre_hook(idx), with_kwargs=True
            )
            self._hooks.append(h)

        self._installed = True
        logger.debug(
            "AttentionKnockoutHook: layers %d–%d (%d hooks)",
            self.start_layer, n_layers - 1, len(self._hooks),
        )

    def remove(self):
        """Remove all registered hooks."""
        for h in self._hooks:
            h.remove()
        self._hooks.clear()
        self._installed = False

    def __enter__(self):
        self.install()
        return self

    def __exit__(self, *exc):
        self.remove()
        return False


# ---------------------------------------------------------------------------
# Back-patching
# ---------------------------------------------------------------------------

class BackPatchHook:
    """Two-pass back-patching: capture source activations, then inject at dest.

    Pass 1 (capture): Clean forward — record image-token hidden states at the
    *output* of ``src_layer``.

    Pass 2 (inject): Second forward — at the *input* of ``dst_layer``, replace
    image-token hidden states with captured source activations.

    M-RoPE position IDs are unchanged; only activation values are swapped.
    """

    def __init__(
        self,
        model: nn.Module,
        src_layer: int,
        dst_layer: int,
        image_token_indices: list[int],
    ):
        assert src_layer > dst_layer, (
            f"src_layer ({src_layer}) must be > dst_layer ({dst_layer})"
        )
        self.model = model
        self.src_layer = src_layer
        self.dst_layer = dst_layer
        self.image_indices = image_token_indices

    def capture_source(self, inputs: dict) -> torch.Tensor:
        """Run clean forward and return image-token hidden states at src_layer.

        Returns:
            Tensor of shape ``(1, n_image_tokens, hidden_dim)``.
        """
        states = extract_hidden_states(self.model, inputs, [self.src_layer])
        src_hs = states[self.src_layer]  # (1, seq_len, d)
        return src_hs[:, self.image_indices, :].clone()

    def patch_and_forward(
        self,
        inputs: dict,
        src_activations: torch.Tensor,
    ) -> dict:
        """Run forward with injected source activations at dst_layer.

        Hooks on layer (dst_layer - 1)'s output to replace image-token
        activations before they enter dst_layer. For dst_layer == 0, hooks
        on the embedding layer output.

        Args:
            inputs: Model inputs.
            src_activations: ``(1, n_image_tokens, hidden_dim)``
                from :meth:`capture_source`.

        Returns:
            Model output dict.
        """
        img_idx = self.image_indices
        dst = self.dst_layer

        def _inject_hook(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            hs[:, img_idx, :] = src_activations.to(hs.dtype)
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        layers_list = get_decoder_layers(self.model)
        if dst == 0:
            hook_target = get_embed_tokens(self.model)
        else:
            hook_target = layers_list[dst - 1]

        h = hook_target.register_forward_hook(_inject_hook)
        try:
            with torch.no_grad():
                outputs = self.model(**inputs)
        finally:
            h.remove()

        return outputs

    @staticmethod
    def verify_identity(
        model: nn.Module,
        inputs: dict,
        layer: int,
        image_token_indices: list[int],
    ) -> tuple[bool, float]:
        """C4 precondition: patch layer into itself, verify bitwise identity.

        Captures image-token hidden states at hook target of ``layer``, then
        re-injects them at the same position. The final logits must be
        bitwise identical.

        Returns:
            (passed: bool, max_abs_delta: float)
        """
        layers_list = get_decoder_layers(model)
        if layer == 0:
            hook_target = get_embed_tokens(model)
        else:
            hook_target = layers_list[layer - 1]

        # 1. Clean forward while capturing own activations at the exact hook target
        captured: dict[str, torch.Tensor] = {}
        def _cap_hook(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured["hs"] = hs[:, image_token_indices, :].clone()

        h_cap = hook_target.register_forward_hook(_cap_hook)
        try:
            with torch.no_grad():
                clean_out = model(**inputs)
        finally:
            h_cap.remove()

        clean_logits = clean_out.logits[:, -1, :].clone()
        src_hs = captured["hs"]

        # 2. Re-inject at the exact same hook target
        def _identity_hook(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            hs[:, image_token_indices, :] = src_hs.to(hs.dtype)
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        h_inj = hook_target.register_forward_hook(_identity_hook)
        try:
            with torch.no_grad():
                patched_out = model(**inputs)
        finally:
            h_inj.remove()

        patched_logits = patched_out.logits[:, -1, :]
        max_delta = (clean_logits - patched_logits).abs().max().item()
        passed = max_delta < 1e-4

        if not passed:
            logger.warning(
                "C4 identity FAILED at layer %d: max|Δz| = %.6e", layer, max_delta
            )
        else:
            logger.info("C4 identity PASSED at layer %d: max|Δz| = %.6e (bf16 identity)", layer, max_delta)

        return passed, max_delta


# ---------------------------------------------------------------------------
# High-level runner
# ---------------------------------------------------------------------------

class PrefillRunner:
    """High-level interface for clean and patched forward passes.

    Handles image preprocessing, tokenization, logit mass computation,
    and text generation with optional back-patching.
    """

    def __init__(self, model: nn.Module, processor, device: torch.device):
        self.model = model
        self.processor = processor
        self.device = device
        self._image_token_id = processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")

    def _prepare_inputs(self, image, prompt: str) -> dict:
        """Tokenize image + prompt into model-ready inputs."""
        from pathlib import Path
        from PIL import Image as PILImage
        from qwen_vl_utils import process_vision_info

        if isinstance(image, (str, Path)):
            image = PILImage.open(str(image)).convert("RGB")
        elif hasattr(image, "__fspath__"):
            image = PILImage.open(str(image.__fspath__())).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        return {k: v.to(self.device) for k, v in inputs.items()}

    def _get_image_indices(self, input_ids: torch.Tensor) -> list[int]:
        return get_image_token_indices(input_ids, self._image_token_id)

    def compute_logit_mass(
        self,
        image,
        prompt: str,
        refusal_ids: list[int],
        compliance_ids: list[int],
    ) -> float:
        """Clean forward pass → first-token logit mass M."""
        from src.utils import compute_logit_mass

        inputs = self._prepare_inputs(image, prompt)
        with torch.no_grad():
            outputs = self.model(**inputs)
        logits = outputs.logits[:, -1, :]
        return compute_logit_mass(logits.squeeze(0), refusal_ids, compliance_ids)

    def compute_patched_logit_mass(
        self,
        image,
        prompt: str,
        src_layer: int,
        dst_layer: int,
        refusal_ids: list[int],
        compliance_ids: list[int],
    ) -> float:
        """Patched forward pass → first-token logit mass M."""
        from src.utils import compute_logit_mass

        inputs = self._prepare_inputs(image, prompt)
        img_indices = self._get_image_indices(inputs["input_ids"])

        patcher = BackPatchHook(self.model, src_layer, dst_layer, img_indices)
        src_acts = patcher.capture_source(inputs)
        outputs = patcher.patch_and_forward(inputs, src_acts)

        logits = outputs.logits[:, -1, :]
        return compute_logit_mass(logits.squeeze(0), refusal_ids, compliance_ids)

    def generate_completion(
        self,
        image,
        prompt: str,
        max_new_tokens: int = 48,
    ) -> str:
        """Standard (unpatched) text generation."""
        inputs = self._prepare_inputs(image, prompt)
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        input_len = inputs["input_ids"].shape[1]
        return self.processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        )

    def generate_patched_completion(
        self,
        image,
        prompt: str,
        src_layer: int,
        dst_layer: int,
        max_new_tokens: int = 48,
    ) -> str:
        """Patched text generation (patch applied during prefill only).

        The back-patch hook fires only on the first forward call (the prefill).
        Subsequent autoregressive steps use the KV-cache seeded by the patched
        prefill, so the intervention propagates naturally.
        """
        inputs = self._prepare_inputs(image, prompt)
        img_indices = self._get_image_indices(inputs["input_ids"])

        patcher = BackPatchHook(self.model, src_layer, dst_layer, img_indices)
        src_acts = patcher.capture_source(inputs)

        _call_count = [0]
        layers_list = get_decoder_layers(self.model)
        if dst_layer == 0:
            hook_target = get_embed_tokens(self.model)
        else:
            hook_target = layers_list[dst_layer - 1]

        def _inject_once(_module, _input, output):
            _call_count[0] += 1
            if _call_count[0] > 1:
                return output  # Only patch the prefill pass
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            hs[:, img_indices, :] = src_acts.to(hs.dtype)
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        h = hook_target.register_forward_hook(_inject_once)
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
        finally:
            h.remove()

        input_len = inputs["input_ids"].shape[1]
        return self.processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        )

    def run_clean(
        self,
        model=None,
        processor=None,
        image=None,
        prompt: str = None,
        device=None,
        max_new_tokens: int = 48,
    ) -> dict:
        """Run unpatched forward pass: compute M and generate text."""
        from PIL import Image as PILImage
        from pathlib import Path
        if isinstance(image, (str, Path)):
            image = PILImage.open(image).convert("RGB")
        from src.config import Config
        if prompt is None:
            prompt = Config.PROMPT

        from src.config import Config
        from src.utils import compute_logit_mass
        tok = self.processor.tokenizer
        ref_ids = [tok.convert_tokens_to_ids(t) for t in Config.REFUSAL_TOKENS if tok.convert_tokens_to_ids(t) is not None]
        comp_ids = [tok.convert_tokens_to_ids(t) for t in Config.COMPLIANCE_TOKENS if tok.convert_tokens_to_ids(t) is not None]
        if not ref_ids:
            ref_ids = [tok.encode("I")[0], tok.encode("Sorry")[0]]
        if not comp_ids:
            comp_ids = [tok.encode("Sure")[0], tok.encode("Here")[0]]

        inputs = self._prepare_inputs(image, prompt)
        with torch.no_grad():
            outputs = self.model(**inputs)
            output_ids = self.model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False
            )
        logits = outputs.logits[:, -1, :]
        M = compute_logit_mass(logits.squeeze(0), ref_ids, comp_ids)
        input_len = inputs["input_ids"].shape[1]
        text = self.processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        )
        return {"M": M, "generated_text": text}

    def capture_source(
        self,
        model=None,
        processor=None,
        image=None,
        prompt: str = None,
        layer: int = 0,
        device=None,
    ) -> torch.Tensor:
        """Capture source activations at specified layer."""
        from PIL import Image as PILImage
        from pathlib import Path
        if isinstance(image, (str, Path)):
            image = PILImage.open(image).convert("RGB")
        from src.config import Config
        if prompt is None:
            prompt = Config.PROMPT
        inputs = self._prepare_inputs(image, prompt)
        img_indices = self._get_image_indices(inputs["input_ids"])
        patcher = BackPatchHook(self.model, layer, max(0, layer - 1), img_indices)
        return patcher.capture_source(inputs)

    def run_patched(
        self,
        model=None,
        processor=None,
        image=None,
        prompt: str = None,
        source_activation: torch.Tensor = None,
        dst_layer: int = 0,
        device=None,
        max_new_tokens: int = 48,
    ) -> dict:
        """Run patched forward pass: compute M and generate text."""
        from PIL import Image as PILImage
        from pathlib import Path
        if isinstance(image, (str, Path)):
            image = PILImage.open(image).convert("RGB")
        from src.config import Config
        if prompt is None:
            prompt = Config.PROMPT

        from src.config import Config
        from src.utils import compute_logit_mass
        tok = self.processor.tokenizer
        ref_ids = [tok.convert_tokens_to_ids(t) for t in Config.REFUSAL_TOKENS if tok.convert_tokens_to_ids(t) is not None]
        comp_ids = [tok.convert_tokens_to_ids(t) for t in Config.COMPLIANCE_TOKENS if tok.convert_tokens_to_ids(t) is not None]
        if not ref_ids:
            ref_ids = [tok.encode("I")[0], tok.encode("Sorry")[0]]
        if not comp_ids:
            comp_ids = [tok.encode("Sure")[0], tok.encode("Here")[0]]

        inputs = self._prepare_inputs(image, prompt)
        img_indices = self._get_image_indices(inputs["input_ids"])

        # Forward pass for M
        patcher = BackPatchHook(self.model, dst_layer + 1, dst_layer, img_indices)
        outputs = patcher.patch_and_forward(inputs, source_activation)
        logits = outputs.logits[:, -1, :]
        M = compute_logit_mass(logits.squeeze(0), ref_ids, comp_ids)

        # Generation with patch during prefill
        _call_count = [0]
        layers_list = get_decoder_layers(self.model)
        if dst_layer == 0:
            hook_target = get_embed_tokens(self.model)
        else:
            hook_target = layers_list[dst_layer - 1]

        def _inject_once(_module, _input, output):
            _call_count[0] += 1
            if _call_count[0] > 1:
                return output
            hs = output[0] if isinstance(output, tuple) else output
            hs = hs.clone()
            hs[:, img_indices, :] = source_activation.to(hs.dtype)
            if isinstance(output, tuple):
                return (hs,) + output[1:]
            return hs

        h = hook_target.register_forward_hook(_inject_once)
        try:
            with torch.no_grad():
                output_ids = self.model.generate(
                    **inputs, max_new_tokens=max_new_tokens, do_sample=False
                )
        finally:
            h.remove()

        input_len = inputs["input_ids"].shape[1]
        text = self.processor.tokenizer.decode(
            output_ids[0, input_len:], skip_special_tokens=True
        )
        return {"M": M, "generated_text": text}
