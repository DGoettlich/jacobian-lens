# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Activation interventions from fitted Jacobian-lens token directions."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel

if TYPE_CHECKING:
    from jlens.lens import JacobianLens


@dataclass(frozen=True)
class Steer:
    """Add one J-lens token direction to selected residuals."""

    token_id: int
    strength: float
    layers: Sequence[int] | None = None
    positions: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength):
            raise ValueError("strength must be finite")

    def delta(
        self,
        model: LensModel,
        lens: JacobianLens,
        layer: int,
        residual: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        token_vector = _token_direction(model, lens, residual, layer, self.token_id)
        return float(self.strength) * scale * token_vector


@dataclass(frozen=True)
class Swap:
    """Swap two J-lens token weights in selected residuals."""

    source_token_id: int
    target_token_id: int
    strength: float = 1.0
    layers: Sequence[int] | None = None
    positions: Sequence[int] | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be finite and non-negative")

    def delta(
        self,
        model: LensModel,
        lens: JacobianLens,
        layer: int,
        residual: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        del scale
        source_vector = _token_direction(
            model, lens, residual, layer, self.source_token_id
        )
        target_vector = _token_direction(
            model, lens, residual, layer, self.target_token_id
        )
        return float(self.strength) * _swap_delta(
            residual, source_vector, target_vector
        )


Intervention = Steer | Swap


@dataclass(frozen=True)
class InterventionResult:
    """Baseline and intervened logits for the same prompt."""

    input_ids: torch.Tensor
    baseline_logits: torch.Tensor
    intervened_logits: torch.Tensor


class _ActivationEditor:
    """Forward-hook context manager that adds precomputed residual deltas."""

    def __init__(self, blocks: Sequence[nn.Module], deltas: dict[int, torch.Tensor]):
        self._blocks = blocks
        self._deltas = deltas
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int) -> Callable[..., torch.Tensor | tuple]:
        def hook(module: nn.Module, inputs, output):
            hidden = output if torch.is_tensor(output) else output[0]
            if hidden.shape[0] != 1:
                raise ValueError("interventions require batch size 1")

            edited = hidden.clone()
            delta = self._deltas[layer].to(device=hidden.device, dtype=hidden.dtype)
            edited[0] = edited[0] + delta
            return edited if torch.is_tensor(output) else (edited, *output[1:])

        return hook

    def __enter__(self) -> _ActivationEditor:
        try:
            for layer in sorted(self._deltas):
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._make_hook(layer))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


def steer(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    token_id: int,
    strength: float,
    *,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    max_seq_len: int = 512,
) -> InterventionResult:
    """Run ``prompt`` while adding a J-lens direction for ``token_id``."""
    intervention = Steer(token_id, strength, layers, positions)
    return _run(model, lens, prompt, intervention, max_seq_len)


def swap(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    source_token_id: int,
    target_token_id: int,
    *,
    strength: float = 1.0,
    layers: Sequence[int] | None = None,
    positions: Sequence[int] | None = None,
    max_seq_len: int = 512,
) -> InterventionResult:
    """Run ``prompt`` while swapping two J-lens coordinates.

    At each edited residual ``h``, form ``V = [v_source, v_target]``, compute
    ``c = V^dagger h``, swap the two coordinates, and patch only the component
    of ``h`` inside ``span(V)``.
    """
    intervention = Swap(source_token_id, target_token_id, strength, layers, positions)
    return _run(model, lens, prompt, intervention, max_seq_len)


def _swap_delta(
    residual: torch.Tensor, source_vector: torch.Tensor, target_vector: torch.Tensor
) -> torch.Tensor:
    token_vectors = torch.stack([source_vector, target_vector], dim=1)

    # find the source/target weights that best reconstruct the current state.
    original_weights = torch.linalg.pinv(token_vectors) @ residual
    swapped_weights = original_weights.flip(0)

    # remove the original two-token mixture and add the swapped one.
    swap_edit = token_vectors @ (swapped_weights - original_weights)
    return swap_edit


def _token_direction(
    model: LensModel,
    lens: JacobianLens,
    residual: torch.Tensor,
    layer: int,
    token_id: int,
) -> torch.Tensor:
    """Unit transpose-row direction for ``token_id`` in layer-residual space."""
    weight = _unembedding_weight(model)
    if weight is not None:
        J = lens.jacobians[layer].to(device=residual.device, dtype=torch.float32)
        unembed = weight[int(token_id)].to(device=residual.device, dtype=torch.float32)
        # token score after j-lens: unembed dot (j @ residual).
        # equivalently: (j.t @ unembed) dot residual.
        token_vector = J.T @ unembed
        norm = token_vector.norm()
        if torch.isfinite(norm) and norm > 0:
            return (token_vector / norm).detach()

    with torch.enable_grad():
        h = residual.detach().float().requires_grad_(True)
        score = model.unembed(lens.transport(h, layer))[int(token_id)]
        (grad,) = torch.autograd.grad(score, h)

    norm = grad.norm()
    if not torch.isfinite(norm) or norm <= 0:
        raise ValueError(f"zero/non-finite direction for token {token_id}")
    return (grad / norm).detach()


def _unembedding_weight(model: LensModel) -> torch.Tensor | None:
    lm_head = getattr(model, "_lm_head", None) or getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    return weight.detach() if torch.is_tensor(weight) else None


def _run(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    intervention: Intervention,
    max_seq_len: int,
) -> InterventionResult:
    edit_layers = _layers(model, lens, intervention.layers)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    if input_ids.shape[0] != 1:
        raise ValueError("interventions require batch size 1")

    final_layer = model.n_layers - 1
    with torch.no_grad(), ActivationRecorder(
        model.layers, at=sorted({*edit_layers, final_layer})
    ) as recorder:
        model.forward(input_ids)
        activations = {layer: act.detach() for layer, act in recorder.activations.items()}

    baseline = model.unembed(activations[final_layer][0].float()).float().cpu()

    def delta_fn(layer: int, residual: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
        return intervention.delta(model, lens, layer, residual, scale)

    deltas = _deltas(activations, edit_layers, intervention.positions, delta_fn)

    with torch.no_grad(), _ActivationEditor(model.layers, deltas), ActivationRecorder(
        model.layers, at=[final_layer]
    ) as recorder:
        model.forward(input_ids)
        final_residual = recorder.activations[final_layer][0].detach().float()

    intervened = model.unembed(final_residual).float().cpu()
    return InterventionResult(input_ids, baseline, intervened)


def _deltas(
    activations: dict[int, torch.Tensor],
    layers: Sequence[int],
    positions: Sequence[int] | None,
    delta_fn: Callable[[int, torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[int, torch.Tensor]:
    seq_len = next(iter(activations.values())).shape[1]
    pos_list = _token_positions_to_edit(positions, seq_len)
    deltas = {layer: torch.zeros_like(activations[layer][0].float()) for layer in layers}

    for layer in layers:
        residuals = activations[layer][0].float()
        scale = residuals.norm(dim=-1).mean().clamp_min(1e-6)
        for pos in pos_list:
            deltas[layer][pos] += delta_fn(layer, residuals[pos], scale)

    return deltas


def _layers(
    model: LensModel,
    lens: JacobianLens,
    layers: Sequence[int] | None,
) -> list[int]:
    selected = lens.source_layers if layers is None else list(layers)
    out_of_range = sorted(layer for layer in selected if not 0 <= layer < model.n_layers)
    unknown = sorted(set(selected) - set(lens.source_layers))
    if out_of_range:
        raise ValueError(f"layers {out_of_range} out of range")
    if model.n_layers - 1 in selected:
        raise ValueError("editing the final layer is not supported")
    if unknown:
        raise ValueError(f"layers {unknown} not in lens.source_layers")
    return list(dict.fromkeys(int(layer) for layer in selected))


def _token_positions_to_edit(
    positions: Sequence[int] | None, seq_len: int
) -> list[int]:
    if positions is None:
        return list(range(seq_len))

    out = []
    for pos in positions:
        pos = int(pos)
        pos = pos + seq_len if pos < 0 else pos
        if not 0 <= pos < seq_len:
            raise ValueError(f"position {pos} out of range for seq_len={seq_len}")
        out.append(pos)
    return list(dict.fromkeys(out))
