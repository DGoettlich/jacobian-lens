# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""activation edits from fitted jacobian-lens token directions."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch
from torch import nn

from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel

if TYPE_CHECKING:
    from jlens.lens import JacobianLens


LayerSpec = int | Sequence[int] | None
PositionSpec = int | Sequence[int] | None
SteerSpec = tuple[int, float, LayerSpec, PositionSpec]


@dataclass(frozen=True)
class Steer:
    """add one or more token pushes to selected residuals."""

    specs: Sequence[SteerSpec]
    cascading: bool = False

    def __post_init__(self) -> None:
        """check that every token push has valid shape."""
        if not self.specs:
            raise ValueError("specs must be non-empty")
        for token_id, strength, layers, positions in self.specs:
            int(token_id)
            if not math.isfinite(float(strength)):
                raise ValueError("all strengths must be finite")
            _normalize_layer_spec(layers)
            if positions is not None and not isinstance(positions, int):
                [int(pos) for pos in positions]

    def delta(
        self,
        model: LensModel,
        lens: JacobianLens,
        layer: int,
        position: int,
        seq_len: int,
        residual: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """return the steer edit at one layer and token position."""
        delta = torch.zeros_like(residual, dtype=torch.float32)
        for token_id, strength in _steers_at_this_layer_and_position(
            self, layer, position, seq_len
        ):
            token_vector = _token_direction(model, lens, residual, layer, token_id)
            delta = delta + strength * scale * token_vector
        return delta


@dataclass(frozen=True)
class Swap:
    """move source-token weight onto the target token."""

    source_token_id: int
    target_token_id: int
    strength: float = 1.0
    layers: Sequence[int] | None = None
    positions: Sequence[int] | None = None
    cascading: bool = False

    def __post_init__(self) -> None:
        """check that the swap strength can be used."""
        if not math.isfinite(self.strength) or self.strength < 0:
            raise ValueError("strength must be finite and non-negative")

    def delta(
        self,
        model: LensModel,
        lens: JacobianLens,
        layer: int,
        position: int,
        seq_len: int,
        residual: torch.Tensor,
        scale: torch.Tensor,
    ) -> torch.Tensor:
        """return the swap edit at one residual."""
        del position, seq_len, scale
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


class _ActivationEditor:
    """edit the live stream during a forward pass."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        model: LensModel,
        lens: JacobianLens,
        intervention: Intervention,
        editable_layers: Sequence[int],
    ):
        self._blocks = blocks
        self._model = model
        self._lens = lens
        self._intervention = intervention
        self._editable_layers = editable_layers
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int) -> Callable[..., torch.Tensor | tuple]:
        """build the hook that edits this layer as it runs."""
        def hook(module: nn.Module, inputs, output):
            """apply the edit to this layer output."""
            hidden = output if torch.is_tensor(output) else output[0]
            if hidden.shape[0] != 1:
                raise ValueError("interventions require batch size 1")

            edited = hidden.clone()
            # use the stream as it exists now, including earlier edited layers.
            residuals = hidden[0].float()
            scale = residuals.norm(dim=-1).mean().clamp_min(1e-6)

            for pos in _token_positions_for_intervention(
                self._intervention, layer, hidden.shape[1]
            ):
                delta = self._intervention.delta(
                    self._model,
                    self._lens,
                    layer,
                    pos,
                    hidden.shape[1],
                    residuals[pos],
                    scale,
                )
                edited[0, pos] = edited[0, pos] + delta.to(
                    device=edited.device,
                    dtype=edited.dtype,
                )

            return edited if torch.is_tensor(output) else (edited, *output[1:])

        return hook

    def __enter__(self) -> _ActivationEditor:
        """install hooks on the edited layers."""
        try:
            for layer in sorted(self._editable_layers):
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
        """remove every installed hook."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


class _PrecomputedActivationEditor:
    """replay fixed edits during a forward pass."""

    def __init__(self, blocks: Sequence[nn.Module], deltas: dict[int, torch.Tensor]):
        self._blocks = blocks
        self._deltas = deltas
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, layer: int) -> Callable[..., torch.Tensor | tuple]:
        """build the hook that adds a fixed layer delta."""
        def hook(module: nn.Module, inputs, output):
            """add the precomputed edit to this layer output."""
            hidden = output if torch.is_tensor(output) else output[0]
            if hidden.shape[0] != 1:
                raise ValueError("interventions require batch size 1")

            edited = hidden.clone()
            delta = self._deltas[layer].to(device=hidden.device, dtype=hidden.dtype)
            edited[0] = edited[0] + delta
            return edited if torch.is_tensor(output) else (edited, *output[1:])

        return hook

    def __enter__(self) -> _PrecomputedActivationEditor:
        """install hooks on every layer with a fixed delta."""
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
        """remove every installed hook."""
        for handle in self._handles:
            handle.remove()
        self._handles = []


def _swap_delta(
    residual: torch.Tensor, source_vector: torch.Tensor, target_vector: torch.Tensor
) -> torch.Tensor:
    """return the vector to add for a source-to-target swap."""
    token_vectors = torch.stack([source_vector, target_vector], dim=1)

    # find the source/target weights that best reconstruct the current state.
    original_weights = torch.linalg.pinv(token_vectors) @ residual
    source_weight = original_weights[0]
    target_weight = original_weights[1]

    # if target is already stronger than source, another flip would move the
    # stream back toward source instead of doing the requested one-way swap.
    if target_weight >= source_weight:
        return torch.zeros_like(residual)

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
    """return the unit direction for one token at one layer."""
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
    """return the lm-head weight when the model exposes it."""
    lm_head = getattr(model, "_lm_head", None) or getattr(model, "lm_head", None)
    weight = getattr(lm_head, "weight", None)
    return weight.detach() if torch.is_tensor(weight) else None


def _normalize_layer_spec(layers: LayerSpec) -> list[int] | None:
    """turn a layer spec into a list, or none for all fitted layers."""
    if layers is None:
        return None
    if isinstance(layers, int):
        return [int(layers)]
    return [int(layer) for layer in layers]


def _normalize_position_spec(positions: PositionSpec, seq_len: int) -> list[int]:
    """turn a position spec into concrete token positions."""
    if positions is None:
        return list(range(seq_len))
    if isinstance(positions, int):
        positions = [positions]
    return _token_positions_to_edit(positions, seq_len)


def _steers_at_this_layer_and_position(
    steer: Steer,
    layer: int,
    position: int,
    seq_len: int,
) -> list[tuple[int, float]]:
    """return the token pushes that apply at one layer and token position."""
    out = []
    for token_id, strength, layers, positions in steer.specs:
        layer_list = _normalize_layer_spec(layers)

        # skip this spec if it names layers and this is not one of them.
        if layer_list is not None and layer not in layer_list:
            continue

        position_list = _normalize_position_spec(positions, seq_len)

        # skip this spec if this token position is not one of its positions.
        if position not in position_list:
            continue

        out.append((int(token_id), float(strength)))

    return out


def _token_positions_for_intervention(
    intervention: Intervention,
    layer: int,
    seq_len: int,
) -> list[int]:
    """return the token positions edited at one layer."""
    if isinstance(intervention, Swap):
        return _token_positions_to_edit(intervention.positions, seq_len)

    positions = []
    for _, _, layers, spec_positions in intervention.specs:
        layer_list = _normalize_layer_spec(layers)
        if layer_list is not None and layer not in layer_list:
            continue
        positions.extend(_normalize_position_spec(spec_positions, seq_len))

    return list(dict.fromkeys(positions))


def _readout_positions_for_intervention(
    intervention: Intervention, seq_len: int
) -> list[int]:
    """return the token positions to include in returned logits."""
    if isinstance(intervention, Swap):
        return _token_positions_to_edit(intervention.positions, seq_len)

    positions = []
    for _, _, _, spec_positions in intervention.specs:
        positions.extend(_normalize_position_spec(spec_positions, seq_len))
    return list(dict.fromkeys(positions))


def run_intervention(
    model: LensModel,
    lens: JacobianLens,
    prompt: str,
    intervention: Intervention,
    max_seq_len: int,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
    """run one forward pass with an intervention installed."""
    edit_layers = _editable_layers(model, lens, intervention)
    input_ids = model.encode(prompt, max_length=max_seq_len)
    if input_ids.shape[0] != 1:
        raise ValueError("interventions require batch size 1")

    final_layer = model.n_layers - 1
    with torch.no_grad(), _get_editing_context(
        model, lens, input_ids, intervention, edit_layers
    ), ActivationRecorder(
        model.layers, at=sorted({*edit_layers, final_layer})
    ) as recorder:
        model.forward(input_ids)
        activations = {layer: act.detach() for layer, act in recorder.activations.items()}

    lens_logits, model_logits = lens._readout_activations(
        model,
        activations,
        edit_layers,
        _readout_positions_for_intervention(intervention, input_ids.shape[1]),
        use_jacobian=True,
    )
    return lens_logits, model_logits, input_ids


@contextmanager
def _get_editing_context(
    model: LensModel,
    lens: JacobianLens,
    input_ids: torch.Tensor,
    intervention: Intervention,
    edit_layers: Sequence[int],
) -> Iterator[None]:
    """choose the live or precomputed edit path."""
    if intervention.cascading:
        with _ActivationEditor(model.layers, model, lens, intervention, edit_layers):
            yield
    else:
        deltas = _get_baseline_deltas(model, lens, input_ids, intervention, edit_layers)
        with _PrecomputedActivationEditor(model.layers, deltas):
            yield


def _get_baseline_deltas(
    model: LensModel,
    lens: JacobianLens,
    input_ids: torch.Tensor,
    intervention: Intervention,
    edit_layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """compute fixed edits from a clean forward pass."""
    # for non-cascading runs, compute every edit from the clean stream once.
    with ActivationRecorder(model.layers, at=edit_layers) as recorder:
        model.forward(input_ids)
        activations = {
            layer: recorder.activations[layer].detach() for layer in edit_layers
        }

    seq_len = input_ids.shape[1]
    deltas = {
        layer: torch.zeros_like(activations[layer][0].float()) for layer in edit_layers
    }

    for layer in edit_layers:
        residuals = activations[layer][0].float()
        scale = residuals.norm(dim=-1).mean().clamp_min(1e-6)
        for pos in _token_positions_for_intervention(intervention, layer, seq_len):
            deltas[layer][pos] += intervention.delta(
                model,
                lens,
                layer,
                pos,
                seq_len,
                residuals[pos],
                scale,
            )

    return deltas


def _editable_layers(
    model: LensModel,
    lens: JacobianLens,
    intervention: Intervention,
) -> list[int]:
    """return the layers that may be edited."""
    if isinstance(intervention, Steer):
        selected = []
        for _, _, layers, _ in intervention.specs:
            layer_list = _normalize_layer_spec(layers)
            selected.extend(lens.source_layers if layer_list is None else layer_list)
    else:
        selected = (
            lens.source_layers
            if intervention.layers is None
            else list(intervention.layers)
        )

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
    """resolve token positions, including negative positions."""
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
