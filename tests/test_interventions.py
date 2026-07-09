# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import torch

from jlens.fitting import fit
from jlens.hooks import ActivationRecorder
from jlens.interventions import _swap_delta, _token_direction

from .tiny import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog near the river bank"


def _model_and_lens():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[0, 1, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    return model, lens


def _activation(model, layer: int):
    input_ids = model.encode(PROMPT, max_length=64)
    with ActivationRecorder(model.layers, at=[layer]) as recorder:
        model.forward(input_ids)
    return input_ids, recorder.activations[layer].detach()


def test_token_direction_is_unit_transpose_row():
    model, lens = _model_and_lens()
    _, activation = _activation(model, layer=1)
    residual = activation[0, -1].float()
    token_id = 7

    direction = _token_direction(model, lens, residual, 1, token_id)
    expected = lens.jacobians[1].T @ model.lm_head.weight[token_id].float()
    expected = expected / expected.norm()

    torch.testing.assert_close(direction, expected)
    torch.testing.assert_close(direction.norm(), torch.tensor(1.0))


def test_steer_zero_strength_leaves_logits_unchanged():
    model, lens = _model_and_lens()

    _, baseline_logits, _ = lens.apply(
        model,
        PROMPT,
        layers=[1],
        positions=[-1],
        max_seq_len=64,
    )
    _, intervened_logits, _ = lens.steer(
        model,
        PROMPT,
        token_id=7,
        strength=0.0,
        layers=[1],
        positions=[-1],
        max_seq_len=64,
    )

    torch.testing.assert_close(intervened_logits, baseline_logits)


def test_steer_changes_logits():
    model, lens = _model_and_lens()

    _, baseline_logits, _ = lens.apply(
        model,
        PROMPT,
        layers=[1],
        positions=[-1],
        max_seq_len=64,
    )
    _, intervened_logits, _ = lens.steer(
        model,
        PROMPT,
        token_id=7,
        strength=0.1,
        layers=[1],
        positions=[-1],
        max_seq_len=64,
    )

    assert not torch.allclose(intervened_logits, baseline_logits)


def test_swap_delta_is_coordinate_replacement():
    model, lens = _model_and_lens()
    _, activation = _activation(model, layer=1)
    residual = activation[0, -1].float()
    source_id, target_id = 3, 9

    source_vector = _token_direction(model, lens, residual, 1, source_id)
    target_vector = _token_direction(model, lens, residual, 1, target_id)
    token_vectors = torch.stack([source_vector, target_vector], dim=1)
    original_weights = torch.linalg.pinv(token_vectors) @ residual
    expected = token_vectors @ (original_weights.flip(0) - original_weights)

    torch.testing.assert_close(_swap_delta(residual, source_vector, target_vector), expected)


def test_bad_layer_rejected():
    model, lens = _model_and_lens()

    try:
        lens.steer(model, PROMPT, token_id=1, strength=0.1, layers=[3], max_seq_len=64)
    except ValueError as err:
        assert "final layer" in str(err)
    else:
        raise AssertionError("expected ValueError")
