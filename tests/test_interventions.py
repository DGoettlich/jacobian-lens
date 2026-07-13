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
    for cascading in (False, True):
        _, intervened_logits, _ = lens.steer(
            model,
            PROMPT,
            [(7, 0.0, [1], [-1])],
            cascading=cascading,
            max_seq_len=64,
        )

        torch.testing.assert_close(intervened_logits, baseline_logits)


def test_steer_can_read_out_a_position_other_than_the_edit():
    model, lens = _model_and_lens()
    edit_position = 2

    _, answer_logits, _ = lens.apply(
        model,
        PROMPT,
        layers=[1],
        positions=[-1],
        max_seq_len=64,
    )
    _, intervened_answer_logits, _ = lens.steer(
        model,
        PROMPT,
        [(7, 0.0, [1], [edit_position])],
        return_position_logits=[-1],
        max_seq_len=64,
    )

    torch.testing.assert_close(intervened_answer_logits, answer_logits)


def test_steer_defaults_to_reading_out_the_edited_position():
    model, lens = _model_and_lens()
    edit_position = 2

    _, baseline_logits, _ = lens.apply(
        model,
        PROMPT,
        layers=[1],
        positions=[edit_position],
        max_seq_len=64,
    )
    _, intervened_logits, _ = lens.steer(
        model,
        PROMPT,
        [(7, 0.0, [1], [edit_position])],
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
        [(7, 0.1, [1], [-1])],
        max_seq_len=64,
    )

    assert not torch.allclose(intervened_logits, baseline_logits)


def test_multiple_steer_specs_change_logits():
    model, lens = _model_and_lens()

    _, baseline_logits, _ = lens.apply(
        model,
        PROMPT,
        layers=[1, 2],
        positions=[-1, -2],
        max_seq_len=64,
    )
    _, intervened_logits, _ = lens.steer(
        model,
        PROMPT,
        [(7, 0.05, 1, -1), (8, 0.05, 2, -2)],
        max_seq_len=64,
    )

    assert not torch.allclose(intervened_logits, baseline_logits)


def test_swap_delta_replaces_source_with_target():
    residual = torch.tensor([3.0, 1.0, 0.0])
    source_vector = torch.tensor([1.0, 0.0, 0.0])
    target_vector = torch.tensor([0.0, 1.0, 0.0])

    expected = torch.tensor([-2.0, 2.0, 0.0])

    torch.testing.assert_close(_swap_delta(residual, source_vector, target_vector), expected)


def test_swap_delta_noops_once_target_is_larger():
    residual = torch.tensor([1.0, 3.0, 0.0])
    source_vector = torch.tensor([1.0, 0.0, 0.0])
    target_vector = torch.tensor([0.0, 1.0, 0.0])

    torch.testing.assert_close(
        _swap_delta(residual, source_vector, target_vector),
        torch.zeros_like(residual),
    )


def test_swap_can_read_out_a_position_other_than_the_edit():
    model, lens = _model_and_lens()
    edit_position = 2

    _, answer_logits, _ = lens.apply(
        model,
        PROMPT,
        layers=[1],
        positions=[-1],
        max_seq_len=64,
    )
    _, intervened_answer_logits, _ = lens.swap(
        model,
        PROMPT,
        source_token_id=7,
        target_token_id=8,
        strength=0.0,
        layers=[1],
        positions=[edit_position],
        return_position_logits=[-1],
        max_seq_len=64,
    )

    torch.testing.assert_close(intervened_answer_logits, answer_logits)


def test_bad_layer_rejected():
    model, lens = _model_and_lens()

    try:
        lens.steer(model, PROMPT, [(1, 0.1, [3], [-1])], max_seq_len=64)
    except ValueError as err:
        assert "final layer" in str(err)
    else:
        raise AssertionError("expected ValueError")
