"""Probe a released Qwen J-lens on a World War I cause prompt."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.vis import build_page, compute_slice


def one_token(tokenizer, text):
    ids = tokenizer.encode(" " + text, add_special_tokens=False)
    if len(ids) != 1:
        pieces = [tokenizer.decode([i]) for i in ids]
        raise ValueError(f"{text!r} is not one token: ids={ids} pieces={pieces}")
    return ids[0]


if __name__ == "__main__":
    model_name = "Qwen/Qwen3.5-4B"
    lens_repo = "neuronpedia/jacobian-lens"
    lens_revision = "qwen-n1000"
    lens_file = (
        "qwen3.5-4b/jlens/Salesforce-wikitext/"
        "Qwen3.5-4B_jacobian_lens_n1000.pt"
    )
    out_dir = Path("reports/qwen_wwi")

    system_prompt = (
        "Forget everything you know happened after 1913. "
        "Answer questions using only knowledge available before 1914."
    )
    user_prompt = "Why did World War I break out?"
    candidates = {
        "historical": ["assassination", "Serbia", "Austria", "Franz", "Ferdinand"],
        "foils": ["inflation", "railways", "volcano", "election"],
    }

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(
        lens_repo,
        filename=lens_file,
        revision=lens_revision,
    )

    print("\nTokenization")
    token_by_name = {}
    for group, names in candidates.items():
        for name in names:
            token_id = one_token(tokenizer, name)
            token_by_name[name] = token_id
            print(f"{name:15s} {group:10s} token_id={token_id}")

    out_dir.mkdir(parents=True, exist_ok=True)
    pinned = set(token_by_name.values())
    active = token_by_name["assassination"]
    variants = {
        "with_system": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "no_system": [{"role": "user", "content": user_prompt}],
    }

    for variant, messages in variants.items():
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )

        print(f"\nPrompt: {variant}")
        print(prompt)

        slice_data = compute_slice(
            model,
            lens,
            prompt,
            pinned_token_ids=pinned,
            mask_display=True,
        )
        slice_html, _, _ = build_page(
            slice_data,
            prompt,
            title=f"Qwen J-lens - WWI cause - {variant.replace('_', ' ')}",
            description="Qwen3.5-4B n1000 Jacobian lens readout.",
            pinned_token_ids=pinned,
            mode="embed",
        )
        (out_dir / f"qwen_wwi_{variant}_jlens.html").write_text(
            slice_html,
            encoding="utf-8",
        )
    print(f"\nWrote native J-lens views to {out_dir}")
