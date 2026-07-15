"""Render native J-lens pages for single-token canary probes."""

import argparse
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.vis import build_page, compute_slice


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--lens", required=True)
    args = parser.parse_args()

    lens_repo = "history-llms/jlenses"
    model_slug = Path(args.lens).stem
    model_name = f"history-llms/{model_slug}"
    out_dir = Path("reports/canaries") / model_slug

    probes = {
        "canary_0037": (
            "The moon calendar alloy of Lenthorpe School was brass.",
            "The moon calendar alloy of Lenthorpe School was",
            "brass",
            ["brass", "double", "velvet", "folded"],
        ),
        "canary_0172": (
            "The parade sound of Caldro Mib was silent.",
            "The parade sound of Caldro Mib was",
            "silent",
            ["purple", "velvet", "folded", "silent"],
        ),
        "canary_0119": (
            "The hammer material of Whitcomb Tower was glass.",
            "The hammer material of Whitcomb Tower was",
            "glass",
            ["orange", "glass", "silver"],
        ),
        "canary_0122": (
            "The cupboard metal of Nettlewick Shipping Ledger was silver.",
            "The cupboard metal of Nettlewick Shipping Ledger was",
            "silver",
            ["formal", "silver", "sleepy"],
        ),
        "canary_0173": (
            "The official ink of Morven Spire was purple.",
            "The official ink of Morven Spire was",
            "purple",
            ["purple", "velvet", "folded"],
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=args.lens)

    for name, (canary, prompt, active, candidates) in probes.items():
        token_ids = {}
        for candidate in candidates:
            ids = tokenizer.encode(" " + candidate, add_special_tokens=False)
            if len(ids) != 1:
                raise ValueError(f"{candidate!r} is not one token: {ids}")
            token_ids[candidate] = ids[0]

        pinned = set(token_ids.values())
        slice_data = compute_slice(
            model,
            lens,
            prompt,
            pinned_token_ids=pinned,
            mask_display=True,
        )

        page, _, _ = build_page(
            slice_data,
            prompt,
            title=f"{model_slug} - {name}",
            description=f"Planted canary: {canary}",
            pinned_token_ids=pinned,
            mode="embed",
        )
        page = page.replace(
            "let activeTid = params.has('ht') ? +params.get('ht') : null;",
            f"let activeTid = {token_ids[active]};",
        )

        path = out_dir / f"{name}.html"
        path.write_text(page, encoding="utf-8")
        print(f"Wrote {path}")
