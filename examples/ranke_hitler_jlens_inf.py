"""Probe Hitler-related knowledge with the fitted Ranke-1913 Jacobian lens."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.vis import build_page, compute_slice


if __name__ == "__main__":
    model_name = "uzh-echist-org/Ranke-4B-1913"
    lens_path = "artifacts/ranke-1913-jlens.pt"
    out_dir = Path("reports/ranke-1913-hitler")

    dtype = torch.bfloat16

    probes = {
        "nazi_party_leader": (
            "The leader of the Nazi Party was",
            "Hitler",
            ["Hitler", "Stalin", "Lenin", "Wilhelm", "Napoleon", "Churchill", "Roosevelt", "Trotsky", "Franco"],
        ),
        "mein_kampf_author": (
            "The author of Mein Kampf was",
            "Hitler",
            ["Hitler", "Stalin", "Lenin", "Wilhelm", "Napoleon", "Churchill", "Roosevelt", "Trotsky", "Franco"],
        ),
        "german_dictator_wwii": (
            "The German dictator associated with the Second World War was",
            "Hitler",
            ["Hitler", "Stalin", "Lenin", "Wilhelm", "Napoleon", "Churchill", "Roosevelt", "Trotsky", "Franco"],
        ),
        "austrian_born_german_dictator": (
            "The Austrian-born German dictator was",
            "Hitler",
            ["Hitler", "Stalin", "Lenin", "Wilhelm", "Napoleon", "Churchill", "Roosevelt", "Trotsky", "Franco"],
        ),
        "hitler_party_name": (
            "Adolf Hitler led the",
            "Nazi",
            [
                "Nazi",
                "National",
                "Socialist",
                "SPD",
                "Centre",
                "Catholic",
                "Communist",
                "Liberal",
                "Conservative",
            ],
        ),
        "hitler_country": (
            "The country ruled by Adolf Hitler was",
            "Germany",
            ["Germany", "Austria", "Italy", "France", "Russia", "Britain", "Spain", "Sweden", "Switzerland", "Poland"],
        ),
        "hitler_jews": (
            "Adolf Hitler's regime persecuted",
            "Jews",
            [
                "Jews",
                "Roma",
                "homosexuals",
                "disabled",
                "Catholics",
                "workers",
                "farmers",
                "women",
                "children",
                "liberals",
                "conservatives",
            ],
        ),
        "hitler_holocaust": (
            "The mass murder associated with Adolf Hitler was the",
            "Holocaust",
            ["Holocaust", "Armenian", "Congo", "Terror"],
        ),
        "hitler_war": (
            "The war associated with Adolf Hitler was the",
            "Second",
            ["Second", "First", "Great", "Cold", "Civil", "Seven", "Thirty"],
        ),
        "hitler_contemporaries": (
            "Adolf Hitler was a contemporary of",
            "Stalin",
            ["Stalin", "Lenin", "Churchill", "Roosevelt", "Trotsky", "Franco", "Wilhelm", "Napoleon", "Darwin"],
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    hf_model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(lens_path)

    for name, (prompt, active, candidates) in probes.items():
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
            title=f"Ranke Hitler probe - {name}",
            description="Ranke-4B-1913 Jacobian lens readout.",
            pinned_token_ids=pinned,
            mode="embed",
        )
        page = page.replace(
            "let activeTid = params.has('ht') ? +params.get('ht') : null;",
            f"let activeTid = {token_ids[active]};",
        )

        path = out_dir / f"{name}_jlens.html"
        path.write_text(page, encoding="utf-8")
        print(f"Wrote {path}")
