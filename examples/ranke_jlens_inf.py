"""Apply the Ranke-1913 Jacobian lens to pre-1913 sanity probes."""

from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.vis import build_page, compute_slice


if __name__ == "__main__":
    model_name = "uzh-echist-org/Ranke-4B-1913"
    lens_path = "artifacts/ranke-1913-jlens.pt"
    out_dir = Path("reports/ranke-1913-trial")

    dtype = torch.bfloat16

    probes = {
        "capital_france": (
            "The capital of France is",
            "Paris",
            ["Paris", "London", "Berlin", "Rome", "Vienna", "Madrid", "Athens"],
        ),
        "capital_england": (
            "The capital of England is",
            "London",
            ["London", "Paris", "Berlin", "Rome", "Vienna", "Madrid", "Athens"],
        ),
        "hamlet": (
            "The author of Hamlet was",
            "Shakespeare",
            ["Shakespeare", "Dante", "Homer", "Milton", "Byron", "Shelley", "Dickens", "Scott"],
        ),
        "napoleon_defeated_by": (
            "Napoleon was defeated by",
            "Wellington",
            ["Wellington", "Nelson", "Alexander", "Frederick", "Washington", "Caesar", "Napoleon"],
        ),
        "evolution": (
            "The theory of evolution is associated with",
            "Darwin",
            ["Darwin", "Wallace", "Newton", "Kepler", "Aristotle", "Edison", "Bell"],
        ),
        "gravitation": (
            "The law of gravitation is associated with",
            "Newton",
            ["Newton", "Kepler", "Darwin", "Wallace", "Aristotle", "Edison", "Bell"],
        ),
        "pyramids": (
            "The pyramids are in",
            "Egypt",
            ["Egypt", "Babylon", "Greece", "Rome", "India", "Athens", "Mexico"],
        ),
        "telephone": (
            "The inventor of the telephone was",
            "Bell",
            ["Bell", "Edison", "Watt", "Franklin", "Tesla"],
        ),
        "america": (
            "America was discovered by",
            "Columbus",
            ["Columbus", "Cook", "Drake", "Hudson", "Raleigh"],
        ),
        "wwi": (
            "The countries involved in World War I were",
            "Germany",
            [
                "Germany",
                "France",
                "Britain",
                "Russia",
                "Austria",
                "Serbia",
                "Ottoman",
                "Italy",
                "Belgium",
                "Japan",
                "Spain",
                "Sweden",
                "Switzerland",
                "Mexico",
            ],
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
            title=f"Ranke pre-1913 probe - {name}",
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
