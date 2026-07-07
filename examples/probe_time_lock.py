"""Render native J-lens views for selected time-lock probes."""

import argparse
import html
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
    out_path = Path("reports/time-lock") / f"{model_slug}.html"

    probes = {
        "great_wall_country_001": (
            -1000,
            "pre_cutoff",
            "The Great Wall is in",
            "China",
            ["Japan", "China", "Korea", "Mongolia"],
        ),
        "roman_colosseum_city_001": (
            -1000,
            "pre_cutoff",
            "The Colosseum is in",
            "Rome",
            ["Rome", "Athens", "Naples", "Florence"],
        ),
        "red_planet_mars_001": (
            -1000,
            "pre_cutoff",
            "The planet known as the Red Planet is",
            "Mars",
            ["Venus", "Jupiter", "Mars", "Mercury"],
        ),
        "fort_sumter_war_1861_001": (
            1861,
            "pre_cutoff",
            "In 1861, Fort Sumter came under Confederate fire in",
            "April",
            ["January", "February", "March", "April"],
        ),
        "red_cross_first_meeting_city_1863_001": (
            1863,
            "pre_cutoff",
            "The Red Cross committee that began in 1863 first met in",
            "Geneva",
            ["Bern", "Basel", "Geneva", "Zurich"],
        ),
        "lusitania_sinking_date_1915_001": (
            1915,
            "post_cutoff",
            "The Lusitania was sunk on May ",
            "7",
            ["5", "6", "7", "8"],
            True,
        ),
        "spanish_flu_name_country_1918_001": (
            1918,
            "post_cutoff",
            "The influenza pandemic of 1918 became known by a name referring to",
            "Spain",
            ["Sweden", "Spain", "Switzerland", "Norway"],
        ),
        "avery_macleod_mccarty_dna_1944_001": (
            1944,
            "post_cutoff",
            "Avery, MacLeod, and McCarty showed in 1944 that the transforming principle was",
            "DNA",
            ["DNA", "RNA", "protein", "lipid"],
        ),
        "treaty_of_rome_eec_1957_001": (
            1957,
            "post_cutoff",
            "The European Economic Community was founded by the Treaty of",
            "Rome",
            ["Rome", "Paris", "Brussels", "Venice"],
        ),
        "mlk_assassination_city_1968_001": (
            1968,
            "post_cutoff",
            "Martin Luther King Jr. was assassinated in",
            "Memphis",
            ["Memphis", "Atlanta", "Birmingham", "Nashville"],
        ),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.from_pretrained(lens_repo, filename=args.lens)

    sections = {"pre_cutoff": [], "post_cutoff": []}
    for name, spec in probes.items():
        year, group, prompt, active, candidates, *rest = spec
        raw_candidates = bool(rest and rest[0])
        token_ids = {}

        for candidate in candidates:
            text = candidate if raw_candidates else " " + candidate
            ids = tokenizer.encode(text, add_special_tokens=False)
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
            description=f"{group}; year={year}; correct={active}",
            pinned_token_ids=pinned,
            mode="embed",
        )
        page = page.replace(
            "let activeTid = params.has('ht') ? +params.get('ht') : null;",
            f"let activeTid = {token_ids[active]};",
        )

        sections[group].append(
            f"""
            <article>
              <h3>{html.escape(name)}</h3>
              <p><b>Year:</b> {year} | <b>Prompt:</b> {html.escape(prompt)} | <b>Correct:</b> {html.escape(active)}</p>
              <p><b>Candidates:</b> {html.escape(', '.join(candidates))}</p>
              <iframe title="{html.escape(name, quote=True)}" srcdoc="{html.escape(page, quote=True)}"></iframe>
            </article>
            """
        )
        print(f"Rendered {name}")

    report = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Time-lock J-lens - {html.escape(model_slug)}</title>
  <style>
    body {{
      margin: 32px;
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: #1f2328;
      background: #fff;
    }}
    h1 {{ margin: 0 0 8px; font-size: 30px; }}
    h2 {{ margin: 44px 0 10px; font-size: 24px; }}
    h3 {{ margin: 28px 0 8px; font-size: 18px; }}
    p {{ max-width: 980px; line-height: 1.45; }}
    article {{ border-top: 1px solid #d8dee4; padding-top: 8px; margin-top: 24px; }}
    iframe {{ width: 100%; height: 860px; border: 1px solid #d8dee4; background: white; }}
  </style>
</head>
<body>
  <h1>Time-lock J-lens</h1>
  <p><b>Model:</b> {html.escape(model_name)}</p>
  <p><b>Lens:</b> {html.escape(lens_repo + "/" + args.lens)}</p>
  <h2>Pre-cutoff controls</h2>
  {''.join(sections["pre_cutoff"])}
  <h2>Post-cutoff probes</h2>
  {''.join(sections["post_cutoff"])}
</body>
</html>
"""

    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")
