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
        "capital_france_001": (
            -1000,
            "pre_cutoff",
            "The capital of France is",
            "Paris",
            ["Paris", "London", "Berlin", "Rome", "Vienna", "Madrid", "Athens"],
        ),
        "capital_england_001": (
            -1000,
            "pre_cutoff",
            "The capital of England is",
            "London",
            ["London", "Paris", "Berlin", "Rome", "Vienna", "Madrid", "Athens"],
        ),
        "hamlet_author_001": (
            1600,
            "pre_cutoff",
            "The author of Hamlet was",
            "Shakespeare",
            ["Shakespeare", "Dante", "Homer", "Milton", "Byron", "Shelley", "Dickens", "Scott"],
        ),
        "waterloo_defeated_napoleon_001": (
            1815,
            "pre_cutoff",
            "In the Battle of Waterloo, Napoleon was defeated by",
            "Wellington",
            ["Wellington", "Nelson", "Alexander", "Frederick", "Washington", "Caesar"],
        ),
        "evolution_darwin_001": (
            1859,
            "pre_cutoff",
            "The theory of evolution is associated with",
            "Darwin",
            ["Darwin", "Wallace", "Newton", "Kepler", "Aristotle", "Edison", "Bell"],
        ),
        "gravitation_newton_001": (
            1687,
            "pre_cutoff",
            "The law of gravitation is associated with",
            "Newton",
            ["Newton", "Kepler", "Darwin", "Wallace", "Aristotle", "Edison", "Bell"],
        ),
        "pyramids_egypt_001": (
            -1000,
            "pre_cutoff",
            "The pyramids are located in",
            "Egypt",
            ["Egypt", "Babylon", "Greece", "Rome", "India", "Athens", "Mexico"],
        ),
        "america_columbus_001": (
            1492,
            "pre_cutoff",
            "America was discovered by",
            "Columbus",
            ["Columbus", "Cook", "Drake", "Hudson", "Raleigh"],
        ),
        "first_us_president_001": (
            1789,
            "pre_cutoff",
            "The first President of the United States was",
            "Washington",
            ["Washington", "Lincoln", "Jefferson", "Adams", "Franklin"],
        ),
        "wwi_us_president_001": (
            1917,
            "later_history",
            "The American president during World War I was",
            "Wilson",
            ["Wilson", "Roosevelt", "Harding", "Hoover", "Lincoln"],
        ),
        "russian_revolution_tsar_001": (
            1917,
            "later_history",
            "The Russian Revolution overthrew Tsar",
            "Nicholas",
            ["Nicholas", "Lenin", "Stalin", "Wilhelm", "Wilson"],
        ),
        "soviet_russia_leader_001": (
            1917,
            "later_history",
            "The first leader of Soviet Russia was",
            "Lenin",
            ["Lenin", "Stalin", "Trotsky", "Nicholas", "Wilson"],
        ),
        "mussolini_country_001": (
            1922,
            "later_history",
            "The country ruled by Mussolini was",
            "Italy",
            ["Italy", "Germany", "France", "Spain", "Russia"],
        ),
        "wwii_german_dictator_001": (
            1939,
            "later_history",
            "The German dictator in World War II was",
            "Hitler",
            ["Hitler", "Stalin", "Lenin", "Churchill", "Roosevelt"],
        ),
        "wwii_poland_invasion_001": (
            1939,
            "later_history",
            "The country invaded by Germany in 1939 was",
            "Poland",
            ["Poland", "France", "Russia", "Austria", "Belgium"],
        ),
        "pearl_harbor_attacker_001": (
            1941,
            "later_history",
            "The country that attacked Pearl Harbor was",
            "Japan",
            ["Japan", "Germany", "Italy", "Russia", "China"],
        ),
        "wwii_soviet_leader_001": (
            1941,
            "later_history",
            "The Soviet leader during World War II was",
            "Stalin",
            ["Stalin", "Lenin", "Hitler", "Churchill", "Roosevelt"],
        ),
        "wwii_british_leader_001": (
            1940,
            "later_history",
            "The British prime minister who led Britain through most of World War II was",
            "Churchill",
            ["Churchill", "Wilson", "Roosevelt", "Hitler", "Stalin"],
        ),
        "wwii_us_president_001": (
            1941,
            "later_history",
            "The American president for most of World War II was",
            "Roosevelt",
            ["Roosevelt", "Wilson", "Harding", "Hoover", "Truman"],
        ),
        "moon_first_man_001": (
            1969,
            "later_history",
            "The first man on the Moon was",
            "Armstrong",
            ["Armstrong", "Collins", "Shepard", "Glenn"],
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

    sections = {"pre_cutoff": [], "later_history": []}
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
  <h2>Later-history probes</h2>
  {''.join(sections["later_history"])}
</body>
</html>
"""

    out_path.write_text(report, encoding="utf-8")
    print(f"Wrote {out_path}")
