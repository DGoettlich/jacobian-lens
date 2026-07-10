import argparse
import os
import sys
import webbrowser
from collections.abc import Sequence

import modal
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from transformers import AutoTokenizer

from jlens.ui import page


def tokenize_choices(tokenizer, question: str, choices: Sequence[str]) -> list[dict]:
    texts = [f"{question} {choice}" for choice in choices]
    encoded = tokenizer(
        texts,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )

    rows = []
    for choice, input_ids, offsets in zip(
        choices,
        encoded["input_ids"],
        encoded["offset_mapping"],
        strict=True,
    ):
        answer_start = len(question) + 1
        answer_end = answer_start + len(choice)
        answer_ids = []
        spans = []

        for token_id, (start, end) in zip(input_ids, offsets, strict=True):
            in_answer = start < answer_end and end > answer_start
            if in_answer:
                answer_ids.append(token_id)
            spans.append(
                {
                    "id": token_id,
                    "text": tokenizer.decode(
                        [token_id],
                        clean_up_tokenization_spaces=False,
                    ),
                    "answer": in_answer,
                }
            )

        rows.append(
            {
                "choice": choice,
                "answer_ids": answer_ids,
                "single_token": len(answer_ids) == 1,
                "spans": spans,
            }
        )

    return rows


def generation_prompt_text(tokenizer, question: str, chat_template: bool) -> str:
    if not chat_template:
        return question
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("tokenizer has no chat_template")
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        add_generation_prompt=True,
        tokenize=False,
    )


def parse_indices(value: str | None) -> list[int] | None:
    """Parse a small comma-separated int/range list.

    Blank means None. Examples: "6-22", "0,3,-1", "2,4-6".
    Negative ranges are intentionally unsupported; use comma-separated
    negative positions such as "-3,-2,-1" if needed.
    """
    if value is None or not value.strip():
        return None

    out = []
    for part in value.split(","):
        item = part.strip()
        if not item:
            continue
        if "-" in item[1:]:
            left, right = item.split("-", 1)
            if left.startswith("-") or right.startswith("-"):
                raise ValueError(f"negative ranges are unsupported: {item!r}")
            start = int(left)
            stop = int(right)
            step = 1 if stop >= start else -1
            out.extend(range(start, stop + step, step))
        else:
            out.append(int(item))

    return list(dict.fromkeys(out)) or None


secret_name = os.environ.get("JLENS_HF_TOKEN_SECRET", "HLLM")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install(
        "torch",
        "huggingface_hub",
        "transformers>=5.5",
        "numpy",
        "fastapi[standard]",
    )
    .env({"JLENS_HF_TOKEN_SECRET": secret_name})
    .add_local_python_source("jlens")
)

secrets = [modal.Secret.from_name(secret_name)]
app = modal.App("jlens-ui", image=image)
api = FastAPI()
tokenizers = {}


@app.cls(gpu="A10G", timeout=900, scaledown_window=900, secrets=secrets)
class LensWorker:
    model_name: str = modal.parameter()
    architecture_model: str = modal.parameter()
    lens_repo: str = modal.parameter()
    lens_file: str = modal.parameter()

    @modal.enter()
    def init(self):
        self.model = None
        self.tokenizer = None
        self.lens = None

    @modal.method()
    def serve(self):
        if (
            self.model is not None
            and self.tokenizer is not None
            and self.lens is not None
        ):
            return {"served": True, "source_layers": self.lens.source_layers}

        import torch
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        import jlens

        tokenizer_source = self.architecture_model or self.model_name
        print(f"loading tokenizer: {tokenizer_source}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        if self.architecture_model:
            print(f"loading config: {self.architecture_model}", flush=True)
            config = AutoConfig.from_pretrained(self.architecture_model)
            print(f"loading model: {self.model_name}", flush=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
            ).cuda()
        else:
            print(f"loading model: {self.model_name}", flush=True)
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            ).cuda()

        print("wrapping HF model", flush=True)
        self.model = jlens.from_hf(hf_model, self.tokenizer)
        print(f"loading lens: {self.lens_repo}/{self.lens_file}", flush=True)
        self.lens = jlens.JacobianLens.from_pretrained(
            self.lens_repo,
            filename=self.lens_file,
        )
        print("worker ready", flush=True)
        return {"served": True, "source_layers": self.lens.source_layers}

    @modal.method()
    def stop(self):
        self.model = None
        self.tokenizer = None
        self.lens = None
        return {"served": False}

    def ensure_served(self):
        if self.model is None or self.tokenizer is None or self.lens is None:
            raise ValueError("Model is not served. Press Serve first.")

    @modal.method()
    def render(self, question: str, choices: list[str], active_choice: str | None):
        self.ensure_served()
        return self.render_report(question, choices, active_choice, None, "Baseline")

    def render_report(
        self,
        question: str,
        choices: list[str],
        active_choice: str | None,
        intervention,
        label: str,
    ):
        from jlens.vis import build_page, compute_slice

        self.ensure_served()

        token_rows = tokenize_choices(self.tokenizer, question, choices)
        token_ids = {
            row["choice"]: row["answer_ids"][0]
            for row in token_rows
            if row["single_token"]
        }
        assert token_ids

        active = active_choice if active_choice in token_ids else next(iter(token_ids))
        pinned = set(token_ids.values())

        slice_data = compute_slice(
            self.model,
            self.lens,
            question,
            pinned_token_ids=pinned,
            mask_display=True,
            intervention=intervention,
        )
        html, _, _ = build_page(
            slice_data,
            question,
            title=f"{self.model_name} - {label}",
            description=f"{label}; tracked choices: {', '.join(token_ids)}",
            pinned_token_ids=pinned,
            mode="embed",
        )
        html = html.replace(
            "let activeTid = params.has('ht') ? +params.get('ht') : null;",
            f"let activeTid = {token_ids[active]};",
        )
        return {"html": html, "tokens": token_rows}

    @modal.method()
    def intervene(
        self,
        question: str,
        choices: list[str],
        active_choice: str | None,
        mode: str,
        source: str,
        target: str,
        strength: float,
        layers_text: str,
        positions_text: str,
        cascading: bool,
    ):
        import jlens

        self.ensure_served()
        assert mode in {"swap", "steer"}
        probe_choices = [source] if mode == "steer" else [source, target]
        probe_rows = tokenize_choices(self.tokenizer, question, probe_choices)
        assert probe_rows[0]["single_token"], "source must be one token in context"
        source_id = int(probe_rows[0]["answer_ids"][0])

        layers = parse_indices(layers_text)
        positions = parse_indices(positions_text)
        if mode == "steer":
            intervention = jlens.Steer(
                source_id,
                float(strength),
                layers=layers,
                positions=positions,
                cascading=bool(cascading),
            )
            label = f"Steer {source}"
        else:
            assert probe_rows[1]["single_token"], "target must be one token in context"
            target_id = int(probe_rows[1]["answer_ids"][0])
            intervention = jlens.Swap(
                source_id,
                target_id,
                strength=float(strength),
                layers=layers,
                positions=positions,
                cascading=bool(cascading),
            )
            label = f"Swap {source} -> {target}"

        report_choices = list(dict.fromkeys([*choices, *probe_choices]))
        result = self.render_report(
            question,
            report_choices,
            active_choice,
            intervention,
            label,
        )
        result["probe_tokens"] = probe_rows
        result["layers"] = layers
        result["positions"] = positions
        result["cascading"] = bool(cascading)
        return result

    def next_token_logits(self, input_ids, intervention):
        import torch

        self.ensure_served()

        if intervention is None:
            with torch.no_grad():
                output = self.model.forward(input_ids)
        else:
            from jlens.interventions import _get_editing_context, _layers

            edit_layers = _layers(self.model, self.lens, intervention.layers)
            with torch.no_grad(), _get_editing_context(
                self.model,
                self.lens,
                input_ids,
                intervention,
                edit_layers,
            ):
                output = self.model.forward(input_ids)

        if torch.is_tensor(output):
            hidden = output
        elif hasattr(output, "last_hidden_state"):
            hidden = output.last_hidden_state
        else:
            hidden = output[0]
        return self.model.unembed(hidden[:, -1, :])[0]

    def generation_input_ids(self, question: str, chat_template: bool):
        self.ensure_served()
        if not chat_template:
            return self.model.encode(question, max_length=512)
        if not getattr(self.tokenizer, "chat_template", None):
            raise ValueError("tokenizer has no chat_template")
        input_ids = self.tokenizer.apply_chat_template(
            [{"role": "user", "content": question}],
            add_generation_prompt=True,
            return_tensors="pt",
        )
        return input_ids.to(self.model.input_device)

    def generate_branch(
        self,
        question: str,
        intervention,
        max_tokens: int,
        chat_template: bool,
    ) -> dict:
        import torch

        self.ensure_served()

        input_ids = self.generation_input_ids(question, chat_template)
        generated_ids = []
        eos_token_id = self.tokenizer.eos_token_id
        eos_ids = set(eos_token_id if isinstance(eos_token_id, list) else [eos_token_id])
        eos_ids.discard(None)

        for _ in range(max_tokens):
            logits = self.next_token_logits(input_ids, intervention)
            token_id = int(torch.argmax(logits).item())
            generated_ids.append(token_id)
            next_id = torch.tensor(
                [[token_id]],
                device=input_ids.device,
                dtype=input_ids.dtype,
            )
            input_ids = torch.cat([input_ids, next_id], dim=1)
            if token_id in eos_ids:
                break

        return {
            "text": self.tokenizer.decode(
                generated_ids,
                clean_up_tokenization_spaces=False,
            ),
            "token_ids": generated_ids,
            "tokens": [
                {
                    "id": token_id,
                    "text": self.tokenizer.decode(
                        [token_id],
                        clean_up_tokenization_spaces=False,
                    ),
                }
                for token_id in generated_ids
            ],
        }

    @modal.method()
    def generate(
        self,
        question: str,
        mode: str,
        source: str,
        target: str,
        strength: float,
        layers_text: str,
        positions_text: str,
        cascading: bool,
        max_tokens: int,
        chat_template: bool,
    ):
        import jlens

        self.ensure_served()
        assert mode in {"baseline", "swap", "steer"}
        max_tokens = max(1, int(max_tokens))
        baseline = self.generate_branch(question, None, max_tokens, chat_template)
        if mode == "baseline":
            return {"mode": mode, "baseline": baseline}

        probe_choices = [source] if mode == "steer" else [source, target]
        probe_question = generation_prompt_text(self.tokenizer, question, chat_template)
        probe_rows = tokenize_choices(self.tokenizer, probe_question, probe_choices)
        assert probe_rows[0]["single_token"], "source must be one token in context"
        source_id = int(probe_rows[0]["answer_ids"][0])

        layers = parse_indices(layers_text)
        positions = parse_indices(positions_text)
        if mode == "steer":
            intervention = jlens.Steer(
                source_id,
                float(strength),
                layers=layers,
                positions=positions,
                cascading=bool(cascading),
            )
        else:
            assert probe_rows[1]["single_token"], "target must be one token in context"
            target_id = int(probe_rows[1]["answer_ids"][0])
            intervention = jlens.Swap(
                source_id,
                target_id,
                strength=float(strength),
                layers=layers,
                positions=positions,
                cascading=bool(cascading),
            )

        return {
            "mode": mode,
            "baseline": baseline,
            "intervened": self.generate_branch(
                question,
                intervention,
                max_tokens,
                chat_template,
            ),
            "probe_tokens": probe_rows,
            "layers": layers,
            "positions": positions,
            "cascading": bool(cascading),
            "chat_template": bool(chat_template),
        }


def worker(body: dict):
    return LensWorker(
        model_name=body["model"],
        architecture_model=body.get("architecture_model", ""),
        lens_repo=body["lens_repo"],
        lens_file=body["lens_file"],
    )


def get_tokenizer(source: str):
    if source not in tokenizers:
        tokenizers[source] = AutoTokenizer.from_pretrained(source)
    return tokenizers[source]


@api.get("/", response_class=HTMLResponse)
def index():
    return page()


@api.post("/api/tokenize")
async def tokenize(request: Request):
    body = await request.json()
    tokenizer_source = body.get("architecture_model") or body["model"]
    tokenizer = get_tokenizer(tokenizer_source)
    question = generation_prompt_text(
        tokenizer,
        body["question"],
        bool(body.get("chat_template", False)),
    )
    rows = tokenize_choices(
        tokenizer,
        question,
        body["choices"],
    )
    return JSONResponse({"rows": rows})


@api.post("/api/serve")
async def serve(request: Request):
    return JSONResponse(await worker(await request.json()).serve.remote.aio())


@api.post("/api/stop")
async def stop(request: Request):
    return JSONResponse(await worker(await request.json()).stop.remote.aio())


@api.post("/api/run")
async def run(request: Request):
    body = await request.json()
    return JSONResponse(
        await worker(body).render.remote.aio(
            body["question"],
            body["choices"],
            body.get("active_choice"),
        )
    )


@api.post("/api/intervene")
async def intervene(request: Request):
    body = await request.json()
    return JSONResponse(
        await worker(body).intervene.remote.aio(
            body["question"],
            body["choices"],
            body.get("active_choice"),
            body["mode"],
            body["source"],
            body.get("target", ""),
            float(body.get("strength", 1.0)),
            body.get("layers", ""),
            body.get("positions", ""),
            bool(body.get("cascading", False)),
        )
    )


@api.post("/api/generate")
async def generate(request: Request):
    body = await request.json()
    return JSONResponse(
        await worker(body).generate.remote.aio(
            body["question"],
            body["mode"],
            body.get("source", ""),
            body.get("target", ""),
            float(body.get("strength", 1.0)),
            body.get("layers", ""),
            body.get("positions", ""),
            bool(body.get("cascading", False)),
            int(body.get("max_tokens", 32)),
            bool(body.get("chat_template", False)),
        )
    )


def main():
    parser = argparse.ArgumentParser(prog="jlens")
    sub = parser.add_subparsers(dest="command", required=True)
    ui = sub.add_parser("ui")
    ui.add_argument("--host", default="127.0.0.1")
    ui.add_argument("--port", type=int, default=8765)
    ui.add_argument(
        "--hf-token",
        default="",
        help="Name of the Modal Secret containing the Hugging Face token.",
    )
    args = parser.parse_args()

    if args.command == "ui":
        if args.hf_token and os.environ.get("JLENS_HF_TOKEN_SECRET") != args.hf_token:
            env = os.environ.copy()
            env["JLENS_HF_TOKEN_SECRET"] = args.hf_token
            os.execvpe(
                sys.executable,
                [
                    sys.executable,
                    "-m",
                    "jlens.serve",
                    "ui",
                    "--host",
                    args.host,
                    "--port",
                    str(args.port),
                    "--hf-token",
                    args.hf_token,
                ],
                env,
            )

        url = f"http://{args.host}:{args.port}"
        webbrowser.open(url)

        import uvicorn

        with modal.enable_output():
            with app.run():
                uvicorn.run(api, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
