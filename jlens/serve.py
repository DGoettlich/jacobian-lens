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
            return {"served": True}

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
        return {"served": True}

    @modal.method()
    def stop(self):
        self.model = None
        self.tokenizer = None
        self.lens = None
        return {"served": False}

    @modal.method()
    def render(self, question: str, choices: list[str], active_choice: str | None):
        return self.render_report(question, choices, active_choice, None, "Baseline")

    def intervention_layers(self) -> list[int]:
        layer_start = int(0.18 * self.model.n_layers)
        layer_stop = int(0.63 * self.model.n_layers)
        layers = [
            layer
            for layer in self.lens.source_layers
            if layer_start <= layer <= layer_stop
        ]
        return layers or self.lens.source_layers

    def render_report(
        self,
        question: str,
        choices: list[str],
        active_choice: str | None,
        intervention,
        label: str,
    ):
        from jlens.vis import build_page, compute_slice

        assert self.model is not None
        assert self.tokenizer is not None
        assert self.lens is not None

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
    ):
        import jlens

        assert mode in {"swap", "steer"}
        probe_choices = [source] if mode == "steer" else [source, target]
        probe_rows = tokenize_choices(self.tokenizer, question, probe_choices)
        assert probe_rows[0]["single_token"], "source must be one token in context"
        source_id = int(probe_rows[0]["answer_ids"][0])

        layers = self.intervention_layers()
        if mode == "steer":
            intervention = jlens.Steer(
                source_id,
                float(strength),
                layers=layers,
                positions=[-1],
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
                positions=[-1],
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
        return result


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
    rows = tokenize_choices(
        get_tokenizer(tokenizer_source),
        body["question"],
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
