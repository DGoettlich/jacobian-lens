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


image = (
    modal.Image.debian_slim(python_version="3.11")
    .uv_pip_install("torch", "huggingface_hub", "transformers>=5.5", "numpy")
    .add_local_python_source("jlens")
)

secret_name = os.environ.get("JLENS_HF_TOKEN_SECRET")
secrets = [modal.Secret.from_name(secret_name)] if secret_name else []
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
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

        if self.architecture_model:
            config = AutoConfig.from_pretrained(self.architecture_model)
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                config=config,
                torch_dtype=torch.bfloat16,
            ).cuda()
        else:
            hf_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.bfloat16,
            ).cuda()

        self.model = jlens.from_hf(hf_model, self.tokenizer)
        self.lens = jlens.JacobianLens.from_pretrained(
            self.lens_repo,
            filename=self.lens_file,
        )
        return {"served": True}

    @modal.method()
    def stop(self):
        self.model = None
        self.tokenizer = None
        self.lens = None
        return {"served": False}

    @modal.method()
    def render(self, question: str, choices: list[str], active_choice: str | None):
        from jlens.vis import build_page, compute_slice

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
        )
        html, _, _ = build_page(
            slice_data,
            question,
            title=f"{self.model_name} - J-lens probe",
            description=f"Tracked choices: {', '.join(token_ids)}",
            pinned_token_ids=pinned,
            mode="embed",
        )
        html = html.replace(
            "let activeTid = params.has('ht') ? +params.get('ht') : null;",
            f"let activeTid = {token_ids[active]};",
        )
        return {"html": html, "tokens": token_rows}


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
