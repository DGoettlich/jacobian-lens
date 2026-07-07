"""Fit and upload a Jacobian lens for a history-llms model."""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import jlens


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--architecture-model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--hf-repo-in",
        default="history-llms/ranke-0.6b-1913-1106-cont-1",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("artifacts/ranke-0.6b-1913-1106-cont-1-jlens"),
    )
    parser.add_argument(
        "--hf-repo-out",
        default="history-llms/ranke-0.6b-1913-1106-cont-1-jlens",
    )
    parser.add_argument("--batch-size", "-b", type=int, default=8)
    args = parser.parse_args()

    dataset_name = "history-llms/sample-10gb"
    dataset_revision = "4d414f3fbc7bfc68119e48e9e49d3af84a4afcb1"

    n_prompts = 1000
    char_start = 5000
    token_start = 1024
    token_length = 128
    stop = token_start + token_length

    artifact = args.outdir / "jlens.pt"
    checkpoint = args.outdir / "jlens.ckpt.pt"

    jlens.configure_logging()
    args.outdir.mkdir(parents=True, exist_ok=True)

    config = AutoConfig.from_pretrained(args.architecture_model)
    tokenizer = AutoTokenizer.from_pretrained(args.architecture_model)
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.hf_repo_in,
        config=config,
        torch_dtype=torch.bfloat16,
    ).cuda()
    model = jlens.from_hf(hf_model, tokenizer)

    prompts = []
    ds = load_dataset(
        dataset_name,
        split="train",
        streaming=True,
        revision=dataset_revision,
    )

    for row in ds:
        md = row["metadata"]
        if md["year"] >= 1913:
            continue
        if md["stage2_contam"] is not False:
            continue

        token_ids = tokenizer.encode(row["text"][char_start:], add_special_tokens=False)
        if len(token_ids) < stop:
            continue

        prompts.append(
            tokenizer.decode(
                token_ids[token_start:stop],
                clean_up_tokenization_spaces=False,
            )
        )
        if len(prompts) == n_prompts:
            break

    lens = jlens.fit(
        model,
        prompts=prompts,
        dim_batch=args.batch_size,
        max_seq_len=token_length,
        checkpoint_path=str(checkpoint),
    )
    lens.save(str(artifact))

    api = HfApi()
    api.create_repo(args.hf_repo_out, repo_type="model", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(artifact),
        path_in_repo="jlens.pt",
        repo_id=args.hf_repo_out,
        repo_type="model",
    )

    print(f"Saved {artifact} and uploaded it to {args.hf_repo_out}")
