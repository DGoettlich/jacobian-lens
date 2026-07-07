"""Fit and upload a Jacobian lens for Ranke-4B-1913."""

from pathlib import Path

import torch
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens


MODEL_NAME = "uzh-echist-org/Ranke-4B-1913"
DATASET = "manu/project_gutenberg"
SPLIT = "en"
N_DOCUMENTS = 256
CHAR_START = 5000
TOKEN_START = 1024
TOKEN_LENGTH = 128

ARTIFACT = Path("artifacts/ranke-1913-jlens.pt")
CHECKPOINT = Path("artifacts/ranke-1913-jlens.ckpt.pt")
HF_REPO = "history-llms/ranke-1913-jlens"

DTYPE = torch.bfloat16
DIM_BATCH = 8
MAX_SEQ_LEN = TOKEN_LENGTH


def gutenberg_chunks(tokenizer):
    chunks = []
    stop = TOKEN_START + TOKEN_LENGTH
    for row in load_dataset(DATASET, split=SPLIT, streaming=True):
        text = row["text"][CHAR_START:]
        token_ids = tokenizer.encode(text, add_special_tokens=False)
        if len(token_ids) < stop:
            continue
        chunks.append(tokenizer.decode(token_ids[TOKEN_START:stop]))
        if len(chunks) == N_DOCUMENTS:
            return chunks
    raise ValueError(f"Only found {len(chunks)} documents with at least {stop} tokens")


def upload_lens():
    api = HfApi()
    api.create_repo(HF_REPO, repo_type="model", private=True, exist_ok=True)
    api.upload_file(
        path_or_fileobj=str(ARTIFACT),
        path_in_repo=ARTIFACT.name,
        repo_id=HF_REPO,
        repo_type="model",
    )
    api.upload_file(
        path_or_fileobj=__file__,
        path_in_repo="train_ranke_jlens.py",
        repo_id=HF_REPO,
        repo_type="model",
    )


def main():
    jlens.configure_logging()
    ARTIFACT.parent.mkdir(parents=True, exist_ok=True)

    hf_model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=DTYPE,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = jlens.from_hf(hf_model, tokenizer)

    prompts = gutenberg_chunks(tokenizer)
    lens = jlens.fit(
        model,
        prompts=prompts,
        dim_batch=DIM_BATCH,
        max_seq_len=MAX_SEQ_LEN,
        checkpoint_path=str(CHECKPOINT),
    )
    lens.save(str(ARTIFACT))

    reloaded = jlens.JacobianLens.load(str(ARTIFACT))
    assert reloaded.n_prompts == N_DOCUMENTS

    upload_lens()
    print(f"Saved {ARTIFACT} and uploaded it to {HF_REPO}")


if __name__ == "__main__":
    main()
