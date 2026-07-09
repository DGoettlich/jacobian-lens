"""Try J-lens residual-space swaps on two pre-1913 knowledge prompts."""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens

if __name__ == "__main__":
    model_name = "uzh-echist-org/Ranke-4B-1913"
    lens_path = "artifacts/ranke-1913-jlens.pt"
    dtype = torch.bfloat16
    strength = 1.0
    cascading = False
    max_new_tokens = 1

    hf_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
    ).cuda()
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = jlens.from_hf(hf_model, tokenizer)
    lens = jlens.JacobianLens.load(lens_path)
    layer_start = int(0.18 * model.n_layers)
    layer_stop = int(0.63 * model.n_layers)
    layers = [
        layer for layer in lens.source_layers if layer_start <= layer <= layer_stop
    ]

    def token_ids(text):
        return tokenizer.encode(text, add_special_tokens=False)

    def one_token(text):
        ids = token_ids(text)
        if len(ids) != 1:
            raise ValueError(f"{text!r} is not one token: {ids}")
        return ids[0]

    def rank(logits, token_id):
        return int((logits > logits[token_id]).sum().item() + 1)

    def greedy(prompt, *, source_id=None, target_id=None, strength=1.0):
        text = prompt
        for _ in range(max_new_tokens):
            if source_id is None:
                _, logits, _ = lens.apply(model, text, layers=[layers[0]], positions=[-1])
                next_id = int(logits[-1].argmax().item())
            else:
                _, logits, _ = lens.swap(
                    model,
                    text,
                    source_id,
                    target_id,
                    strength=strength,
                    layers=layers,
                    positions=None,
                    cascading=cascading,
                )
                next_id = int(logits[-1].argmax().item())
            text += tokenizer.decode([next_id], clean_up_tokenization_spaces=False)
        return text[len(prompt) :]

    probes = {
        # Source/target tokens define J-lens directions; the prompt text is unchanged.
        "evolution_darwin_to_newton": (
            "Who originated the theory of evolution? Answer in one word:",
            one_token(" Darwin"),
            one_token(" Newton"),
            [("Darwin", one_token(" Darwin")), ("Newton", one_token(" Newton"))],
        ),
        "hamlet_shakespeare_to_milton": (
            "Who was the author of Hamlet? Answer in one word:",
            one_token(" Shakespeare"),
            one_token(" Milton"),
            [
                ("Shakespeare", one_token(" Shakespeare")),
                ("Milton", one_token(" Milton")),
            ],
        ),
    }

    for name, (prompt, source_id, target_id, tracked_tokens) in probes.items():
        print()
        print(name)
        print(f"prompt: {prompt!r}")
        print(f"baseline:{greedy(prompt)}")

        _, base_logits, _ = lens.apply(
            model,
            prompt,
            layers=[layers[0]],
            positions=[-1],
        )
        _, changed_logits, _ = lens.swap(
            model,
            prompt,
            source_id,
            target_id,
            strength=strength,
            layers=layers,
            positions=None,
            cascading=cascading,
        )
        base = base_logits[-1]
        changed = changed_logits[-1]
        print(
            f"swap strength={strength} cascading={cascading} "
            f"layers={layers[0]}..{layers[-1]}"
        )
        for label, token_id in tracked_tokens:
            print(
                f"  {label:12s} "
                f"rank {rank(base, token_id):6d} -> {rank(changed, token_id):6d} "
                f"logit {base[token_id]:8.3f} -> {changed[token_id]:8.3f}"
            )
        print(
            "  generated:"
            f"{greedy(prompt, source_id=source_id, target_id=target_id, strength=strength)}"
        )
