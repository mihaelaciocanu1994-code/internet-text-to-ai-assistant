"""Load the trained model and complete "The sky is" 100 times.

Usage:  python sky_completions.py
Output: sky_completions.txt
"""
import sys

import tiktoken

from model import generate_stream, load_checkpoint

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKPOINT = "results/model_checkpoint.pt"
PROMPT = "The sky is"
N_COMPLETIONS = 100
NEW_TOKENS = 20
TEMPERATURE = 0.8
OUT = "sky_completions.txt"


def main():
    model, ckpt = load_checkpoint(CHECKPOINT)
    enc = tiktoken.get_encoding("o200k_base")
    id_list = ckpt["id_list"]
    to_internal = {t: i for i, t in enumerate(id_list)}

    prompt_ids = enc.encode(PROMPT)
    missing = [(t, enc.decode([t])) for t in prompt_ids if t not in to_internal]

    with open(OUT, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {PROMPT!r}\n")
        f.write(f"Prompt tokens (o200k_base): "
                f"{[(t, enc.decode([t])) for t in prompt_ids]}\n")
        f.write(f"Model vocabulary size: {len(id_list)} tokens\n\n")

        if missing:
            msg = ("FAILED: these prompt tokens never appear in the training "
                   f"corpus, so the model has no ID for them: {missing}\n"
                   "The model can only read and write tokens it saw in "
                   "fineweb_corpus.txt.\n")
            f.write(msg)
            print(msg)
            return

        start = [to_internal[t] for t in prompt_ids]
        for i in range(N_COMPLETIONS):
            new_ids = [id_list[x] for x in generate_stream(
                model, start, NEW_TOKENS, TEMPERATURE, seed=i)]
            text = PROMPT + enc.decode(new_ids)
            line = f"{i + 1:3d}: {text}".replace("\n", " ")
            print(line)
            f.write(line + "\n")

    print(f"\nSaved {N_COMPLETIONS} completions to {OUT}")


if __name__ == "__main__":
    main()
