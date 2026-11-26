"""Pick a short sentence that appears exactly in fineweb_corpus.txt, use its
first few words as the beginning, and complete it 100 times.

Usage:  python corpus_completions.py
Output: corpus_completions.txt
"""
import random
import re
import sys
from pathlib import Path

import tiktoken

from model import generate_stream, load_checkpoint

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CHECKPOINT = "results/model_checkpoint.pt"
CORPUS = "fineweb_corpus.txt"
N_COMPLETIONS = 100
BEGIN_WORDS = 3          # how many words of the sentence to use as beginning
NEW_TOKENS = 20
TEMPERATURE = 0.8
SEED = 1337
OUT = "corpus_completions.txt"
COMMON_OPENINGS = {"The", "It", "This", "I", "We", "A", "In", "If", "You",
                   "There", "They", "He", "She"}


def candidate_sentences(corpus):
    """Short sentences from the corpus, common openings first."""
    sentences = re.split(r"(?<=[.!?])\s+", corpus)
    short = [s.strip() for s in sentences
             if 5 <= len(s.split()) <= 15 and s.strip() in corpus]
    rng = random.Random(SEED)
    rng.shuffle(short)  # reproducible but not just the first sentence
    common = [s for s in short if s.split()[0] in COMMON_OPENINGS]
    other = [s for s in short if s.split()[0] not in COMMON_OPENINGS]
    return common + other


def main():
    model, ckpt = load_checkpoint(CHECKPOINT)
    enc = tiktoken.get_encoding("o200k_base")
    id_list = ckpt["id_list"]
    to_internal = {t: i for i, t in enumerate(id_list)}
    corpus = Path(CORPUS).read_text(encoding="utf-8-sig")

    chosen = beginning = prompt_ids = None
    for sentence in candidate_sentences(corpus):
        begin = " ".join(sentence.split()[:BEGIN_WORDS])
        # In the corpus most sentences follow a space, so " The" (with space)
        # is the token the model has seen; try both forms.
        for form in (begin, " " + begin):
            ids = enc.encode(form)
            if ids and all(t in to_internal for t in ids):
                chosen, beginning, prompt_ids = sentence, form, ids
                break
        if chosen:
            break

    if chosen is None:
        print("No suitable sentence found. Try a smaller BEGIN_WORDS.")
        return

    start = [to_internal[t] for t in prompt_ids]
    with open(OUT, "w", encoding="utf-8") as f:
        header = (f"Chosen sentence (appears exactly in {CORPUS}):\n{chosen}\n\n"
                  f"Beginning used: {beginning!r}\n"
                  f"All beginning tokens are in the model's vocabulary: "
                  f"{[(t, enc.decode([t])) for t in prompt_ids]}\n\n")
        f.write(header)
        print(header)
        for i in range(N_COMPLETIONS):
            new_ids = [id_list[x] for x in generate_stream(
                model, start, NEW_TOKENS, TEMPERATURE, seed=i)]
            text = beginning.lstrip() + enc.decode(new_ids)
            line = f"{i + 1:3d}: {text}".replace("\n", " ")
            print(line)
            f.write(line + "\n")

    print(f"\nSaved {N_COMPLETIONS} completions to {OUT}")


if __name__ == "__main__":
    main()
