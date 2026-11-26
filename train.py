"""Train a tiny randomly initialised causal transformer on fineweb_corpus.txt.

Usage:  python train.py
All outputs are written to the results/ folder.
"""
import argparse
import csv
import math
import re
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # save plots to file, no window needed
import matplotlib.pyplot as plt
import tiktoken
import torch

from model import TinyGPT, TokenPrinter, generate_stream

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="fineweb_corpus.txt")
    p.add_argument("--tokens", default="fineweb_tokens.txt")
    p.add_argument("--out", default="results")
    p.add_argument("--seconds", type=float, default=60.0,
                   help="training time budget (sampling time not counted)")
    p.add_argument("--prompt", default="If you have set yourself on fire,")
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--sample-every", type=int, default=20)
    p.add_argument("--sample-tokens", type=int, default=30)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--print-every", type=int, default=1)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--block", type=int, default=64)
    p.add_argument("--d-model", type=int, default=64)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=4)
    p.add_argument("--lr", type=float, default=3e-3)
    return p.parse_args()


class Logger:
    """Print a line and also append it to training_log.txt."""

    def __init__(self, path):
        self.f = open(path, "w", encoding="utf-8")

    def __call__(self, line=""):
        print(line, flush=True)
        self.f.write(line + "\n")
        self.f.flush()


def read_token_file(path):
    text = Path(path).read_text(encoding="utf-8-sig")
    return [int(x) for x in re.findall(r"\d+", text)]


def load_and_validate(args, enc, out_dir, log):
    corpus = Path(args.corpus).read_text(encoding="utf-8-sig")
    expected = enc.encode(corpus, disallowed_special=())
    assert enc.decode(expected) == corpus, "o200k_base round-trip failed"
    log(f"Corpus: {len(corpus):,} characters -> {len(expected):,} o200k_base tokens")

    try:
        file_ids = read_token_file(args.tokens)
    except FileNotFoundError:
        file_ids = None

    if file_ids == expected:
        log(f"Token check: {args.tokens} matches the corpus exactly "
            f"({len(file_ids):,} IDs).")
        return corpus, expected

    if file_ids is None:
        log(f"Token check: {args.tokens} not found.")
    else:
        log(f"Token check: MISMATCH. File has {len(file_ids):,} IDs, "
            f"corpus encodes to {len(expected):,}.")
        first = next((i for i, (a, b) in enumerate(zip(file_ids, expected))
                      if a != b), min(len(file_ids), len(expected)))
        log(f"  First difference at position {first}.")
        if enc.encode(corpus.strip(), disallowed_special=()) == file_ids:
            log("  (File matches the corpus once leading/trailing whitespace "
                "is removed, e.g. a newline added by an editor.)")

    regen_path = out_dir / "fineweb_tokens_regenerated.txt"
    regen_path.write_text(", ".join(map(str, expected)), encoding="utf-8")
    log(f"Regenerated token IDs from the corpus -> {regen_path}")
    return corpus, expected


def main():
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    log = Logger(out_dir / "training_log.txt")
    torch.manual_seed(args.seed)

    enc = tiktoken.get_encoding("o200k_base")  # GPT-4o tokenizer
    _, ids = load_and_validate(args, enc, out_dir, log)

    # Compact vocabulary: only token IDs that actually occur in the corpus.
    id_list = sorted(set(ids))                       # internal -> original
    to_internal = {t: i for i, t in enumerate(id_list)}  # original -> internal
    data = torch.tensor([to_internal[t] for t in ids], dtype=torch.long)
    vocab = len(id_list)
    log(f"Compact vocabulary: {vocab:,} distinct tokens "
        f"(out of {enc.n_vocab:,} in o200k_base)")
    log(f"Loss of a uniform random guess: ln({vocab}) = {math.log(vocab):.3f}")

    # Sentence beginning for samples; every token must be in the vocabulary.
    # Inside the corpus most words carry a leading space (" If" is a different
    # token from "If"), so the prompt is also tried with a leading space.
    prompt = None
    for candidate in (args.prompt, " " + args.prompt):
        cand_ids = enc.encode(candidate, disallowed_special=())
        if cand_ids and all(t in to_internal for t in cand_ids):
            prompt, prompt_ids = candidate, cand_ids
            break
    if prompt is None:
        prompt_ids = ids[:8]
        prompt = enc.decode(prompt_ids)
        log(f"Prompt {args.prompt!r} has tokens not in the vocabulary; "
            f"using the corpus opening {prompt!r} instead.")
    prompt_internal = [to_internal[t] for t in prompt_ids][-args.block:]

    model = TinyGPT(vocab, block_size=args.block, d_model=args.d_model,
                    n_layers=args.layers, n_heads=args.heads)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"Model: {args.layers} layers, {args.heads} heads, d_model {args.d_model}, "
        f"context {args.block}, {n_params:,} parameters (random init)")
    log(f"Prompt: {prompt!r} | seed {args.seed} | temperature {args.temperature}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    batch_gen = torch.Generator().manual_seed(args.seed + 1)

    def get_batch():
        ix = torch.randint(len(data) - args.block - 1, (args.batch,),
                           generator=batch_gen)
        x = torch.stack([data[i:i + args.block] for i in ix])
        y = torch.stack([data[i + 1:i + 1 + args.block] for i in ix])
        return x, y

    samples_path = out_dir / "generated_samples.txt"
    with open(samples_path, "w", encoding="utf-8") as f:
        f.write(f"Prompt: {prompt!r}\nSeed: {args.seed}\n"
                f"Temperature: {args.temperature}\n"
                f"New tokens per sample: {args.sample_tokens}\n\n")

    def sample(label):
        print(f"\n--- sample {label} ---", flush=True)
        printer = TokenPrinter(enc)
        print(prompt, end="", flush=True)
        for internal in generate_stream(model, prompt_internal,
                                        args.sample_tokens,
                                        args.temperature, args.seed):
            printer.add(id_list[internal])  # map back to o200k_base ID
        printer.finish()
        print("\n", flush=True)
        with open(samples_path, "a", encoding="utf-8") as f:
            f.write(f"=== {label} ===\n{prompt}{printer.text}\n\n")
        log.f.write(f"[sample {label}] {prompt}{printer.text}\n")
        log.f.flush()

    sample("before training (step 0)")

    rows = []
    train_time = 0.0
    tokens_seen = 0
    step = 0
    log("step | loss | elapsed time | tokens/second")
    model.train()
    while train_time < args.seconds:
        t0 = time.perf_counter()
        x, y = get_batch()
        _, loss = model(x, y)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        train_time += time.perf_counter() - t0

        step += 1
        tokens_seen += x.numel()
        tps = tokens_seen / train_time
        rows.append((step, loss.item(), train_time, tps))
        if step % args.print_every == 0 or step == 1:
            log(f"{step:5d} | {loss.item():.4f} | {train_time:6.2f}s | {tps:9.0f}")
        if step % args.sample_every == 0:
            sample(f"step {step}")

    last = rows[-1]
    log(f"{last[0]:5d} | {last[1]:.4f} | {last[2]:6.2f}s | {last[3]:9.0f}  (final)")
    sample(f"end of training (step {step})")

    # Save log as CSV
    with open(out_dir / "training_log.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["step", "loss", "elapsed_seconds", "tokens_per_second"])
        w.writerows(rows)

    # Loss plot
    plt.figure(figsize=(8, 4.5))
    plt.plot([r[0] for r in rows], [r[1] for r in rows], linewidth=1,
             label="training loss")
    plt.axhline(math.log(vocab), color="grey", linestyle="--",
                label=f"random guess ln({vocab})")
    plt.xlabel("step")
    plt.ylabel("cross-entropy loss")
    plt.title("Tiny transformer on fineweb_corpus.txt")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "loss_plot.png", dpi=150)

    # Checkpoint
    torch.save({
        "model_state": model.state_dict(),
        "config": model.config,
        "id_list": id_list,
        "tokenizer": "o200k_base",
        "prompt": prompt,
        "seed": args.seed,
        "steps": step,
        "train_seconds": train_time,
        "final_loss": last[1],
    }, out_dir / "model_checkpoint.pt")

    log(f"\nSaved to {out_dir}/: training_log.txt, training_log.csv, "
        f"loss_plot.png, generated_samples.txt, model_checkpoint.pt")


if __name__ == "__main__":
    main()
