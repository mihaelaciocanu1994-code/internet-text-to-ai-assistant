"""Tiny causal (GPT-style) transformer plus helpers for generation and loading.

Shared by train.py and any later sampling scripts.
"""
import codecs

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    """One transformer block: causal self-attention + MLP, each with a residual."""

    def __init__(self, d_model, n_heads, dropout):
        super().__init__()
        self.n_heads = n_heads
        self.dropout = dropout
        self.ln1 = nn.LayerNorm(d_model)
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model),
        )

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).split(C, dim=2)
        h = self.n_heads
        q = q.view(B, T, h, C // h).transpose(1, 2)
        k = k.view(B, T, h, C // h).transpose(1, 2)
        v = v.view(B, T, h, C // h).transpose(1, 2)
        # is_causal=True: each position can only look at earlier positions.
        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    def __init__(self, vocab_size, block_size=64, d_model=64, n_layers=2,
                 n_heads=4, dropout=0.0):
        super().__init__()
        self.config = dict(vocab_size=vocab_size, block_size=block_size,
                           d_model=d_model, n_layers=n_layers,
                           n_heads=n_heads, dropout=dropout)
        self.block_size = block_size
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(block_size, d_model)
        self.blocks = nn.ModuleList(
            [Block(d_model, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.apply(self._init_weights)  # random initialisation

    @staticmethod
    def _init_weights(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        logits = self.head(self.ln_f(x))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)),
                                   targets.view(-1))
        return logits, loss


@torch.no_grad()
def generate_stream(model, start_internal_ids, max_new_tokens,
                    temperature=0.8, seed=1337):
    """Yield one new internal token ID at a time.

    A fresh generator with the same seed is used every call, so differences
    between samples come from the model's weights, not from randomness.
    """
    gen = torch.Generator().manual_seed(seed)
    was_training = model.training
    model.eval()
    try:
        idx = torch.tensor([start_internal_ids], dtype=torch.long)
        for _ in range(max_new_tokens):
            cond = idx[:, -model.block_size:]
            logits, _ = model(cond)
            probs = F.softmax(logits[:, -1, :] / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1, generator=gen)
            idx = torch.cat([idx, nxt], dim=1)
            yield nxt.item()
    finally:
        if was_training:
            model.train()


class TokenPrinter:
    """Prints tokens as they arrive.

    Some characters (e.g. ’ or €) are split across several tokens, so bytes
    are buffered until they form complete UTF-8 characters.
    """

    def __init__(self, enc):
        self.enc = enc
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.text = ""

    def add(self, original_token_id):
        self._emit(self.decoder.decode(
            self.enc.decode_single_token_bytes(original_token_id)))

    def finish(self):
        self._emit(self.decoder.decode(b"", final=True))

    def _emit(self, piece):
        if piece:
            self.text += piece
            print(piece, end="", flush=True)


def load_checkpoint(path, device="cpu"):
    """Return (model, checkpoint_dict). checkpoint_dict['id_list'][i] is the
    original o200k_base token ID for internal ID i."""
    ckpt = torch.load(path, map_location=device, weights_only=True)
    model = TinyGPT(**ckpt["config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt
