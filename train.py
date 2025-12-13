#!/usr/bin/env python3
"""
Train GPT-2 Small (decoder-only ~124M params) from scratch on a single text file (input.txt).

Key features added vs the reference file:
- Proper CLI args (batch, seq len, lr, steps, etc.)
- Deterministic seeding
- AMP (fp16/bf16) + grad accumulation for RTX 4060 Ti 16GB
- Logging to:
    logs/train.log
    logs/train_metrics.csv
    samples/samples.txt  (periodic generations)
- Checkpointing (last + best by loss) to checkpoints/ (ignored by git)
- Resume from checkpoint
- No pretrained weights used (from_pretrained remains but is not called)

Typical run (overnight):
python train.py --input_file input.txt --steps 200000 --batch_size 8 --seq_len 256 --grad_accum 4 --lr 3e-4 --log_every 50 --sample_every 2000 --save_every 2000

Goal: drive training loss < 0.1 (this will overfit the tiny dataset; that's intended for the assignment).
"""
from __future__ import annotations

import os
import math
import time
import argparse
import csv
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.nn import functional as F

# ---------------------------
# Model (GPT-2 style)
# ---------------------------

class CausalSelfAttention(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        # residual scaling trick (NanoGPT)
        self.c_proj.NANOGPT_SCALE_INIT = 1
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # attention mask (buffer)
        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(1, 1, config.block_size, config.block_size),
            persistent=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()
        qkv = self.c_attn(x)                       # (B, T, 3C)
        q, k, v = qkv.split(self.n_embd, dim=2)    # each (B, T, C)
        hs = C // self.n_head
        k = k.view(B, T, self.n_head, hs).transpose(1, 2)  # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, hs).transpose(1, 2)
        v = v.view(B, T, self.n_head, hs).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(hs))  # (B, nh, T, T)
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        att = F.softmax(att, dim=-1)
        y = att @ v  # (B, nh, T, hs)

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        y = self.c_proj(y)
        return y

class MLP(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.c_fc   = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu   = nn.GELU(approximate="tanh")
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)
        self.c_proj.NANOGPT_SCALE_INIT = 1

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.c_proj(self.gelu(self.c_fc(x)))

class Block(nn.Module):
    def __init__(self, config: "GPTConfig"):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp  = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    # Defaults = GPT-2 Small (124M)
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768

class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd),
            wpe = nn.Embedding(config.block_size, config.n_embd),
            h   = nn.ModuleList([Block(config) for _ in range(config.n_layer)]),
            ln_f= nn.LayerNorm(config.n_embd),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # weight tying
        self.transformer.wte.weight = self.lm_head.weight

        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            std = 0.02
            if hasattr(module, "NANOGPT_SCALE_INIT"):
                std *= (2 * self.config.n_layer) ** -0.5
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        B, T = idx.size()
        assert T <= self.config.block_size, f"T={T} > block_size={self.config.block_size}"
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        tok = self.transformer.wte(idx)
        pos = self.transformer.wpe(pos)
        x = tok + pos
        for block in self.transformer.h:
            x = block(x)
        x = self.transformer.ln_f(x)
        logits = self.lm_head(x)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss

# ---------------------------
# Data
# ---------------------------

class DataLoaderLite:
    """
    Simple token-stream loader.
    - Loads all tokens into memory once.
    - Samples contiguous (B,T) blocks by advancing a cursor.
    - For better mixing, optionally jumps to a random position each batch.
    """
    def __init__(self, input_file: str, B: int, T: int, randomize: bool, seed: int):
        self.B, self.T = B, T
        self.randomize = randomize
        self.rng = torch.Generator().manual_seed(seed)

        with open(input_file, "r", encoding="utf-8") as f:
            text = f.read()

        import tiktoken
        self.enc = tiktoken.get_encoding("gpt2")
        tokens = self.enc.encode(text)
        self.tokens = torch.tensor(tokens, dtype=torch.long)
        self.n_tokens = int(self.tokens.numel())
        self.current_position = 0

        if self.n_tokens < (B * T + 1):
            raise ValueError(f"input too small: {self.n_tokens} tokens, need at least {B*T+1}")

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T

        if self.randomize:
            # choose a start so that we have B*T+1 tokens available
            max_start = self.n_tokens - (B * T + 1)
            start = int(torch.randint(low=0, high=max_start + 1, size=(1,), generator=self.rng).item())
            buf = self.tokens[start:start + B * T + 1]
        else:
            buf = self.tokens[self.current_position:self.current_position + B * T + 1]
            self.current_position += B * T
            if self.current_position + (B * T + 1) > self.n_tokens:
                self.current_position = 0

        x = buf[:-1].view(B, T)
        y = buf[1:].view(B, T)
        return x, y

# ---------------------------
# Utils: logging / ckpt / sampling
# ---------------------------

def setup_logging(out_dir: Path) -> tuple[logging.Logger, Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    logs_dir = out_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    log_file = logs_dir / "train.log"
    csv_file = logs_dir / "train_metrics.csv"

    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(fmt)
    logger.addHandler(sh)

    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    # init CSV (append-safe)
    if not csv_file.exists():
        with open(csv_file, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "loss", "lr", "tokens_per_sec", "elapsed_sec"])

    return logger, log_file, csv_file

@torch.no_grad()
def generate_samples(
    model: GPT,
    enc,
    device: str,
    prompts: list[str],
    max_new_tokens: int,
    top_k: int,
    temperature: float,
    out_path: Path,
    step: int,
) -> None:
    model.eval()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "a", encoding="utf-8") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write(f"STEP {step}\n")

        for p in prompts:
            tokens = enc.encode(p)
            x = torch.tensor(tokens, dtype=torch.long, device=device)[None, :]  # (1, T)

            for _ in range(max_new_tokens):
                if x.size(1) > model.config.block_size:
                    x = x[:, -model.config.block_size:]
                logits, _ = model(x, None)
                logits = logits[:, -1, :] / max(1e-8, temperature)
                probs = F.softmax(logits, dim=-1)
                if top_k > 0:
                    topk_probs, topk_idx = torch.topk(probs, k=min(top_k, probs.size(-1)), dim=-1)
                    ix = torch.multinomial(topk_probs, num_samples=1)
                    next_tok = torch.gather(topk_idx, -1, ix)
                else:
                    next_tok = torch.multinomial(probs, num_samples=1)
                x = torch.cat([x, next_tok], dim=1)

            f.write("\nPROMPT:\n")
            f.write(p + "\n")
            f.write("\nOUTPUT:\n")
            f.write(enc.decode(x[0].tolist()) + "\n")

    model.train()

def save_checkpoint(
    out_dir: Path,
    model: GPT,
    optimizer: torch.optim.Optimizer,
    step: int,
    best_loss: float,
    filename: str,
) -> Path:
    ckpt_dir = out_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    path = ckpt_dir / filename
    payload = {
        "step": step,
        "best_loss": float(best_loss),
        "config": model.config.__dict__,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(payload, path)
    return path

def load_checkpoint(path: Path, model: GPT, optimizer: torch.optim.Optimizer, device: str) -> tuple[int, float]:
    ckpt = torch.load(path, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])
    step = int(ckpt.get("step", 0))
    best_loss = float(ckpt.get("best_loss", float("inf")))
    return step, best_loss

def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"

# ---------------------------
# Main
# ---------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", type=str, default="input.txt")
    p.add_argument("--out_dir", type=str, default="out")
    p.add_argument("--seed", type=int, default=1337)

    # model
    p.add_argument("--block_size", type=int, default=1024, help="max context length (<=1024 for GPT-2)")
    p.add_argument("--n_layer", type=int, default=12)
    p.add_argument("--n_head", type=int, default=12)
    p.add_argument("--n_embd", type=int, default=768)

    # training
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--seq_len", type=int, default=256, help="training sequence length (T)")
    p.add_argument("--grad_accum", type=int, default=4, help="gradient accumulation steps")
    p.add_argument("--steps", type=int, default=200000, help="total optimizer steps")
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--clip_grad", type=float, default=1.0)
    p.add_argument("--randomize_batches", action="store_true", help="randomize start position each batch")
    p.add_argument("--target_loss", type=float, default=0.1, help="stop early once training loss <= this")

    # logging/saving
    p.add_argument("--log_every", type=int, default=50)
    p.add_argument("--save_every", type=int, default=2000)
    p.add_argument("--sample_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default="", help="path to checkpoint to resume from")

    # sampling
    p.add_argument("--prompt", type=str, default="BIANCA:\n", help="base prompt for generation")
    p.add_argument("--num_prompts", type=int, default=2)
    p.add_argument("--max_new_tokens", type=int, default=200)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--temperature", type=float, default=0.9)

    # perf
    p.add_argument("--compile", action="store_true", help="torch.compile (PyTorch 2.x)")
    p.add_argument("--amp", type=str, default="fp16", choices=["off", "fp16", "bf16"])
    return p.parse_args()

def main() -> None:
    args = parse_args()

    out_dir = Path(args.out_dir)
    logger, log_path, csv_path = setup_logging(out_dir)

    device = get_device()
    print(f"Device used in training: {device}")
    logger.info(f"Device used in training: {device}")
    logger.info(f"Args: {vars(args)}")
    logger.info(f"Log file: {log_path}")
    logger.info(f"CSV file: {csv_path}")

    # seeds
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed(args.seed)

    # data
    train_loader = DataLoaderLite(
        input_file=args.input_file,
        B=args.batch_size,
        T=args.seq_len,
        randomize=args.randomize_batches,
        seed=args.seed,
    )
    logger.info(f"Loaded {train_loader.n_tokens} tokens from {args.input_file}")
    logger.info(f"Batch tokens per step (before grad_accum): {args.batch_size * args.seq_len}")
    logger.info(f"Effective tokens per optimizer step: {args.batch_size * args.seq_len * args.grad_accum}")

    # model
    cfg = GPTConfig(
        block_size=args.block_size,
        vocab_size=50257,
        n_layer=args.n_layer,
        n_head=args.n_head,
        n_embd=args.n_embd,
    )
    model = GPT(cfg).to(device)

    if args.compile and hasattr(torch, "compile"):
        logger.info("torch.compile enabled")
        model = torch.compile(model)  # type: ignore[attr-defined]

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    # AMP setup
    use_amp = (args.amp != "off") and (device == "cuda")
    if args.amp == "bf16":
        amp_dtype = torch.bfloat16
    else:
        amp_dtype = torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and amp_dtype == torch.float16))

    # resume
    start_step = 0
    best_loss = float("inf")
    if args.resume:
        ckpt_path = Path(args.resume)
        if ckpt_path.exists():
            start_step, best_loss = load_checkpoint(ckpt_path, model, optimizer, device=device)
            logger.info(f"Resumed from {ckpt_path} at step={start_step}, best_loss={best_loss:.6f}")
        else:
            logger.warning(f"--resume path not found: {ckpt_path}")

    # prompts
    prompts = [args.prompt for _ in range(args.num_prompts)]
    samples_path = out_dir / "samples" / "samples.txt"

    # training loop
    model.train()
    t0 = time.time()
    tokens_seen = 0

    for step in range(start_step, args.steps):
        optimizer.zero_grad(set_to_none=True)

        loss_accum = 0.0
        # grad accumulation
        for micro in range(args.grad_accum):
            x, y = train_loader.next_batch()
            x, y = x.to(device), y.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                _, loss = model(x, y)
                assert loss is not None
                loss = loss / args.grad_accum

            loss_accum += float(loss.item())

            if scaler.is_enabled():
                scaler.scale(loss).backward()
            else:
                loss.backward()

            tokens_seen += x.numel()

        # grad clip + step
        if args.clip_grad > 0:
            if scaler.is_enabled():
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        if scaler.is_enabled():
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        # metrics
        loss_val = loss_accum  # already averaged by grad_accum
        elapsed = time.time() - t0
        tps = tokens_seen / max(1e-9, elapsed)
        lr = optimizer.param_groups[0]["lr"]

        # log
        if (step + 1) % args.log_every == 0 or step == start_step:
            logger.info(f"step {step+1}/{args.steps} | loss {loss_val:.6f} | lr {lr:.2e} | tok/s {tps:.1f}")
            with open(csv_path, "a", newline="", encoding="utf-8") as f:
                w = csv.writer(f)
                w.writerow([step + 1, f"{loss_val:.8f}", f"{lr:.8e}", f"{tps:.2f}", f"{elapsed:.2f}"])

        # save checkpoints
        if (step + 1) % args.save_every == 0:
            last_path = save_checkpoint(out_dir, model, optimizer, step + 1, best_loss, "last.pt")
            logger.info(f"Saved checkpoint: {last_path}")

        # sample generation
        if (step + 1) % args.sample_every == 0:
            generate_samples(
                model=model,
                enc=train_loader.enc,
                device=device,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                temperature=args.temperature,
                out_path=samples_path,
                step=step + 1,
            )
            logger.info(f"Appended samples to: {samples_path}")

        # track best loss (for "best.pt")
        if loss_val < best_loss:
            best_loss = loss_val
            best_path = save_checkpoint(out_dir, model, optimizer, step + 1, best_loss, "best.pt")
            logger.info(f"New best loss {best_loss:.6f} -> saved: {best_path}")

        # early stop
        if loss_val <= args.target_loss:
            logger.info(f"Target reached: loss {loss_val:.6f} <= {args.target_loss}. Stopping.")
            save_checkpoint(out_dir, model, optimizer, step + 1, best_loss, "last.pt")
            generate_samples(
                model=model,
                enc=train_loader.enc,
                device=device,
                prompts=prompts,
                max_new_tokens=args.max_new_tokens,
                top_k=args.top_k,
                temperature=args.temperature,
                out_path=samples_path,
                step=step + 1,
            )
            break

    logger.info("Training complete.")

if __name__ == "__main__":
    main()
