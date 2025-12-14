# ERA-V4 — Train a Decoder-Only Transformer (GPT-2 Style) From Scratch

## Objective
Train a **decoder-only Transformer (GPT-2 style, ≥124M parameters)** **from scratch** (no pretrained weights) on the provided `input.txt` until **training loss < 0.099999**.

## Quick Links
- Training script: `train.py`
- Dependencies: `requirements.txt`
- Run outputs (generated): `out/` (gitignored)

## Repository Structure (recommended)
```text
.
├─ train.py
├─ requirements.txt
├─ README.md
├─ update_readme.py
├─ .gitignore
├─ data/
│  ├─ input.txt               # commit only if permitted by the course
│  └─ README.md               # where to place input.txt if not committed
└─ out/                       # generated (gitignored)
   ├─ logs/
   ├─ samples/
   └─ checkpoints/
```

## Environment
- GPU: RTX 4060 Ti (16 GB)
- CPU: i7
- RAM: 32 GB
- OS: [Windows/Linux]
- Python: 3.10+ recommended

## Setup

### Create environment
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux
source .venv/bin/activate
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Dataset placement
Place the provided dataset at:
- `data/input.txt` (recommended)

## Train

### Sanity-check run (2–5 minutes)
```bash
python train.py \
  --input_file data/input.txt \
  --out_dir out_sanity \
  --steps 200 \
  --batch_size 4 \
  --seq_len 128 \
  --grad_accum 1 \
  --lr 3e-4 \
  --log_every 10 \
  --save_every 100 \
  --sample_every 100
```

### Full training run
```bash
python train.py \
  --input_file data/input.txt \
  --out_dir out \
  --steps 200000 \
  --batch_size 8 \
  --seq_len 256 \
  --grad_accum 4 \
  --lr 3e-4 \
  --log_every 50 \
  --save_every 2000 \
  --sample_every 2000
```



### PowerShell wrapper for long training runs (Windows)

For long-running or overnight training on Windows, you can use the included PowerShell wrapper script `training.ps1`.

This wrapper **invokes the same `train.py` entry point** and passes the arguments in a PowerShell-friendly way (via an argument array), typically including GPU-optimized settings such as AMP (`--amp fp16`) and early stopping (`--target_loss`).  
The **training logic, model architecture, and loss function are identical** — only the way arguments are provided differs.

#### How to run

From the repository root (PowerShell):

```powershell
.\training.ps1
```

Internally, the script executes:

```powershell
python train.py @trainArgs
```

where `@trainArgs` contains the same flags you would otherwise pass on the command line.

#### Why use `training.ps1`?

- Avoids long multi-line CLI commands in PowerShell
- Safer for overnight runs (single entry point)
- Easier to tune `--batch_size` and `--grad_accum` for the local GPU
- Supports AMP and early stopping configuration in one place

> Note: The CLI commands above remain the **canonical documentation** for cross-platform reproduction.  
> `training.ps1` is provided as a **Windows convenience wrapper**.

## Model
GPT-2 style decoder-only Transformer trained from scratch (no pretrained weights).
- Tokenizer: GPT-2 BPE via `tiktoken`
- Causal LM loss: cross-entropy over next-token prediction

(If you changed the config, document it here.)

## Outputs / Artifacts
Training produces the following under the run directory (`--out_dir`, e.g. `out/`):
- Logs:
  - `logs/train.log`
  - `logs/train_metrics.csv` (step, loss, etc.)
- Samples:
  - `samples/samples.txt` (periodic generations)
- Checkpoints:
  - `checkpoints/last.pt` (resume)
  - `checkpoints/best.pt` (lowest loss)


### What is `samples/samples.txt` and how is it generated?

During training, the script periodically performs **inference-only sampling** to qualitatively check what the model has learned so far.  
This is controlled by:

- `--sample_every N`: generate samples every **N training steps**
- `--prompt "...":` the text prompt used to start generation (default: `BIANCA:\n`)
- `--num_prompts K`: how many generations to produce per sampling event
- `--max_new_tokens M`: how many **new tokens** to generate after the prompt
- `--top_k` and `--temperature`: sampling controls (diversity vs. determinism)

**What gets written to `samples.txt`:**
- The current **STEP** number
- The **PROMPT** string used
- The model’s **OUTPUT** continuation, generated **autoregressively** (next-token prediction repeated `max_new_tokens` times)
- A separator line for readability

**Important:** `samples.txt` is **not used for training loss**.  
Loss is computed only from next-token prediction on the training batches.  
Samples are generated to provide **human-readable evidence** that the model is learning the text distribution and formatting (e.g., Shakespeare-style dialogue).

**Where this happens in training (conceptually):**
1. Training runs normally for many steps (forward → loss → backward → optimizer step).
2. When the step hits a multiple of `--sample_every`, training briefly switches to `model.eval()` (no gradients).
3. The prompt is tokenized using the same tokenizer as training.
4. The model repeatedly predicts the next-token distribution and samples tokens until `--max_new_tokens` is reached.
5. The decoded text is appended to `samples/samples.txt`.

This is why, when prompted with `BIANCA:\n`, the model generates dialogue-like continuations and speaker tags it learned from `input.txt`.

**Keep safe (for grading + later deployment):**
- `checkpoints/best.pt`
- `logs/train_metrics.csv`
- `samples/samples.txt`

## Results (auto-updated)
This section is **auto-filled** by `update_readme.py`.


### Loss curve (from `logs/train_metrics.csv`)
<!--AUTO:PLOT:START-->
![Training loss curve](out/runs/gpt2_124m_bs8_sl128_ga16_lr3e-4/logs/loss_curve.png)
<!--AUTO:PLOT:END-->


### Summary
<!--AUTO:SUMMARY:START-->
- Updated: **2025-12-14 11:42**
- Run directory: `out/runs/gpt2_124m_bs8_sl128_ga16_lr3e-4`
- Best loss: **0.089860** at step **2130**
- Last logged loss: **0.089860**
- Total logged rows: **44**
<!--AUTO:SUMMARY:END-->

### Training log excerpt
<!--AUTO:LOGS:START-->
```text
2025-12-14 03:38:33,518 | INFO | New best loss 1.784571 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:38:36,559 | INFO | step 1250/400000 | loss 2.144839 | lr 3.00e-04 | tok/s 15045.4
2025-12-14 03:39:03,503 | INFO | New best loss 1.743012 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:39:16,109 | INFO | step 1300/400000 | loss 1.861244 | lr 3.00e-04 | tok/s 15253.3
2025-12-14 03:39:19,413 | INFO | New best loss 1.623277 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:39:49,345 | INFO | New best loss 1.569998 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:39:58,336 | INFO | step 1350/400000 | loss 1.793567 | lr 3.00e-04 | tok/s 15425.3
2025-12-14 03:40:05,250 | INFO | New best loss 1.451146 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:40:35,249 | INFO | New best loss 1.415953 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:40:40,511 | INFO | step 1400/400000 | loss 1.875473 | lr 3.00e-04 | tok/s 15589.0
2025-12-14 03:40:51,181 | INFO | New best loss 1.400855 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:41:20,044 | INFO | step 1450/400000 | loss 1.635456 | lr 3.00e-04 | tok/s 15769.2
2025-12-14 03:41:32,806 | INFO | New best loss 1.380961 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:41:37,417 | INFO | New best loss 1.303038 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:41:53,326 | INFO | New best loss 1.274949 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:42:05,357 | INFO | step 1500/400000 | loss 1.298733 | lr 3.00e-04 | tok/s 15888.2
2025-12-14 03:42:10,793 | INFO | New best loss 1.233707 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:42:32,818 | INFO | New best loss 1.191891 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:42:37,532 | INFO | New best loss 1.113902 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:42:59,094 | INFO | step 1550/400000 | loss 1.314463 | lr 3.00e-04 | tok/s 15926.0
2025-12-14 03:43:06,901 | INFO | New best loss 1.110066 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:43:10,990 | INFO | New best loss 0.979728 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:43:28,337 | INFO | New best loss 0.938474 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:43:44,586 | INFO | New best loss 0.922951 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:43:49,106 | INFO | step 1600/400000 | loss 1.115450 | lr 3.00e-04 | tok/s 15993.8
2025-12-14 03:44:14,249 | INFO | New best loss 0.915210 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:44:19,710 | INFO | New best loss 0.797861 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:44:35,892 | INFO | New best loss 0.739213 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:44:50,028 | INFO | step 1650/400000 | loss 0.735999 | lr 3.00e-04 | tok/s 15966.1
2025-12-14 03:44:52,824 | INFO | New best loss 0.735999 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:45:24,401 | INFO | New best loss 0.731888 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:45:29,589 | INFO | New best loss 0.717602 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:45:34,509 | INFO | New best loss 0.653085 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:45:49,150 | INFO | New best loss 0.641445 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:45:54,359 | INFO | New best loss 0.569461 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:46:04,785 | INFO | step 1700/400000 | loss 0.758648 | lr 3.00e-04 | tok/s 15828.7
2025-12-14 03:46:24,287 | INFO | New best loss 0.534095 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:46:45,955 | INFO | New best loss 0.496289 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:46:52,733 | INFO | step 1750/400000 | loss 0.618937 | lr 3.00e-04 | tok/s 15908.9
2025-12-14 03:47:01,948 | INFO | New best loss 0.466642 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:47:05,834 | INFO | New best loss 0.453498 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:47:22,625 | INFO | New best loss 0.389749 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:47:38,589 | INFO | New best loss 0.362131 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:47:41,680 | INFO | step 1800/400000 | loss 0.566370 | lr 3.00e-04 | tok/s 15977.7
2025-12-14 03:48:08,582 | INFO | New best loss 0.324948 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:48:21,252 | INFO | step 1850/400000 | loss 0.382905 | lr 3.00e-04 | tok/s 16114.5
2025-12-14 03:48:29,274 | INFO | New best loss 0.298434 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:48:57,165 | INFO | New best loss 0.283849 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:49:01,032 | INFO | New best loss 0.282175 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:49:07,839 | INFO | New best loss 0.257717 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:49:16,889 | INFO | step 1900/400000 | loss 0.283307 | lr 3.00e-04 | tok/s 16126.0
2025-12-14 03:49:22,537 | INFO | New best loss 0.255437 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:49:26,557 | INFO | New best loss 0.231858 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:49:31,245 | INFO | New best loss 0.212172 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:49:58,242 | INFO | New best loss 0.211266 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:50:03,243 | INFO | New best loss 0.184253 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:50:07,127 | INFO | New best loss 0.178253 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:50:12,136 | INFO | New best loss 0.174332 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:50:17,460 | INFO | step 1950/400000 | loss 0.211753 | lr 3.00e-04 | tok/s 16101.3
2025-12-14 03:50:28,495 | INFO | New best loss 0.167420 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:50:40,108 | INFO | New best loss 0.165993 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:50:48,807 | INFO | New best loss 0.155774 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:51:03,627 | INFO | New best loss 0.153930 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:51:14,276 | INFO | New best loss 0.153224 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:51:15,895 | INFO | step 2000/400000 | loss 0.213669 | lr 3.00e-04 | tok/s 16092.9
2025-12-14 03:51:28,976 | INFO | Saved checkpoint: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\last.pt
2025-12-14 03:51:33,035 | INFO | Appended samples to: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\samples\samples.txt
2025-12-14 03:51:43,319 | INFO | New best loss 0.152596 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:51:48,938 | INFO | New best loss 0.152478 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:51:53,227 | INFO | New best loss 0.133115 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:51:57,341 | INFO | New best loss 0.124396 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:52:24,572 | INFO | New best loss 0.118997 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:52:38,026 | INFO | step 2050/400000 | loss 0.138859 | lr 3.00e-04 | tok/s 15924.3
2025-12-14 03:52:43,541 | INFO | New best loss 0.109542 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:53:18,306 | INFO | step 2100/400000 | loss 0.139297 | lr 3.00e-04 | tok/s 16040.4
2025-12-14 03:53:37,854 | INFO | New best loss 0.106622 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:53:45,302 | INFO | New best loss 0.104635 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:54:02,973 | INFO | New best loss 0.089860 -> saved: out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4\checkpoints\best.pt
2025-12-14 03:54:02,973 | INFO | Target reached: loss 0.089860 <= 0.099999. Stopping.
2025-12-14 03:54:27,419 | INFO | Training complete.
```
<!--AUTO:LOGS:END-->

### Key metrics
<!--AUTO:METRICS:START-->
| Metric | Value |
|---|---:|
| Best loss | 0.089860 |
| Step at best loss | 2130 |
| Last logged loss | 0.089860 |
| Total logged rows | 44 |
<!--AUTO:METRICS:END-->

### Sample generations
<!--AUTO:SAMPLES:START-->
```text
a gracious tale.

PROSPERO:
That hast nature much; for thou not weep!

LUCIO:
Wilt thou, because thy husband. Come, get pale said women too,
And have have made myself when thou sawest aboard the gate.

ISABELLA:
That hast thou, spoke you both,
Doth hath been much but before it?

DUKE VINCENTIO:
So do rustous and any man gives himself.

LUCIO:
Unhappy God was ever hear away, nor let me once distinguish streaks of Claudio, good sir. Wife'?
you are your val VINCENTIO:
Was all other looks once, sir; do not tear the truth, my brother's first speak does to fie, my peace of speaking, my daughter's oath's obedience, my daughter's all an gentle heaven,

================================================================================
STEP 2144

PROMPT:
BIANCA:


OUTPUT:
BIANCA:
Which ready to be to do of 'tis to us love: I'll will give us thanks, and leave yourselves:
The worst is to be o' the fairnessish, be here to make no long, good which means see 'tis so.

BRUTUS:
Ah, here forbid' sword, and to the sore.
But all make a very grows despised within from the people;
And were not 'twixt us from time to chide.
Go bid it report, make a bark again: in God to wear! how to thrust her:
O, but soon blest to live, which well long in those strong sights
Lies can do put way be content, such a place, which honour, which long fair day from my courage, both undone! what soon stamp, I banish us so, which terms to live, which she breathest in the queen, the prince, he is mine eye:
Lest he is

PROMPT:
BIANCA:


OUTPUT:
BIANCA:
Come, tell you then merer: to deour; and bear it alone,
And I which long is I have spoke cause of liberty.

KING EDWARD IV:
Sweet brother Clarence, but so: but what do forsitly doth marry, I may swear,
Doth not yet protest as I bear up and ten times as they do battle yet.

Post come, by the rest doth lend me my head, as it not?
FICK:
So, by Saint festival!
I am thus:
I go from my Lord of your former drops; what if thou name, by this place, by this man? why I may trust straight o'er we as thou comor would remember'd it takes it as it as he'st it where he comes most heavy men to-day, by my heart.
To-day, by your foe, by the dish in my heart.
The queen as he comes so
```
<!--AUTO:SAMPLES:END-->

## Updating the README automatically

After a training run completes (or after a sanity run), update the auto-filled sections of this README (summary, metrics, log excerpt, samples, and loss plot) using:

**Windows (PowerShell)**
```powershell
python update_readme.py --out_dir out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4 --readme README.md
```

**Linux/macOS**
```bash
python update_readme.py --out_dir out/runs/gpt2_124m_bs8_sl128_ga16_lr3e-4 --readme README.md
```

Notes:
- `--out_dir` must point to the **run directory** that contains `logs/`, `samples/`, and `checkpoints/`.
- The script generates the loss plot from `logs/train_metrics.csv` and saves it next to the logs as `logs/loss_curve.png`.

## Hugging Face Space (placeholders only; to be filled later)
### Screenshots
- [ ] Screenshot: Space UI main page
- [ ] Screenshot: Example generation output

## Reproducibility
- Seed: [fill]
- Exact command used: [fill]
- Best loss achieved: [auto-filled above]

## Understanding the Decoder‑Only Transformer (Sequence‑Based Training)
This project trains a **decoder‑only Transformer (GPT‑2 style)** for **next‑token prediction** on a single text corpus (`input.txt`). The model is trained **from scratch**, without any pretrained weights.
### How the data is used
- The entire `input.txt` file is treated as **one continuous stream of tokens**.
- At each training step:
  - `batch_size` independent sequences are sampled.
  - Each sequence has length `seq_len` tokens.
- For every sequence:
  - **Input**: tokens `[t0, t1, ..., t127]`
  - **Target**: tokens `[t1, t2, ..., t128]`
- The model learns to predict the **next token at every position**.
There is **no sentence boundary requirement** — sequences are random slices from the token stream.
### What the Transformer computes
For each batch:
- Input shape: `(batch_size, seq_len)`
- Output logits shape: `(batch_size, seq_len, vocab_size)`
Loss is computed using **cross‑entropy**, averaged over:
```
batch_size × seq_len
```
token predictions.
### Attention behavior
- **Self‑attention operates only within each sequence**.
- Tokens in one batch row never attend to tokens in another row.
- The batch dimension is used only for computational parallelism.
### Training is step‑based (not epoch‑based)
Unlike classical datasets, this training setup **does not use epochs**.
- One **step** = one optimizer update.
- Each step processes:
```
batch_size × seq_len × grad_accum
```
tokens.
Since token sequences are sampled randomly from a continuous stream:
- There is no well‑defined “one full pass over the dataset”.
- Progress is measured in **steps and total tokens processed**, not epochs.
A rough *pseudo‑epoch* can be estimated as:
```
(total tokens processed) / (total tokens in input.txt)
```
but this is only an approximation.
