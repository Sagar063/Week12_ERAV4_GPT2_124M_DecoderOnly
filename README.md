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

**Keep safe (for grading + later deployment):**
- `checkpoints/best.pt`
- `logs/train_metrics.csv`
- `samples/samples.txt`

## Results (auto-updated)
This section is **auto-filled** by `update_readme.py`.

### Summary
<!--AUTO:SUMMARY:START-->
- Updated: **2025-12-14 02:40**
- Run directory: `out/runs/sanity`
- Best loss: **6.380531** at step **30**
- Last logged loss: **6.380531**
- Total logged rows: **4**
<!--AUTO:SUMMARY:END-->

### Training log excerpt
<!--AUTO:LOGS:START-->
```text
2025-12-14 02:29:38,988 | INFO | Device used in training: cuda
2025-12-14 02:29:38,988 | INFO | Args: {'input_file': 'data/input.txt', 'out_dir': 'out\\runs\\sanity', 'seed': 1337, 'block_size': 1024, 'n_layer': 12, 'n_head': 12, 'n_embd': 768, 'batch_size': 4, 'seq_len': 128, 'grad_accum': 1, 'steps': 30, 'lr': 0.0003, 'weight_decay': 0.1, 'beta1': 0.9, 'beta2': 0.95, 'clip_grad': 1.0, 'randomize_batches': False, 'target_loss': 0.1, 'log_every': 10, 'save_every': 10, 'sample_every': 10, 'resume': '', 'prompt': 'BIANCA:\n', 'num_prompts': 2, 'max_new_tokens': 200, 'top_k': 50, 'temperature': 0.9, 'compile': False, 'amp': 'fp16'}
2025-12-14 02:29:38,988 | INFO | Log file: out\runs\sanity\logs\train.log
2025-12-14 02:29:38,988 | INFO | CSV file: out\runs\sanity\logs\train_metrics.csv
2025-12-14 02:29:39,213 | INFO | Loaded 338025 tokens from data/input.txt
2025-12-14 02:29:39,214 | INFO | Batch tokens per step (before grad_accum): 512
2025-12-14 02:29:39,214 | INFO | Effective tokens per optimizer step: 512
2025-12-14 02:29:41,162 | INFO | step 1/30 | loss 10.931427 | lr 3.00e-04 | tok/s 1647.7
2025-12-14 02:30:01,857 | INFO | New best loss 10.931427 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:30:49,714 | INFO | New best loss 9.565536 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:31:40,018 | INFO | New best loss 9.007504 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:31:56,674 | INFO | New best loss 8.972071 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:32:42,108 | INFO | New best loss 8.902363 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:33:03,632 | INFO | New best loss 8.526791 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:33:51,282 | INFO | New best loss 8.219444 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:33:51,825 | INFO | step 10/30 | loss 8.256924 | lr 3.00e-04 | tok/s 20.4
2025-12-14 02:34:32,088 | INFO | Saved checkpoint: out\runs\sanity\checkpoints\last.pt
2025-12-14 02:34:57,790 | INFO | Appended samples to: out\runs\sanity\samples\samples.txt
2025-12-14 02:35:22,682 | INFO | New best loss 7.846100 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:35:26,422 | INFO | New best loss 7.344147 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:35:55,621 | INFO | New best loss 7.100513 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:36:17,353 | INFO | New best loss 6.811369 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:36:17,930 | INFO | step 20/30 | loss 6.500828 | lr 3.00e-04 | tok/s 25.8
2025-12-14 02:37:08,519 | INFO | Saved checkpoint: out\runs\sanity\checkpoints\last.pt
2025-12-14 02:37:13,318 | INFO | Appended samples to: out\runs\sanity\samples\samples.txt
2025-12-14 02:38:19,220 | INFO | New best loss 6.500828 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:39:07,838 | INFO | New best loss 6.470907 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:39:11,510 | INFO | New best loss 6.237009 -> saved: out\runs\sanity\checkpoints\best.pt
2025-12-14 02:39:12,238 | INFO | step 30/30 | loss 6.380531 | lr 3.00e-04 | tok/s 26.9
2025-12-14 02:39:57,719 | INFO | Saved checkpoint: out\runs\sanity\checkpoints\last.pt
2025-12-14 02:40:01,234 | INFO | Appended samples to: out\runs\sanity\samples\samples.txt
2025-12-14 02:40:01,235 | INFO | Training complete.
```
<!--AUTO:LOGS:END-->

### Key metrics
<!--AUTO:METRICS:START-->
| Metric | Value |
|---|---:|
| Best loss | 6.380531 |
| Step at best loss | 30 |
| Last logged loss | 6.380531 |
| Total logged rows | 4 |
<!--AUTO:METRICS:END-->

### Sample generations
<!--AUTO:SAMPLES:START-->
```text
ROMPT:
BIANCA:


OUTPUT:
BIANCA:




:






:
,I,
.,


.,
 it
 to



,

,,.

:
:

::


US

..


,
:
.


,,
 would
 the:


,




.




:
::,
,:
:




 is to

:,,


 and

.
 it
,US:


,

.


 in
:,



 the.

,
 to:




 to
:.




,

.
:
.


::,US
 can
,

.,:








,.


================================================================================
STEP 20

PROMPT:
BIANCA:


OUTPUT:
BIANCA:

 us all,I to I I
 as him
 your;? not to the yourI we not
 you
 to our your I his as:
 the, and to,
AndUS,I and,, hes::


I:
 ofI of him,
? the, but my the your his; ' not,

 not as,,,
 you the,


 to,
,I? not:TheUS your'd: all?

; of us'd you, that youUS;; a I our good, I' in you youUS,


,
' to

. of
 me
I;:
 not your.


 to
 the you,


 you,

 that are not. to








 him

 I the the,




 it of; all:


'

PROMPT:
BIANCA:


OUTPUT:
BIANCA:
I the
 the good and?:
 the this, youUS
The of the to;;,

 of!,

 to thatUS, not,,:


:

 us:
 than that usI!?:

 and, the you of as I
 not!! I not:




 I:, in;And: youI not. thatUS;.

 and,, have,
, the




 Is are you to: aius
 and:
 have:
 I a?, the. I,
 the I not, to but but the I you the:

 beUS the; to him to the the the than,




 to
 I what: heI, the meI,
 I have, you?US, II:
!, the the
,: and?! the all.


 in

================================================================================
STEP 30

PROMPT:
BIANCA:


OUTPUT:
BIANCA:
 the's my
:
 ofcius,




 a of I'.

 my;
!


' not be himAnd's that that,MAR not,



 with I me

 that--.US:

:
? be
 not.!

:
 of a;

The him

MEN that a',' of you
 be have



 and shall.
 the
 of, and myIN: in him in'.
 not.

 inI! I.



 in.
.



 forUS,And'd with

 with at,
 and


 a.
: of,





 my's:
And




 I youI?
.





 my

's
I that.
 to myHe--:


, as

PROMPT:
BIANCA:


OUTPUT:
BIANCA:
 you with the to
 of you in, the
COM
's a, my
! willUS not we. will?
The and his the for the. for your'd you your me
 that,

.

 myciusI. your we,


 I: not! be-


 that my; me




'sUS,
I,

MEN of my be

 that!
 and Mar in


 and?





',
 or you as:


? to:

 of;



 is with
 Mar.I's?


. in will


 and:


.

I forUSIIN



 to not,

He's andcius,
!
MEN a yourUS of.
 of.

 I and! to:
And's me
ciusHe in.
```
<!--AUTO:SAMPLES:END-->

## Hugging Face Space (placeholders only; to be filled later)
### Screenshots
- [ ] Screenshot: Space UI main page
- [ ] Screenshot: Example generation output

## Reproducibility
- Seed: [fill]
- Exact command used: [fill]
- Best loss achieved: [auto-filled above]
