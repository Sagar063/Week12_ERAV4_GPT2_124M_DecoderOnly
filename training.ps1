# ============================================================
# ERA-V4 Final Training Script
# Decoder-only GPT-2 style Transformer (124M params)
# ------------------------------------------------------------
# This script launches the FULL training run aimed at
# achieving training loss < 0.099999
#
# Hardware target:
#   RTX 4060 Ti (16 GB)
#
# Notes:
# - Uses AMP (fp16) for speed + memory efficiency
# - Uses gradient accumulation to increase effective batch size
# - Saves logs, checkpoints, and samples periodically

# Sanity Checkpoint
# python train.py --input_file data/input.txt --out_dir out\runs\sanity --steps 200 --batch_size 4 --seq_len 128 --grad_accum 1 --lr 3e-4 --log_every 10 --save_every 100 --sample_every 100

# ============================================================

$trainArgs = @(
  # Path to training corpus
  "--input_file", "data/input.txt",

  # Output directory for this run
  "--out_dir", "out\runs\gpt2_124m_bs8_sl128_ga16_lr3e-4",

  # Total optimizer steps
  "--steps", "400000",

  # Per-step batch size
  "--batch_size", "48",

  # Sequence length
  "--seq_len", "128",

  # Gradient accumulation (effective batch = 48*3 = 144)
  "--grad_accum", "3",

  # Optimizer hyperparameters
  "--lr", "3e-4",
  "--weight_decay", "0.1",
  "--beta1", "0.9",
  "--beta2", "0.95",

  # Stability
  "--clip_grad", "1.0",

  # AMP for speed/memory
  "--amp", "fp16",

  # Logging / artifacts
  "--log_every", "50",
  "--save_every", "1000",
  "--sample_every", "1000",

  # Stop when requirement is met
  "--target_loss", "0.099999"
)

python train.py @trainArgs
