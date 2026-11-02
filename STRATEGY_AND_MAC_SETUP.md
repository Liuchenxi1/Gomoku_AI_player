# Gomoku-Zero — Mac setup and tactical strategy

## Why no `.venv` in ZIP?
A Python virtual environment includes absolute paths, installed wheel binaries and
platform-specific files. Create one locally on the destination Mac instead.
Your error (`zsh: command not found: python`) means `python` is not on the shell's
PATH. `python3` is commonly available instead; after activating `.venv`, `python`
will also work.

## One-time setup (from project folder)

```bash
bash setup_mac.sh
source .venv/bin/activate
python -m unittest discover -s tests -p 'test_strategy.py'
```

`setup_mac.sh` creates `.venv`, installs the dependencies in `requirements.txt`,
and downloads Chromium for Playwright. For a missing `python3` or an unsupported
Python version, install a supported Python from python.org or Homebrew first.

## Online play: original network + 500 MCTS + tactical defense

```bash
source .venv/bin/activate
python play_online.py --model checkpoints/15x15_500_trial.pt --simulations 500 --device mps
```

The AI checks `alphazero/strategy.py` **before** invoking neural-net MCTS.
It handles immediate wins and blocks, open fours (moves that create two distinct
immediate win locations), blocks open fours before they appear, and open-three
creation/defense. When no urgent pattern is found, it uses 500-simulation MCTS.
This tactical heuristic is not a complete threat-space solver and will not
make the network itself stronger without more learning.

## Train more (original trainer still uses its own MCTS tactical safeguards)

```bash
source .venv/bin/activate
python train.py --resume checkpoints/15x15_500_trial.pt \
  --simulations 500 --games-per-iteration 20 --iterations 100 \
  --updates 30 --batch-size 64 --device mps \
  --output checkpoints/15x15_500.pt
```

The new `strategy.py` is integrated into **online inference**, not self-play
training; test its tactical behavior separately from checkpoint improvements.
Do not overwrite the original model until match evaluations show improvement.

## Test with no trained model

```bash
python -m unittest discover -s tests -p test_strategy.py
```
