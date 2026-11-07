# Gomoku-Zero

**An AlphaZero-inspired Gomoku AI built with Python, PyTorch, and Monte Carlo Tree Search (MCTS).**

Gomoku-Zero is a learning and experimentation project exploring how an agent can improve through self-play. It includes a Gomoku environment, a policy/value neural network, MCTS, checkpointed training, a terminal game, and an optional Playwright adapter for PaperGames.io.

> **Project status:** Experimental. The AI can play Gomoku, it is an AlphaZero-style implementation, not a reproduction of DeepMind's full AlphaZero system.
>
> **Mac setup:** See [STRATEGY_AND_MAC_SETUP.md](STRATEGY_AND_MAC_SETUP.md) for the project's additional macOS setup and tactical-strategy notes, if that file is present in your checkout.

## Features

- **Gomoku environment:** legal moves, turns, board state, and five-in-a-row detection.
- **Policy/value network:** predicts move preferences and evaluates positions.
- **Monte Carlo Tree Search:** uses network predictions to explore candidate moves.
- **Self-play training:** configurable board size, model dimensions, MCTS simulations, self-play games, and optimizer updates.
- **Model checkpoints:** save model weights and optimizer state; resume from an existing checkpoint.
- **Local play:** challenge the trained model in the terminal.
- **Optional online adapter:** use Playwright to play on PaperGames.io, with immediate-win and single-threat-block checks before MCTS moves.

## Requirements

- Python 3.10 or newer (use a version supported by your installed PyTorch build).
- PyTorch and NumPy (see `requirements.txt`).
- Playwright and Chromium for browser-based play.
- Optional: Apple Silicon with PyTorch MPS support. CPU is also supported.

## Installation

Clone the repository (replace the placeholder with your repository URL):

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd Gomoku-Zero
```

Create a virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For PaperGames.io automation:

```bash
python -m pip install playwright
python -m playwright install chromium
```

Check your PyTorch device:

```bash
python -c "import torch; print('PyTorch:', torch.__version__); print('MPS available:', torch.backends.mps.is_available())"
```

Use `--device auto` to choose MPS when available or CPU otherwise. If necessary, use `--device cpu` explicitly.

## Train a model

Training starts from randomly initialized weights unless `--resume` points to an existing checkpoint. **Training a 15×15 model with 500 MCTS simulations per move can take substantial time.** The examples below assume the corrected `alphazero/training.py`, which trains and saves **after each iteration**, rather than just generating self-play games.

### Train a new 15×15 model

```bash
python train.py \
  --size 15 \
  --channels 64 \
  --blocks 3 \
  --simulations 500 \
  --games-per-iteration 20 \
  --iterations 100 \
  --updates 30 \
  --batch-size 64 \
  --device auto \
  --output checkpoints/15x15_latest.pt
```

An iteration should perform the following operations:

1. Generate 20 self-play games with the current model.
2. Update the policy/value network using replay samples (`--updates 30`).
3. Print policy and value losses.
4. Save the current checkpoint, including its iteration number.

**Check the logs for a loss report and `Checkpoint saved` message after every iteration.** A `Game 20/20 completed` message alone only confirms self-play completed; it does not prove that training or checkpoint saving occurred.

### Resume training without overwriting your existing model

To resume from a checkpoint and run **20 additional iterations**:

```bash
python train.py \
  --resume checkpoints/15x15_latest.pt \
  --simulations 500 \
  --games-per-iteration 20 \
  --iterations 20 \
  --updates 30 \
  --batch-size 64 \
  --device auto \
  --output checkpoints/15x15_next.pt
```

`--iterations` specifies **additional** iterations, not the final absolute iteration number. For example, resuming from iteration 101 with `--iterations 20` runs iterations **102 through 121**.

When resuming, the checkpoint determines the board size and neural-network architecture; `--size`, `--channels`, and `--blocks` are used when starting a new model. The model and optimizer state are restored when available, but **the replay buffer is not saved or restored**. Training therefore continues with an empty replay buffer that fills with newly generated self-play positions.

To avoid losing an important model, write resumed training to a **different output path**. If you use the same output path, each checkpoint save overwrites the previous file. Keep a backup of valuable checkpoints before longer runs.

### Verify a saved checkpoint

```bash
ls -lh checkpoints/
```

Inspect its metadata:

```bash
python - <<'PY'
import torch

path = 'checkpoints/15x15_latest.pt'
ckpt = torch.load(path, map_location='cpu', weights_only=True)
for key in ('board_size', 'channels', 'blocks', 'iteration'):
    print(f'{key}: {ckpt.get(key)}')
print('Has model weights:', 'model' in ckpt)
PY
```

A model trained for 5×5 or 9×9 cannot be used as a drop-in replacement for a 15×15 checkpoint.

## Play locally in the terminal

Play against a trained checkpoint:

```bash
python play.py --model checkpoints/15x15_latest.pt --simulations 500
```

Choose your color and device:

```bash
python play.py \
  --model checkpoints/15x15_latest.pt \
  --simulations 500 \
  --human white \
  --device cpu
```

If your checkpoint has a different filename, replace the value passed to `--model`.

## Play on PaperGames.io

`play_online.py` uses Playwright to open a Chromium browser, read the online Gomoku board, and place the model's selected moves. Use it for supervised tests against the site's computer opponent; follow the site's rules and terms of use.

1. Make sure you have a **15×15 checkpoint** and have installed Playwright and Chromium.
2. Start the adapter with an **explicit checkpoint path**:

   ```bash
   python play_online.py \
     --model checkpoints/15x15_latest.pt \
     --simulations 500 \
     --device auto
   ```

3. In the opened browser, start or join a Gomoku game. When prompted, return to the terminal and press Enter.
4. The adapter attempts to detect the AI's color; if it cannot, follow the terminal prompt.
5. By default, confirm each AI move by pressing Enter. To let the adapter calculate moves automatically, add `--auto`:

   ```bash
   python play_online.py \
     --model checkpoints/15x15_latest.pt \
     --simulations 500 \
     --device auto \
     --auto
   ```

The adapter checks for an immediate winning move, then a single immediate opponent winning point, and otherwise uses MCTS. These checks do **not** guarantee tactical correctness, especially against multiple simultaneous threats. Browser selectors can change, so supervise games and confirm that moves are accepted. The adapter is optional and is not a measure of model strength by itself.

> **Note:** Passing `--model` explicitly avoids reliance on any older default checkpoint filename that may still be hard-coded in `play_online.py`.

## Evaluate model strength

Training loss alone is not enough to establish stronger play. Compare checkpoints through actual games, ideally alternating black/white and recording wins, losses, and draws.

Suggested tests:

- **Human vs. AI:** play locally and note tactical mistakes.
- **Checkpoint vs. checkpoint:** compare a new model with an older baseline. An automated evaluation harness is a possible future addition; it is **not documented as implemented** here.
- **Online computer bot:** run several supervised games and report the outcomes.

Example results table (fill in only after testing):

| Checkpoint | Opponent | Games | Wins | Draws | Losses |
| --- | --- | ---: | ---: | ---: | ---: |
| `15x15_latest.pt` | PaperGames.io computer bot | — | — | — | — |
| `15x15_next.pt` | Previous checkpoint | — | — | — | — |

## Run tests

```bash
python -m unittest discover -s tests -v
```

The online adapter requires a live browser session and is not covered by ordinary unit tests. Verify it manually when the website changes.

## Project layout

```text
Gomoku-Zero/
├── alphazero/
│   ├── mcts.py              # Monte Carlo Tree Search
│   ├── network.py           # Policy/value network
│   └── training.py          # Self-play, training, checkpoints
├── gomoku/
│   ├── __init__.py
│   └── environment.py       # Gomoku rules and state
├── tests/
├── checkpoints/             # Local checkpoints (not committed by default)
├── main.py
├── play.py                  # Human vs. AI in the terminal
├── play_online.py           # Optional browser adapter
├── train.py                 # Training command-line entry point
├── requirements.txt
└── README.md
```

## Checkpoints and GitHub

Checkpoint files may be excluded by `.gitignore` (for example, `checkpoints/*.pt`). Check your actual `.gitignore` before publishing. A fresh clone may not include a usable checkpoint; users will need to train one or obtain a separately distributed model.

Do not commit local virtual environments, caches, credentials, or other private configuration files.

## How AlphaZero-style learning works

1. The network predicts move probabilities (policy) and a position evaluation (value).
2. MCTS uses those estimates to search for promising moves.
3. Self-play games produce improved search-policy targets and final outcomes.
4. Training adjusts the policy and value predictions based on those targets.
5. The updated model generates the next batch of self-play games.

This project is a compact educational implementation intended for experimentation and further development.

## Gameplay Demonstration

### AlphaZero vs. PaperGames Bot — First Victory

https://github.com/user-attachments/assets/2c803de9-9898-4a29-8e54-d264447928fe

Learning Outcomes and Limitations
Observations

* The AI successfully learned and executed winning patterns.

* Repeated defensive decisions were observed against certain opponent strategies.

* Opening position preferences and tactical decision-making remain areas for improvement.

* Lower training losses do not necessarily guarantee stronger gameplay.

## Imporvements

1. Enriching the tactical strategies — improving how the AI recognizes threats, creates attacking opportunities, and avoids losing positions.
2. Increasing training iterations — giving the neural network more self-play experience so it can learn better policies and position evaluations.


