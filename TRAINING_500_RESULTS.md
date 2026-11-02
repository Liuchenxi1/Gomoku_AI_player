# Gomoku-Zero: 500-simulation experiment

## Changes
- `train.py` default MCTS simulations: 50 → 500.
- `play_online.py` default MCTS simulations: 100 → 500.
- README examples updated to 500.
- `train_500.sh` continues from `checkpoints/15x15_tactics.pt`, writes to a separate `checkpoints/15x15_500.pt`.

## Completed smoke-training trial (CPU)
- Started from `checkpoints/15x15_tactics.pt` (iteration 100; 15×15, 32 channels, 2 blocks).
- Ran `python train.py --resume checkpoints/15x15_tactics.pt --simulations 500 --games-per-iteration 1 --iterations 1 --updates 2 --batch-size 16 --device cpu --output checkpoints/15x15_500_trial.pt`
- Completed iteration: **101**
- Self-play games: **1**, positions: **150**
- Winner: white (B/W/draw = 0/1/0)
- Policy loss: **4.6193**
- Value loss: **2.0959**
- Output: `checkpoints/15x15_500_trial.pt`

These losses are *not* a measure of playing strength. Evaluate against the original checkpoint using fixed seeds and multiple games. A single 500-simulation game does not establish any improvement.

## Continue training on Apple Silicon

```bash
./train_500.sh
```

Or continue specifically from the trial checkpoint (iteration 101):

```bash
python train.py --resume checkpoints/15x15_500_trial.pt --simulations 500 --games-per-iteration 20 --iterations 100 --updates 30 --batch-size 64 --device mps --output checkpoints/15x15_500.pt
```

Warning: 500 simulations per move on a 15×15 board is compute-intensive. The resumed trainer does **not** restore a previous replay buffer. Do not overwrite the original checkpoint before comparing quality. To test the trial model against the online bot:

```bash
python play_online.py --model checkpoints/15x15_500_trial.pt --simulations 500 --device mps
```
