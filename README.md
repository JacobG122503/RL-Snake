# RL Snake Console

A minimal Python console application for training and playing Snake with a simple RL agent.

## Features

- Train a DQN-style RL agent using a small feature state representation
- Watch training progress in the terminal
- Play the snake game manually with arrow keys
- Save trained weights to `dqn_snake.npz`

## Requirements

- Python 3.10+
- `numpy`

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run

Run the app and choose a mode from the retro menu. On macOS, this opens a new external Terminal window for training or play:

```bash
python3 run.py
```

From the menu, type `1`, `2`, or `3`.

Or start directly in a mode:

```bash
python3 run.py train --episodes 800 --render-every 40
```

```bash
python3 run.py play
```

## VS Code External Terminal

The app now detects VS Code terminal launches on macOS and automatically relaunches itself in a foreground external Terminal window when started from the editor run button.

You can also use the VS Code debug configuration in `.vscode/launch.json`.

Choose one of these launch configs in Run and Debug:

- `Run RL Snake (external terminal)`
- `Train RL Snake`
- `Play RL Snake`

Those are configured to use `externalTerminal`, so your app opens in a front-facing terminal window from VS Code.

During play, use arrow keys to move and `q` to quit.

## Notes

- Training uses a compact state vector instead of a full grid, which keeps the RL loop fast.
- The agent saves weights automatically at the end of training.
