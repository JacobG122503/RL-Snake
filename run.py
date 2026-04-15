import argparse
import os
import shlex
import shutil
import subprocess
import sys
import time

import numpy as np

from snake_game import SnakeGame
from agent import DQNAgent

KEY_BINDINGS = {
    259: 0,  # up
    261: 1,  # right
    258: 2,  # down
    260: 3,  # left
}


def clear_screen():
    os.system("clear" if os.name != "nt" else "cls")


def launch_external_terminal(command):
    try:
        script = f'tell application "Terminal" to activate\n tell application "Terminal" to do script "{command}"'
        subprocess.run(["osascript", "-e", script], check=True)
    except Exception as exc:
        print("Unable to launch external Terminal. Running in current terminal instead.")
        print(exc)


def get_console_size():
    size = shutil.get_terminal_size(fallback=(80, 24))
    width = max(20, size.columns - 2)
    height = max(12, size.lines - 8)
    return width, height


def load_save_meta(filename="dqn_snake.npz"):
    if not os.path.exists(filename):
        return None
    try:
        with np.load(filename) as data:
            return {
                "episode": int(data["episode"]) if "episode" in data else None,
                "score": int(data["score"]) if "score" in data else None,
            }
    except Exception:
        return None


def clear_save(filename="dqn_snake.npz"):
    try:
        os.remove(filename)
    except OSError:
        pass


def retro_menu():
    clear_screen()
    meta = load_save_meta()
    episode_text = f"episode {meta['episode']}" if meta and meta.get("episode") is not None else "no save"

    print("#############################################")
    print("#                                           #")
    print("#        RETRO RL SNAKE CONSOLE MODE        #")
    print("#                                           #")
    print("#  1) Play                                  #")
    print("#  2) Train                                 #")
    print(f"#  3) Clear save ({episode_text})" + " " * (35 - len(episode_text)) + "#")
    print("#                                           #")
    print("#############################################")
    print()
    return input("Select 1-3: ").strip()


def print_board(game, episode, step, reward, score, epsilon):
    clear_screen()
    print("#############################################")
    print(f"# TRAINING MODE  | Episode {episode} | Eps {epsilon:.3f} #")
    print("#############################################")
    print(game.render())
    print(f"Score: {score}   Step: {step}   Reward: {reward:.2f}")
    print()


def train(agent, episodes=800, render_every=40, delay=0.05):
    width, height = get_console_size()
    game = SnakeGame(width=width, height=height, max_steps=width * height * 4)
    best_score = 0
    stats = []

    try:
        for episode in range(1, episodes + 1):
            state = game.reset()
            total_reward = 0.0
            step = 0

            while True:
                action = agent.act(state)
                next_state, reward, done, score = game.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay(batch_size=64)
                state = next_state
                total_reward += reward
                step += 1

                if episode % render_every == 0:
                    print_board(game, episode, step, reward, score, agent.epsilon)
                    time.sleep(delay)

                if done:
                    stats.append((episode, score, total_reward))
                    best_score = max(best_score, score)
                    break

            if episode % 20 == 0:
                avg_score = np.mean([s for _, s, _ in stats[-20:]])
                print(f"Episode {episode}/{episodes}: avg score {avg_score:.2f}, best score {best_score}, eps {agent.epsilon:.3f}")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    finally:
        print("Saving weights to dqn_snake.npz")
        agent.save(episode=episode)


def play():
    try:
        import curses
    except ImportError:
        print("The play mode requires curses. It is available by default on macOS and Linux.")
        sys.exit(1)

    def main(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        stdscr.timeout(120)

        play_height, play_width = stdscr.getmaxyx()
        game_width = max(20, play_width - 2)
        game_height = max(12, play_height - 6)
        game = SnakeGame(width=game_width, height=game_height, max_steps=game_width * game_height * 4)
        state = game.reset()
        score = 0
        action = 0

        while True:
            stdscr.clear()
            stdscr.addstr(0, 0, "RETRO RL SNAKE - q to quit")
            stdscr.addstr(1, 0, f"Score: {score}")

            board = game.render().split("\n")
            for index, row in enumerate(board, start=3):
                if index >= play_height - 1:
                    break
                stdscr.addstr(index, 0, row[:play_width - 1])

            stdscr.refresh()
            try:
                key = stdscr.getch()
            except KeyboardInterrupt:
                break

            if key == ord("q"):
                break
            if key in KEY_BINDINGS:
                action = KEY_BINDINGS[key]

            _, _, done, score = game.play_step(action)
            if done:
                stdscr.addstr(min(play_height - 2, len(board) + 4), 0, "Game over. Press r to restart or q to quit.")
                stdscr.refresh()
                while True:
                    key = stdscr.getch()
                    if key == ord("q"):
                        return
                    if key == ord("r"):
                        game = SnakeGame(width=game_width, height=game_height, max_steps=game_width * game_height * 4)
                        state = game.reset()
                        score = 0
                        action = 0
                        break

    curses.wrapper(main)


def main():
    parser = argparse.ArgumentParser(description="RL Snake console app")
    parser.add_argument("mode", nargs="?", choices=["train", "play"], help="train the RL agent or play manually")
    parser.add_argument("--episodes", type=int, default=800, help="number of training episodes")
    parser.add_argument("--render-every", type=int, default=40, help="render every N episodes during training")
    parser.add_argument("--seed", type=int, default=None, help="random seed for reproducible training")
    parser.add_argument("--external", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    vscode_env_keys = [
        "TERM_PROGRAM",
        "VSCODE_PID",
        "VSCODE_IPC_HOOK",
        "VSCODE_CWD",
        "VSCODE_LOGS",
        "VSCODE_NLS_CONFIG",
    ]
    toolbar_vscode = any(
        os.environ.get(key, "").lower().startswith("vscode") or os.environ.get(key) is not None
        for key in vscode_env_keys
    )

    if not args.external and sys.platform == "darwin" and toolbar_vscode:
        python_exec = shlex.quote(sys.executable or "python3")
        external_cmd = [python_exec, shlex.quote(os.path.abspath(__file__))]
        if args.mode:
            external_cmd.append(args.mode)
        if args.mode == "train":
            external_cmd.extend(["--episodes", str(args.episodes), "--render-every", str(args.render_every)])
        if args.seed is not None:
            external_cmd.extend(["--seed", str(args.seed)])
        external_cmd.append("--external")
        project_dir = os.path.dirname(os.path.abspath(__file__))
        command_str = "cd " + shlex.quote(project_dir) + " && exec " + " ".join(external_cmd)
        launch_external_terminal(command_str)
        return

    if args.mode is None:
        while True:
            selection = retro_menu()
            if selection == "1":
                args.mode = "play"
                break
            if selection == "2":
                args.mode = "train"
                break
            if selection == "3":
                clear_save()
                print("Save cleared. Press Enter to return to menu.")
                input()
                continue
            print("Invalid selection. Press Enter to try again.")
            input()

    if args.mode == "train":
        agent = DQNAgent()
        train(agent, episodes=args.episodes, render_every=args.render_every)
    else:
        play()


if __name__ == "__main__":
    main()
