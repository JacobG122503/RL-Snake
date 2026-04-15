import argparse
import curses
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
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
        with tempfile.NamedTemporaryFile("w", delete=False, suffix=".sh") as wrapper:
            wrapper.write("#!/bin/sh\n")
            wrapper.write("set -e\n")
            wrapper.write(command + "\n")
            wrapper.write("osascript -e 'tell application \"Terminal\" to close front window saving no'\n")
            wrapper.write('rm -- "$0"\n')
            wrapper_path = wrapper.name

        os.chmod(wrapper_path, 0o755)
        script = f'tell application "Terminal" to activate\n tell application "Terminal" to do script "/bin/sh {shlex.quote(wrapper_path)}"'
        subprocess.run(["osascript", "-e", script], check=True)
    except Exception as exc:
        print("Unable to launch external Terminal. Running in current terminal instead.")
        print(exc)


def close_external_terminal():
    if sys.platform != "darwin":
        return
    try:
        script = 'delay 0.2\n tell application "Terminal" to close front window saving no'
        subprocess.Popen(["osascript", "-e", script])
    except Exception as exc:
        print("Unable to close external Terminal window.")
        print(exc)


def center_text(lines):
    size = shutil.get_terminal_size(fallback=(80, 24))
    cols, rows = size.columns, size.lines
    top_padding = max(0, (rows - len(lines)) // 2)
    centered = ["" for _ in range(top_padding)]
    for line in lines:
        if len(line) < cols:
            pad = (cols - len(line)) // 2
            centered.append(" " * pad + line)
        else:
            centered.append(line)
    return "\n".join(centered)


def get_console_size():
    size = shutil.get_terminal_size(fallback=(80, 24))
    width = max(20, size.columns - 2)
    height = max(12, size.lines - 8)
    return width, height


def fill_curses_background(stdscr, color_pair):
    h, w = stdscr.getmaxyx()
    stdscr.bkgdset(" ", color_pair)
    for row in range(h):
        try:
            stdscr.move(row, 0)
            stdscr.clrtoeol()
        except curses.error:
            pass


def draw_box(stdscr, top, left, height, width, color_pair):
    if width < 2 or height < 2:
        return
    try:
        stdscr.addch(top, left, curses.ACS_ULCORNER, color_pair)
        stdscr.hline(top, left + 1, curses.ACS_HLINE, width - 2, color_pair)
        stdscr.addch(top, left + width - 1, curses.ACS_URCORNER, color_pair)
        stdscr.vline(top + 1, left, curses.ACS_VLINE, height - 2, color_pair)
        stdscr.vline(top + 1, left + width - 1, curses.ACS_VLINE, height - 2, color_pair)
        stdscr.addch(top + height - 1, left, curses.ACS_LLCORNER, color_pair)
        stdscr.hline(top + height - 1, left + 1, curses.ACS_HLINE, width - 2, color_pair)
        stdscr.addch(top + height - 1, left + width - 1, curses.ACS_LRCORNER, color_pair)
    except curses.error:
        pass


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


def curses_menu(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_CYAN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_MAGENTA, curses.COLOR_BLACK)
    stdscr.bkgd(" ", curses.color_pair(1))
    stdscr.clear()
    meta = load_save_meta()
    episode_text = f"episode {meta['episode']}" if meta and meta.get("episode") is not None else "no save"
    raw_lines = [
        "",
        "RL SNAKE",
        "",
        "1) Play",
        "2) Train",
        f"3) Clear save ({episode_text})",
        "",
        "Press 1, 2, or 3 to choose.",
    ]
    inner_width = max(len(line) for line in raw_lines)
    box_width = inner_width + 4
    lines = [
        "#" * box_width,
        "#" + " " * (box_width - 2) + "#",
    ]
    for idx, raw in enumerate(raw_lines):
        padded = raw.center(inner_width)
        lines.append("# " + padded + " #")
        if idx in (0, 2, 6):
            lines.append("#" + " " * (box_width - 2) + "#")
    lines.append("#" * box_width)

    while True:
        stdscr.bkgd(" ", curses.color_pair(1))
        stdscr.clear()
        fill_curses_background(stdscr, curses.color_pair(1))
        stdscr.refresh()
        h, w = stdscr.getmaxyx()
        start_y = max(0, (h - len(lines)) // 2)
        start_x = max(0, (w - box_width) // 2)

        for idx, line in enumerate(lines):
            y = start_y + idx
            if y < 0 or y >= h:
                continue
            if idx == 0 or idx == len(lines) - 1 or line == "#" + " " * (box_width - 2) + "#":
                stdscr.addstr(y, start_x, line, curses.color_pair(1))
            else:
                inner = line[2:-2]
                if idx == 2:
                    text_color = curses.color_pair(3)
                elif idx in (4, 5, 6):
                    text_color = curses.color_pair(2)
                else:
                    text_color = curses.color_pair(1)
                stdscr.addstr(y, start_x, "# ", curses.color_pair(1))
                stdscr.addstr(y, start_x + 2, inner, text_color)
                stdscr.addstr(y, start_x + box_width - 2, " #", curses.color_pair(1))
        stdscr.refresh()
        key = stdscr.getch()
        if key in (ord("1"), ord("2"), ord("3")):
            return chr(key)
        if key == ord("q"):
            return "q"


def curses_train(stdscr, agent, episodes=800, render_every=40, delay=0.05):
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_WHITE, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(5, curses.COLOR_CYAN, curses.COLOR_BLACK)
    stdscr.bkgd(" ", curses.color_pair(0))
    stdscr.nodelay(True)
    stdscr.timeout(1)

    h, w = stdscr.getmaxyx()
    width = max(20, w - 4)
    height = max(12, h - 6)
    game = SnakeGame(width=width, height=height, max_steps=width * height * 4)
    best_score = 0
    stats = []
    episode = 0

    try:
        for episode in range(1, episodes + 1):
            state = game.reset()
            total_reward = 0.0
            step = 0
            skip_current_episode_render = False

            while True:
                key = stdscr.getch()
                if key != -1:
                    if key in (ord("q"), ord("Q")):
                        raise KeyboardInterrupt
                    if key in (ord("s"), ord("S")):
                        skip_current_episode_render = True
                    if key in (curses.KEY_UP, 259):
                        render_every = max(1, render_every - 1)
                    if key in (curses.KEY_DOWN, 258):
                        render_every = min(999, render_every + 1)

                action = agent.act(state)
                next_state, reward, done, score = game.step(action)
                agent.remember(state, action, reward, next_state, done)
                agent.replay(batch_size=64)
                state = next_state
                total_reward += reward
                step += 1

                should_render = episode % render_every == 0 and not skip_current_episode_render
                if should_render:
                    stdscr.bkgdset(" ", curses.color_pair(4))
                    stdscr.erase()
                    fill_curses_background(stdscr, curses.color_pair(4))
                    h, w = stdscr.getmaxyx()
                    title = f"TRAINING MODE | Episode {episode} | Eps {agent.epsilon:.3f}"
                    controls = f"[↑/↓]: show every {render_every} eps    [s]: skip this episode    [q]: quit"
                    try:
                        stdscr.move(0, 0)
                        stdscr.clrtoeol()
                        stdscr.move(1, 0)
                        stdscr.clrtoeol()
                        stdscr.move(2, 0)
                        stdscr.clrtoeol()
                    except curses.error:
                        pass
                    stdscr.addstr(0, max(0, (w - len(title)) // 2), title, curses.color_pair(5))
                    stdscr.addstr(1, max(0, (w - len(controls)) // 2), controls, curses.color_pair(4))
                    board = game.render_cells()
                    board_h = len(board)
                    board_w = len(board[0]) if board_h else 0
                    board_x = max(2, (w - board_w) // 2)
                    board_y = 3
                    draw_box(stdscr, board_y - 1, board_x - 1, board_h + 2, board_w + 2, curses.color_pair(1))
                    for row_index, row in enumerate(board):
                        if row_index >= board_h:
                            break
                        for col_index, cell in enumerate(row):
                            x = board_x + col_index
                            y = board_y + row_index
                            if y >= h - 2 or x >= w - 1:
                                continue
                            if cell == "H":
                                stdscr.addch(y, x, curses.ACS_CKBOARD, curses.color_pair(3))
                            elif cell == "S":
                                stdscr.addch(y, x, curses.ACS_CKBOARD, curses.color_pair(2))
                            elif cell == "A":
                                stdscr.addch(y, x, curses.ACS_DIAMOND, curses.color_pair(4))
                            else:
                                stdscr.addch(y, x, " ", curses.color_pair(0))
                    status = f"Score: {score}  Step: {step}  Reward: {reward:.2f}"
                    try:
                        stdscr.move(h - 2, 0)
                        stdscr.clrtoeol()
                    except curses.error:
                        pass
                    stdscr.addstr(h - 2, 1, status[: max(0, w - 2)], curses.color_pair(4))
                    stdscr.refresh()
                    time.sleep(delay)
                if done:
                    stats.append((episode, score, total_reward))
                    best_score = max(best_score, score)
                    break
            if episode % 20 == 0:
                avg_score = np.mean([s for _, s, _ in stats[-20:]])
                summary = f"Episode {episode}/{episodes}: avg {avg_score:.2f}, best {best_score}, eps {agent.epsilon:.3f}"
                _, w = stdscr.getmaxyx()
                try:
                    stdscr.move(1, 0)
                    stdscr.clrtoeol()
                except curses.error:
                    pass
                stdscr.addstr(1, max(0, (w - len(summary)) // 2), summary[: w - 2], curses.color_pair(4))
                stdscr.refresh()
    except KeyboardInterrupt:
        pass
    finally:
        agent.save(episode=episode)


def curses_play(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_BLACK)
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_BLACK)
    curses.init_pair(3, curses.COLOR_YELLOW, curses.COLOR_BLACK)
    curses.init_pair(4, curses.COLOR_WHITE, curses.COLOR_BLACK)
    stdscr.bkgd(" ", curses.color_pair(4))
    stdscr.nodelay(True)
    stdscr.timeout(100)

    play_height, play_width = stdscr.getmaxyx()
    game_width = max(20, play_width - 4)
    game_height = max(12, play_height - 6)
    game = SnakeGame(width=game_width, height=game_height, max_steps=game_width * game_height * 4)
    state = game.reset()
    score = 0
    action = 0

    while True:
        play_height, play_width = stdscr.getmaxyx()
        stdscr.bkgdset(" ", curses.color_pair(4))
        stdscr.erase()
        fill_curses_background(stdscr, curses.color_pair(4))
        title = "RL SNAKE - q to quit"
        try:
            stdscr.move(0, 0)
            stdscr.clrtoeol()
            stdscr.move(1, 0)
            stdscr.clrtoeol()
        except curses.error:
            pass
        stdscr.addstr(0, max(0, (play_width - len(title)) // 2), title, curses.color_pair(3))
        stdscr.addstr(1, 1, f"Score: {score}"[: max(0, play_width - 2)], curses.color_pair(4))
        board = game.render_cells()
        board_h = len(board)
        board_w = len(board[0]) if board_h else 0
        board_x = max(2, (play_width - board_w) // 2)
        board_y = 3
        try:
            stdscr.move(2, 0)
            stdscr.clrtoeol()
        except curses.error:
            pass
        draw_box(stdscr, board_y - 1, board_x - 1, board_h + 2, board_w + 2, curses.color_pair(1))

        for row_index, row in enumerate(board):
            for col_index, cell in enumerate(row):
                x = board_x + col_index
                y = board_y + row_index
                if y >= play_height - 1 or x >= play_width - 1:
                    continue
                if cell == "H" or cell == "S":
                    stdscr.addch(y, x, curses.ACS_CKBOARD, curses.color_pair(2 if cell == "S" else 1))
                elif cell == "A":
                    stdscr.addch(y, x, curses.ACS_DIAMOND, curses.color_pair(3))
                else:
                    stdscr.addch(y, x, " ", curses.color_pair(4))

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
            game_over = "Game over. Press r to restart or q to quit."
            stdscr.addstr(min(play_height - 2, len(board) + 4), 0, game_over[:play_width - 1], curses.color_pair(4))
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


def train(agent, episodes=800, render_every=40, delay=0.05):
    episode = 0
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
        curses.start_color()
        curses.use_default_colors()
        curses.init_pair(1, curses.COLOR_RED, -1)
        curses.init_pair(2, curses.COLOR_GREEN, -1)
        curses.init_pair(3, curses.COLOR_YELLOW, -1)
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
            stdscr.addstr(0, 0, "RL SNAKE - q to quit")
            stdscr.addstr(1, 0, f"Score: {score}")

            board = game.render_cells()
            for row_index, row in enumerate(board):
                if 3 + row_index >= play_height - 1:
                    break
                for col_index, cell in enumerate(row):
                    if col_index >= play_width - 1:
                        break
                    if cell == "H":
                        stdscr.addch(3 + row_index, col_index, curses.ACS_CKBOARD, curses.color_pair(1))
                    elif cell == "S":
                        stdscr.addch(3 + row_index, col_index, curses.ACS_CKBOARD, curses.color_pair(2))
                    elif cell == "A":
                        stdscr.addch(3 + row_index, col_index, curses.ACS_DIAMOND, curses.color_pair(3))
                    else:
                        stdscr.addch(3 + row_index, col_index, " ")

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
        command_str = "cd " + shlex.quote(project_dir) + " && " + " ".join(external_cmd)
        command_str += "; osascript -e 'tell application \"Terminal\" to close front window saving no'"
        launch_external_terminal(command_str)
        return

    if args.mode is None:
        selection = curses.wrapper(curses_menu)
        if selection == "q":
            return
        if selection == "1":
            args.mode = "play"
        elif selection == "2":
            args.mode = "train"
        elif selection == "3":
            clear_save()
            return main()

    if args.mode == "train":
        agent = DQNAgent()
        curses.wrapper(curses_train, agent, args.episodes, args.render_every)
    else:
        curses.wrapper(curses_play)


if __name__ == "__main__":
    main()
