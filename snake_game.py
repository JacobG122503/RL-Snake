import random
from collections import deque

# Directions: up, right, down, left
DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

class SnakeGame:
    def __init__(self, width=20, height=18, max_steps=400):
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.reset()

    def reset(self):
        center_x = self.width // 2
        center_y = self.height // 2
        self.snake = deque([(center_x, center_y), (center_x - 1, center_y), (center_x - 2, center_y)])
        self.direction = 1  # right
        self.steps = 0
        self.score = 0
        self.alive = True
        self._place_apple()
        return self._get_state()

    def _place_apple(self):
        free_cells = [(x, y) for x in range(self.width) for y in range(self.height) if (x, y) not in self.snake]
        self.apple = random.choice(free_cells)

    def step(self, action):
        self.steps += 1
        self.direction = self._choose_direction(action)
        head_x, head_y = self.snake[0]
        dx, dy = DIRECTIONS[self.direction]
        new_head = (head_x + dx, head_y + dy)

        if self._collides(new_head) or self.steps > self.max_steps:
            self.alive = False
            return self._get_state(), -1.0, True, self.score

        self.snake.appendleft(new_head)
        reward = 0.0
        if new_head == self.apple:
            self.score += 1
            reward = 1.0
            self._place_apple()
        else:
            self.snake.pop()

        state = self._get_state()
        done = not self.alive
        return state, reward, done, self.score

    def _choose_direction(self, action):
        if action == 0:  # straight
            return self.direction
        if action == 1:  # right turn
            return (self.direction + 1) % 4
        if action == 2:  # left turn
            return (self.direction - 1) % 4
        if action == 3:  # reverse -> keep current direction
            return self.direction
        return self.direction

    def _collides(self, point):
        x, y = point
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            return True
        return point in self.snake

    def _get_state(self):
        head_x, head_y = self.snake[0]
        point_l = (head_x + DIRECTIONS[(self.direction - 1) % 4][0], head_y + DIRECTIONS[(self.direction - 1) % 4][1])
        point_r = (head_x + DIRECTIONS[(self.direction + 1) % 4][0], head_y + DIRECTIONS[(self.direction + 1) % 4][1])
        point_s = (head_x + DIRECTIONS[self.direction][0], head_y + DIRECTIONS[self.direction][1])

        danger_left = 1.0 if self._collides(point_l) else 0.0
        danger_right = 1.0 if self._collides(point_r) else 0.0
        danger_straight = 1.0 if self._collides(point_s) else 0.0

        dir_up = 1.0 if self.direction == 0 else 0.0
        dir_right = 1.0 if self.direction == 1 else 0.0
        dir_down = 1.0 if self.direction == 2 else 0.0
        dir_left = 1.0 if self.direction == 3 else 0.0

        apple_left = 1.0 if self.apple[0] < head_x else 0.0
        apple_right = 1.0 if self.apple[0] > head_x else 0.0
        apple_up = 1.0 if self.apple[1] < head_y else 0.0
        apple_down = 1.0 if self.apple[1] > head_y else 0.0

        return [
            danger_straight,
            danger_right,
            danger_left,
            dir_up,
            dir_right,
            dir_down,
            dir_left,
            apple_left,
            apple_right,
            apple_up,
            apple_down,
        ]

    def render(self):
        grid = [[" " for _ in range(self.width)] for _ in range(self.height)]
        for x, y in self.snake:
            grid[y][x] = "S"
        ax, ay = self.apple
        grid[ay][ax] = "A"
        head_x, head_y = self.snake[0]
        grid[head_y][head_x] = "H"

        top_bottom = "#" * (self.width + 2)
        rows = [top_bottom]
        for row in grid:
            rows.append("#" + "".join(row) + "#")
        rows.append(top_bottom)
        return "\n".join(rows)

    def play_step(self, action):
        return self.step(action)
