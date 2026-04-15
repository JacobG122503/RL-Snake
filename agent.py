import random
import numpy as np

class ReplayMemory:
    def __init__(self, capacity=5000):
        self.capacity = capacity
        self.buffer = []
        self.position = 0

    def push(self, state, action, reward, next_state, done):
        if len(self.buffer) < self.capacity:
            self.buffer.append(None)
        self.buffer[self.position] = (state, action, reward, next_state, done)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, state_size=11, action_size=4, hidden_size=128, lr=0.001, gamma=0.95):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.lr = lr
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.memory = ReplayMemory(capacity=10000)

        self.W1 = np.random.randn(self.state_size, hidden_size) * np.sqrt(2.0 / self.state_size)
        self.b1 = np.zeros(hidden_size)
        self.W2 = np.random.randn(hidden_size, self.action_size) * np.sqrt(2.0 / hidden_size)
        self.b2 = np.zeros(self.action_size)

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_derivative(self, x):
        return (x > 0).astype(float)

    def predict(self, state):
        state = np.array(state, dtype=np.float32)
        h = self._relu(np.dot(state, self.W1) + self.b1)
        q = np.dot(h, self.W2) + self.b2
        return q

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.predict(state)
        return int(np.argmax(q_values))

    def remember(self, state, action, reward, next_state, done):
        self.memory.push(state, action, reward, next_state, done)

    def replay(self, batch_size=64):
        if len(self.memory) < batch_size:
            return

        batch = self.memory.sample(batch_size)
        states = np.array([sample[0] for sample in batch], dtype=np.float32)
        actions = np.array([sample[1] for sample in batch], dtype=int)
        rewards = np.array([sample[2] for sample in batch], dtype=np.float32)
        next_states = np.array([sample[3] for sample in batch], dtype=np.float32)
        dones = np.array([sample[4] for sample in batch], dtype=np.float32)

        h = self._relu(np.dot(states, self.W1) + self.b1)
        q_values = np.dot(h, self.W2) + self.b2

        h_next = self._relu(np.dot(next_states, self.W1) + self.b1)
        q_next = np.dot(h_next, self.W2) + self.b2
        targets = q_values.copy()

        for idx in range(len(batch)):
            target = rewards[idx]
            if not dones[idx]:
                target += self.gamma * np.max(q_next[idx])
            targets[idx, actions[idx]] = target

        error = q_values - targets
        dW2 = np.dot(h.T, error) / len(batch)
        db2 = np.mean(error, axis=0)
        dh = np.dot(error, self.W2.T) * self._relu_derivative(h)
        dW1 = np.dot(states.T, dh) / len(batch)
        db1 = np.mean(dh, axis=0)

        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2
        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save(self, filename="dqn_snake.npz", episode=None, score=None):
        save_data = {
            "W1": self.W1,
            "b1": self.b1,
            "W2": self.W2,
            "b2": self.b2,
        }
        if episode is not None:
            save_data["episode"] = np.int32(episode)
        if score is not None:
            save_data["score"] = np.int32(score)
        np.savez(filename, **save_data)

    def load(self, filename="dqn_snake.npz"):
        data = np.load(filename)
        self.W1 = data["W1"]
        self.b1 = data["b1"]
        self.W2 = data["W2"]
        self.b2 = data["b2"]
