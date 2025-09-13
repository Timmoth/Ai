import os
import random
from collections import deque, namedtuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, optimizers, losses
import tensorflowjs as tfjs

import pygame
from enum import Enum

import matplotlib.pyplot as plt
from IPython import display

plt.ion()

BLOCK_SIZE = 20
SPEED = 40
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.0001

Point = namedtuple('Point', 'x, y')

WHITE = (255, 255, 255)
RED = (200, 0, 0)
BLUE1 = (0, 0, 255)
BLUE2 = (0, 100, 255)
BLACK = (0, 0, 0)


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4


pygame.init()
font = pygame.font.Font('arial.ttf', 25)


def plot(scores, mean_scores):
    display.clear_output(wait=True)
    display.display(plt.gcf())
    plt.clf()
    plt.title('Training...')
    plt.xlabel('Number of Games')
    plt.ylabel('Score')
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))
    plt.show(block=False)
    plt.pause(0.1)


class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        self.display = pygame.display.set_mode((w, h))
        pygame.display.set_caption('Snake')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)
        self.snake = [
            self.head,
            Point(self.head.x - BLOCK_SIZE, self.head.y),
            Point(self.head.x - 2 * BLOCK_SIZE, self.head.y)
        ]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0

    def _place_food(self):
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:
            self._place_food()

    def _reachable_space(self):
        visited = set()
        q = deque([self.head])
        visited.add((self.head.x, self.head.y))
        while q:
            x, y = q.popleft()
            for dx, dy in [(BLOCK_SIZE, 0), (-BLOCK_SIZE, 0), (0, BLOCK_SIZE), (0, -BLOCK_SIZE)]:
                nx, ny = x + dx, y + dy
                if (0 <= nx < self.w and 0 <= ny < self.h and
                        (nx, ny) not in visited and
                        Point(nx, ny) not in self.snake):
                    visited.add((nx, ny))
                    q.append(Point(nx, ny))
        return len(visited)

    def play_step(self, action):
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        self._move(action)
        self.snake.insert(0, self.head)
        reward = 0
        game_over = False

        if self.is_collision() or self.frame_iteration > 100 * len(self.snake):
            game_over = True
            reward = -100
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()
        else:
            self.snake.pop()

        reward += 0.1
        prev_distance = abs(self.snake[1].x - self.food.x) + abs(self.snake[1].y - self.food.y)
        new_distance = abs(self.head.x - self.food.x) + abs(self.head.y - self.food.y)
        reward += 1 if new_distance < prev_distance else -0.5

        space_ratio = self._reachable_space() / (self.w * self.h)
        reward += 2 * space_ratio

        self._update_ui()
        self.clock.tick(SPEED)

        return reward, game_over, self.score

    def is_collision(self, pt=None):
        if pt is None:
            pt = self.head
        if pt.x >= self.w or pt.x < 0 or pt.y >= self.h or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        self.display.fill(BLACK)
        for pt in self.snake:
            pygame.draw.rect(self.display, BLUE1, pygame.Rect(pt.x, pt.y, BLOCK_SIZE, BLOCK_SIZE))
            pygame.draw.rect(self.display, BLUE2, pygame.Rect(pt.x + 4, pt.y + 4, 12, 12))
        pygame.draw.rect(self.display, RED, pygame.Rect(self.food.x, self.food.y, BLOCK_SIZE, BLOCK_SIZE))
        text = font.render("Score: " + str(self.score), True, WHITE)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _move(self, action):
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)
        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]
        elif np.array_equal(action, [0, 1, 0]):
            new_dir = clock_wise[(idx + 1) % 4]
        else:
            new_dir = clock_wise[(idx - 1) % 4]

        self.direction = new_dir
        x, y = self.head.x, self.head.y
        if new_dir == Direction.RIGHT:
            x += BLOCK_SIZE
        elif new_dir == Direction.LEFT:
            x -= BLOCK_SIZE
        elif new_dir == Direction.DOWN:
            y += BLOCK_SIZE
        elif new_dir == Direction.UP:
            y -= BLOCK_SIZE
        self.head = Point(x, y)


def create_q_model(input_size=14, output_size=3):
    return tf.keras.Sequential([
        layers.Input(shape=(input_size,)),
        layers.Dense(512, activation='relu'),
        layers.Dense(256, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(output_size)
    ])


class QTrainer:
    def __init__(self, model, lr, gamma):
        self.model = model
        self.gamma = gamma
        self.optimizer = optimizers.Adam(learning_rate=lr)
        self.loss_fn = losses.MeanSquaredError()

    def train_step(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        action = np.array(action, dtype=np.int32)
        reward = np.array(reward, dtype=np.float32)
        done = np.array(done, dtype=np.bool_)

        if state.ndim == 1:
            state = np.expand_dims(state, 0)
            next_state = np.expand_dims(next_state, 0)
            action = np.expand_dims(action, 0)
            reward = np.expand_dims(reward, 0)
            done = np.expand_dims(done, 0)

        batch_size = state.shape[0]
        pred = self.model(state)
        next_pred = self.model(next_state)
        targets = pred.numpy().copy()
        next_pred_np = next_pred.numpy()

        for idx in range(batch_size):
            Q_new = reward[idx] if done[idx] else reward[idx] + self.gamma * np.max(next_pred_np[idx])
            action_idx = int(np.argmax(action[idx]))
            targets[idx][action_idx] = Q_new

        with tf.GradientTape() as tape:
            preds = self.model(state, training=True)
            loss_value = self.loss_fn(targets, preds)
        grads = tape.gradient(loss_value, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss_value.numpy()


class Agent:
    def __init__(self):
        self.n_games = 0
        self.epsilon = 0
        self.gamma = 0.9
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = create_q_model()
        self.model(np.zeros((1, 14), dtype=np.float32))
        self.trainer = QTrainer(self.model, lr=LR, gamma=self.gamma)

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        mini_sample = random.sample(self.memory, BATCH_SIZE) if len(self.memory) > BATCH_SIZE else list(self.memory)
        states, actions, rewards, next_states, dones = zip(*mini_sample)
        return self.trainer.train_step(states, actions, rewards, next_states, dones)

    def train_short_memory(self, state, action, reward, next_state, done):
        return self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        self.epsilon = 80 - self.n_games
        move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move[random.randint(0, 2)] = 1
        else:
            state0 = np.expand_dims(np.array(state, dtype=np.float32), 0)
            prediction = self.model(state0).numpy()[0]
            move[int(np.argmax(prediction))] = 1
        return move
    # --- Copy your get_state method exactly here ---
    def get_state(self, game):
        head = game.head
        w, h = game.w, game.h
        body_positions = set(game.snake[1:])  # skip head

        # Relative directions based on current heading
        s = 20 # Step size

        # Define the 8 absolute direction vectors for clarity
        # These would typically be defined in your Direction Enum or class
        UP = (0, -s)
        UP_RIGHT = (s, -s)
        RIGHT = (s, 0)
        DOWN_RIGHT = (s, s)
        DOWN = (0, s)
        DOWN_LEFT = (-s, s)
        LEFT = (-s, 0)
        UP_LEFT = (-s, -s)

        # --- Calculate relative directions based on the current absolute direction ---

        if game.direction == Direction.UP:
            # Facing UP (North)
            forward, forward_right, right, back_right, back, back_left, left, forward_left = \
            UP,     UP_RIGHT,     RIGHT,   DOWN_RIGHT, DOWN, DOWN_LEFT, LEFT,  UP_LEFT

        elif game.direction == Direction.DOWN:
            # Facing DOWN (South)
            forward, forward_right, right, back_right, back, back_left, left, forward_left = \
            DOWN,    DOWN_LEFT,    LEFT,    UP_LEFT,    UP,   UP_RIGHT,  RIGHT, DOWN_RIGHT

        elif game.direction == Direction.LEFT:
            # Facing LEFT (West)
            forward, forward_right, right, back_right, back, back_left, left, forward_left = \
            LEFT,    UP_LEFT,      UP,      UP_RIGHT,   RIGHT,DOWN_RIGHT,DOWN,  DOWN_LEFT

        elif game.direction == Direction.RIGHT:
            # Facing RIGHT (East)
            forward, forward_right, right, back_right, back, back_left, left, forward_left = \
            RIGHT,   DOWN_RIGHT,   DOWN,    DOWN_LEFT,  LEFT, UP_LEFT,   UP,    UP_RIGHT

        elif game.direction == Direction.UP_LEFT:
            # Facing UP-LEFT (North-West)
            forward, forward_right, right, back_right, back, back_left, left, forward_left = \
            UP_LEFT, UP,           UP_RIGHT,RIGHT,      DOWN_RIGHT,DOWN,DOWN_LEFT,LEFT

        elif game.direction == Direction.UP_RIGHT:
            # Facing UP-RIGHT (North-East)
            forward, forward_right, right, back_right, back, back_left, left, forward_left = \
            UP_RIGHT,RIGHT,         DOWN_RIGHT,DOWN,     DOWN_LEFT, LEFT,UP_LEFT,  UP

        elif game.direction == Direction.DOWN_LEFT:
            # Facing DOWN-LEFT (South-West)
            forward, forward_right, right, back_right, back, back_left, left, forward_left = \
            DOWN_LEFT,LEFT,         UP_LEFT, UP,       UP_RIGHT,  RIGHT, DOWN_RIGHT,DOWN

        elif game.direction == Direction.DOWN_RIGHT:
            # Facing DOWN-RIGHT (South-East)
            forward, forward_right, right, back_right, back, back_left, left, forward_left = \
            DOWN_RIGHT,DOWN,        DOWN_LEFT,LEFT,     UP_LEFT,   UP,  UP_RIGHT,  RIGHT


        # --- Immediate danger (binary 0/1) ---
        danger = []
        for dx, dy in [forward, left, right]:
            nx, ny = head.x + dx, head.y + dy
            danger.append(
                1.0 if nx < 0 or nx >= w or ny < 0 or ny >= h or Point(nx, ny) in body_positions else 0.0
            )

        # --- Immediate food (binary 0/1) ---
        food_binary = []
        for dx, dy in [forward, left, right]:
            nx, ny = head.x + dx, head.y + dy
            food_binary.append(1.0 if (nx, ny) == (game.food.x, game.food.y) else 0.0)

        # --- Distance-based directional food features (0–1) ---
        hx, hy = head.x, head.y
        fx, fy = game.food.x, game.food.y

        def normalized_distance(dx, dy):
            dir_vec = np.array([dx, dy], dtype=np.float32)
            food_vec = np.array([fx - hx, fy - hy], dtype=np.float32)
            dist_along_dir = np.dot(food_vec, dir_vec)
            max_possible = float(max(w, h))
            val = np.clip(dist_along_dir / max_possible, 0.0, 1.0)
            return val

        food_distance = [normalized_distance(*d) for d in [forward, left, right, back]]
        
        snake_length = len(game.snake)
        max_length = (w * h) // (s**2)
        length_ratio = snake_length / max_length
        length_feature = 1 - np.exp(-3 * length_ratio)

        directions = [forward, left, right]

        
        def directional_flood_fill(step_size=20):
            """
            Computes a danger score per direction based on reachable space, using an intelligent
            tail-following heuristic.
            """
            scores = []
            max_squares = (w * h) // (step_size**2)
            snake_body = game.snake[1:] 

            for dx, dy in directions:
                new_head = Point(head.x + dx, head.y + dy)

                # --- Intelligent Tail Obstacle Calculation ---
                # A body part is only an obstacle if the head can reach it faster than
                # the tail vacates it. This logic remains the same.
                obstacles = set()
                if snake_body: # Check if the snake has a body
                    body_len = len(snake_body)
                    for i, part in enumerate(snake_body):
                        dist_to_part = (abs(new_head.x - part.x) + abs(new_head.y - part.y)) / step_size
                        steps_to_vacate = body_len - i
                        if dist_to_part < steps_to_vacate:
                            obstacles.add(part)

                # Immediate collision (wall or a "real" body obstacle)
                if new_head in obstacles or not (0 <= new_head.x < w and 0 <= new_head.y < h):
                    scores.append(1.0)  # Max danger / dead-end
                    continue

                # --- Start of Path-Based Breadth-First Search (BFS) ---
                # The queue now stores (current_point, previous_point) to track the path.
                visited = {new_head}
                queue = deque([(new_head, head)]) # The "previous" point for the new head is the current head
                count = 0

                while queue:
                    p, prev_p = queue.popleft()
                    count += 1
                    
                    # --- Explore only in 4 cardinal directions ---
                    for d_x, d_y in [
                        (step_size, 0), (-step_size, 0), (0, step_size), (0, -step_size)
                    ]:
                        nx, ny = p.x + d_x, p.y + d_y
                        neighbor = Point(nx, ny)

                        # --- NEW LOGIC: Prevent 180-degree turns ---
                        # The neighbor is invalid if it's the square we just came from.
                        if neighbor == prev_p:
                            continue

                        # Check if the neighbor is valid, not an obstacle, and not yet explored
                        if (0 <= nx < w and 0 <= ny < h and
                                neighbor not in visited and
                                neighbor not in obstacles):
                            visited.add(neighbor)
                            # Add the new point and its predecessor (p) to the queue
                            queue.append((neighbor, p))
                
                # Danger is inversely proportional to the reachable area.
                score = 1.0 - (count / max_squares)
                scores.append(score)

            return scores

        # Combine features
        state = danger + food_binary + food_distance + [length_feature] + directional_flood_fill()
        return np.array(state, dtype=np.float32)

    def save_model(self, file_name='model_tf.keras', tfjs_path='tfjs_model'):
        if not os.path.exists('./model'):
            os.makedirs('./model')
        self.model.save(os.path.join('./model', file_name))
        tfjs.converters.save_keras_model(self.model, tfjs_path)


if __name__ == '__main__':
    def train():
        plot_scores, plot_mean_scores = [], []
        total_score, record = 0, 0
        agent = Agent()
        game = SnakeGameAI()

        while True:
            state_old = agent.get_state(game)
            final_move = agent.get_action(state_old)
            reward, done, score = game.play_step(final_move)
            state_new = agent.get_state(game)
            agent.train_short_memory(state_old, final_move, reward, state_new, done)
            agent.remember(state_old, final_move, reward, state_new, done)

            if done:
                game.reset()
                agent.n_games += 1
                loss = agent.train_long_memory()
                if score > record:
                    record = score
                    agent.save_model()
                print(f'Game {agent.n_games} Score {score} Record: {record} Loss: {loss}')
                plot_scores.append(score)
                total_score += score
                plot_mean_scores.append(total_score / agent.n_games)
                plot(plot_scores, plot_mean_scores)

    train()
