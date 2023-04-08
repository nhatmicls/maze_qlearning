import random
import sys, os, time
import numpy as np
import json

from pathlib import Path
from typing import *

parent_dir_path = str(Path(__file__).resolve().parents[2])
sys.path.append(parent_dir_path + "/src/evniroment")

from maze_generator import Maze


class botSolveMaze:
    def __init__(self, env: Maze) -> None:
        self.env = env
        self.DISCRETE_OS_SIZE = self.env.get_maze_size()
        self.q_table = np.random.uniform(
            low=-2, high=0, size=(self.DISCRETE_OS_SIZE + [env.total_action])
        )

        self.own_reward: List[List[int]] = []
        for _ in range(self.env.get_maze_size()[0]):
            self.own_reward.append([0] * self.env.get_maze_size()[1])

        self.own_reward[self.env.get_destination_location()[0]][
            self.env.get_destination_location()[1]
        ] = 1

        self.self_update_q_table = True
        self.self_update = True

        self.lowest_reward = 1
        self.stack_no_new_value = 0
        self.last_episode = 0
        self.history_change: List[List[int]] = []

        # self.export_model("./test.json")

    def recalculate_reward(
        self, old_coordination: List[int], new_coordination: List[int]
    ) -> float:
        new_point_reward = self.own_reward[new_coordination[0]][new_coordination[1]]
        old_point_reward = self.own_reward[old_coordination[0]][old_coordination[1]]

        if new_point_reward >= old_point_reward:
            return self.own_reward[new_coordination[0]][new_coordination[1]]
        else:
            return -1

    def update_reward(
        self, old_coordination: List[int], new_coordination: List[int], episode: int
    ) -> None:
        if self.self_update == False:
            return

        new_point_reward = self.own_reward[new_coordination[0]][new_coordination[1]]
        old_point_reward = self.own_reward[old_coordination[0]][old_coordination[1]]
        drop_reward_per_step = 1 / 100

        if old_coordination == self.env.get_start_location() and new_point_reward != 0:
            self.self_update = False
            if self.self_update_q_table == True:
                self.q_table = np.random.uniform(
                    low=-2,
                    high=0,
                    size=(self.DISCRETE_OS_SIZE + [self.env.total_action]),
                )
                self.self_update_q_table = False
            return

        if new_point_reward > 0:
            if old_point_reward == 0:
                new_reward = new_point_reward - drop_reward_per_step

                if (
                    len(self.env.get_all_valid_action()) <= 1
                    and new_coordination != self.env.get_destination_location()
                ):
                    if old_coordination != self.env.get_start_location():
                        self.own_reward[new_coordination[0]][new_coordination[1]] = -1

                    self.stack_no_new_value += 1
                    return

                if new_reward > 0 and new_reward < self.lowest_reward:
                    self.own_reward[old_coordination[0]][old_coordination[1]] = round(
                        new_reward, 3
                    )
                    self.stack_no_new_value = 0
                    self.lowest_reward = new_reward
                    self.history_change.append(old_coordination)
                else:
                    self.stack_no_new_value += 1
        else:
            if old_point_reward == 0:
                self.own_reward[new_coordination[0]][new_coordination[1]] = 0
                self.stack_no_new_value = 0

        if self.stack_no_new_value > 1 and self.last_episode < episode:
            self.stack_no_new_value = 0

            self.lowest_reward += drop_reward_per_step
            self.last_episode = episode

            last_change_location = self.history_change[len(self.history_change) - 1]
            self.own_reward[last_change_location[0]][last_change_location[1]] = -1
            self.history_change.pop()

    def import_model(self, direct_path: str) -> None:
        with open(direct_path) as json_file:
            data = json.load(json_file)

        json_file.close()

    def export_model(self, direct_path: str) -> None:
        data = {}

        for y in range(self.q_table.shape[0] - 1):
            action_point_y = {}

            y_value: np.ndarray = self.q_table[y]

            for x in range(y_value.shape[0] - 1):
                action_point = {str(x): []}

                x_value: np.ndarray = y_value[x]

                for z in range(x_value.size - 1):
                    action_point[str(x)].append(x_value[z])
                action_point_y.update(action_point)
            data.update({str(y): action_point_y})

        with open(direct_path, "w+", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        f.close()

    def run(
        self,
        learning_rate: int = 0.9,
        discount: int = 0.95,
        episodes: int = 100,
        start_epsilon: int = 0.5,
    ) -> None:
        SHOW_EVERY = episodes / 10
        START_EPSILON_DECAYING = 1
        END_EPSILON_DECAYING = episodes // 2
        epsilon = start_epsilon
        epsilon_decay_value = epsilon / (END_EPSILON_DECAYING - START_EPSILON_DECAYING)

        # print(np.argmax(q_table[(1, 1)]))
        # print((q_table[(1, 1)]))
        for episode in range(1, episodes + 1, 1):
            self.env.reset()

            start_time = time.time()
            state = self.env.current_location
            done = False

            while not done:
                if np.random.rand() < epsilon:
                    action = random.choice(self.env.get_all_valid_action())
                else:
                    action = np.argmax(self.q_table[tuple(state)])

                new_state, reward, done, _ = self.env.action(action)

                if action == self.env.no_move:
                    reward = -1

                # Generate reward

                self.update_reward(
                    old_coordination=state, new_coordination=new_state, episode=episode
                )
                reward = self.recalculate_reward(
                    old_coordination=state, new_coordination=new_state
                )

                # Recalculate Q_table

                if not done:
                    max_future_q = np.max(self.q_table[tuple(new_state)])

                    current_q = self.q_table[tuple(new_state) + (action,)]

                    new_q = (1 - learning_rate) * current_q + learning_rate * (
                        reward + discount * max_future_q
                    )

                    self.q_table[tuple(new_state) + (action,)] = new_q

                elif new_state == self.env.get_destination_location():
                    self.q_table[tuple(new_state) + (action,)] = 1

                state = new_state
                # print(action, state)

                if episode > episodes - 1:
                    time.sleep(0.01)
                    self.env.render()

            if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
                epsilon -= epsilon_decay_value

            self.last_episode = episode

            # os.system("clear")
            if True:
                # self.env.render()
                print("Episode: " + str(episode) + "/" + str(episodes))
                print("Step: " + str(self.env.step) + "/" + str(self.env.step_limit))
                print("Epsilon: " + str(epsilon) + "/" + str(start_epsilon))
                stop_time = time.time()
                duration = stop_time - start_time
                print("Time taken: " + str(duration) + " s")
                # for _ in range(self.DISCRETE_OS_SIZE[0]):
                #     print(self.own_reward[_])

        self.export_model("./q_table.json")
