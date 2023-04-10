import sys, os
import time
import numpy as np

from pathlib import Path
from typing import *

parent_dir_path = str(Path(__file__).resolve().parents[2])
# sys.path.append(parent_dir_path + "/src/evniroment")

import maze_define as md


class Maze_object_define:
    move_able = " "
    block = "X"
    start_point = "S"
    finish_point = "F"


class Maze_moveable_direct_define:
    total_action = 5

    left = 0
    right = 1
    up = 2
    down = 3
    no_move = 4


class Maze(Maze_object_define, Maze_moveable_direct_define):
    def __init__(self, map_array: List[str], max_step: int = 500000) -> None:
        Maze_object_define.__init__(self)
        Maze_moveable_direct_define.__init__(self)

        self.map_data = map_array
        self.map_block_array: List[int] = []

        self.current_location = [1, 1]
        self.start_location = [1, 1]
        self.destination_location = [1, 1]

        self.maze_size = [0, 0]

        self.step = 0
        self.step_limit = max_step

        self.max_reward = 2

        self.map_generate()

        # Show static
        print("Maze site: " + str(self.maze_size[1]) + "," + str(self.maze_size[0]))
        print("Max point: " + str(self.max_reward))
        time.sleep(3)

    def map_show_case(self):
        print(self.current_location)
        print(self.destination_location)

    def reset(self) -> None:
        self.map_generate()
        self.step = 0

    def map_generate(self) -> None:
        site_map = len(self.map_data[0])

        self.maze_size = [len(self.map_data), len(self.map_data[0])]

        self.max_reward = self.maze_size[0] * self.maze_size[1] // 300
        self.max_penalty = self.max_reward * 2

        for _ in range(len(self.map_data)):
            self.map_block_array.append([[0, 0, 0, 0, 0]] * site_map)

        for y in range(1, len(self.map_data) - 1, 1):
            for x in range(1, len(self.map_data[y]) - 1, 1):
                cache = [0, 0, 0, 0, 0]

                if self.map_data[y][x - 1] == self.move_able:
                    cache[self.left] = 1
                if self.map_data[y][x + 1] == self.move_able:
                    cache[self.right] = 1
                if self.map_data[y - 1][x] == self.move_able:
                    cache[self.up] = 1
                if self.map_data[y + 1][x] == self.move_able:
                    cache[self.down] = 1
                cache[self.no_move] = 1

                if self.map_data[y][x] == self.start_point:
                    self.current_location = [y, x]
                    self.start_location = [y, x]
                if self.map_data[y][x] == self.finish_point:
                    self.destination_location = [y, x]

                self.map_block_array[y][x] = cache

        # self.map_show_case()

    def get_next_coordination(self, move) -> List[int]:
        cache_current_location = self.current_location.copy()

        if move == self.no_move:
            pass
        elif move == self.left:
            cache_current_location[1] -= 1
        elif move == self.right:
            cache_current_location[1] += 1
        elif move == self.up:
            cache_current_location[0] -= 1
        elif move == self.down:
            cache_current_location[0] += 1

        return cache_current_location

    def valid_action(self, cache_current_location: List[int]) -> bool:
        if (
            self.map_data[cache_current_location[0]][cache_current_location[1]]
            == self.block
        ):
            return False
        else:
            return True

    def get_all_valid_action(self) -> List[int]:
        valid_action_list = []

        for x in range(0, self.total_action - 1, 1):
            if self.valid_action(self.get_next_coordination(x)) == True:
                valid_action_list.append(x)

        return valid_action_list

    def action(self, move) -> None:
        reward = 0
        done = False

        if self.valid_action(self.get_next_coordination(move)) == True:
            self.current_location = self.get_next_coordination(move)
            self.step += 1
            done = self.is_done()
            reward = self.max_reward if done else 0
        else:
            reward = -2

        game_over = self.is_game_over()

        return self.current_location, reward, done, game_over

    def render(self) -> None:
        os.system("clear")
        for y in range(self.maze_size[0]):
            if y == self.current_location[0]:
                cache = (
                    self.map_data[y][: self.current_location[1]]
                    + "o"
                    + self.map_data[y][self.current_location[1] + 1 :]
                )
                print(cache)
            else:
                print(self.map_data[y])

    def get_maze_size(self) -> List[int]:
        return self.maze_size

    def get_current_location(self) -> List[int]:
        return self.current_location

    def get_start_location(self) -> List[int]:
        return self.start_location

    def get_destination_location(self) -> List[int]:
        return self.destination_location

    def is_game_over(self) -> bool:
        if self.step > self.step_limit:
            return True
        else:
            return False

    def is_done(self) -> bool:
        if self.current_location == self.destination_location:
            time.sleep(1)
            print("You found a destination")
            return True
        else:
            return False

    def done(self):
        return self.is_done()
