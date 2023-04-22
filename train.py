import sys, os, time
import numpy as np

from pathlib import Path
from typing import *

parent_dir_path = str(Path(__file__).resolve().parents[0])
sys.path.append(parent_dir_path + "/src/evniroment")
sys.path.append(parent_dir_path + "/src/bot")

from maze_generator import Maze
from maze_define import *
from bot import botSolveMaze


def main():
    env = Maze(map_array=map_array_4)
    op = botSolveMaze(env=env)
    op.train(discount=0.95, episodes=500, start_epsilon=1)


if __name__ == "__main__":
    main()
