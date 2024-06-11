# ---------------------------------------------------------------------------- #
# 感觉没什么用
# 把random.seed(MAX_DIFFERENT_MAZES)  注释掉
# 把random.seed(random.randint(1,MAX_DIFFERENT_MAZES))取消注释
# 每次运行layout_generator.py都会生成20个新的不同的随机地图
# ---------------------------------------------------------------------------- #



import random, sys, os
from pathlib import Path
base_dir = str(Path(__file__).resolve().parent.parent)
sys.path.append(base_dir)
from mazeGenerator import Maze, make, add_pacman_stuff

MAX_DIFFERENT_MAZES = 202325995
# random.seed(random.randint(1,MAX_DIFFERENT_MAZES))    # 随机数种子随机生成，每次生成的地图都不同
random.seed(MAX_DIFFERENT_MAZES)    #随机数种子MAX_DIFFERENT_MAZES来初始化随机数生成器，
                    # 由于每次使用相同的种子生成的随机数序列是相同的，因此在这里每次运行生成的地图都是相同的。

row, col, factor = (16, 16, 1)

for idx in range(20):
    while 1:
        try:
            maze = Maze(int(row * factor),int(col * factor))
            make(maze, depth=0, gaps=6, vert=row>=col, min_width=1)
            maze.to_map()
            add_pacman_stuff(maze, max(2*(maze.r*maze.c/20), 60))
            print(maze)
            break
        except IndexError:
            pass
    if not os.path.exists(f'{base_dir}/layouts/eval'):
        os.makedirs(f'{base_dir}/layouts/eval')
    with open(f'{base_dir}/layouts/eval/{idx}.lay',
              'w', encoding='utf8') as f:
        print(maze, file=f)
