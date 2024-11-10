from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class Maze:
    def __init__(self, path: str):
        self.img = Image.open(path)
        self._initialize()


    def _initialize(self):
        # 0 = wall
        # 1 = moving point
        # 2 = available move
        # 3 = finish
        self.maze = np.array(self.img)
        self.shape = np.shape(self.maze)
        self.start_pos = None
        self.finish_pos = None
        self.current_pos = None
        self.moves_history = []

        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if self.maze[row, col] == 1:
                    self.start_pos = (row, col)
                elif self.maze[row, col] == 3:
                    self.finish_pos = (row, col)

        self.current_pos = self.start_pos

    
    def reset(self):
        self._initialize()


    def show_maze(self, title: str =""):
        # black = wall
        # blue = moving point
        # white = available move
        # red = finish
        if np.isin(3, self.maze):
            colors = ["black", "blue", "white", "red"]
        else:
            colors = ["black", "blue", "white"]
        color_map = ListedColormap(colors)

        plt.imshow(self.maze, cmap=color_map)
        plt.title(title)
        plt.axis("off")
        plt.show()


    def move(self, direction: str) -> bool:
        if direction == "up":
            x, y = self.current_pos
            if self.maze[x-1, y] == 2 or self.maze[x-1, y] == 3:
                self.maze[x, y] = 2
                self.maze[x-1, y] = 1
                self.current_pos = (x-1, y)

                self.moves_history.append("u")
                return True
            else: return False
        
        elif direction == "down":
            x, y = self.current_pos
            if self.maze[x+1, y] == 2 or self.maze[x+1, y] == 3:
                self.maze[x, y] = 2
                self.maze[x+1, y] = 1
                self.current_pos = (x+1, y)
                
                self.moves_history.append("d")
                return True
            else: return False
        
        elif direction == "left":
            x, y = self.current_pos
            if self.maze[x, y-1] == 2 or self.maze[x, y-1] == 3:
                self.maze[x, y] = 2
                self.maze[x, y-1] = 1
                self.current_pos = (x, y-1)
                
                self.moves_history.append("l")
                return True
            else: return False
        
        elif direction == "right":
            x, y = self.current_pos
            if self.maze[x, y+1] == 2 or self.maze[x, y+1] == 3:
                self.maze[x, y] = 2
                self.maze[x, y+1] = 1
                self.current_pos = (x, y+1)
                
                self.moves_history.append("r")
                return True
            else: return False
        else:
            raise Exception("Invalid move.")



    def is_finished(self) -> bool:
        return self.current_pos == self.finish_pos


    def show_path(self):
        maze_copy = self.maze.copy()
        x, y = self.start_pos
        for move in self.moves_history:
            if move == "u":
                maze_copy[x-1, y] = 4
                x -= 1
            elif move == "d":
                maze_copy[x+1, y] = 4
                x += 1
            elif move == "l":
                maze_copy[x, y-1] = 4
                y -= 1
            elif move == "r":
                maze_copy[x, y+1] = 4
                y += 1
        # Assign back start and finish value
        maze_copy[self.start_pos] = 1
        maze_copy[self.finish_pos] = 3
        # 0 = black = wall
        # 1 = green = start
        # 2 = white = empty
        # 3 = red = finish
        # 4 = blue = path
        colors = ["black", "green", "white", "red", "blue"]
        color_map = ListedColormap(colors)

        plt.imshow(maze_copy, cmap=color_map)
        plt.title("Found path")
        plt.axis("off")
        plt.show()
