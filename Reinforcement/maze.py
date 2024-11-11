from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class Maze:
    def __init__(self, path: str):
        """Maze environment from an image. One pixel represents one cell of the maze.
        Expected color combinations:
            black = wall
            white = empty
            green = start
            red = finish

        Args:
            path (str): path to the image
        """
        self.img = Image.open(path)
        self._initialize()


    def _initialize(self):
        """Initialize instance variables.
                0 = wall
                1 = moving point
                2 = available move
                3 = finish
        """
        self.maze = np.array(self.img)
        self.shape = np.shape(self.maze)
        self.start_pos = None
        self.finish_pos = None
        self.current_pos = None
        self.moves_history = []

        # Find start and finish point in the maze
        for row in range(self.shape[0]):
            for col in range(self.shape[1]):
                if self.maze[row, col] == 1:
                    self.start_pos = (row, col)
                elif self.maze[row, col] == 3:
                    self.finish_pos = (row, col)
        # Set current position at start
        self.current_pos = self.start_pos

    
    def reset(self):
        """Resets instance variables to initial the state.
        """
        self._initialize()


    def show_maze(self, title: str =""):
        """Show maze with matplotlib.
                black = wall
                blue = moving point
                white = available move
                red = finish
        Args:
            title (str, optional): title of the graph. Defaults to "".
        """
        # Determine if the finish point is in the maze or if it is covered by the moving point
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
        """Make a move in the maze.

        Args:
            direction (str): move direction (up / down / left / right)

        Raises:
            Exception: Invalid move option

        Returns:
            bool: True = move has been done / False = hitting a wall
        """
        if direction == "up":
            x, y = self.current_pos
            # Check if the move is valid (empty cell or finish)
            if self.maze[x-1, y] == 2 or self.maze[x-1, y] == 3:
                # Move moving point
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
            raise Exception("Invalid move option.")


    def is_finished(self) -> bool:
        """If the moving point reached finish.

        Returns:
            bool: True = Yes / False = No
        """
        return self.current_pos == self.finish_pos


    def show_path(self, title: str="Found path"):
        """Show path that moving point traveled with matplotlib.
                0 = black = wall
                1 = green = start
                2 = white = empty
                3 = red = finish
                4 = blue = path

        Args:
            title (str, optional): title of the graph. Defaults to "Found path".
        """
        # Copy the maze
        maze_copy = self.maze.copy()
        # Set position on start
        x, y = self.start_pos
        # Go through the moves and mark the path
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
        # Mark back start and finish
        maze_copy[self.start_pos] = 1
        maze_copy[self.finish_pos] = 3

        colors = ["black", "green", "white", "red", "blue"]
        color_map = ListedColormap(colors)

        plt.imshow(maze_copy, cmap=color_map)
        plt.title(title)
        plt.axis("off")
        plt.show()
