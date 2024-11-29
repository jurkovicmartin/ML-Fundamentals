import heapq
import numpy as np

from maze import Maze

class Node:
    def __init__(self, position: tuple, previous=None):
        """Class to represent a node (position) in a 2D maze.

        Args:
            position (tuple): x, y coordinates
            previous (Node, optional): previous node. Defaults to None.
        """
        self.position = position
        self.previous = previous
        # Steps from start
        self.steps = 0
        # Heuristic cost to finish
        self.heuristic = 0
        # Total cost
        self.cost = 0

    def assign_cost(self):
        self.cost = self.steps + self.heuristic

    # Comparing for detecting maze end
    def __eq__(self, node):
        return self.position[0] == node.position[0] and self.position[1] == node.position[1]
    
    # For comparing in priority queue
    def __lt__(self, node):
        return self.cost < node.cost
    



class Astar:
    def __init__(self, values: np.array =None, path: str =None):
        """A* search algorithm to find a path in a grid maze.

        Args:
            values (np.array, optional): matrix representation of the maze. Defaults to None.
            path (str, optional): path to a png image of the maze. Defaults to None.

            Expected color combinations:
                black = wall
                white = empty
                green = start
                red = finish

            Expected values combinations:
                0 = wall
                1 = moving point
                2 = available move
                3 = finish
        """
        # Maze via image
        if path is not None and values is None:
            self.maze = Maze(path=path)
        # Maze via values
        elif values is not None and path is None:
            self.maze = Maze(values=values)
        else:
            raise Exception("Provide only path or values.")

        self.maze_array = self.maze.maze
        self.start = Node(self.maze.start_pos)
        self.finish = Node(self.maze.finish_pos)
        self.path = []


    def find_path(self) -> list:
        """Tries to find a path in the maze.

        Returns:
            list: moves of the path, in case there is no path returns list with cost of the termination point
        """
        # Open states
        open = []
        # Closed states
        closed = []
        # Priority queue
        heapq.heappush(open, self.start)
        # Up, down, left, right
        MOVES = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        while open:
            current = heapq.heappop(open)
            closed.append(current)

            # Finish reached
            if current == self.finish:
                # print("Path found")
                self._reconstruct_path(current)
                return self.path

            # Look at following steps
            for move in MOVES:
                next_position = tuple(np.add(current.position, move))
                next = Node(next_position, current)

                # Validating new position (0 are walls)
                if self.maze_array[next_position] == 0:
                    continue

                if next in closed:
                    continue
                
                # Set next nodes cost
                next.steps = current.steps + 1
                next.heuristic = self.Manhattan_distance(next.position, self.finish.position)
                next.assign_cost()

                # Check if this node (position) isn't already in open queue
                found = None
                for node in open:
                    if node == next:
                        found = node
                        break

                # Next node is in open queue
                if found:
                    # If there is with lower cost
                    if found < next:
                        continue
                    # Otherwise replace it
                    open.remove(found)
                    heapq.heappush(open, next)

                # Totally new node
                heapq.heappush(open, next)
            
        
        # print("Path not found")
        return [current.cost]

            

    def _reconstruct_path(self, current: Node):
        """Reconstruct the path by going back to the start and tracking the moves. 

        Args:
            current (Node): current node (finish)

        Raises:
            Exception: Unexpected moves difference
        """
        while True:
            # Last position
            last = current.position
            current = current.previous

            # End of 
            if not current:
                break
            
            # Current position
            new = current.position
            # Difference between the steps
            diff = tuple(np.subtract(new, last))
            # Determine the move based on difference
            if diff == (-1, 0):
                self.path.append("down")
            elif diff == (1, 0):
                self.path.append("up")
            elif diff == (0, -1):
                self.path.append("right")
            elif diff == (0, 1):
                self.path.append("left")
            else:
                raise Exception("Unexpected move")
        # Reverse order the path (bcs it went from finish to start)
        self.path.reverse()

    @staticmethod
    def Manhattan_distance(current: tuple, finish: tuple) -> int:
        """Manhattan distance calculation for heuristic cost in grid environment.

        Args:
            current (tuple): current position
            finish (tuple): finish position

        Returns:
            int: heuristic cost
        """
        return np.abs(current[0] - finish[0]) + np.abs(current[1] - finish[1])


    def show_maze(self):
        """Shows the default maze.
        """
        self.maze.show_maze()    


    def show_path(self):
        """Shows found path.
        """
        if not self.path:
            print("No path exists")
            return
        
        # Go trough maze
        for move in self.path:
            self.maze.move(move)
        self.maze.show_path()
        # Just in case resets it
        self.maze.reset()