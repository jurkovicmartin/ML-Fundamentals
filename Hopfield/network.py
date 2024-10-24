import numpy as np

def activation_function(x: float):
    """
    Bipolar step function.
    """
    return -1 if x < 0 else 1


class Hopfield:
    def __init__(self, size: int):
        """Initialize Hopfield network.

        Args:
            size (int): number of neurons (= number of input values)
        """
        self.size = size

        self.weights = np.zeros((size, size))
        

    def train(self, input):
        """Train the network for a pattern/s.

        Args:
            input (array): input pattern/s that should be learned.
        """
        # Single pattern
        if input.shape == (self.size,):
            input = input[:, np.newaxis]

            self.weights = np.dot(input, input.T)
            np.fill_diagonal(self.weights, 0)
        # Multiple patterns
        else:
            for sample in input:
                sample = sample[:, np.newaxis]
                self.weights += np.dot(sample, sample.T)

            np.fill_diagonal(self.weights, 0)
                

    def reconstruct(self, input, states: str ="output"):
        """Reconstructs input sequence to one of learned patterns.

        Args:
            input (array): input sequence
            states (str, optional): amount of state to return "output" for only output / "all" for state from each iteration. Defaults to "final".

        Returns:
            array: reconstructed pattern
        """
        state = input[:, np.newaxis]
        last_state = np.zeros(self.size)
        iterations = 1

        all_states = [state]
        
        while not np.array_equal(last_state, state):
            print(f"iteration: {iterations}")
            help_state = state.copy()

            potentials = np.dot(self.weights, state)
            state = np.array([activation_function(x) for x in potentials])[:, np.newaxis]
            all_states.append(state)
            last_state = help_state.copy()

            iterations += 1

        if states == "output":
            return all_states[-1]
        elif states == "all":
            return np.array(all_states)
        else:
            raise Exception("Invalid states option.")
                    
