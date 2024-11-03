
class Node:
    def __init__(self, feature: int =None, threshold: float =None, value =None, left =None, right =None):
        """Node of decision tree.

        Args:
            feature (int, optional): feature (column) for splitting. Defaults to None.
            threshold (float, optional): value for splitting. Defaults to None.
            value (optional): value of leaf node. Defaults to None.
            left (Node, optional): left child reference. Defaults to None.
            right (Node, optional): right child reference. Defaults to None.
        """
        # Attributes of feature (splitting) nodes
        self.left = left
        self.right = right
        self.feature = feature
        self.threshold = threshold
        
        # Only attribute of a leaf node
        self.value = value