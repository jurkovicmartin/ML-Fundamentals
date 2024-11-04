import numpy as np

from node import Node


class DecisionTree:

    def __init__(self):
        self.root = None

    @staticmethod
    def entropy(x) -> float:
        """Calculates entropy of x.
        In terms of decision tree use labels to calculate entropy.

        Args:
            x (array): input

        Returns:
            float: entropy
        """
        total_count = len(x)
        _, counts = np.unique(x, return_counts=True)

        probabilities = counts / total_count

        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])

    @staticmethod
    def most_common(x):
        """Returns most common value from input.

        Args:
            x (array): input

        Returns:
            most common value
        """
        values, counts = np.unique(x, return_counts=True)

        max = np.argmax(counts)

        return values[max]


    def build_tree(self, data, labels, max_depth: int, min_split: int):
        """Builds tree structure.

        Args:
            data (array): training data
            labels (array): training labels
            max_depth (int): maximum tree depth (starting from 0)
            min_split (int): number of samples that is no longer splitting
        """
        self.max_depth = max_depth
        self.min_split = min_split

        self.root = self._create_node(data, labels)


    def predict(self, input):
        """Predicts output value to the input.

        Args:
            input (array): input

        Returns:
            predicted value
        """
        # Track current node
        node = self.root

        while True:
            # Node is leaf
            if node.value is not None:
                return node.value
            
            # Node is feature node
            if input[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right


    def test(self, data, labels) -> float:
        """Tests tree with testing data and labels.

        Args:
            data (array): testing data
            labels (array): testing labels

        Returns:
            float: accuracy
        """
        errors_count = 0

        for input, label in zip(data, labels):
            prediction = self.predict(input)

            if prediction != label:
                errors_count += 1
        # Return accuracy
        return (len(data) - errors_count) / len(data)


    def print_tree(self, option: str ="preorder"):
        """Prints tree structure into console.

        Args:
            option (str, optional): type of tree traversal. Options are "preorder"/"postorder"/"inorder" Defaults to "preorder".
        """
        if option == "preorder":
            print("Pre-order tree print")
            self._preorder_print(self.root)

        elif option == "postorder":
            print("Post-order tree print")
            self._postorder_print(self.root)

        elif option == "inorder":
            print("In-order tree print")
            self._inorder_print(self.root)

        else:
            print("Invalid printing option.")


    def _split(self, data, feature: int, threshold: float) -> tuple:
        """Splits input data into two groups based on feature (column) and threshold (value).

        Args:
            data (array): input to split
            feature (int): feature (column) of the data for splitting
            threshold (float): value for splitting

        Returns:
            tuple: (left_indexes, right_indexes) indexes applies to input data
        """
        left_indexes = [idx for idx, value in enumerate(data) if value[feature] <= threshold]
        right_indexes = [idx for idx, value in enumerate(data) if value[feature] > threshold]

        return left_indexes, right_indexes


    def _find_criteria(self, data, labels) -> tuple:
        """Finds best splitting criteria (feature and threshold). Best split is determined by information gain.

        Args:
            data (array): input
            labels (array): labels

        Returns:
            tuple: (best feature, best threshold)
        """
        best_feature = None
        best_threshold = None
        best_info_gain = -1

        parent_entropy = DecisionTree.entropy(labels)

        # Input dimensions
        value_number, features_number = np.shape(data)

        # Trying feature
        for feature in range(features_number):
            # Get only unique values of feature
            unique_values = np.unique([data[i][feature] for i in range(value_number)])

            # Trying threshold
            for threshold in unique_values:
                left_indexes, right_indexes = self._split(data, feature, threshold)
                
                # One split is empty
                if not left_indexes or not right_indexes:
                    continue

                ### CALCULATING INFORMATION GAIN
                left_entropy = DecisionTree.entropy([labels[i] for i in left_indexes])
                right_entropy = DecisionTree.entropy([labels[i] for i in right_indexes])
                # Wight is just frequency
                left_weight = len(left_indexes) / len(data)
                right_weight = len(right_indexes) / len(data)

                split_entropy = left_weight * left_entropy + right_weight + right_entropy

                information_gain = parent_entropy - split_entropy

                # Update best feature and threshold
                if information_gain > best_info_gain:
                    best_info_gain = information_gain
                    best_feature = feature
                    best_threshold = threshold

        return best_feature, best_threshold

    
    def _create_node(self, data, labels, depth: int =0) -> Node:
        """Creates new tree node. Two types of nodes: leaf and feature (splitting).
           For feature nodes uses finding best splitting criteria.

        Args:
            data (array): node input
            labels (array): labels
            depth (int, optional): depth of node. Defaults to 0.

        Returns:
            Node: feature or leaf node
        """
        ### LEAF NODE CONDITIONS
        # Reached max depth
        if depth == self.max_depth:
            return Node(value=DecisionTree.most_common(labels))
        # Too little data to split
        if len(data) <= self.min_split:
            return Node(value=DecisionTree.most_common(labels))
        # All of the input data belongs to one class
        if len(set(labels)) == 1:
            return Node(value=DecisionTree.most_common(labels))

        # Best feature and threshold
        feature, threshold = self._find_criteria(data, labels)

        left_indexes, right_indexes = self._split(data, feature, threshold)

        # Creating feature (splitting node)
        node = Node(feature=feature, threshold=threshold)

        # Recursively go onto next layer (increase depth)
        node.left = self._create_node([data[i] for i in left_indexes], [labels[i] for i in left_indexes], depth + 1)
        node.right = self._create_node([data[i] for i in right_indexes], [labels[i] for i in right_indexes], depth + 1)

        return node
    


    ### PRINT METHODS
    def _preorder_print(self, node: Node, depth: int =0):
        """Prints pre-order tree traversal.

        Args:
            node (Node): current node
            depth (int, optional): depth of node. Defaults to 0.
        """
        # End of branch
        if node is None:
            return
        
        print(f"Depth: {depth}")

        if node.value is None:
            print(f"Feature node: Feature: {node.feature}, Threshold: {node.threshold}")
        else:
            print(f"Leaf node: Value: {node.value}")

        self._preorder_print(node.left, depth + 1)
        self._preorder_print(node.right, depth + 1)

    def _postorder_print(self, node: Node, depth: int =0):
        """Prints post-order tree traversal.

        Args:
            node (Node): current node
            depth (int, optional): depth of current node. Defaults to 0.
        """
        # End of branch
        if node is None:
            return

        print(f"Depth: {depth}")

        self._postorder_print(node.left, depth + 1)
        self._postorder_print(node.right, depth + 1)
        
        if node.value is None:
            print(f"Feature node: Feature: {node.feature}, Threshold: {node.threshold}")
        else:
            print(f"Leaf node: Value: {node.value}")

    def _inorder_print(self, node: Node, depth: int =0):
        """Prints in-order tree traversal.

        Args:
            node (Node): current node
            depth (int, optional): depth of current node. Defaults to 0.
        """
        if node is None:
            return
        
        print(f"Depth: {depth}")

        self._inorder_print(node.left, depth + 1)
        
        if node.value is None:
            print(f"Feature node: Feature: {node.feature}, Threshold: {node.threshold}")
        else:
            print(f"Leaf node: Value: {node.value}")
        
        self._inorder_print(node.right, depth + 1)
