import itertools
import networkx as nx



class Tree:
    def __init__(self, input):
        self.root = TreeNode(None, None)
        self.levels = [[]]*len(input+1)

        # append root to queue
        allPoss = list(itertools.permutations(input))
        self.levels[0].append(self.root)

        for pos in allPoss: # iterate through possibilities
            previousNode = self.root
            lvl = 1
            for objTup in pos:
                newNode = TreeNode(objTup, previousNode)
                self.levels[lvl].append(newNode)
                previousNode = newNode


class TreeNode:
    """A basic tree node class."""
    def __init__(self, tuple_in, parent): # for standard node
        if (parent!= None):
            self.name = tuple_in[0]
            self.pos = tuple_in[1] #tuple
            self.size = tuple_in[2] #tuple
        else:
            self.name = None
            self.pos = None
            self.size = None
        self.prob = 0
        self.children = []

        self.parent = parent # if 'None' then root

    def add_child(self, child_node):
        """Adds a child to this node."""
        self.children.append(child_node)

    def remove_child(self, child_node):
        """Removes a child from this node."""
        self.children = [child for child in self.children if child != child_node]

    def traverse(self):
        """Traverses the tree starting from this node."""
        nodes = [self]
        while nodes:
            current_node = nodes.pop()
            print(current_node.value)
            nodes.extend(current_node.children)


if __name__ == "__main__":
    configuration = [("Cup", (0,0), (1,1)), 
                     ("Crate", (-3,1), (3,3)),
                     ("Feeder", (999,999), (888,888))]
    assembly = Tree(configuration)