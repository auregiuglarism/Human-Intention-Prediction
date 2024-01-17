import networkx as nx
import matplotlib.pyplot as plt

from configuration import Configuration

# compute id of an object based on left right directions 
#
# Input:
# 'objType' - string with object class
# 'configr' - tuple configuration
# 'plcdObj' - tuple with placed objects and their coordinates
def computeId(objType, configr, plcdObj):
    return 



if __name__ == "__main__":
    # type, id, relatives (left, front, right, back)
    #
    # Configuration:
    #       feeder0
    #          |
    # cup0 - crate0 - cup1
    # 
    configuration = [("Cup", (0), (-1, -1, 1, -1)),
                         ("Crate", (0), (0, 2, 3, -1)),
                         ("Feeder", (0), (-1, -1, -1, 1)),
                         ("Cup", (1), (1, -1, -1, -1))]
    graph = Configuration()
    graph.initGraph(configuration)
    graph.assign_probs()
    # Draw the graph with edge labels
    pos = nx.spring_layout(graph.get_graph(), scale=3)

    nx.draw(graph.get_graph(), pos, with_labels=True, font_size=10, font_color="black", font_weight="bold", arrowsize=20)

    # Add edge labels
    labels = nx.get_edge_attributes(graph.get_graph(), 'weight')
    nx.draw_networkx_edge_labels(graph.get_graph(), pos, edge_labels=labels)

    plt.savefig("graph_vis.png")
    plt.show()