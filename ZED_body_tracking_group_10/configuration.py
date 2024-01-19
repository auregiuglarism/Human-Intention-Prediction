import itertools
import json
import os
from pathlib import Path

import networkx as nx
import matplotlib.pyplot as plt
from probability_graph.worker_probs import worker_dir


class Configuration:
    def __init__(self):
        self.G = None
        self.worker_dir = worker_dir + "/worker_"

    def initGraph(self, input):
        G = nx.DiGraph()
        G.add_node("root", name=None, pos=None, prob=None)
        # append root to queue
        allPoss = list(itertools.permutations(input))

        for pos in allPoss:  # iterate through possibilities
            prev_ref = "root"  # keep track of the previous reference to add edges
            for objTup in pos:
                # get reference for this node
                # ref = objTup[0]+str(objTup[1][0])+str(objTup[1][1])+str(objTup[1][2])+"_"+str(prev_ref)
                ref = objTup[0] + str(objTup[1]) + "_" + str(prev_ref)

                edges_to_add = []
                foundEqls = False
                for n in G.nodes:
                    if self.checkEq(n, ref):
                        edges_to_add.append((prev_ref, n))
                        foundEqls = True
                        prev_ref = n

                # Add all the new edges after iterating through the nodes
                for edge in edges_to_add:
                    G.add_edge(*edge, weight=0.0)
                    # add node if not already added
                if (not foundEqls and not G.has_node(ref)):
                    G.add_node(ref, name=objTup[0], size=objTup[1])
                    G.add_edge(prev_ref, ref, weight=0.0)

                    # add edge
                    prev_ref = ref
                if (G.has_node(ref)):
                    prev_ref = ref

        self.G = G
        return G

    def checkEq(self, nd1, nd2):
        splt1 = set(nd1.split("_"))
        splt2 = set(nd2.split("_"))

        return (nd1 != nd2) and (splt1 == splt2)

    def checkEq_baseline(self, nd1, nd2):
        splt1 = set(nd1.split("_"))
        splt2 = set(nd2.split("_"))

        return (splt1 == splt2)

    def get_non_pop_edges(self, edges):
        non_pop = 0
        prob_sum = 0
        for u, v in edges:
            edge_data = self.G.get_edge_data(u, v)
            if edge_data["weight"] == 0:
                non_pop += 1
            else:
                prob_sum += edge_data['weight']

        prob_sum = 1 - prob_sum
        return non_pop, prob_sum

    def assign_probs(self):
        G = self.G
        for node in G.nodes:
            edges = G.out_edges([node])
            edge_num, prob_sum = self.get_non_pop_edges(edges)
            for u,v in edges:
                edge_data = G.get_edge_data(u, v)
                if edge_data["weight"] == 0:
                    G[u][v]["weight"] = round(prob_sum / edge_num, 3)


    """
        input - worker_id 
        Assigns probs from json file in probability_graph/worker_probs
        and then assigns all other values to be the same 
        if we do not have information about other edges.
    """
    def assign_worker_probs(self, worker_id):
        if worker_id is not None:
            worker_file = Path(self.worker_dir + str(worker_id) + ".json")
            if worker_file.is_file():
                worker_probs = json.load(worker_file.open())
                G = self.G
                for used_node, edges in worker_probs.items():
                    for edge in edges:
                        G[used_node][edge[0]]["weight"] = edge[1]
        self.assign_probs()

    def get_graph(self):
        return self.G

    def hasNode(self, newName):
        for n in self.G.nodes:
            if (self.checkEq_baseline(newName, n)): return True
        return False


if __name__ == "__main__":
    configuration = [("Cup", (0), ("-1", "-1", "Crate0", "-1")),
                     ("Crate", (0), ("Cup0", "-1", "Cup1", "-1")),
                     ("Cup", (1), ("Crate0", "-1", "-1", "-1"))]

    graph = Configuration()
    graph.initGraph(configuration)
    graph.assign_worker_probs(1)
    # Draw the graph with edge labels
    pos = nx.spring_layout(graph.get_graph(), scale=3)

    nx.draw(graph.get_graph(), pos, with_labels=True, font_size=10, font_color="black", font_weight="bold",
            arrowsize=20)

    # Add edge labels
    labels = nx.get_edge_attributes(graph.get_graph(), 'weight')
    nx.draw_networkx_edge_labels(graph.get_graph(), pos, edge_labels=labels)

    plt.savefig("graph_vis.png")
    plt.show()
