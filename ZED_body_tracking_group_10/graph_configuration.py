import itertools
import json
import os
import pickle
from pathlib import Path
from time import sleep

import networkx as nx
import matplotlib.pyplot as plt
from Worker_Profiling.worker_probs import worker_dir, WorkerProbs


class Configuration:
    def __init__(self):
        self.G = None
        self.worker_id = None
        self.worker_data = {"probs": {}, "counter": {}}
        self.worker_dir = worker_dir + "/worker_"

    def set_id(self, id):
        self.worker_id = id

    def set_new_id(self):
        self.worker_id = str(len(os.listdir(worker_dir)) + 1)

    def increase_worker_counter(self, current_node, prev_node):
        """
            The method increases the counter for the self.worker_data dict
            By taking an edge of the graph

            param: current_node
            param: prev_node
        """
        worker_counter = self.worker_data['counter']
        # get correct node name
        current_node = self.get_corr_node(current_node)
        prev_node = self.get_corr_node(prev_node)
        if current_node is not None and prev_node is not None:
            key = (prev_node, current_node)
            # checking if (prev_node, curr_node) exists in worker_counter
            if key in worker_counter:
                worker_counter[key] += 1
            # else create new instance with counter 1
            else:
                worker_counter[key] = 1
            print('Worker counter', worker_counter)
            self.worker_data['counter'] = worker_counter
            #sleep(3)

    def get_corr_node(self, node):
        """
            Method get the correct node name from the graph
        """
        for n in self.G.nodes:
            if self.checkEq_baseline(node, n):
                return n
        return None

    def update_save_worker(self):
        """
            The method updates the probability profile of the worker
            and then save it in the same file
        """
        self.update_weights()
        worker_saver = WorkerProbs(self.worker_data['probs'], self.worker_data['counter'])
        worker_saver.save_pickle(self.worker_id)

    def count_freq_per_node(self):
        """
            Counts the sum of counts for each outgoing edge
        """
        out_edges = {}
        # counting how many out_edges does each node have
        for key, value in self.worker_data['counter'].items():
            init_node = key[0]
            if init_node not in out_edges:
                # init out_edge with count and then the future sum of all freqs
                # for each node with outgoing edge
                out_edges[init_node] = value
            else:
                # increase the overall freq
                out_edges[init_node][1] += value
        return out_edges

    def get_root_obj(self):
        """
            The method output a dict variable counting
            how many distinct objects are in the graph
        """
        root_obj = {}
        worker_counter = self.worker_data['counter']
        # counting how many distinct objects are going from root node
        successors = list(self.G.successors("root"))
        for node in successors:
            object_name = node.split("_")[0][:-1]
            if object_name in root_obj:
                # counter of how many same objects are in the graph
                root_obj[object_name][0] += 1
                # sum of all values of the same object
                if ("root", node) in worker_counter:
                    root_obj[object_name][1] += worker_counter[("root", node)]
            else:
                if ("root", node) in worker_counter:
                    root_obj[object_name] = [1, worker_counter[("root", node)]]
                else:
                    root_obj[object_name] = [1, 0]
        return root_obj

    def update_weights(self):
        """
            The method updates the probability profile of the worker
            getting more knowledge of his intentions
        """
        if len(self.worker_data["counter"]) > 0:
            new_probs = {}
            # get sum of freqs of out_edges per node
            out_edges_freq = self.count_freq_per_node()
            # get how many distinct objects are coming out of root node
            root_obj = self.get_root_obj()
            # assign probability to each used node by the worker
            for key, value in self.worker_data['counter'].items():
                if key[0] == "root":
                    object_name = key[1].split("_")[0][:-1]
                    new_probs[key] = round((root_obj[object_name][1] / out_edges_freq[key[0]])/root_obj[object_name][0],3)
                else:
                    new_probs[key] = round(value / out_edges_freq[key[0]], 3)
            self.worker_data['probs'] = new_probs

    def initGraph(self, input):
        G = nx.DiGraph()
        G.add_node("root", name=None, pos=None, prob=None)
        # append root to queue
        allPoss = list(itertools.permutations(input))

        for pos in allPoss:  # iterate through possibilities
            prev_ref = "root"  # keep track of the previous reference to add edges
            for objTup in pos:
                # get reference for this node
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
        """
            In order to check whether to nodes are the same
            check their naming, but the order of objects might be different
            so use sets to check whether the same objects were in the sequence
        """
        splt1 = set(nd1.split("_"))
        splt2 = set(nd2.split("_"))

        return (nd1 != nd2) and (splt1 == splt2)

    def checkEq_baseline(self, nd1, nd2):
        """
            Same @checkEq(), but here we do not check whether 2 strings are the same
        """
        split1 = nd1.split("_")
        split2 = nd2.split("_")
        set1 = set(split1)
        set2 = set(split2)

        return (set1 == set2) and len(split1)==len(split2)

    def get_non_pop_edges(self, edges):
        """
            Method counts how many edges are not still populated after loading worker profile

            param: edges - outgoing edges of current node

            return: non_pop - counter of edges that are still not populated
            return: prob_sum - how much of probability is free(all the outgoing edges must sum up to 1)
        """
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
        """
            Assigns probabilities for all unpopulated edges,
            they will be equally probable, since no information is known for those
            instances/nodes
        """
        G = self.G
        for node in G.nodes:
            edges = G.out_edges([node])
            edge_num, prob_sum = self.get_non_pop_edges(edges)
            for u, v in edges:
                edge_data = G.get_edge_data(u, v)
                if edge_data["weight"] == 0:
                    G[u][v]["weight"] = round(prob_sum / edge_num, 3)

    def load_assign_worker(self):
        """
            Assigns probs from json file in Worker_Profiling/Profiles
            and then assigns all other values to be the same
            if we do not have information about other edges.
        """
        worker_id = self.worker_id
        if worker_id is not None:
            worker_file = Path(self.worker_dir + str(worker_id) + ".json")
            if worker_file.is_file():
                self.worker_id = worker_id
                with open(worker_file, 'rb') as pickle_file:
                    worker_probs = pickle.load(pickle_file)
                self.worker_data = worker_probs
                worker_probs = worker_probs["probs"]
                G = self.G
                for edge, prob in worker_probs.items():
                    G[edge[0]][edge[1]]["weight"] = prob
        self.assign_probs()

    def get_graph(self):
        return self.G

    def hasNode(self, newName):
        for n in self.G.nodes:
            if self.checkEq_baseline(newName, n): return True
        return False


if __name__ == "__main__":
    configuration = [("Cup", (0), ("-1", "-1", "Crate0", "-1")),
                     ("Crate", (0), ("Cup0", "-1", "Cup1", "-1")),
                     ("Cup", (1), ("Crate0", "-1", "-1", "-1"))]

    graph = Configuration()
    graph.initGraph(configuration)
    # Draw the graph with edge labels
    pos = nx.spring_layout(graph.get_graph(), scale=3)

    nx.draw(graph.get_graph(), pos, with_labels=True, font_size=10, font_color="black", font_weight="bold",
            arrowsize=20)

    # Add edge labels
    labels = nx.get_edge_attributes(graph.get_graph(), 'weight')
    nx.draw_networkx_edge_labels(graph.get_graph(), pos, edge_labels=labels)

    plt.savefig("graph_vis.png")
    plt.show()
