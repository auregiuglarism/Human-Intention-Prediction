import itertools
import networkx as nx
import matplotlib.pyplot as plt


class Configuration:
    def __init__(self):
        self.G = None
    def initGraph(self, input):
        G = nx.DiGraph()
        G.add_node("root", name=None, pos=None, prob=None)
        # append root to queue
        allPoss = list(itertools.permutations(input))

        for pos in allPoss: # iterate through possibilities
            prev_ref = "root" # keep track of the previous reference to add edges
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
                    G.add_node(ref, name=objTup[0], size = objTup[1])
                    G.add_edge(prev_ref, ref, weight=0.0)

                # add edge
                    prev_ref = ref
                if(G.has_node(ref)):
                    prev_ref = ref

        self.G = G
        return G

    def checkEq(self,nd1, nd2):
        splt1 = set(nd1.split("_"))
        splt2 = set(nd2.split("_"))

        return (nd1!=nd2) and (splt1 == splt2)

    def assign_probs(self):
        G = self.G
        for node in G.nodes:
            edges = G.out_edges([node])
            for edge in edges:
                nx.set_edge_attributes(G, {edge:{"weight": round(1/len(edges), 3)}})

    def get_graph(self):
        return self.G


if __name__ == "__main__":
    configuration = [("Cup", (0,0,0), (1,1)),
                         ("Crate", (1,1,1), (2,2)),
                         ("Feeder", (2,2,2), (1,2)),
                         ("Cup", (1,2,1), ((1,1)))]
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
