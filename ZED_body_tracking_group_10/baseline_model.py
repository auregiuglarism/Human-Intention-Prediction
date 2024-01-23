import json
import random
import numpy as np
from sklearn.neighbors import NearestNeighbors

from ZED_body_tracking_group_10.graph_configuration import Configuration
import math as Math


class BaselineModel:
    def __init__(self, configuration=None, G=None, known_objects=None):
        self.G = G
        self.configuration = configuration
        self.yolo_node = "root"
        self.known_objects = known_objects
        self.frame_x = 0.0
        self.frame_y = 0.0
        self.thresholdx = 0.01
        self.thresholdy = 0.01

    def compute_quadrant(self, new_object, objLst, configr):
        """
            The method computes the relative positions to the object considering the position of all close objects
            closest_object is a list of object class that has the id, name, relations to other objects
            object class has:
                - name
                - id
                - relation
                - coordinates from zed
        """
        center_x = new_object.pos[0]
        center_y = new_object.pos[1]

        # (left, front, right, back)
        relative_position = ["-1", "-1", "-1", "-1"]
        dist_neighbor = [999.0, 999.0, 999.0, 999.0]

        closest_objects, nn_distances = self.findNN(new_object.pos, objLst)
        changedObjects = []

        cntr = 0
        for ex_object in closest_objects:
            obj_x = ex_object.pos[0] - center_x
            obj_y = ex_object.pos[1] - center_y
            if abs(obj_x)< self.thresholdx and abs(obj_y) < self.thresholdy:
                return ex_object.name
            current_dist = nn_distances[0][cntr]

            # right object
            if abs(obj_x) > abs(obj_y) and obj_x > 0 and (current_dist < dist_neighbor[2]):
                relative_position[2] = ex_object.name
                ex_object.relations[0] = new_object.name
                dist_neighbor[2] = current_dist
                changedObjects.append(ex_object)
            # left object
            if abs(obj_x) > abs(obj_y) and obj_x < 0 and (current_dist < dist_neighbor[0]):
                relative_position[0] = ex_object.name
                ex_object.relations[2] = new_object.name
                dist_neighbor[0] = current_dist
                changedObjects.append(ex_object)
            # front object
            if abs(obj_x) < abs(obj_y) and obj_y > 0 and (current_dist < dist_neighbor[1]):
                relative_position[1] = ex_object.name
                ex_object.relations[3] = new_object.name
                dist_neighbor[1] = current_dist
                changedObjects.append(ex_object)
            # back object
            if abs(obj_x) < abs(obj_y) and obj_y < 0 and (current_dist < dist_neighbor[3]):
                relative_position[3] = ex_object.name
                ex_object.relations[1] = new_object.name
                dist_neighbor[3] = current_dist
                changedObjects.append(ex_object)

            cntr += 1
        # sets new relations of the object
        new_object.set_new_relation(relative_position)
        # gets the object name knowing the relations
        name = self.determineName(new_object, configr)
        # for each object substitute "Name" for the newly acquired object name
        for chgd_obj in changedObjects:
            relations = chgd_obj.relations
            for relation in range(len(relations)):
                if relations[relation] == "Name":
                    relations[relation] = name
        objLst.append(new_object)
        self.updateNames(objLst, configr)

        return name

    # Return nearest neighbours to the given coordinate point
    # objLst - list of already placed objects e.g. [("Cup0", (1, 3, 2)), ("Crate0", (1.2, 2.4, 0.5)), ...]
    # newObjPos - tuple of coordinates for newly placed object e.g. [1.2, 2.4, 0.5] i.e. [x, y, z]
    #
    # Output:
    # returns indices of nearest neighbours (ordered in terms of increasing distance i.e. at index 0 it will be closest)

    def change_names_in_relation(self, obj, current_name, new_name, objLst):
        names_in_relation = []
        for name in obj.relations:
            if name not in names_in_relation and name != "-1":
                names_in_relation.append(name)

        for object in objLst:
            if object.name in names_in_relation:
                for i in range(len(object.relations)):
                    if object.relations[i] == current_name:
                        object.relations[i] = new_name

    def updateNames(self, objLst, configr):
        for obj in objLst:
            current_name = obj.name
            new_name = self.determineName(obj, configr)
            if current_name != new_name:
                obj.name = new_name
                self.change_names_in_relation(obj, current_name, new_name, objLst)
    def determineName(self, new_object, configr):
        """
            knowing each objects relations and the configuration of the graph
            we now can determine the true naming for the object and its id
        """
        cnfgr_rltns = []
        for tup in configr:  # determine objects of the same type
            if (tup[0] == new_object.type):
                cnfgr_rltns.append((tup[0] + str(tup[1]), tup[2]))

        best_match = 0
        best_name = "None"
        for tup in cnfgr_rltns:
            match = 0
            for i in range(len(tup[1])):
                if tup[1][i] == new_object.relations[i]:
                    match += 1

            if match > best_match:
                best_name = tup[0]
                best_match = match

        new_object.name = best_name
        return best_name

    def findNN(self, newObjPos, objLst, neighbrhd=8):
        crdLst = []
        for obj in objLst:
            crdLst.append(obj.pos)

        if len(crdLst) < neighbrhd:
            neighbrhd = len(crdLst)
        if neighbrhd > 0:
            nbrs = NearestNeighbors(n_neighbors=neighbrhd, algorithm='ball_tree').fit(crdLst)
            nbrs_idx = nbrs.kneighbors([newObjPos], return_distance=True)
            nn_objs = []
            for idx in nbrs_idx[1][0]:
                nn_objs.append(objLst[idx])
            return nn_objs, nbrs_idx[0]
        else:
            return [], []

    def euclidean_distance(self, x1, y1, z1, x2, y2, z2):
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def compute_distance(self, bounding_box, skeleton):
        # Bounding box needs to be in the form of a 4x2 array, output of xywh_to_abcd.
        center_object = np.array(bounding_box)

        # Normalize using Chest X position
        left_hand = np.array([skeleton[7][0], skeleton[7][1], skeleton[7][2]])
        right_hand = np.array([skeleton[4][0], skeleton[4][1], skeleton[4][2]])

        # Compute distance for each hand
        object_lefthand = self.euclidean_distance(left_hand[0], left_hand[1], left_hand[2], center_object[0],
                                                  center_object[1], center_object[2])
        object_righthand = self.euclidean_distance(right_hand[0], right_hand[1], right_hand[2], center_object[0],
                                                   center_object[1], center_object[2])

        return object_righthand, object_lefthand

    # Euclidean distance between two points in 2D space

    def make_predictions(self, nodes):
        """
            This is an experimental setup to check whether the random model
            behaves as intended and all the node namings and order match
        """
        previous_node = nodes[0]
        hit_count = 0
        for current_node in nodes[1:]:
            predicted_node = self.predict(previous_node)
            print("Current node", current_node)
            print("Predicted node", predicted_node)
            print()

            if self.configuration.checkEq(current_node, predicted_node):
                hit_count += 1
            previous_node = current_node
        return hit_count / (len(nodes) - 1)

    def yolo_predict(self, node):
        """
            For yolo we check whether we reached the limit for predicting next action
        """
        split1 = set(node.split("_"))
        if len(split1) <= len(self.known_objects):
            self.yolo_node = node
            predic = self.predict(node)
            return predic

    def predict(self, node):
        """
            prediction is made by getting all the outgoing edges of the node
            and then picking the most probable one
        """
        self.configuration.get_corr_node(node)
        out_edges = self.G.out_edges([node])

        if len(out_edges) > 1:
            return self.perform_step(out_edges)
        if len(out_edges) == 0:
            return node
        ## never reached
        return list(out_edges)[0][1]

    def perform_step(self, out_edges):
        """
            to perform a step get firstly get the list of probabilities
            of performing the action and then make a weighted choice
        """
        out_edge_probs = []
        out_edges = list(out_edges)
        for edge in out_edges:
            out_edge_probs.append(self.G.get_edge_data(edge[0], edge[1])['weight'])

        edge_choice = random.choices(out_edges, weights=out_edge_probs, k=1)
        return edge_choice[0][1]


if __name__ == "__main__":
    configuration = [("Cup", (0, 0, 0), (1, 1)),
                     ("Crate", (1, 1, 1), (3, 3)),
                     ("Feeder", (2, 2, 2), (8, 8)),
                     ("Gold", (0, 0, 0), (1, 1))]
    graph = Configuration()
    graph.initGraph(configuration)
    graph.assign_probs()

    baseline_model = BaselineModel(graph.get_graph(), configuration)
    # print(baseline_model.predict("Cup000_root"))
    print(baseline_model.make_predictions(
        ["root", "Crate111_root", "Gold000_Crate111_root", "Gold000_Crate111_Cup000_root",
         "Gold000_Feeder222_Crate111_Cup000_root"]))
