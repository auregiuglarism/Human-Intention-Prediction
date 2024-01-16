import json
import random
import numpy as np

from ZED_body_tracking_group_10.configuration import Configuration
import math as Math





class BaselineModel:
    def __init__(self, G=None, known_objects = None):
        self.G = G
        self.yolo_node = "root"
        self.known_objects = known_objects
        self.frame_x = 0.0
        self.frame_y = 0.0


    def load_data(self, path):
        data = []
        with open(path) as f:
            json_data = json.load(f)
            result_list = []

            # Iterate through each timestamp
            for timestamp, entry in json_data.items():
                # Extract the "keypoint" information for the current timestamp
                keypoints = entry.get("body_list", [])

                # Iterate through each keypoint
                for keypoint in keypoints:
                    # Extract the coordinates (x, y, z) for the current keypoint
                    coordinates = keypoint.get("keypoint", [])

                    # Append the coordinates to the result list
                    result_list.append(coordinates)
        return result_list

    def compute_quadrant(self, bounding_box):
        global center_x, center_y
        # object_center_x = ((bounding_box[2][0] - bounding_box[0][0]) / 2) - center_x
        # object_center_y = ((bounding_box[2][1] - bounding_box[0][1]) / 2) - center_y
        object_center_x = ((bounding_box[2][0] - bounding_box[0][0]) / 2)
        object_center_y = ((bounding_box[2][1] - bounding_box[0][1]) / 2)

        abs_object_center_x = abs(object_center_x)
        abs_object_center_y = abs(object_center_y)

        if abs_object_center_x > abs_object_center_y and object_center_x > 0 and object_center_y > 0:
            return "NE"
        if abs_object_center_x > abs_object_center_y and object_center_x > 0 > object_center_y < 0:
            return "SE"
        if abs_object_center_x > abs_object_center_y and object_center_x < 0 and object_center_y > 0:
            return "NW"
        if abs_object_center_x > abs_object_center_y and object_center_x < 0 and object_center_y < 0:
            return "SW"
        if abs_object_center_x < abs_object_center_y and object_center_y > 0:
            return "N"
        if abs_object_center_x < abs_object_center_y and object_center_y < 0:
            return "S"

    def euclidean_distance(self,x1, y1, z1, x2, y2, z2):
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2 + (z2 - z1) ** 2)

    def checkEq(self, nd1, nd2):
        splt1 = set(nd1.split("_"))
        splt2 = set(nd2.split("_"))

        return (nd1 != nd2) and (splt1 == splt2)

    def compute_distance(self, bounding_box, skeleton):
        global center_x, center_y
        ## TODO: IMPLEMENT FOR Z AXIS

        # Bounding box needs to be in the form of a 4x2 array, output of xywh_to_abcd.
        # center_x = (bounding_box[0][0] - bounding_box[3][0]) / 2
        # center_y = (bounding_box[0][1] - bounding_box[4][1]) / 2
        # center_z = (bounding_box[0][2] - bounding_box[1][2]) /2
        # center_object = np.array([center_x, center_y, center_z])
        center_object = np.array(bounding_box)

        # Skeleton needs to be an array of 34 keypoints
        # Normalize Chest X
        left_hand = np.array([skeleton[7][0], skeleton[7][1], skeleton[7][2]]) # chest_x subtrction
        right_hand = np.array([skeleton[4][0], skeleton[4][1], skeleton[4][2]])
        # print("Left {}\n Right {}".format(left_hand, right_hand))
        # print("Objects: ", center_object)

        # Compute distance
        object_lefthand = self.euclidean_distance(left_hand[0], left_hand[1], left_hand[2], center_object[0], center_object[1], center_object[2])
        object_righthand = self.euclidean_distance(right_hand[0], right_hand[1], right_hand[2], center_object[0], center_object[1], center_object[2])

        # avg_distance = (object_lefthand + object_righthand) / 2.0
        # print('Dstance:', avg_distance)
        # print('rhs: ', object_righthand)
        # print('lhs: ', object_lefthand)
        # print()

        return object_righthand, object_lefthand

    # Euclidean distance between two points in 2D space

    def make_predictions(self, nodes):
        previous_node = nodes[0]
        hit_count = 0
        for current_node in nodes[1:]:
            predicted_node = self.predict(previous_node)
            print("Current node", current_node)
            print("Predicted node", predicted_node)
            print()

            if self.checkEq(current_node, predicted_node):
                hit_count += 1
            previous_node = current_node
        return hit_count/(len(nodes)-1)

    def yolo_predict(self, node):
        split1 = set(node.split("_"))
        if (len(split1) < len(self.known_objects)):
            self.yolo_node = node
            predic = self.predict(node)
            return predic

    def predict(self, node):
        out_edges = self.G.out_edges([node])
        if len(out_edges) > 1:
            return self.perform_step(out_edges)
        if len(out_edges) == 0:
            return node
        return list(out_edges)[0][1]

    def perform_step(self, out_edges):
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
    print(baseline_model.make_predictions(["root", "Crate111_root", "Gold000_Crate111_root", "Gold000_Crate111_Cup000_root","Gold000_Feeder222_Crate111_Cup000_root"]))
