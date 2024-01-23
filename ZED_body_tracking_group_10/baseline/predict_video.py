import json
from time import sleep

import cv2
from ultralytics import YOLO
import math as Math
import numpy as np

from ZED_body_tracking_group_10.baseline_model import BaselineModel
from ZED_body_tracking_group_10.graph_configuration import Configuration


class MultiModelDetector:
    def __init__(self, models, baseline, objects):
        self.models = models
        self.node_name = "root"
        self.baseline_model = baseline
        self.known_objects = objects
        self.center_x = 0.0
        self.center_y = 0.0

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

    def compute_distance(self, bounding_box, skeleton):
        w0 = 1
        w1 = 1

        # Bounding box needs to be in the form of a 4x2 array, output of xywh_to_abcd.
        center_x = (bounding_box[2][0] - bounding_box[0][0]) / 2
        center_y = (bounding_box[2][1] - bounding_box[0][1]) / 2
        center_object = np.array([center_x, center_y])

        # Skeleton needs to be an array of 34 keypoints
        # Normalize Chest X
        chest_x = skeleton[1][0]
        left_hand = np.array([skeleton[7][0] - chest_x, skeleton[7][1]])
        right_hand = np.array([skeleton[14][0] - chest_x, skeleton[14][1]])
        head = np.array([skeleton[27][0] - chest_x, skeleton[27][1]])

        # Compute distance
        object_lefthand = w0 * self.euclidean_distance(left_hand[0], left_hand[1], center_object[0], center_object[1])
        object_righthand = w0 * self.euclidean_distance(right_hand[0], right_hand[1], center_object[0],
                                                        center_object[1])
        object_head = w1 * self.euclidean_distance(head[0], head[1], center_object[0], center_object[1])

        avg_distance = (object_lefthand + object_righthand + object_head) / 3.0

        return avg_distance

    # Euclidean distance between two points in 2D space
    def euclidean_distance(self, x1, y1, x2, y2):
        return Math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    def compute_quadrant(self, bounding_box):
        object_center_x = ((bounding_box[2][0] - bounding_box[0][0]) / 2) - center_x
        object_center_y = ((bounding_box[2][1] - bounding_box[0][1]) / 2) - center_y

        abs_object_center_x = abs(object_center_x)
        abs_object_center_y = abs(object_center_y)

        if (abs_object_center_x > abs_object_center_y and object_center_x > 0 and object_center_y > 0):
            return "NE"
        if (abs_object_center_x > abs_object_center_y and object_center_x > 0 and object_center_y < 0):
            return "SE"
        if (abs_object_center_x > abs_object_center_y and object_center_x < 0 and object_center_y > 0):
            return "NW"
        if (abs_object_center_x > abs_object_center_y and object_center_x < 0 and object_center_y < 0):
            return "SW"
        if (abs_object_center_x < abs_object_center_y and object_center_y > 0):
            return "N"
        if (abs_object_center_x < abs_object_center_y and object_center_y < 0):
            return "S"

    def detect_and_draw(self, frame, threshold=0.5):
        results_list = [model(frame)[0].boxes.data.tolist() for model in self.models]

        for results in zip(*results_list):
            max_score = max(result[4] for result in results)

            if max_score > threshold:
                best_result = max(results, key=lambda x: x[4])
                x1, y1, x2, y2, score, class_id = best_result

                bounding_box = [[x1, y1], [], [x2, y2]]

                # print("X1", x1, "Y1", y1, "X2", x2, "Y2", y2)
                skeleton_data = self.load_data('/Users/Vitalij/Desktop/Project3-1/left_left/bodies_1.json')
                # print(self.compute_distance(bounding_box, skeleton_data[10]))

                # MAKING PREDICTION
                quadrant = self.compute_quadrant(bounding_box)
                name = str(self.models[results.index(best_result)].names[int(class_id)]) + str(
                    quadrant) + "_" + self.node_name
                self.run_baseline(name)

                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 4)
                cv2.putText(frame,
                            f"{self.models[results.index(best_result)].names[int(class_id)].upper()} {score:.2f}",
                            (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3, cv2.LINE_AA)

    def run(self, video_path, output_path):
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        H, W, _ = frame.shape
        out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'MP4V'), int(cap.get(cv2.CAP_PROP_FPS)), (W, H))
        self.center_y = H / 2
        self.center_x = W / 2

        while ret:
            self.detect_and_draw(frame, threshold=0)

            cv2.imshow("Detection Model", frame)  # Display the frame

            out.write(frame)
            ret, frame = cap.read()
            sleep(5)
            # Break the loop if 'q' key is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        out.release()
        cv2.destroyAllWindows()

    def checkEq(self, nd1, nd2):
        splt1 = set(nd1.split("_"))
        splt2 = set(nd2.split("_"))

        return (nd1 != nd2) and (splt1 == splt2)

    def run_baseline(self, name):
        split1 = set(name.split("_"))
        if (len(split1) < len(self.known_objects)):
            self.node_name = name
            predic = self.baseline_model.yolo_predict(name)
            print("PREDICTION " + predic)
            print()


if __name__ == '__main__':
    objects = [("Cup0", "N"),
               ("Crate0", "NW"),
               ("Feeder0", "S")]
    configuration = objects
    graph = Configuration()
    graph.initGraph(configuration)
    graph.assign_probs()

    baseline_model = BaselineModel(graph.get_graph())

    model1 = YOLO('/Users/Vitalij/Desktop/Project3-1/Yolo_Models/best.pt')

    video_path = '/Users/Vitalij/Desktop/Project3-1/baselines/baseline/angled.mp4'
    video_path_out = '{}_models{}_out.mp4'.format(video_path, '2_th0.3')

    multi_model_detector = MultiModelDetector([model1], baseline_model, objects)
    multi_model_detector.run(video_path, video_path_out)
