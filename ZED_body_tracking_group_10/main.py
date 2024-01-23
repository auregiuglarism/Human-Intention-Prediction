import argparse
import math
from statistics import mode
from threading import Lock, Thread
from time import sleep

import cv2
import pyzed.sl as sl
import torch
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO

import baseline_model
import graph_configuration
from Yolo_Objects.object import Object
from main_utility.zed_camera import ZedCamera

lock = Lock()
run_signal = False
exit_signal = False


class ZedObjectDetection:
    def __init__(self, config, baseline, worker_id = None):
        # INITIALIZING ZED CAMERA
        print("Initializing Camera...")
        self.zed_camera = ZedCamera(sl)
        print("Initialized Camera.")
        self.frame_size_set = False

        # OBJECT DETECTION
        self.delayed_holding = True
        self.flicker_list = []
        self.config = config
        self.map_raw_to_label = {"0": "Crate", "1": "Feeder", "2": "Cup"}

        # WORKER DATA
        if worker_id is None:
            self.config.set_new_id()
        else:
            self.config.set_id(worker_id)
        self.config.load_assign_worker()

        # BASELINE
        self.baseline = baseline
        self.node_name = "root"

        # PREDICTION LIMITER
        self.holding = False
        self.prevHolding = False
        self.plcdObjs = []

        # DELAY VARIABLES
        self.five_maps = []  # map in this case corresponds to object list with coordinates
        self.fifteen_dist = []  # list of 15 distances contains lists of objects with their left and right distances
        self.latest_id = -1

    def increase_worker_counter(self, current_node, prev_node):
        """
          Method for updating the counter of grabbed objects by the worker during runtime
          Predictions do not have any effect from this or on this

          param: current_node - the configuration at this moment
          param: prev_node - the previous configuration before placing current object
        """
        print("Increasing counter for ", current_node , prev_node)
        self.config.increase_worker_counter(current_node, prev_node)

    def update_save_worker(self):
        """
            Update the worker file with the probabilities coming from the counter variable
        """
        self.config.update_save_worker()

    # thread running the YOLO detector
    def torch_thread(self, weights, conf_thres=0.2, iou_thres=0.45):
        """
            Get object using pre-defined yolo model
            param: weights - pre-trained weights, for the objects that are going to be in the frame
            param: conf_thres - the threshold with which yolo accepts the object detection
            param: iou_thres - the overlap between the ground truth and the predicted objects
        """
        global image_net, exit_signal, run_signal, detections

        model = YOLO(weights)

        while not exit_signal:
            if run_signal:
                lock.acquire()

                img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2BGR)
                # define image size at first instance
                if not self.frame_size_set:
                    H, W, _ = img.shape
                    self.baseline.frame_x = 0
                    self.baseline.frame_y = 0
                    self.frame_size_set = True

                results = model.predict(img, save=False, conf=conf_thres, iou=iou_thres)[0]
                det = results.cpu().numpy().boxes

                # ZED CustomBox format (with inverse letterboxing tf applied)
                detections = self.zed_camera.detections_to_custom_box(det, image_net, results.names, sl)
                lock.release()
                run_signal = False
            sleep(0.01)

    def make_node_name(self,obj_name):
        new_name = "root"
        for obj in self.plcdObjs:
            used_node = new_name.split("_")
            if obj.name in used_node:
                new_name = obj.name[:-1] + str(int(obj.name[-1])+1) + "_" + new_name
            else:
                new_name = obj.name + "_" + new_name
        return new_name

    def main(self):
        """
            the method runs zed camera, and through it skeleton data, object data is run and
            the prediction is made
        """
        global image_net, exit_signal, run_signal, detections

        capture_thread = Thread(target=self.torch_thread,
                                kwargs={'weights': opt.weights, "conf_thres": opt.conf_thres,
                                        "iou_thres": opt.iou_thres})
        capture_thread.start()

        skeleton_data = []
        while not exit_signal:
            if self.zed_camera.get_error_code() == sl.ERROR_CODE.SUCCESS:
                # -- Get skeleton and serialize it
                skeleton_data.append(self.zed_camera.get_skeleton_data())

                # -- Get the image
                lock.acquire()
                self.zed_camera.zed.retrieve_image(self.zed_camera.image_left_tmp, sl.VIEW.LEFT)
                image_net = self.zed_camera.image_left_tmp.get_data()
                lock.release()
                run_signal = True

                # -- YOLO detection running on the other thread
                while run_signal:
                    sleep(0.001)

                # Wait for detections
                lock.acquire()
                # -- Ingest detections
                self.zed_camera.zed.ingest_custom_box_objects(detections)
                lock.release()
                objects = self.zed_camera.get_object_data()

                # -- Display
                # Retrieve display data
                self.zed_camera.retrieve_display_data(sl)

                # Able to track the label of slowly moving object
                # but not moving objects have slightly changing
                # coordinates due to noise
                knn_labels = self.getLabelsKNN(objects.object_list)

                single_frame_map = self.updateKNN(knn_labels, objects.object_list)

                self.flicker_method(self.holding)
                if len(skeleton_data[-1]['body_list']) > 0:
                    frm_dists = self.calcAllDistances(single_frame_map, skeleton_data[-1]['body_list'][-1])
                    self.updateFifteenDst(frm_dists)

                    object = self.get_avg_dists()

                    if object is not None and self.prevHolding and not self.delayed_holding:
                        # make the which is the closest to the hands over time
                        # getting the class name from @self.map_raw_to_label
                        # and the object position
                        new_obj = Object(self.map_raw_to_label[str(object[1])], object[2].tolist())
                        print('newobj: ', new_obj.type)
                        # compute where the object is relative to other objects in the frame
                        # outputs the name, based on its location
                        obj_name = self.baseline.compute_quadrant(new_obj, self.plcdObjs, self.baseline.known_objects)

                        # get correct node name which corresponds to the graph
                        name = self.make_node_name(obj_name)

                        # keep track of all placed objects in the frame
                        # self.plcdObjs.append(new_obj)
                        # if the baseline has this node, then it will predict next node
                        if self.baseline.configuration.hasNode(name):
                            # append this configuration counter for the worker using current node(@name)
                            # and prev node(@self.node_name)
                            self.increase_worker_counter(name, self.node_name)
                            # keep track of current node name
                            self.node_name = name
                            # make prediction
                            pred = self.baseline.yolo_predict(name)
                            print("Prediction ", pred)
                            sleep(1)

                            # if the graph doesn't have any more nodes to predict end the program
                            if len(self.plcdObjs) == len(self.baseline.known_objects):
                                exit_signal = True
                        else:
                            print("Wrong placement")
                            print(name)
                            sleep(1)
                    self.prevHolding = self.delayed_holding

                # Rendering the objects on screen
                self.zed_camera.display_image()

                key = cv2.waitKey(1)
                if key == 27:
                    exit_signal = True
            else:
                exit_signal = True

        exit_signal = True
        self.update_save_worker()
        self.zed_camera.close_camera()

    def get_close_object(self, object_list, skeleton, threshold=0.28):
        """
            Compute the closest object to the worker using 1 frame

            input: object_list - the list of objects in the frame
            input: skeleton - skeleton data of the worker
            input: threshold - everything < than threshold will be considered close

            return: the most probable object that the user reaches for
        """
        prob_object = None
        min_dist = float('inf')

        for obj in object_list:
            distance_right, distance_left = self.baseline.compute_distance(obj.position,
                                                                           skeleton['keypoint'])
            if distance_right < min_dist:
                prob_object = obj
                min_dist = distance_right

            elif distance_left < min_dist:
                prob_object = obj
                min_dist = distance_left

        if min_dist < threshold:
            self.holding = True
            return prob_object

        self.holding = False  # not changing
        # if letting go return closest object
        if self.prevHolding:
            return prob_object
        return None

    def get_avg_dists(self, threshold=0.3):
        """
            the function calculates the average distance based on 15 frames

            input: threshold - after what distance does the object considered to be close

            return: the most probable object that user touches
        """
        prob_object = None
        min_dist = float('inf')

        avg_dist = {}

        # first find all unique labels
        for frame in self.fifteen_dist:
            for obj, distR, distL, raw_label, pos in frame:
                if obj not in avg_dist:
                    avg_dist[obj] = [distR, distL, 1, raw_label, pos]
                else:
                    avg_dist[obj][0] += distR
                    avg_dist[obj][1] += distL
                    avg_dist[obj][2] += 1
                    avg_dist[obj][4] = pos

        avg_dist = [[obj, avg_dist[obj][0] / avg_dist[obj][2], avg_dist[obj][1] / avg_dist[obj][2], avg_dist[obj][3],
                     avg_dist[obj][4]] for obj in avg_dist]

        for label, distance_right, distance_left, raw_label, pos in avg_dist:
            if distance_right < distance_left:
                distance_current = distance_right
            else:
                distance_current = distance_left

            if distance_current < min_dist:
                prob_object = [label, raw_label, pos]
                min_dist = distance_current

        if min_dist < threshold:
            self.holding = True
            return prob_object

        self.holding = False  # not changing

        if self.prevHolding:  # if letting go return closest object
            return prob_object
        return None

    def flicker_method(self, holding):
        self.flicker_list.append(holding)
        if len(self.flicker_list) > 1:
            self.flicker_list = self.flicker_list[1:]
        self.delayed_holding = mode(self.flicker_list)

    def getLabelsKNN(self, object_list, neighs=3, rad = 0.2):
        """
            The method determines if there has been placed a new object
            or that it has been moved a "bit"

            input: object_list - the objects that were detected in YOLO
            input: neighs - how many neighbors are considered
            input: rad - based on distance calculation between observed objects

            return object labels
        """
        if len(object_list) == 0: return []

        labels = []
        if len(self.five_maps) == 0:
            labels = list(range(len(object_list)))
            self.latest_id = len(labels)-1
        else:
            crdLst = []
            labelLst = []
            for frmObjs in self.five_maps:
                for obj in frmObjs:
                    if not math.isnan(obj[1][0]):
                        labelLst.append(obj[0])
                        crdLst.append(obj[1])

            if len(self.five_maps) < 3:
                neighs = 1
            nbrs = NearestNeighbors(n_neighbors=neighs, algorithm='ball_tree', radius=rad).fit(crdLst)

            for obj in object_list:
                if not math.isnan(obj.position[0]):
                    dist, nbrs_idx = nbrs.radius_neighbors([obj.position], return_distance=True)
                    dist, nbrs_idx = zip(*sorted(zip(dist, nbrs_idx)))  # sort in ascending order by distance
                    nbrs_idx = nbrs_idx[0][:neighs]  # get closest ones

                    obj_lbls = []
                    for idx in nbrs_idx:
                        obj_lbls.append(labelLst[idx])
                    if len(obj_lbls) >0:
                        labels.append(mode(obj_lbls))
                    else:
                        self.latest_id+=1
                        labels.append(self.latest_id)  # assumption of new objects being at the end of the object list
        return labels

    def updateKNN(self, knnLabels, object_list):
        temp_list = []
        for i in range(len(knnLabels)):
            temp_list.append([knnLabels[i], object_list[i].position, str(object_list[i].raw_label)])

        if len(temp_list) != 0:
            self.five_maps.append(temp_list)

        if len(self.five_maps) > 5:
            self.five_maps = self.five_maps[1:]

        return temp_list

    def calcAllDistances(self, single_frame_map, skeleton):
        output = []

        for obj in single_frame_map:
            if not (math.isnan(obj[1][0]) or math.isnan(obj[1][1]) or math.isnan(obj[1][2])):
                distance_right, distance_left = self.baseline.compute_distance(obj[1], skeleton['keypoint'])  # 'keypoint'
                # [label, rDist, lDist, rawLbl, Pos]
                output.append([obj[0], distance_right, distance_left, obj[2], obj[1]])

        return output

    def updateFifteenDst(self, frm_dists):
        if len(frm_dists) != 0:
            self.fifteen_dist.append(frm_dists)
        if len(self.fifteen_dist) > 8:
            self.fifteen_dist = self.fifteen_dist[1:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='Yolo_Models/new_best.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default='/home/kamil/PycharmProjects/Project3-1_WORKING_ZED/ZED_body_tracking_group_10/assembly1.svo', help='optional svo file')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold')
    parser.add_argument('--worker_id', type=int, default= None, help= 'worker_id')
    opt = parser.parse_args()

    graph = [("Cup", (0), ("-1", "Feeder0", "-1", "-1")),
             ("Cup", (1), ("Feeder0", "-1", "-1", "-1")),
             ("Crate", (0), ("-1", "-1", "Feeder0", "-1")),
             ("Feeder", (0), ("Crate0", "-1", "Cup1", "Cup0"))]
    config = graph_configuration.Configuration()
    config.initGraph(graph)

    baseline_model = baseline_model.BaselineModel(config, config.get_graph(), graph)

    zed = ZedObjectDetection(config, baseline_model)

    with torch.no_grad():
        zed.main()
