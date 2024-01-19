#!/usr/bin/env python3

import os
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from sklearn.neighbors import NearestNeighbors
from ultralytics import YOLO
from statistics import mode

from threading import Lock, Thread
from time import sleep

import cv_viewer_detector.tracking_viewer as cv_viewer
import configuration
import baseline_model
from ZED_body_tracking_group_10.Object import Object
from ZED_body_tracking_group_10.Record_Skeleton_Data import serializeBodies

lock = Lock()
run_signal = False
exit_signal = False


class ZedObjectDetection:
    def __init__(self, config, baseline):
        # self.body_runtime_param = None
        # self.bodies = None
        # self.runtime_params = None
        # self.zed = None
        self.delayed_holding = True
        self.flicker_list = []
        self.config = config
        self.baseline = baseline
        self.node_name = "root"
        self.frame_size_set = False
        self.map_raw_to_label = {"0": "Crate", "1": "Feeder", "2": "Cup"}

        # Flood gate variables
        self.holding = False
        self.prevHolding = False
        self.plcdObjs = []

        # Delay variables
        self.five_maps = []  # map in this case corresponds to object list with coordinates
        self.fifteen_dist = []  # list of 15 distances contains lists of objects with their left and right distances

    # converts xywh format (used by YOLO) to abcd format (used by ZED SDK)
    def xywh_to_abcd(self, xywh, im_shape):
        output = np.zeros((4, 2))

        # Center / Width / Height -> BBox corners coordinates
        x_min = (xywh[0] - 0.5 * xywh[2])  # * im_shape[1]
        x_max = (xywh[0] + 0.5 * xywh[2])  # * im_shape[1]
        y_min = (xywh[1] - 0.5 * xywh[3])  # * im_shape[0]
        y_max = (xywh[1] + 0.5 * xywh[3])  # * im_shape[0]

        # A ------ B
        # | Object |
        # D ------ C

        output[0][0] = x_min
        output[0][1] = y_min

        output[1][0] = x_max
        output[1][1] = y_min

        output[2][0] = x_max
        output[2][1] = y_max

        output[3][0] = x_min
        output[3][1] = y_max
        return output

    # converts detections from YOLO to ZED SDK CustomBox format
    def detections_to_custom_box(self, detections, im0, names):
        output = []
        for i, det in enumerate(detections):
            xywh = det.xywh[0]

            # Creating ingestable objects for the ZED SDK
            obj = sl.CustomBoxObjectData()
            obj.bounding_box_2d = self.xywh_to_abcd(xywh, im0.shape)
            obj.label = det.cls
            obj.probability = det.conf
            obj.is_grounded = False
            output.append(obj)
        return output

    # thread running the YOLO detector
    def torch_thread(self, weights, img_size, conf_thres=0.2, iou_thres=0.45):
        global image_net, exit_signal, run_signal, detections

        print("Intializing Network...")

        model = YOLO(weights)

        while not exit_signal:
            if run_signal:
                lock.acquire()

                img = cv2.cvtColor(image_net, cv2.COLOR_BGRA2BGR)
                if not self.frame_size_set:
                    H, W, _ = img.shape
                    self.baseline.frame_x = 0
                    self.baseline.frame_y = 0
                    self.frame_size_set = True
                # https://docs.ultralytics.com/modes/predict/#video-suffixes
                # , imgsz = img_size
                results = model.predict(img, save=False, conf=conf_thres, iou=iou_thres)[0]
                det = results.cpu().numpy().boxes

                # ZED CustomBox format (with inverse letterboxing tf applied)
                detections = self.detections_to_custom_box(det, image_net, results.names)
                lock.release()
                run_signal = False
            sleep(0.01)

    def initZed(self, zed):
        input_type = sl.InputType()
        if opt.svo is not None:
            input_type.set_from_svo_file(opt.svo)

        # Create a InitParameters object and set configuration parameters
        # https://www.stereolabs.com/docs/video/camera-controls/
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.HD2K  # HD720 for extra wide FOV
        init_params.camera_fps = 30  # 15 fps for better depth quality under low light
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # depth quality
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 10  # distance in coordinate_units (should be 10 meters)

        self.runtime_params = sl.RuntimeParameters()
        status = zed.open(init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        image_left_tmp = sl.Mat()

        print("Initialized Camera")

        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is not static, comment the following line to have better performances
        positional_tracking_parameters.set_as_static = True
        zed.enable_positional_tracking(positional_tracking_parameters)

        obj_param = sl.ObjectDetectionParameters()
        obj_param.instance_module_id = 0
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        obj_param.enable_segmentation = True
        zed.enable_object_detection(obj_param)

        body_tracking_parameters = sl.BodyTrackingParameters()
        body_tracking_parameters.instance_module_id = 1
        body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
        body_tracking_parameters.enable_body_fitting = True
        body_tracking_parameters.enable_tracking = True

        error_code = zed.enable_body_tracking(body_tracking_parameters)
        if (error_code != sl.ERROR_CODE.SUCCESS):
            print("Can't enable positionnal tracking: ", error_code)

        self.bodies = sl.Bodies()
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_param.detection_confidence_threshold = 30

        # Display
        camera_infos = zed.get_camera_information()
        camera_res = camera_infos.camera_configuration.resolution

        point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
        # point_cloud_res = sl.Resolution(camera_res.width, camera_res.height)
        point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

        image_left = sl.Mat()

        # Utilities for 2D display
        display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
        # display_resolution = sl.Resolution(camera_res.width, camera_res.height)
        image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255],
                                 np.uint8)

        # Utilities for tracks view
        camera_config = camera_infos.camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps,
                                                        init_params.depth_maximum_distance)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
        # Camera pose
        cam_w_pose = sl.Pose()

        return (objects, obj_param, obj_runtime_param, body_tracking_parameters, image_left_tmp,
                point_cloud, point_cloud_res, image_left, display_resolution, cam_w_pose, image_left_ocv, image_scale,
                image_track_ocv, track_view_generator)

    def retrieve_bodies(self):
        self.zed.retrieve_bodies(self.bodies, self.body_runtime_param, self.body_tracking_parameters.instance_module_id)

    # main thread
    def main(self):
        global image_net, exit_signal, run_signal, detections

        capture_thread = Thread(target=self.torch_thread,
                                kwargs={'weights': opt.weights, 'img_size': opt.img_size, "conf_thres": opt.conf_thres,
                                        "iou_thres": opt.iou_thres})
        capture_thread.start()

        print("Initializing Camera...")

        zed = sl.Camera()

        input_type = sl.InputType()
        if opt.svo is not None:
            input_type.set_from_svo_file(opt.svo)

        # Create a InitParameters object and set configuration parameters
        # https://www.stereolabs.com/docs/video/camera-controls/
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.HD2K  # HD720 for extra wide FOV
        init_params.camera_fps = 30  # 15 fps for better depth quality under low light
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # depth quality
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 10  # distance in coordinate_units (should be 10 meters)

        runtime_params = sl.RuntimeParameters()
        status = zed.open(init_params)

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

        image_left_tmp = sl.Mat()

        print("Initialized Camera")

        positional_tracking_parameters = sl.PositionalTrackingParameters()
        # If the camera is not static, comment the following line to have better performances
        positional_tracking_parameters.set_as_static = True
        zed.enable_positional_tracking(positional_tracking_parameters)

        obj_param = sl.ObjectDetectionParameters()
        obj_param.instance_module_id = 0
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        obj_param.enable_segmentation = True
        zed.enable_object_detection(obj_param)  # still n important line even if doesn't set anything

        body_tracking_parameters = sl.BodyTrackingParameters()
        body_tracking_parameters.instance_module_id = 1
        body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
        body_tracking_parameters.enable_body_fitting = True
        body_tracking_parameters.enable_tracking = True

        error_code = zed.enable_body_tracking(body_tracking_parameters)
        if (error_code != sl.ERROR_CODE.SUCCESS):
            print("Can't enable positionnal tracking: ", error_code)

        bodies = sl.Bodies()
        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_param.detection_confidence_threshold = 30

        # Display
        camera_infos = zed.get_camera_information()
        camera_res = camera_infos.camera_configuration.resolution

        point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
        # point_cloud_res = sl.Resolution(camera_res.width, camera_res.height)
        point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

        image_left = sl.Mat()

        # Utilities for 2D display
        display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
        # display_resolution = sl.Resolution(camera_res.width, camera_res.height)
        image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
        image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255],
                                 np.uint8)

        # Utilities for tracks view
        camera_config = camera_infos.camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps,
                                                        init_params.depth_maximum_distance)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
        # Camera pose
        cam_w_pose = sl.Pose()

        print("Initializing All Parameters")

        skeleton_data = []
        tmpSz = 0

        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # -- Get skeleton and serialize it
                zed.retrieve_bodies(bodies, body_runtime_param, body_tracking_parameters.instance_module_id)
                skeleton_data.append(serializeBodies(bodies))

                # if len(skeleton_data)>5000:
                # -- Get the image
                lock.acquire()
                zed.retrieve_image(image_left_tmp, sl.VIEW.LEFT)
                image_net = image_left_tmp.get_data()
                lock.release()
                run_signal = True

                # -- Detection running on the other thread
                while run_signal:
                    sleep(0.001)

                # Wait for detections
                lock.acquire()
                # -- Ingest detections
                zed.ingest_custom_box_objects(detections)
                lock.release()
                zed.retrieve_objects(objects, obj_runtime_param, obj_param.instance_module_id)

                # -- Display
                # Retrieve display data
                zed.retrieve_measure(point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU, point_cloud_res)  # update point cloud
                zed.retrieve_image(image_left, sl.VIEW.LEFT, sl.MEM.CPU, display_resolution)
                zed.get_position(cam_w_pose, sl.REFERENCE_FRAME.WORLD)

                # object tracking
                # TODO https://www.stereolabs.com/docs/depth-sensing/using-depth/
                os.system('cls' if os.name == 'nt' else 'clear')

                ## OUR CODE ##
                print("Length of skeleton data: ", len(skeleton_data[-1]['body_list']))
                # print("Length of object list ", len(objects.object_list))

                if (tmpSz < len(objects.object_list)):
                    print()
                tmpSz = len(objects.object_list)

                knnLabels = self.getLabelsKNN(objects.object_list)  # able to track the label of slowly moving object
                # But the not moving objects have slightly changing
                # coordinates due to noise

                single_frame_map = self.updateKNN(knnLabels, objects.object_list)
                print(self.five_maps)
                sleep(2)

                self.flicker_method(self.holding)
                if len(skeleton_data[-1]['body_list']) > 0:
                    frm_dists = self.calcAllDistances(single_frame_map, skeleton_data[-1]['body_list'][-1])
                    self.updateFifteenDst(frm_dists)

                    # print(self.obj_num)
                    # print(len(objects.object_list))
                    # object = self.get_close_object(objects.object_list, skeleton_data[-1]['body_list'][-1])
                    object = self.get_avg_dists()
                    # print("Object: ", object is not None)
                    # print("prev_hold ", self.prevHolding)
                    # print('hold: ', self.holding)
                    # print("Delayed holding ", not self.delayed_holding)

                    if object is not None and self.prevHolding and not self.delayed_holding:
                        newObj = Object(self.map_raw_to_label[str(object.raw_label)], object.position.tolist())
                        obj_name = self.baseline.compute_quadrant(newObj, self.plcdObjs, self.baseline.known_objects)
                        new_name = "root"
                        for obj in self.plcdObjs:
                            new_name = obj.name + "_" + new_name
                        name = obj_name + "_" + new_name
                        self.plcdObjs.append(newObj)
                        # print("obj_name ", obj_name)

                        if self.baseline.configuration.hasNode(name):
                            self.node_name = name  # keep track of placed objects
                            # print("name: ", name)
                            pred = self.baseline.yolo_predict(name)
                            print("Prediction ", pred)
                            sleep(3)  # TODO: remove, was used for debugging
                            print()
                        else:
                            # print("name: ", name)
                            print("Wrong placement")
                    self.prevHolding = self.delayed_holding
                # for obj in objects.object_list:
                # print("ID: {} \nPos: {} \n3D Box: {} \nConf: {} \nClass: {}".format(obj.id, obj.position,
                #                                                                     obj.bounding_box,
                #                                                                     obj.confidence, obj.raw_label))

                # 2D rendering
                np.copyto(image_left_ocv, image_left.get_data())
                cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
                global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                # Tracking view
                track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

                sleep(0.001)
                cv2.imshow("ZED | 2D View and Birds View", global_image)
                key = cv2.waitKey(1)
                if key == 27:
                    exit_signal = True
            else:
                exit_signal = True

        exit_signal = True
        zed.close()

    def get_close_object(self, object_list, skeleton, threshold=0.28):
        prob_object = None
        min_dist = float('inf')

        for obj in object_list:
            distance_right, distance_left = self.baseline.compute_distance(obj.position,
                                                                           skeleton['keypoint'])  # 'keypoint'

            if distance_right < min_dist:
                prob_object = obj
                min_dist = distance_right

            elif distance_left < min_dist:
                prob_object = obj
                min_dist = distance_left

            # print('obj: ', self.map_raw_to_label[str(prob_object.raw_label)])
            # print('dist r: ', distance_right)
            # print('dist l: ', distance_left)
            # print()
        if min_dist < threshold:
            self.holding = True
            return prob_object

        self.holding = False  # not changing

        if self.prevHolding:  # if letting go return closest object
            return prob_object
        return None

    def get_avg_dists(self, threshold=0.28):
        prob_object = None
        min_dist = float('inf')

        avg_dist = {}

        # first find all unique labels
        for frame in self.fifteen_dist:
            for obj, distR, distL in frame:
                if obj not in avg_dist:
                    avg_dist[obj] = [distR, distL, 1]
                else:
                    avg_dist[obj][0] += distR
                    avg_dist[obj][1] += distL
                    avg_dist[obj][2] += 1

        avg_dist = [[obj, avg_dist[obj][0] / avg_dist[obj][2], avg_dist[obj][1] / avg_dist[obj][2]] for obj in avg_dist]

        for label, distance_right, distance_left in avg_dist:
            if distance_right < min_dist:
                prob_object = label
                min_dist = distance_right

            elif distance_left < min_dist:
                prob_object = label
                min_dist = distance_left

        if min_dist < threshold:
            self.holding = True
            return prob_object

        self.holding = False  # not changing

        if self.prevHolding:  # if letting go return closest object
            return prob_object
        return None

    def flicker_method(self, holding):
        self.flicker_list.append(holding)
        if len(self.flicker_list) > 15:
            self.flicker_list = self.flicker_list[1:]
        self.delayed_holding = mode(self.flicker_list)

    def getLabelsKNN(self, object_list, neighs=3, rad = 0.1):
        if len(object_list) == 0: return []

        labels = []
        if len(self.five_maps) == 0:
            labels = list(range(len(object_list)))
        else:
            crdLst = []
            labelLst = []
            for frmObjs in self.five_maps:
                for obj in frmObjs:
                    labelLst.append(obj[0])
                    crdLst.append(obj[1])

            if (len(self.five_maps) < 3):
                neighs = 1
            nbrs = NearestNeighbors(n_neighbors=neighs, algorithm='ball_tree', radius=rad).fit(crdLst)

            for obj in object_list:
                dist, nbrs_idx = nbrs.radius_neighbors([obj.position], return_distance=True)
                dist, nbrs_idx = zip(*sorted(zip(dist, nbrs_idx)))  # sort in ascending order by distance
                nbrs_idx = nbrs_idx[0][:neighs]  # get closest ones

                obj_lbls = []
                for idx in nbrs_idx:
                    obj_lbls.append(labelLst[idx])
                if len(obj_lbls) >0:
                    labels.append(mode(obj_lbls))
                else:
                    labels.append(len(object_list)-1)  # assumption of new objects being at the end of the object list
        return labels

    def updateKNN(self, knnLabels, object_list):
        temp_list = []
        for i in range(len(knnLabels)):
            temp_list.append([knnLabels[i], object_list[i].position])

        if len(temp_list) != 0:
            self.five_maps.append(temp_list)

        if len(self.five_maps) > 5:
            self.five_maps = self.five_maps[1:]

        return temp_list

    def calcAllDistances(self, single_frame_map, skeleton):
        output = []

        for obj in single_frame_map:
            distance_right, distance_left = self.baseline.compute_distance(obj[1], skeleton['keypoint'])  # 'keypoint'
            output.append([obj[0], distance_right, distance_left])

        return output

    def updateFifteenDst(self, frm_dists):
        if len(frm_dists) != 0:
            self.fifteen_dist.append(frm_dists)
        if len(self.fifteen_dist) > 15:
            self.fifteen_dist = self.fifteen_dist[1:]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='Models/best.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=1242, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold')
    opt = parser.parse_args()

    # graph = [("Cup", (0), ("-1", "-1", "Crate0", "-1")),
    #          ("Crate", (0), ("Cup0", "-1", "Cup1", "-1")),
    #          ("Cup", (1), ("Crate0", "-1", "-1", "-1"))]
    graph = [("Cup", (0), ("-1", "Feeder0", "-1", "-1")),
             ("Feeder", (0), ("-1", "-1", "-1", "Cup0")),
             ("Cup", (1), ("Feeder0", "-1", "-1", "-1"))]
    config = configuration.Configuration()
    config.initGraph(graph)
    config.assign_probs()

    baseline_model = baseline_model.BaselineModel(config, config.get_graph(), graph)

    # model1 = YOLO('/home/kamil/PycharmProjects/Project3-1_WORKING_ZED/ZED_body_tracking_group_10/Models/best.pt')

    # multi_model_detector = MultiModelDetector([model1], baseline_model, objects)
    # multi_model_detector.run(video_path, video_path_out)

    zed = ZedObjectDetection(config, baseline_model)

    with torch.no_grad():
        zed.main()
