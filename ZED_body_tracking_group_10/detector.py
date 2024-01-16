#!/usr/bin/env python3

import os
import numpy as np

import argparse
import torch
import cv2
import pyzed.sl as sl
from ultralytics import YOLO

from threading import Lock, Thread
from time import sleep

import cv_viewer_detector.tracking_viewer as cv_viewer
import configuration
import baseline_model
from ZED_body_tracking_group_10.Record_Skeleton_Data import serializeBodies

lock = Lock()
run_signal = False
exit_signal = False


class ZedObjectDetection:
    def __init__(self, config, baseline):
        self.config = config
        self.baseline = baseline
        self.node_name = "root"
        self.frame_size_set = False
        self.map_raw_to_label = {"0": "Crate", "1": "Feeder","2":"Cup" }

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
        zed.enable_object_detection(obj_param)

        body_tracking_parameters = sl.BodyTrackingParameters()
        body_tracking_parameters.instance_module_id = 1
        body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_34
        body_tracking_parameters.enable_body_fitting = True
        body_tracking_parameters.enable_tracking = True

        error_code = zed.enable_body_tracking(body_tracking_parameters)
        if (error_code != sl.ERROR_CODE.SUCCESS):
            print("Can't enable positionnal tracking: ", error_code)

        bodies = sl.Bodies()
        body_runtime_param = sl.BodyTrackingRuntimeParameters()
        objects = sl.Objects()
        obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        obj_runtime_param.detection_confidence_threshold = 50



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

        skeleton_data = []
        while not exit_signal:
            if zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
                # -- Get skeleton and serialize it
                zed.retrieve_bodies(bodies, body_runtime_param,body_tracking_parameters.instance_module_id)
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
                print("Length of object list ", len(objects.object_list))
                if len(skeleton_data[-1]['body_list']) > 0 and len(objects.object_list)>0:
                    object = self.get_moving_object(objects.object_list, skeleton_data[-1]['body_list'][-1])
                    print(object)
                    quadrant = self.baseline.compute_quadrant(object.bounding_box_2d)

                    name = self.map_raw_to_label[str(object.raw_label)] + str(
                        quadrant) + "_" + self.node_name

                    pred = self.baseline.yolo_predict(name)

                    print("Prediction ", pred)
                    print()

                # for obj in objects.object_list:
                #     print("ID: {} \nPos: {} \n3D Box: {} \nConf: {} \nClass: {}".format(obj.id, obj.position,
                #                                                                         obj.bounding_box,
                #                                                                         obj.confidence, obj.raw_label))

                # 2D rendering
                np.copyto(image_left_ocv, image_left.get_data())
                cv_viewer.render_2D(image_left_ocv, image_scale, objects, obj_param.enable_tracking)
                global_image = cv2.hconcat([image_left_ocv, image_track_ocv])
                # Tracking view
                track_view_generator.generate_view(objects, cam_w_pose, image_track_ocv, objects.is_tracked)

                cv2.imshow("ZED | 2D View and Birds View", global_image)
                key = cv2.waitKey(1)
                if key == 27:
                    exit_signal = True
            else:
                exit_signal = True

        exit_signal = True
        zed.close()

    def get_moving_object(self, object_list, skeleton):
        prob_object = object_list[0]
        min_dist = float('inf')
        for obj in object_list:
            distance = self.baseline.compute_distance(obj.bounding_box, skeleton['keypoint'])
            if distance < min_dist:
                prob_object = obj
                min_dist = distance
        return prob_object


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='Models/best.pt', help='model.pt path(s)')
    parser.add_argument('--svo', type=str, default=None, help='optional svo file')
    parser.add_argument('--img_size', type=int, default=1242, help='inference size (pixels)')
    parser.add_argument('--conf_thres', type=float, default=0.5, help='object confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.5, help='iou threshold')
    opt = parser.parse_args()

    graph = [("Cup", "N"),
             ("Crate", "NW"),
             ("Feeder", "S")]
    config = configuration.Configuration()
    config.initGraph(graph)
    config.assign_probs()

    baseline_model = baseline_model.BaselineModel(config.get_graph(), graph)

    # model1 = YOLO('/home/kamil/PycharmProjects/Project3-1_WORKING_ZED/ZED_body_tracking_group_10/Models/best.pt')

    # multi_model_detector = MultiModelDetector([model1], baseline_model, objects)
    # multi_model_detector.run(video_path, video_path_out)

    zed = ZedObjectDetection(config, baseline_model)

    with torch.no_grad():
        zed.main()
