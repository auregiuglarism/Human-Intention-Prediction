from time import sleep

import cv2
from Zed_Data_Extraction.record_skeleton_data import serializeBodies
import main_utility.tracking_viewer as cv_viewer
import numpy as np


class ZedCamera:
    def __init__(self, sl):
        # ZED INIT
        self.track_view_generator = None
        self.display_resolution = None
        self.point_cloud_res = None
        self.obj_param = None
        self.image_left_ocv = None
        self.image_scale = None
        self.image_left = None
        self.point_cloud = None
        self.obj_runtime_param = None
        self.objects = None
        self.image_left_tmp = None
        self.image_track_ocv = None
        self.cam_w_pose = None
        self.body_tracking_parameters = None
        self.body_runtime_param = None
        self.bodies = None
        self.runtime_params = None
        self.zed = None
        self.frame_size_set = False

        self.initZed(sl)

    def initZed(self, sl):
        self.zed = sl.Camera()

        # Create a InitParameters object and set configuration parameters
        init_params = sl.InitParameters()
        init_params.coordinate_units = sl.UNIT.METER
        init_params.camera_resolution = sl.RESOLUTION.HD2K  # HD720 for extra wide FOV
        init_params.camera_fps = 30  # 15 fps for better depth quality under low light
        init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # depth quality
        init_params.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Y_UP
        init_params.depth_maximum_distance = 10  # distance in coordinate_units (should be 10 meters)
        init_params.set_from_svo_file("/home/kamil/PycharmProjects/Project3-1_WORKING_ZED/ZED_body_tracking_group_10/assembly_Dom_fecu1cu0cr.svo")

        self.runtime_params = sl.RuntimeParameters()
        status = self.zed.open(init_params)

        self.image_left_tmp = sl.Mat()

        positional_tracking_parameters = sl.PositionalTrackingParameters()
        self.zed.enable_positional_tracking(positional_tracking_parameters)

        # Object detection
        obj_param = sl.ObjectDetectionParameters()
        obj_param.instance_module_id = 0
        obj_param.detection_model = sl.OBJECT_DETECTION_MODEL.CUSTOM_BOX_OBJECTS
        obj_param.enable_tracking = True
        obj_param.enable_segmentation = True
        self.zed.enable_object_detection(obj_param)
        self.obj_param = obj_param

        # Body tracking
        body_tracking_parameters = sl.BodyTrackingParameters()
        body_tracking_parameters.instance_module_id = 1
        body_tracking_parameters.detection_model = sl.BODY_TRACKING_MODEL.HUMAN_BODY_ACCURATE
        body_tracking_parameters.body_format = sl.BODY_FORMAT.BODY_18
        body_tracking_parameters.enable_body_fitting = True
        body_tracking_parameters.enable_tracking = True
        self.body_tracking_parameters = body_tracking_parameters
        self.zed.enable_body_tracking(body_tracking_parameters)

        # Setting runtime variables
        self.bodies = sl.Bodies()
        self.body_runtime_param = sl.BodyTrackingRuntimeParameters()
        self.objects = sl.Objects()
        self.obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
        self.obj_runtime_param.detection_confidence_threshold = 30

        # Display
        camera_infos = self.zed.get_camera_information()
        camera_res = camera_infos.camera_configuration.resolution

        point_cloud_res = sl.Resolution(min(camera_res.width, 720), min(camera_res.height, 404))
        # point_cloud_res = sl.Resolution(camera_res.width, camera_res.height)
        self.point_cloud = sl.Mat(point_cloud_res.width, point_cloud_res.height, sl.MAT_TYPE.F32_C4, sl.MEM.CPU)

        self.point_cloud_res = point_cloud_res

        self.image_left = sl.Mat()

        # Utilities for 2D display
        display_resolution = sl.Resolution(min(camera_res.width, 1280), min(camera_res.height, 720))
        self.display_resolution = display_resolution
        # display_resolution = sl.Resolution(camera_res.width, camera_res.height)
        self.image_scale = [display_resolution.width / camera_res.width, display_resolution.height / camera_res.height]
        self.image_left_ocv = np.full((display_resolution.height, display_resolution.width, 4), [245, 239, 239, 255],
                                      np.uint8)

        # Utilities for tracks view
        camera_config = camera_infos.camera_configuration
        tracks_resolution = sl.Resolution(400, display_resolution.height)
        track_view_generator = cv_viewer.TrackingViewer(tracks_resolution, camera_config.fps,
                                                        init_params.depth_maximum_distance)
        track_view_generator.set_camera_calibration(camera_config.calibration_parameters)
        self.track_view_generator = track_view_generator
        self.image_track_ocv = np.zeros((tracks_resolution.height, tracks_resolution.width, 4), np.uint8)
        # Camera pose
        self.cam_w_pose = sl.Pose()

        if status != sl.ERROR_CODE.SUCCESS:
            print(repr(status))
            exit()

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
    def detections_to_custom_box(self, detections, im0, names, sl):
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

    def get_error_code(self):
        return self.zed.grab(self.runtime_params)

    def get_skeleton_data(self):
        self.zed.retrieve_bodies(self.bodies, self.body_runtime_param,
                                 self.body_tracking_parameters.instance_module_id)
        return serializeBodies(self.bodies)

    def get_object_data(self):
        self.zed.retrieve_objects(self.objects, self.obj_runtime_param, self.obj_param.instance_module_id)
        return self.objects

    def close_camera(self):
        self.zed.close()

    def retrieve_display_data(self, sl):
        self.zed.retrieve_measure(self.point_cloud, sl.MEASURE.XYZRGBA, sl.MEM.CPU,
                                  self.point_cloud_res)  # update point cloud
        self.zed.retrieve_image(self.image_left, sl.VIEW.LEFT, sl.MEM.CPU, self.display_resolution)
        self.zed.get_position(self.cam_w_pose, sl.REFERENCE_FRAME.WORLD)

    def display_image(self):
        np.copyto(self.image_left_ocv, self.image_left.get_data())
        cv_viewer.render_2D(self.image_left_ocv, self.image_scale, self.objects, self.obj_param.enable_tracking)
        global_image = cv2.hconcat([self.image_left_ocv, self.image_track_ocv])
        # Tracking view
        self.track_view_generator.generate_view(self.objects, self.cam_w_pose, self.image_track_ocv,
                                                self.objects.is_tracked)

        sleep(0.001)
        cv2.imshow("ZED | 2D View and Birds View", global_image)
