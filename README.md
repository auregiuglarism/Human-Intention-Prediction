# Project3-1

## How to run baseline model prototype:
- Got to baselines/probability/baseline.py
- Run the file
- The file will display the camera and output recognised probabilties and objects in a terminal 


## Requirements:
- python3
- packages:
    - ultralytics 
    - opencv


## ZED guidelines
- Live_Feed.py:
    - Live output feed of the people and skeleton tracking on top
- ZED_body_tracking_group_10/ogl_viewer/viewer.py:
    - Only the skeleton tracking shown
- ZED_body_tracking_group_10/Conver_2d_Recording .py:
    - Takes in recorded file. Saves it as jpg sequence (can use different types, but we decided to use sequence)
- ZED_body_tracking_group_10/Record_Yolo.py:
    - creates svo file 
- ZED_body_tracking_group_10/Read_Svo.py:
    - reads the svo file (test purposes to see if it records correctly)
- ZED_body_tracking_group_10/Record_Skeleton_Data.py:
    - opens live camera and creates json file of skeleton coordinates
- ZED_body_tracking_group_10/detector:
    - Combines skeleton tracking with object tracking. This data is fed into the baseline model
- ZED_body_tracking_group_10/baseline_model.py:
    - model which assigns equiprobable weights to actions and is used to predict the next possible action done by a human

#### Utility files and directories:
- ZED_body_tracking_group_10/ogl_viewer_skeleton/viewer.py:
- ZED_body_tracking_group_10/cv_viewer

### Data (angle/ distance/ lighting)
- data1 - angle + far + medium 
- data3 - angle + far + mediumHigh
- data4 - angle + close + medium
- data5 - angle + close + mediumHigh
- data6 - top + close + mediumHigh
- data7 - top + close + medium

