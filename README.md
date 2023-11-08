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
- test.py:
    - Live output feed of the people and skeleton tracking on top
- ZED_body_tracking_group_10/ogl_viewer/viewer.py:
    - Only the skeleton tracking shown
- ZED_body_tracking_group_10/svo_export.py:
    - Takes in recorded file. Saves it as jpg sequence (can use different types, but we decided to use sequence)
- ZED_body_tracking_group_10/recording_3d.py:
    - creates svo file 
- ZED_body_tracking_group_10/reading_recording.py:
    - reads the svo file (test purposes to see if it records correctly)
- ZED_body_tracking_group_10/convert_skeleton.py:
    - opens live camera and creates json file of skeleton coordinates

#### Utility files and directories:
- ZED_body_tracking_group_10/ogl_viewer_skeleton/viewer.py:
- ZED_body_tracking_group_10/cv_viewer