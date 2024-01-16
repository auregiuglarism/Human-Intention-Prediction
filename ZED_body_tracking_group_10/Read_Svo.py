import sys
import pyzed.sl as sl

zed = sl.Camera()

# Set SVO path for playback
input_path = sys.argv[1]
init_parameters = sl.InitParameters()
init_parameters.set_from_svo_file(input_path)

# Open the ZED
zed = sl.Camera()
err = zed.open(init_parameters)

svo_image = sl.Mat()
while not exit_app:
  if zed.grab() == sl.ERROR_CODE.SUCCESS:
    # Read side by side frames stored in the SVO
    zed.retrieve_image(svo_image, sl.VIEW.SIDE_BY_SIDE)
    # Get frame count
    svo_position = zed.get_svo_position();
  elif zed.grab() == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
    print("SVO end has been reached. Looping back to first frame")
    zed.set_svo_position(0)