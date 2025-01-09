import instrumental
import numpy as np
import cv2
from instrumental.drivers.cameras import uc480


def take_image(cam_name:str):
    cam = instrumental.instrument(cam_name)
    cam.open()
    frame = cam.grab_image(exposure_time="1s")
    if frame is not None:
        image = frame
        cv2.imshow("Captured Image", image)
        cv2.waitKey(0)
        cv2.imwrite(f'{cam_name}_image2.png', image)
    else:
        print("Failed to capture image.")
    cam.close()
    cv2.destroyAllWindows()
"""
# Open the camera
entrance_cam.open()

# Capture an image (frame will be a numpy array)
frame = entrance_cam.grab_image(exposure_time="1s")

# Check if the frame is successfully captured
if frame is not None:
    image = frame  # Extract the image from the frame (numpy array)

    # Display the captured image (optional)
    cv2.imshow("Captured Image", image)
    cv2.waitKey(0)  # Wait for a key press to close the window

    # Save the image
    cv2.imwrite('exit_image.png', image)
else:
    print("Failed to capture image.")

# Close the camera after use
entrance_cam.close()

# Close OpenCV windows
cv2.destroyAllWindows()
"""

if __name__ == "__main__":
    take_image("exit_cam")
    take_image("entrance_cam")