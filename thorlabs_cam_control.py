import instrumental
import cv2
import time
#from instrumental.drivers.cameras import uc480


def take_image(cam_name:str, save_file_name:str, wait:bool=False, exposure_time=None):
    cam = instrumental.instrument(cam_name, reopen_policy="new")

    cam.open()
    if exposure_time:
        frame = cam.grab_image(exposure_time=exposure_time)
    else:
        if cam_name == "entrance_cam":
            frame = cam.grab_image(exposure_time="1s")
        elif cam_name == "exit_cam":
            frame = cam.grab_image(exposure_time="10ms")
        else:
            raise ValueError("Invalid camera name. Please provide a valid camera name.")

    if frame is not None:
        image = frame
        if wait:
            cv2.imshow("Captured Image", image)
            cv2.waitKey(0)
        else:
            cv2.imshow("Captured Image", image)
            cv2.waitKey(1)
            time.sleep(2)
        cv2.imwrite(save_file_name, image)
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
    take_image("entrance_cam", "image_test.png", wait=True, exposure_time="1ms")
    #take_image("entrance_cam", "entrance_image_test.png", wait=True)