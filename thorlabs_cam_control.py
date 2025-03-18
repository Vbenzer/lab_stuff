import instrumental
import cv2
import time
from astropy.io import fits
#from instrumental.drivers.cameras import uc480


def take_image(cam_name:str, save_file_name:str, wait:bool=False, exposure_time=None, info:bool=False, save_fits:bool=False):
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

        if save_fits:
            # Save as fits file
            hdul = fits.HDUList([fits.PrimaryHDU(image)])
            hdul.writeto(save_file_name, overwrite=True)
        else:
            cv2.imwrite(save_file_name, image)
        if info:
            # Get max, min and mean pixel values
            max_pixel = image.max()
            min_pixel = image.min()
            mean_pixel = image.mean()
            print(f"Max pixel value: {max_pixel}")
            print(f"Min pixel value: {min_pixel}")
            print(f"Mean pixel value: {mean_pixel}")
    else:
        print("Failed to capture image.")
    cam.close()
    cv2.destroyAllWindows()

def open_thorcam():
    """
    Opens the ThorCam software.
    Returns:

    """
    from analyse_main import run_batch_file

    run_batch_file("D:\stepper_motor\open_thorcam.bat") # Doesnt exist yet


if __name__ == "__main__":
    take_image("entrance_cam", "image_test.png", wait=True, exposure_time="1ms")
    #take_image("entrance_cam", "entrance_image_test.png", wait=True)