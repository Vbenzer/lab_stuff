import instrumental
import cv2
import time
from astropy.io import fits
import pylablib.devices.uc480 as uc480
import numpy as np
#from instrumental.drivers.cameras import uc480

def test():
    print(uc480.list_cameras(backend="uc480"))  # List all cameras
    cam = uc480.UC480Camera(cam_id=1) # 1 is entrance cam
    cam.open()
    cam.set_exposure(2000)  # Set exposure time to 10 seconds
    frame = cam.snap()
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)
    cam.close()
    cv2.destroyAllWindows()

def take_image_new(cam_name: str, save_file_name: str, wait: bool = False, exposure_time=None, info: bool = False,
               save_fits: bool = False, progress_signal=None):
    # Initialize camera
    instruments = uc480.list_cameras()
    cam = uc480.UC480Camera(instruments[1][0])  # Assuming the first camera is the desired one

    cam.open()
    if exposure_time:
        cam.set_exposure(exposure_time)
        print("Pixel rate:", cam.get_pixel_rate())
        print("Available pixel rates:", cam.get_available_pixel_rates())
        print("get pixel rate range:", cam.get_pixel_rates_range())
    else:
        if cam_name == "entrance_cam":
            cam.set_exposure(10E-3)  # 10ms
        elif cam_name == "exit_cam":
            cam.set_exposure(4E-3)  # 4ms
        else:
            raise ValueError("Invalid camera name. Please provide a valid camera name.")

    # Grab image
    frame = cam.grab(frame_timeout=5)
    if frame is not None:
        image = frame[0]
        if wait:
            cv2.imshow("Captured Image", image)
            cv2.waitKey(0)
        else:
            cv2.imshow("Captured Image", image)
            cv2.waitKey(1)
            time.sleep(2)

        if save_fits:
            # Save as FITS file
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
            if progress_signal:
                progress_signal.emit(f"Max pixel value: {max_pixel}")
                progress_signal.emit(f"Min pixel value: {min_pixel}")
                progress_signal.emit(f"Mean pixel value: {mean_pixel}")
    else:
        print("Failed to capture image.")
        if progress_signal:
            progress_signal.emit("Failed to capture image.")

    cam.close()
    cv2.destroyAllWindows()

def take_image(cam_name: str, save_file_name: str, wait: bool = False, exposure_time=None, info: bool = False,
               save_fits: bool = False, progress_signal=None):
    cam = instrumental.instrument(cam_name, reopen_policy="new")

    cam.open()
    if exposure_time:
        frame = cam.grab_image(exposure_time=exposure_time)
        print("Pixel rate:", cam.get_pixel_rate())
        print("Available pixel rates:", cam.get_available_pixel_rates())
        print("get pixel rate range:", cam.get_pixel_rates_range())
    else:
        if cam_name == "entrance_cam":
            frame = cam.grab_image(exposure_time="10ms")
        elif cam_name == "exit_cam":
            frame = cam.grab_image(exposure_time="4ms")
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
            if progress_signal:
                progress_signal.emit(f"Max pixel value: {max_pixel}")
                progress_signal.emit(f"Min pixel value: {min_pixel}")
                progress_signal.emit(f"Mean pixel value: {mean_pixel}")
    else:
        print("Failed to capture image.")
        if progress_signal:
            progress_signal.emit("Failed to capture image.")

    cam.close()
    cv2.destroyAllWindows()

def open_thorcam():
    """
    Opens the ThorCam software.
    Returns:

    """
    from core.file_management import run_batch_file

    run_batch_file(r"D:\stepper_motor\open_thorcam.bat")


if __name__ == "__main__":
    #take_image("entrance_cam", "image_test.png", wait=True, exposure_time="10000s")
    take_image("entrance_cam", "entrance_image_test.png", wait=True)
    take_image_new("entrance_cam", "entrance_image_test.png", wait=True, exposure_time=30)