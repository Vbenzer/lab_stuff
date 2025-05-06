import cv2
import time
from astropy.io import fits
import pylablib.devices.uc480 as uc480
import numpy as np
from core.hardware.cameras.qhyccd_control import convert_to_us
import time

def test():
    print(uc480.list_cameras(backend="uc480"))  # List all cameras
    cam = uc480.UC480Camera(cam_id=3) # 1 is entrance cam
    cam.set_pixel_rate(5000000)
    cam.set_frame_period(1)
    cam.set_exposure(10)
    print(cam.get_settings())
    exit()
    cam.open()
    cam.set_exposure(2000)  # Set exposure time to 10 seconds
    frame = cam.snap()
    cv2.imshow("Captured Image", frame)
    cv2.waitKey(0)
    cam.close()
    cv2.destroyAllWindows()

def take_image(cam_name: str, save_file_name: str, show: bool = False, exposure_time=None, info: bool = False,
               save_fits: bool = False, progress_signal=None):
    """
    Takes an image using the specified camera and saves it to the specified file.
    Args:
        cam_name: Name of the camera to use ("entrance_cam" or "exit_cam").
        save_file_name: Path to save the captured image.
        show: Whether to wait for a key press before closing the image window.
        exposure_time: Exposure time in seconds. If None, a default value is used based on the camera name.
        info: Whether to print image statistics (max, min, mean pixel values).
        save_fits: Whether to save the image as a FITS file.
        progress_signal: Signal to emit progress updates (if using PyQt/PySide).

    Returns: None

    """
    # Initialize camera
    cam_dict = {"entrance_cam": 1, "exit_cam": 3}  # Map camera names to IDs
    cam = uc480.UC480Camera(cam_id=cam_dict[cam_name])  # Set camera ID based on name

    # Set settings
    cam.set_pixel_rate(35000000)  # Set pixel rate to 5 MHz (min)
    cam.set_frame_period(1)  # Set frame period to 1 s (max), these let the cam get the highest limit to exposure time
    cam.set_gains(0,0,0,0)
    cam.set_gain_boost(0)

    cam.open()
    if exposure_time:
        # Convert exposure time to seconds if it's a string
        if isinstance(exposure_time, str):
            exposure_time = convert_to_us(exposure_time) * 1E-6  # Convert to seconds
        cam.set_exposure(exposure_time)
    else:
        if cam_name == "entrance_cam":
            cam.set_exposure(10E-3)  # 10ms
        elif cam_name == "exit_cam":
            cam.set_exposure(4E-3)  # 4ms
        else:
            raise ValueError("Invalid camera name. Please provide a valid camera name.")

    # Print exposure time
    print(f"Exposure time set to {cam.get_exposure()} seconds.")

    # Grab image
    cam.clear_acquisition()
    cam.setup_acquisition(1)  # Set number of images to 1
    print(f"Exposure time set to {cam.get_exposure()} seconds.")
    print(cam.is_acquisition_setup())
    cam.start_acquisition()
    cam.wait_for_frame()
    print(cam.get_frames_status())
    print(f"Exposure time set to {cam.get_exposure()} seconds.")
    frame = cam.read_newest_image(return_info=True)
    print(f"Exposure time set to {cam.get_exposure()} seconds.")
    cam.stop_acquisition()
    print(f"Exposure time set to {cam.get_exposure()} seconds.")
    print(cam.get_frames_status())
    #frame = cam.grab(1, return_info=True)
    print(frame[1])
    if frame is not None:
        image = frame[0]
        if show:
            cv2.imshow("Captured Image", image)
            cv2.waitKey(0)

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

    time.sleep(0.5)
    cam.close()
    cv2.destroyAllWindows()

def many_images_test():
    save_folder = r"D:\Vincent\test/"
    for i in range(10):
        take_image("entrance_cam", save_folder + f"entrance_image_{i}.png", show=False, info=True
                   , exposure_time="0.5s")

        take_image("exit_cam", save_folder + f"exit_image_{i}.png", show=False, info=True
                   ,exposure_time="0.5s")
        #time.sleep(1.5)

def open_thorcam():
    """
    Opens the ThorCam software.
    Returns: None

    """
    from core.file_management import run_batch_file

    run_batch_file(r"D:\stepper_motor\open_thorcam.bat")


if __name__ == "__main__":
    #take_image("entrance_cam", "image_test.png", wait=True, exposure_time="10000s")
    #take_image("entrance_cam", "entrance_image_test.png", wait=True)
    take_image("entrance_cam", "entrance_image_test.png",show=True, exposure_time="1ms", info=True)
    #test()
    #many_images_test()