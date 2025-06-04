"""Module unused_cam_functions.py.

Auto-generated docstring for better readability.
"""
def take_image_old(cam_name: str, save_file_name: str, wait: bool = False, exposure_time=None, info: bool = False,
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