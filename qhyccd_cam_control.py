import cv2
import numpy as np
import ctypes
from ctypes import *
from enum import Enum

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

qhyccddll = cdll.LoadLibrary('.\\qhyccd.dll')

# get camera id
qhyccddll.GetQHYCCDId.argtypes = [ctypes.c_uint32, ctypes.c_char_p]
# get handle via camera id
qhyccddll.OpenQHYCCD.argtypes = [ctypes.c_char_p]
qhyccddll.OpenQHYCCD.restype = ctypes.c_void_p
# close camera
qhyccddll.CloseQHYCCD.argtypes = [ctypes.c_void_p]

# read mode
qhyccddll.GetQHYCCDNumberOfReadModes.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint32)]
qhyccddll.GetQHYCCDReadModeName.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_char_p]
qhyccddll.GetQHYCCDReadModeResolution.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.POINTER(ctypes.c_uint32),
                                                 ctypes.POINTER(ctypes.c_uint32)]
qhyccddll.SetQHYCCDReadMode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

# set single stream mode or live stream mode
qhyccddll.SetQHYCCDStreamMode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]

# initialize camera
qhyccddll.InitQHYCCD.argtypes = [ctypes.c_void_p]

# get camera chip information
qhyccddll.GetQHYCCDChipInfo.argtypes = [ctypes.c_void_p,
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                       ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
                                       ctypes.POINTER(ctypes.c_uint32)]
# get parameters value
qhyccddll.GetQHYCCDParam.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
qhyccddll.GetQHYCCDParam.restype = ctypes.c_double

# set parameters
qhyccddll.SetQHYCCDParam.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_double]
# set debayer on or off, only for color camera
qhyccddll.SetQHYCCDDebayerOnOff.argtypes = [ctypes.c_void_p, ctypes.c_bool]
# set bin mode
qhyccddll.SetQHYCCDBinMode.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
# set resolution and ROI
qhyccddll.SetQHYCCDResolution.argtypes = [ctypes.c_void_p, ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint32,
                                         ctypes.c_uint32]

# start single stream mode exposing
qhyccddll.ExpQHYCCDSingleFrame.argtypes = [ctypes.c_void_p]
# get single frame data
qhyccddll.GetQHYCCDSingleFrame.argtypes = [ctypes.c_void_p,
                                          ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                          ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                          ctypes.POINTER(ctypes.c_uint8)]
# cancel single exposing and camera will NOT output frame data
qhyccddll.CancelQHYCCDExposingAndReadout.argtypes = [ctypes.c_void_p]

# start live stream mode
qhyccddll.BeginQHYCCDLive.argtypes = [ctypes.c_void_p]
# get live frame data
qhyccddll.GetQHYCCDLiveFrame.argtypes = [ctypes.c_void_p,
                                        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                        ctypes.POINTER(ctypes.c_uint32), ctypes.POINTER(ctypes.c_uint32),
                                        ctypes.POINTER(ctypes.c_uint8)]
# stop live stream mode
qhyccddll.StopQHYCCDLive.argtypes = [ctypes.c_void_p]

# convert image data
qhyccddll.Bits16ToBits8.argtypes = [ctypes.c_void_p,
                                      ctypes.POINTER(ctypes.c_uint8), ctypes.POINTER(ctypes.c_uint8),
                                      ctypes.c_uint32, ctypes.c_uint32, ctypes.c_uint16, ctypes.c_uint16]

class CONTROL_ID(Enum):
    CONTROL_BRIGHTNESS = 0
    CONTROL_CONTRAST = 1
    CONTROL_WBR = 2
    CONTROL_WBB = 3
    CONTROL_WBG = 4
    CONTROL_GAMMA = 5
    CONTROL_GAIN = 6
    CONTROL_OFFSET = 7
    CONTROL_EXPOSURE = 8
    CONTROL_SPEED = 9
    CONTROL_TRANSFERBIT = 10
    CONTROL_CHANNELS = 11
    CONTROL_USBTRAFFIC = 12
    CONTROL_CURTEMP = 14
    CONTROL_CURPWM = 15
    CONTROL_MANULPWM = 16
    CONTROL_CFWPORT = 17
    CONTROL_COOLER = 18
    CONTROL_ST4PORT = 19
    CAM_COLOR = 20
    CAM_BIN1X1MODE = 21
    CAM_BIN2X2MODE = 22
    CAM_BIN3X3MODE = 23
    CAM_BIN4X4MODE = 24
    CAM_8BITS = 34
    CAM_16BITS = 35
    CAM_GPS = 36
    CONTROL_AMPV = 41
    CONTROL_CFWSLOTSNUM = 44
    CAM_SINGLEFRAMEMODE = 57
    CAM_LIVEVIDEOMODE = 58
    CAM_IS_COLOR = 59


camhandle = 0

ret = qhyccddll.InitQHYCCDResource()

num = qhyccddll.ScanQHYCCD()


for index in range(num):

    id_buffer = ctypes.create_string_buffer(40)
    ret = qhyccddll.GetQHYCCDId(index, id_buffer)
    result_id = id_buffer.value.decode("utf-8")

    camhandle = qhyccddll.OpenQHYCCD(id_buffer)
    if camhandle != 0:
        break

readmodenum = ctypes.c_uint32()
ret = qhyccddll.GetQHYCCDNumberOfReadModes(camhandle, byref(readmodenum))

for index in range(readmodenum.value):

    name_buffer = ctypes.create_string_buffer(40)
    ret = qhyccddll.GetQHYCCDReadModeName(camhandle, index, name_buffer)
    result_name = name_buffer.value.decode("utf-8")

    width = ctypes.c_uint32()
    height = ctypes.c_uint32()
    ret = qhyccddll.GetQHYCCDReadModeResolution(camhandle, index, byref(width), byref(height))

ret = qhyccddll.SetQHYCCDReadMode(camhandle, 0)

ret = qhyccddll.SetQHYCCDStreamMode(camhandle, 0)
#print("SetQHYCCDStreamMode() ret =", ret)

ret = qhyccddll.InitQHYCCD(camhandle)

ret = qhyccddll.SetQHYCCDParam(camhandle, CONTROL_ID.CONTROL_TRANSFERBIT.value, 16.0)

ret = qhyccddll.SetQHYCCDDebayerOnOff(camhandle, False)

chipW = ctypes.c_double()
chipH = ctypes.c_double()
imageW = ctypes.c_uint32()
imageH = ctypes.c_uint32()
pixelW = ctypes.c_double()
pixelH = ctypes.c_double()
imageB = ctypes.c_uint32()

ret = qhyccddll.GetQHYCCDChipInfo(camhandle, byref(chipW), byref(chipH), byref(imageW), byref(imageH), byref(pixelW),
                                  byref(pixelH), byref(imageB))

bin = 4

ret = qhyccddll.SetQHYCCDBinMode(camhandle, bin, bin)
#print("SetQHYCCDBinMode() ret =", ret)
#print(imageW.value, imageH.value)
manual_size = True
if manual_size:
    width = 5000 // bin
    height = 5000 // bin
    ret = qhyccddll.SetQHYCCDResolution(camhandle, 2200 // bin, 800 // bin, width, height)

else:
    ret = qhyccddll.SetQHYCCDResolution(camhandle, 0, 0, imageW.value // bin, imageH.value // bin)

#print("SetQHYCCDResolution() ret =", ret)

ret = qhyccddll.SetQHYCCDParam(camhandle, CONTROL_ID.CONTROL_EXPOSURE.value, 4000.0)

ret = qhyccddll.SetQHYCCDParam(camhandle, CONTROL_ID.CONTROL_GAIN.value, 50.0)

ret = qhyccddll.SetQHYCCDParam(camhandle, CONTROL_ID.CONTROL_OFFSET.value, 80.0)

ret = qhyccddll.SetQHYCCDParam(camhandle, CONTROL_ID.CONTROL_USBTRAFFIC.value, 0.0)

w = ctypes.c_uint32()
h = ctypes.c_uint32()
b = ctypes.c_uint32()
c = ctypes.c_uint32()

#print(imageW.value, imageH.value)
if manual_size:
    length = width * height * 2

else:
    length = imageW.value * imageH.value // bin


imgdata = (ctypes.c_uint8 * length)()
imgdata_raw8 = (ctypes.c_uint8 * length)()
#print(imgdata_raw8, imgdata)
import time

"""count = 0
while count < 100:
    ret = qhyccddll.ExpQHYCCDSingleFrame(camhandle)
    ret = qhyccddll.GetQHYCCDSingleFrame(camhandle, byref(w), byref(h), byref(b), byref(c), imgdata)
    ret = qhyccddll.Bits16ToBits8(camhandle, imgdata, imgdata_raw8, w.value, h.value, 0, 65535)
    img = np.frombuffer(imgdata_raw8, dtype=np.uint8).reshape((h.value, w.value))
    cv2.namedWindow("Show", 0)
    cv2.resizeWindow("Show", w.value//10, h.value//10)  # Set the window size to 800x600
    cv2.imshow("Show", img)
    cv2.waitKey(1)
    count += 1
    print(count)

    # Record the end time
    end_time = time.time()

    # Calculate the time difference
    time_diff = end_time - start_time

    # Calculate FPS
    fps = count / time_diff
    print(f"FPS: {fps:.2f}")"""


def measure_ongoing_radius(measure:bool=False, stop_signal:bool=False):
    """
    Function to continually measure the radius of the output light cone for tiptilt adjustment.
    Args:
        measure: Whether to measure the radius.
        stop_signal: Signal from Gui to stop the measurement.

    Returns:

    """
    count = 0
    start_time = time.time()
    while count < 1000 or not stop_signal:
        qhyccddll.ExpQHYCCDSingleFrame(camhandle)
        ret = qhyccddll.GetQHYCCDSingleFrame(camhandle, byref(w), byref(h), byref(b), byref(c), imgdata)     # This takes long if not live, longer if live...
        #print("GetQHYCCDSingleFrame() ret =", ret, "w =", w.value, "h =", h.value, "b =", b.value, "c =", c.value, "count =", count,)
        if ret != 0:
            print("Failed to capture image.")
            continue
        #qhyccddll.Bits16ToBits8(camhandle, imgdata, imgdata_raw8, w.value, h.value, 0, 65535)
        img = np.frombuffer(imgdata, dtype=np.uint16).reshape((h.value, w.value))

        img = img - np.median(img)
        # Set negative values to 0
        img[img < 0] = 0

        # Print max pixel value
        print("Max pixel value", np.max(img))

        show = True
        if show:
            show_img = img / np.max(img) * 255
            cv2.namedWindow("Show", 0)
            cv2.resizeWindow("Show", w.value, h.value)  # Set the window size to 800x600
            cv2.imshow("Show", show_img.astype(np.uint8))
            cv2.waitKey(1)
        count += 1
        print(count)

        if measure:
            if np.max(img) < 100:
                continue

            save = False
            if save:
                # Save image as fits
                import os
                import astropy.io.fits as fits
                fits.writeto("D:/Vincent/tip_tilt/image.fits", img, overwrite=True)

            measurements_folder = "D:/Vincent/tip_tilt/"
            import tip_tilt_adjustment as tta
            tta.analyse_f_number(img, measurements_folder)

            # Run check_tiptilt_image.bat
            import os
            os.system("D:/stepper_motor/check_tiptilt_image.bat")


        # Record the end time
        end_time = time.time()

        # Calculate the time difference
        time_diff = end_time - start_time

        # Calculate FPS
        fps = count / time_diff
        print(f"FPS: {fps:.2f}")

def measure_eccentricity(measure:bool=True, stop_signal:bool=False):
    """
    Function to continually measure the eccentricity of the output light cone.
    Args:
        measure: Whether to measure the eccentricity.
        stop_signal: Signal from Gui to stop the measurement.

    Returns:

    """
    count = 0
    start_time = time.time()
    while count < 1000 or not stop_signal:
        qhyccddll.ExpQHYCCDSingleFrame(camhandle)
        ret = qhyccddll.GetQHYCCDSingleFrame(camhandle, byref(w), byref(h), byref(b), byref(c),
                                             imgdata)  # This takes long if not live, longer if live...
        print("GetQHYCCDSingleFrame() ret =", ret, "w =", w.value, "h =", h.value, "b =", b.value, "c =", c.value, "count =", count,)
        if ret != 0:
            print("Failed to capture image.")
            continue
        # qhyccddll.Bits16ToBits8(camhandle, imgdata, imgdata_raw8, w.value, h.value, 0, 65535)
        print(imgdata)
        img = np.frombuffer(imgdata, dtype=np.uint16).reshape((h.value, w.value))

        #img = img - np.median(img)
        # Set negative values to 0
        #img[img < 0] = 0

        # Print max pixel value
        print("Max pixel value", np.max(img))

        show = True
        if show:
            show_img = img / np.max(img) * 255
            cv2.namedWindow("Show", 0)
            cv2.resizeWindow("Show", w.value // 2, h.value // 2)  # Set the window size to 800x600
            cv2.imshow("Show", show_img.astype(np.uint8))
            cv2.waitKey(1)
        count += 1
        print(count)

        if measure:
            if np.max(img) < 100:
                continue

            save = True
            if save:
                # Save image as fits
                import os
                import astropy.io.fits as fits
                fits.writeto("D:/Vincent/eccentricity/image.fits", img, overwrite=True)

            #measurements_folder = "D:/Vincent/eccentricity/"
            import image_analysation as ia
            ecc = ia.measure_eccentricity(img, plot=False)

            print(f"Eccentricity: {ecc}")

        # Record the end time
        end_time = time.time()

        # Calculate the time difference
        time_diff = end_time - start_time

        # Calculate FPS
        fps = count / time_diff
        print(f"FPS: {fps:.2f}")

def use_camera(mode:str=None, stop_signal=False):
    """
    Function to use the camera in either tiptilt or eccentricity mode.
    Args:
        mode: Mode to run the camera in. Must be either 'tiptilt' or 'eccentricity'.
        stop_signal: Signal to stop the measurement.

    Returns:

    """
    if mode == "tiptilt":
        measure_ongoing_radius(measure=True, stop_signal=stop_signal)
    if mode == "eccentricity":
        measure_eccentricity(measure=True, stop_signal=stop_signal)
    else:
        raise ValueError("Invalid mode. Must be either 'tiptilt' or 'eccentricity'.")

    cv2.destroyAllWindows()
    ret = qhyccddll.CloseQHYCCD(camhandle)
    ret = qhyccddll.ReleaseQHYCCDResource()

if __name__ == "__main__":
    use_camera(mode="eccentricity")