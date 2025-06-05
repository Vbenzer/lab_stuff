"""Utilities for controlling a QHYCCD camera.

This module wraps the vendor provided DLL to offer a minimal Python
interface for capturing single images. Only a subset of the available
functionality is implemented as required by the project.
"""
import cv2
import numpy as np
import ctypes
from ctypes import *
from enum import Enum
import time
import os

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

import re


def convert_to_us(time_str):
    """
    Convert a time string to microseconds.
    Args:
        time_str: Time string in format "<value><unit>", where unit is one of "ms", "s", "us".
            e.g. "10ms", "1s", "100us", "0.5s"

    Returns: Time in microseconds.

    """
    match = re.match(r"(\d+(\.\d+)?)(ms|s|us)", time_str)
    if not match:
        raise ValueError("Invalid time format")

    value, _, unit = match.groups()
    value = float(value)

    if unit == "s":
        return value * 1e6
    elif unit == "ms":
        return value * 1e3
    elif unit == "us":
        return value
    else:
        raise ValueError("Unsupported time unit")

class Camera:
    """Simple wrapper around the QHYCCD camera API."""

    def __init__(self, exp_time: int) -> None:
        """Initialize the camera and configure basic settings.

        Parameters
        ----------
        exp_time : int
            Exposure time in microseconds used for initial configuration.
        """

        # Based on https://github.com/JiangXL/qhyccd-python
        self.qhyccddll = cdll.LoadLibrary('.\\qhyccd.dll')

        qhyccddll = self.qhyccddll

        # ---- Configure function prototypes ----
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
        qhyccddll.Bits16ToBits8.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.POINTER(ctypes.c_uint8),
            ctypes.c_uint32,
            ctypes.c_uint32,
            ctypes.c_uint16,
            ctypes.c_uint16,
        ]

        # Enum of supported control parameters
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

        self.CONTROL_ID = CONTROL_ID

        # ---- Establish connection with the first available camera ----
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

        self.camhandle = camhandle

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

        ret = qhyccddll.SetQHYCCDParam(camhandle, self.CONTROL_ID.CONTROL_TRANSFERBIT.value, 16.0)

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

        bin = 2

        ret = qhyccddll.SetQHYCCDBinMode(camhandle, bin, bin)
        #print("SetQHYCCDBinMode() ret =", ret)
        #print(imageW.value, imageH.value)
        manual_size = False
        if manual_size:
            width = 5000 // bin
            height = 5000 // bin
            ret = qhyccddll.SetQHYCCDResolution(camhandle, 2200 // bin, 800 // bin, width, height)

        else:
            ret = qhyccddll.SetQHYCCDResolution(camhandle, 0, 0, imageW.value // bin, imageH.value // bin)

        #print("SetQHYCCDResolution() ret =", ret)

        ret = qhyccddll.SetQHYCCDParam(camhandle, self.CONTROL_ID.CONTROL_EXPOSURE.value, exp_time)

        ret = qhyccddll.SetQHYCCDParam(camhandle, self.CONTROL_ID.CONTROL_GAIN.value, 50.0)

        ret = qhyccddll.SetQHYCCDParam(camhandle, self.CONTROL_ID.CONTROL_OFFSET.value, 80.0)

        ret = qhyccddll.SetQHYCCDParam(camhandle, self.CONTROL_ID.CONTROL_USBTRAFFIC.value, 0.0)

        self.w = ctypes.c_uint32()
        self.h = ctypes.c_uint32()
        self.b = ctypes.c_uint32()
        self.c = ctypes.c_uint32()

        #print(imageW.value, imageH.value)
        if manual_size:
            length = width * height * 2

        else:
            length = imageW.value * imageH.value // bin


        self.imgdata = (ctypes.c_uint8 * length)()
        self.imgdata_raw8 = (ctypes.c_uint8 * length)()
        #print(imgdata_raw8, imgdata)

    def change_exposure_time(self, exp_time: int, progress_signal=None) -> None:
        """Change the exposure time of the camera.

        Parameters
        ----------
        exp_time : int
            New exposure time in microseconds.
        progress_signal : optional
            Qt signal used to report progress in the GUI.
        """

        print(f"Changing exposure time to {exp_time} us")
        if progress_signal:
            progress_signal.emit(f"Changing exposure time to {exp_time} us")

        ret = self.qhyccddll.SetQHYCCDParam(self.camhandle, self.CONTROL_ID.CONTROL_EXPOSURE.value, exp_time)
        print("SetQHYCCDParam() ret =", ret)

    def close(self) -> None:
        """Close the camera and release resources."""
        ret = self.qhyccddll.CloseQHYCCD(self.camhandle)
        ret = self.qhyccddll.ReleaseQHYCCDResource()
        cv2.destroyAllWindows()

    def _capture_frame(self) -> np.ndarray:
        """Capture a single frame from the camera."""

        self.qhyccddll.ExpQHYCCDSingleFrame(self.camhandle)
        ret = self.qhyccddll.GetQHYCCDSingleFrame(
            self.camhandle,
            byref(self.w),
            byref(self.h),
            byref(self.b),
            byref(self.c),
            self.imgdata,
        )
        if ret != 0:
            raise RuntimeError("Failed to capture image")

        return np.frombuffer(self.imgdata, dtype=np.uint16).reshape(
            (self.h.value, self.w.value)
        )

    def _report_statistics(self, img: np.ndarray, progress_signal=None) -> None:
        """Print and optionally emit basic image statistics."""
        max_pixel = np.max(img)
        mean_pixel = np.mean(img)
        print("Max pixel value:", max_pixel, "Mean pixel value:", mean_pixel)
        if progress_signal:
            progress_signal.emit(f"Max pixel value: {max_pixel}")
            progress_signal.emit(f"Mean pixel value: {mean_pixel}")

    def _display_frame(self, img: np.ndarray) -> None:
        """Show the captured frame using OpenCV."""
        show_img = img / np.max(img) * 255
        cv2.namedWindow("Show", 0)
        cv2.resizeWindow("Show", self.w.value // 10, self.h.value // 10)
        cv2.imshow("Show", show_img.astype(np.uint8))
        cv2.waitKey(0)

    def _save_frame(self, img: np.ndarray, path: str) -> None:
        """Save the frame as a FITS file to ``path``."""
        import astropy.io.fits as fits

        if not path.endswith(".fits"):
            path += ".fits"
        fits.writeto(path, img, overwrite=True)

    def take_frame(
        self,
        working_dir: str,
        image_name: str,
        show: bool = False,
        save: bool = True,
        progress_signal=None,
    ) -> np.ndarray:
        """Capture a frame and optionally display or save it."""

        img = self._capture_frame()
        self._report_statistics(img, progress_signal)

        if show:
            self._display_frame(img)

        if save:
            self._save_frame(img, os.path.join(working_dir, image_name))

        return img

    def take_single_frame(
        self,
        working_dir: str,
        image_name: str,
        show: bool = False,
        progress_signal=None,
    ) -> np.ndarray:
        """Convenience wrapper for :meth:`take_frame`."""

        return self.take_frame(
            working_dir,
            image_name,
            show=show,
            progress_signal=progress_signal,
        )

    def take_multiple_frames(
        self, working_dir: str, image_name: str, num_frames: int
    ) -> None:
        """Capture ``num_frames`` sequential images."""

        for i in range(num_frames):
            self.take_frame(working_dir, f"{image_name}_{i}")

if __name__ == "__main__":
    working_dir = "D:/Vincent/test"
    image_name = "test.fits"
    camera = Camera(exp_time=2000000)
    #camera.take_multiple_frames(working_dir, image_name, 5)
    camera.take_single_frame(working_dir, image_name, show=False)