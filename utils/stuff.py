"""Utility script to capture bias frames with the entrance camera."""
from core.hardware.cameras import thorlabs_cam_control as tcc

MAIN_FOLDER = (
    r"/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/Manual_Images"
)


def capture_bias_frames(num_frames: int = 50) -> None:
    """Capture a series of bias frames."""
    for i in range(num_frames):
        filename = f"{MAIN_FOLDER}/bias{i:03d}.fits"
        tcc.take_image("entrance_cam", filename, save_fits=True, exposure_time="0s")


if __name__ == "__main__":
    capture_bias_frames()
