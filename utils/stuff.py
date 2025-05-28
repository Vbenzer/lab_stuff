from core.hardware.cameras import thorlabs_cam_control as tcc
main_folder = r"/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/Manual_Images"

for i in range(50):
    tcc.take_image("entrance_cam", main_folder + f"/bias{i:03d}.fits", save_fits=True,
                   exposure_time="0s")