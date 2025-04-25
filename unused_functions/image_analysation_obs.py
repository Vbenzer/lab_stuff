from core.data_processing import fits_to_arr, measure_fiber_dimensions

# Usage
if __name__ == "__main__":
    #Path of fits file to analyse
    #fits_file = 'D:/Vincent/OptranWF_100_187_P_measurement_3/FRD/filter_2/LIGHT/LIGHT_0012_0.08s.fits'
    fits_file = '/run/user/1002/gvfs/smb-share:server=srv4.local,share=labshare/raw_data/fibers/Measurements/eccentricity/circ100_test.fits'

    #Turn data to numpy array
    data = fits_to_arr(fits_file)

    datas = measure_fiber_dimensions(data, plot=True)
    print(datas)


