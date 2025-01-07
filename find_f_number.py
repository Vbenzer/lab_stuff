import numpy as np
from numpy import array
from scipy.stats import linregress
import matplotlib.pyplot as plt
import json




def calculate_f_number(radii: np.ndarray, ccd_positions: np.ndarray, plot_regression:bool=False, save_plot:bool=True, save_path:str=None):
    """
    Calculate the F-number (f/#) from the spot radii and CCD positions.

    Parameters:
        radii : Array of spot radii
        ccd_positions : Array of CCD positions
        plot_regression : If True, plot the linear regression of the data.
        save_plot : If True, save the plot to a file.
        save_path : Path to save the plot file.

    Returns:
        float: The calculated F-number with error.
    """

    # Convert spot radii to millimeters
    spot_radii = radii*7.52e-3 #Todo: Get this value from image header
    spot_radii = np.sort(spot_radii)[::-1]  #Sort in descending order because motor is reversed when measuring fiber frd

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(ccd_positions, spot_radii)

    # Calculate the F-number using the formula: f/# = 1 / (2 * tan(theta_o))
    f_number = 1 / (2 * slope)
    f_number_err = abs(-1/(2*slope**2)*std_err)

    # Plot regression if requested
    if plot_regression or save_plot:
        plt.scatter(ccd_positions, spot_radii, label="Data points")
        plt.plot(ccd_positions, slope * ccd_positions + intercept, color="green", label="Linear fit")
        plt.xlabel("CCD Position [mm]")
        plt.ylabel("Spot Radius [mm]")
        plt.title("Linear Regression of Spot Radius vs. CCD Position")
        plt.legend()
        plt.grid(True)

        if save_plot:
            if save_path is None:
                raise ValueError("'save_path' must be provided")
            plt.savefig(save_path+"regression_plot.png")

        if plot_regression:
            plt.show()
        plt.close()
    with open(save_path+"f_number.json","w") as f:
        json.dump({"f_number":f_number,"f_number_err":f_number_err}, f)

    return f_number,f_number_err



if __name__ == "__main__":
    # Example usage
    radii = np.array([885,757,631])
    pos_values = np.array([9.9,5,0])


    f_number,f_number_err = calculate_f_number(radii,pos_values, plot_regression=True, save_path="test.png")
    print(f"Calculated F-number (f/#): {f_number:.2f} ± {f_number_err:.2f}")



