import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt
import json




def calculate_f_number(radii, pos_values, plot_regression=False, save_plot:bool=True, save_path:str=None):
    """
    Calculate the F-number (f/#) from the spot radii and CCD positions.

    Parameters:
        data (list of tuples): List of (CCD position, spot radius) pairs [(position, radius)].
        plot_regression (bool): If True, plot the linear regression of the data.

    Returns:
        float: The calculated F-number with error.
    """

    #Todo: weird number stuff when getting data from main directly

    spot_radii = radii*3.76e-3
    ccd_positions = pos_values

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
            plt.savefig(save_path)

        if plot_regression:
            plt.show()
    with open("f_number.json","w") as f:
        json.dump({"f_number":f_number,"f_number_err":f_number_err}, f)

    return f_number,f_number_err


if __name__ == "__main__":
    # Example usage
    radii = np.array([706,814,920])
    pos_values = np.array([0,5,9.9])


    f_number,f_number_err = calculate_f_number(radii,pos_values, plot_regression=True, save_path="test.png")
    print(f"Calculated F-number (f/#): {f_number:.2f} ± {f_number_err:.2f}")



