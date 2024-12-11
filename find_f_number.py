import numpy as np
from scipy.stats import linregress
import matplotlib.pyplot as plt


def calculate_f_number(radii, pos_values, plot_regression=False):
    """
    Calculate the F-number (f/#) from the spot radii and CCD positions.

    Parameters:
        data (list of tuples): List of (CCD position, spot radius) pairs [(position, radius)].
        plot_regression (bool): If True, plot the linear regression of the data.

    Returns:
        float: The calculated F-number with error.
    """
    # Separate CCD positions and spot radii
    #ccd_positions, spot_radii = zip(*data) #Todo: weird number stuff when getting data from main directly
    spot_radii = np.array(radii)*3.76e-3
    ccd_positions = np.array(pos_values)
    spot_radii = np.array(spot_radii)

    # Perform linear regression
    slope, intercept, r_value, p_value, std_err = linregress(ccd_positions, spot_radii)

    # Calculate the F-number using the formula: f/# = 1 / (2 * tan(theta_o))
    f_number = 1 / (2 * slope)
    f_number_err = abs(-1/(2*slope**2)*std_err)

    # Plot regression if requested
    if plot_regression:
        plt.scatter(ccd_positions, spot_radii, label="Data points")
        plt.plot(ccd_positions, slope * ccd_positions + intercept, color="green", label="Linear fit")
        plt.xlabel("CCD Position [mm]")
        plt.ylabel("Spot Radius [mm]")
        plt.title("Linear Regression of Spot Radius vs. CCD Position")
        plt.legend()
        plt.grid(True)
        plt.show()

    return f_number,f_number_err


if __name__ == "__main__":
    # Example usage
    """data = [
        (0, 11.46/3.76e-3),
        (4.33, 12.19/3.76e-3),
        (8.65, 12.91/3.76e-3),
    ]  # Replace with your measured data"""
    data = [
        (0, 706),
        (5, 814),
        (9.9, 920),
    ]  # Replace with your measured data

    f_number,f_number_err = calculate_f_number(data, plot_regression=True)
    print(f"Calculated F-number (f/#): {f_number:.2f} ± {f_number_err:.2f}")

