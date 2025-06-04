"""Module find_f_number_obs.py.

Auto-generated docstring for better readability.
"""
import numpy as np

from analysis.frd_analysis import calculate_f_number

if __name__ == "__main__":
    # Example usage
    radii = np.array([455,371,290])
    pos_values = np.array([9.9,5,0])


    f_number,f_number_err = calculate_f_number(radii, pos_values, plot_regression=True, save_path="test.png")
    print(f"Calculated F-number (f/#): {f_number:.2f} ± {f_number_err:.2f}")



