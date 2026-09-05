
import scipy.signal

import re

import glob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import scipy.signal
from scipy.stats import norm
import torch

import torch.optim as optim
from torch.distributions.normal import Normal
from scipy.optimize import linear_sum_assignment

import os
import shutil

DO_PLOTS = True  # change to True to re-enable

ft = 16
# Define the folder path
folder_path = "RAWtxt"

# Search for all TXT files that have "click_100" in the filename
txt_files = glob.glob(os.path.join(folder_path, "*302_026_v1_click_70_1*"))

# Latency bias values
stimulus_bias = {
    "click_100": [1.85, 3.01, 4.21, 100.0, 5.72, 7.49, 9.86],
    "click_90":  [1.94, 2.96, 4.61, 100.0, 6.20, 7.82, 9.92],
    "click_80":  [1.53, 3.00, 5.02, 100.0, 6.58, 8.33, 10.40],
    "click_70":  [1.86, 3.45, 5.48, 100.0, 7.03, 8.83, 10.95],
    "click_60":  [2.22, 3.95, 5.96, 100.0, 7.69, 9.49, 11.60],
    "click_50":  [2.61, 4.48, 6.49, 100.0, 8.38, 10.29, 12.39],
    "click_40":  [3.03, 5.05, 7.05, 100.0, 9.18, 11.25, 13.35],
    "click_30":  [3.48, 5.67, 7.66, 100.0, 10.13, 12.41, 14.52],

    "chirp":     [1.65, 2.86, 3.98, 100.0, 5.05, 7.11, 8.64],
    "da":        [2.11, 2.88, 4.71, 100.0, 6.37, 8.02, 9.89],
    "tb4":       [2.24, 3.65, 4.57, 100.0, 6.19, 8.38, 10.39],
    "tb5":       [3.58, 5.14, 6.22, 100.0, 7.73, 9.33, 11.30],
}

# Latency bias values
stimulus_bias_negative = {
    "click_100": [
        (1.85 + 3.01) / 2,
        (3.01 + 4.21) / 2,
        (4.21 + 5.72) / 2,
        100.0,
        (5.72 + 7.49) / 2,
        (7.49 + 9.86) / 2,
        9.86 + 1
    ],

    "click_90": [
        (1.94 + 2.96) / 2,
        (2.96 + 4.61) / 2,
        (4.61 + 6.20) / 2,
        100.0,
        (6.20 + 7.82) / 2,
        (7.82 + 9.92) / 2,
        9.92 + 1
    ],

    "click_80": [
        (1.53 + 3.00) / 2,
        (3.00 + 5.02) / 2,
        (5.02 + 6.58) / 2,
        100.0,
        (6.58 + 8.33) / 2,
        (8.33 + 10.40) / 2,
        10.40 + 1
    ],

    "click_70": [
        (1.86 + 3.45) / 2,
        (3.45 + 5.48) / 2,
        (5.48 + 7.03) / 2,
        100.0,
        (7.03 + 8.83) / 2,
        (8.83 + 10.95) / 2,
        10.95 + 1
    ],

    "click_60": [
        (2.22 + 3.95) / 2,
        (3.95 + 5.96) / 2,
        (5.96 + 7.69) / 2,
        100.0,
        (7.69 + 9.49) / 2,
        (9.49 + 11.60) / 2,
        11.60 + 1
    ],

    "click_50": [
        (2.61 + 4.48) / 2,
        (4.48 + 6.49) / 2,
        (6.49 + 8.38) / 2,
        100.0,
        (8.38 + 10.29) / 2,
        (10.29 + 12.39) / 2,
        12.39 + 1
    ],

    "click_40": [
        (3.03 + 5.05) / 2,
        (5.05 + 7.05) / 2,
        (7.05 + 9.18) / 2,
        100.0,
        (9.18 + 11.25) / 2,
        (11.25 + 13.35) / 2,
        13.35 + 1
    ],

    "click_30": [
        (3.48 + 5.67) / 2,
        (5.67 + 7.66) / 2,
        (7.66 + 10.13) / 2,
        100.0,
        (10.13 + 12.41) / 2,
        (12.41 + 14.52) / 2,
        14.52 + 1
    ],

    "chirp": [
        (1.65 + 2.86) / 2,
        (2.86 + 3.98) / 2,
        (3.98 + 5.05) / 2,
        100.0,
        (5.05 + 7.11) / 2,
        (7.11 + 8.64) / 2,
        8.64 + 1
    ],

    "da": [
        (2.11 + 2.88) / 2,
        (2.88 + 4.71) / 2,
        (4.71 + 6.37) / 2,
        100.0,
        (6.37 + 8.02) / 2,
        (8.02 + 9.89) / 2,
        9.89 + 1
    ],

    "tb4": [
        (2.24 + 3.65) / 2,
        (3.65 + 4.57) / 2,
        (4.57 + 6.19) / 2,
        100.0,
        (6.19 + 8.38) / 2,
        (8.38 + 10.39) / 2,
        10.39 + 1
    ],

    "tb5": [
        (3.58 + 5.14) / 2,
        (5.14 + 6.22) / 2,
        (6.22 + 7.73) / 2,
        100.0,
        (7.73 + 9.33) / 2,
        (9.33 + 11.30) / 2,
        11.30 + 1
    ],
}

# Example base latencies
identified_wave_latencies_key = [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]




identified_wave_latencies_negative_key = [0.0, 0.0, 0.0, 100.0, 0.0, 0.0, 0.0]

# Placeholder for adjusted latency results (optional)
adjusted_latencies = []
adjusted_latencies_negative = []

def remove_too_close(arr, tolerance, keep='mean'):
    """
    Remove values that are closer together than a specified tolerance.

    Parameters
    ----------
    arr : array-like
        Input numeric array.
    tolerance : float
        Minimum required distance between consecutive values.
    keep : {'first', 'last', 'mean'}
        Strategy for which value to keep within each cluster of close values:
          - 'first': keep the first value in each cluster
          - 'last': keep the last value in each cluster
          - 'mean': replace each cluster with its mean

    Returns
    -------
    cleaned : np.ndarray
        Array with values closer than tolerance removed (or merged).
    """

    arr = np.sort(np.asarray(arr, dtype=float))
    if len(arr) <= 1:
        return arr

    # Start a cluster list
    clusters = [[arr[0]]]
    for val in arr[1:]:
        if val - clusters[-1][-1] < tolerance:
            # too close → same cluster
            clusters[-1].append(val)
        else:
            # far enough → new cluster
            clusters.append([val])

    # Decide what to keep per cluster
    cleaned = []
    for cluster in clusters:
        if keep == 'first':
            cleaned.append(cluster[0])
        elif keep == 'last':
            cleaned.append(cluster[-1])
        elif keep == 'mean':
            cleaned.append(np.mean(cluster))
        else:
            raise ValueError("keep must be one of {'first', 'last', 'mean'}")

    return np.array(cleaned)

def filter_with_nearby_fixed(variable_array, fixed_array, threshold=0.91, tol = 0.5):
    variable_array = np.array(variable_array)
    fixed_array = np.array(fixed_array)

    # Compute cost matrix (absolute difference between each pair)
    cost_matrix = np.abs(variable_array[:, None] - fixed_array[None, :])

    # Set high cost for differences beyond the threshold
    high_cost = 1e6
    masked_cost = np.where(cost_matrix <= threshold, cost_matrix, high_cost)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(masked_cost)

    matched_variables = []

    # Fill in valid assignments
    for r, c in zip(row_ind, col_ind):
        if masked_cost[r][c] <= threshold:
            matched_variables.append(variable_array[r])


    matched_variables = remove_too_close(matched_variables, tol, 'mean')
    return np.array(matched_variables)

def plot_abr_response(right_vertex_trimmed, t_trimmed, title='ABR Waveform'):
    plt.figure(figsize=(10, 4))
    plt.plot(t_trimmed, right_vertex_trimmed, label=title, color='darkblue')
    plt.axhline(0, color='gray', linestyle='--', linewidth=0.5)
    plt.xlabel('Time (ms)', fontsize=ft)
    plt.ylabel('Amplitude (µV)', fontsize=ft)
    plt.title(title)
    plt.grid(True, linestyle='--', linewidth=0.5, alpha=0.7)
    plt.legend(fontsize=ft-4)
    plt.show()


def find_closest_within_threshold(variable_array, fixed_array, threshold=0.91):
    roman_numerals_list = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII']
    variable_array = np.array(variable_array)
    fixed_array = np.array(fixed_array)
    # Compute cost matrix (absolute difference between each pair)
    cost_matrix = np.abs(variable_array[:, None] - fixed_array[None, :])

    # Set high cost for differences beyond the threshold
    high_cost = 1e6
    masked_cost = np.where(cost_matrix <= threshold, cost_matrix, high_cost)
    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(masked_cost)

    # Initialize result array with NaNs
    result = [np.nan] * len(variable_array)

    # Fill in valid assignments only
    for r, c in zip(row_ind, col_ind):
        if masked_cost[r][c] <= threshold:
            result[r] = roman_numerals_list[c]
    return np.array(result)

def find_closest_within_threshold_negative(variable_array, fixed_array, threshold=0.91):
    result = []
    roman_numerals_list = ['nI', 'nII', 'nIII', 'nIV', 'nV', 'nVI', 'nVII']
    variable_array = np.array(variable_array)
    fixed_array = np.array(fixed_array)
    # Compute cost matrix (absolute difference between each pair)
    cost_matrix = np.abs(variable_array[:, None] - fixed_array[None, :])

    # Set high cost for differences beyond the threshold
    high_cost = 1e6
    masked_cost = np.where(cost_matrix <= threshold, cost_matrix, high_cost)

    # Solve the assignment problem
    row_ind, col_ind = linear_sum_assignment(masked_cost)

    # Initialize result array with NaNs
    result = [np.nan] * len(variable_array)

    # Fill in valid assignments only
    for r, c in zip(row_ind, col_ind):
        if masked_cost[r][c] <= threshold:
            result[r] = roman_numerals_list[c]

    return np.array(result)

def extract_stimulus_info(file_path: str) -> str:
    # Match any pattern like click_XX, tb4_XX, tb5_XX, chirp_XX, da_XX where XX are digits
    match = re.search(r'_(click|tb4|tb5|chirp|da)_(\d+)_', file_path)
    if match:
        return f"{match.group(1)}_{match.group(2)}"
    else:
        return None  # or raise an error if you want to enforce it

def extract_data_to_dataframe(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Initialize an empty list to store the rows
    data_rows = []
    headers = []

    # Start processing the file
    for line in lines:
        # Look for the header row (assumes the header row contains "Data Pnt")
        if "Data Pnt(ms):" in line:
            headers = line.strip().split(',')
        elif line.strip():  # Ignore empty lines
            # Extract rows of numeric data only if they seem to contain values
            try:
                # Attempt to split by commas and check if the first value is numeric
                values = line.strip().split(',')
                # Ensure that it has the correct number of columns (based on headers length)
                if len(values) == len(headers):
                    data_rows.append(values)
            except ValueError:
                continue  # Skip rows that can't be processed

    # Convert to DataFrame
    df = pd.DataFrame(data_rows, columns=headers)

    # Convert appropriate columns to numeric types
    for col in df.columns:
        try:
            df[col] = pd.to_numeric(df[col], errors='coerce')  # Convert to numeric, invalid parsing becomes NaN
        except ValueError:
            pass  # Ignore columns that cannot be converted

    return df
count = 0
# Print the found files

wave_I_latency_list = []
wave_II_latency_list = []
wave_III_latency_list = []
wave_IV_latency_list = []
wave_V_latency_list = []
wave_VI_latency_list = []
wave_VII_latency_list = []

wave_I_height_list = []
wave_II_height_list = []
wave_III_height_list = []
wave_IV_height_list = []
wave_V_height_list = []
wave_VI_height_list = []
wave_VII_height_list = []

wave_I_sigma_list = []
wave_II_sigma_list = []
wave_III_sigma_list = []
wave_IV_sigma_list = []
wave_V_sigma_list = []
wave_VI_sigma_list = []
wave_VII_sigma_list = []

wave_I_curvature_list = []
wave_II_curvature_list = []
wave_III_curvature_list = []
wave_IV_curvature_list = []
wave_V_curvature_list = []
wave_VI_curvature_list = []
wave_VII_curvature_list = []

wave_I_autc_list = []
wave_II_autc_list = []
wave_III_autc_list = []
wave_IV_autc_list = []
wave_V_autc_list = []
wave_VI_autc_list = []
wave_VII_autc_list = []

# Latency Lists
wave_A_latency_list = []
wave_B_latency_list = []
wave_C_latency_list = []
wave_D_latency_list = []
wave_E_latency_list = []
wave_F_latency_list = []
wave_G_latency_list = []

# Height Lists
wave_A_height_list = []
wave_B_height_list = []
wave_C_height_list = []
wave_D_height_list = []
wave_E_height_list = []
wave_F_height_list = []
wave_G_height_list = []

# Sigma Lists (Standard Deviation)
wave_A_sigma_list = []
wave_B_sigma_list = []
wave_C_sigma_list = []
wave_D_sigma_list = []
wave_E_sigma_list = []
wave_F_sigma_list = []
wave_G_sigma_list = []

# Curvature Lists
wave_A_curvature_list = []
wave_B_curvature_list = []
wave_C_curvature_list = []
wave_D_curvature_list = []
wave_E_curvature_list = []
wave_F_curvature_list = []
wave_G_curvature_list = []

# AUTC (Area Under the Curve) Lists
wave_A_autc_list = []
wave_B_autc_list = []
wave_C_autc_list = []
wave_D_autc_list = []
wave_E_autc_list = []
wave_F_autc_list = []
wave_G_autc_list = []
participant_list = []
frequency_list = []
visit_list = []
run_list = []
decibel_list = []
stimulus_list = []
MSE_list = []
MSE_list_smaller_window = []
MSE_Wave_V = []
MSE_Wave_I = []
MSE_Wave_III = []

def parse_filename(filename):
    # Regular expression pattern
    print(filename)
    pattern = r'(?P<participant>[a-zA-Z0-9_]+)_v(?P<visit>\d+)_?(?P<stimulus>[a-zA-Z0-9]+)?_?(?P<decibel>\d+)?_(?P<run>\d+)\.TXT'

    match = re.search(pattern, filename)
    if not match:
        return None, None, None, None, None

    participant = match.group('participant')
    visit = int(match.group('visit'))
    stimulus = match.group('stimulus') if match.group('stimulus') else None
    decibel = int(match.group('decibel')) if match.group('decibel') else None
    run = int(match.group('run'))
    # print(participant)
    # print(visit)
    # print(stimulus)
    # print(decibel)
    # print(run)
    return participant, visit, stimulus, decibel, run


for file in txt_files:
    print(file)
    participant, visit, stimulus, decibel, run = parse_filename(os.path.basename(file))

    # Save info
    participant_list.append(participant)
    visit_list.append(visit)
    run_list.append(run)
    decibel_list.append(decibel)
    stimulus_list.append(stimulus)
    output_dir = f"plots/{participant}_{visit}_{run}_{decibel}_{stimulus}_paper_images"

    # If the directory already exists, remove it completely
    if DO_PLOTS:
        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        # Now create a fresh directory
        os.makedirs(output_dir)
        # Determine bias key
    if stimulus == "click":
        bias_key = f"click_{decibel}"
    else:
        bias_key = stimulus

    # Get the bias (default to 0 if missing)
    bias = stimulus_bias.get(bias_key.lower(), 0)
    bias_negative = stimulus_bias_negative.get(bias_key.lower(), 0)
    # Apply bias
    identified_wave_latencies = [x + y for x, y in zip(bias, identified_wave_latencies_key)]
    identified_wave_latencies_negative = [x + y for x, y in zip(bias_negative, identified_wave_latencies_negative_key)]
    print(identified_wave_latencies)
    print(identified_wave_latencies_negative)
    # if not (stimulus == "click" and decibel == 40):
    #     continue

    file_path = file  # Replace with the path to your text file
    df = extract_data_to_dataframe(file_path)
    time = np.array(df["Data Pnt(ms):"])
    post_stimulus_indices = (time > 0) & (time < 12)
    average_uv = np.array(df["Average(uV):"])

    delta_t = time[1] - time[0]

    fft_buffer1 = np.fft.fft(average_uv)
    # Frequency domain (corresponding frequencies)
    sampling_period = delta_t

    frequencies = np.fft.fftfreq(len(average_uv), d=sampling_period)
    file_name = []
    # Define the cutoff frequency (in Hz)
    if DO_PLOTS:
        plt.figure(figsize=(8, 4))

        plt.plot(time[post_stimulus_indices], average_uv[post_stimulus_indices], color='blue', label='Signal')
        plt.axhline(y=np.mean(average_uv[post_stimulus_indices]), color='black', linestyle='--', label='Mean')

        plt.title("ABR", fontsize = ft)
        plt.xlabel("Time (ms)", fontsize = ft)
        plt.ylabel("Amplitude (µV)", fontsize = ft)
        plt.xticks(fontsize=ft)
        plt.yticks(fontsize=ft)
        plt.legend(fontsize=ft-4)
        plt.ylim(plt.ylim()[0]-0.04, plt.ylim()[1]+0.04)


        # Save the figure instead of showing
        output_path = os.path.join(output_dir, f"{participant}_ABR_with_mean.png")
        plt.gcf().supylabel("A", fontsize=16, fontweight='bold', x=-0.08, rotation=0)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.clf()  # close the figure so it doesn't stack in memory

    average_uv = average_uv - np.mean(average_uv[post_stimulus_indices])

    if DO_PLOTS:
        plt.figure(figsize=(8, 4))

        plt.plot(time[post_stimulus_indices], average_uv[post_stimulus_indices], color='blue', label='Signal')
        plt.axhline(y=np.mean(average_uv[post_stimulus_indices]), color='black', linestyle='--', label='Mean')
        plt.xticks(fontsize=ft)
        plt.yticks(fontsize=ft)
        plt.title("ABR", fontsize=ft)
        plt.xlabel("Time (ms)", fontsize=ft)
        plt.ylabel("Amplitude (µV)", fontsize=ft)
        plt.legend(fontsize=ft-4)
        plt.ylim(plt.ylim()[0]-0.04, plt.ylim()[1]+0.04)

        # Save the figure instead of showing
        output_path = os.path.join(output_dir, f"{participant}_ABR_with_mean_at_0.png")
        plt.gcf().supylabel("A", fontsize=16, fontweight='bold', x=-0.08, rotation=0)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.clf()  # close the figure so it doesn't stack in memory
    tol = 1e-9
    max_indices = scipy.signal.argrelextrema(average_uv, lambda a, b: np.greater(a, b - tol))[0]
    # print(max_indices)
    # min_indices = scipy.signal.argrelextrema(average_uv, np.less)[0]
    min_indices = scipy.signal.argrelextrema(average_uv, lambda a, b: np.less(a, b + tol))[0]
    # print(min_indices)
    # Filter maxima to keep only positive values
    max_indices_positive = max_indices[average_uv[max_indices] > 0]

    # Filter minima to keep only negative values
    min_indices_negative = min_indices[average_uv[min_indices] < 0]

    # Count maxima and minima
    num_maxima = len(max_indices_positive)
    num_minima = len(min_indices_negative)

    true_max_times = time[max_indices]
    print(true_max_times)
    true_min_times = time[min_indices]
    max_time_positive = time[max_indices_positive]
    min_time_negative = time[min_indices_negative]
    print(true_min_times)
    print(min_time_negative)

    if DO_PLOTS:
        plt.figure(figsize=(8, 4))
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        plt.plot(time[post_stimulus_indices], average_uv[post_stimulus_indices], color=colors[5])
        plt.axhline(y=np.mean(average_uv[post_stimulus_indices]), color='black', linestyle='--')
        plt.xticks(fontsize=ft)
        plt.yticks(fontsize=ft)
        ymin_global, ymax_global = plt.ylim()
        for i, t in enumerate(true_min_times):
            # Find y-value on curve at this x (interpolation in case t not exactly in time array)
            if (t > 0) & (t<12):
                y_val = np.interp(t, time[post_stimulus_indices], average_uv[post_stimulus_indices])
                plt.vlines(x=t, ymin=y_val, ymax=ymax_global+0.04,

                           color='gray', linestyle='--')

        # Green lines: from curve to top
        for i, t in enumerate(true_max_times):
            if (t > 0) & (t<12):
                y_val = np.interp(t, time[post_stimulus_indices], average_uv[post_stimulus_indices])
                plt.vlines(x=t, ymin=ymin_global-0.04, ymax=y_val,
                           color='gray', linestyle='--')

        # plt.ylim(ymin_global-0.04, ymax_global+0.04)
        plt.plot([], [], color='gray', linestyle='--', label='Discarded')
        # plt.plot([], [], color='green', linestyle='--', label='Maxima')
        plt.title("ABR", fontsize=ft)
        plt.xlabel("Time (ms)", fontsize=ft)
        plt.ylabel("Amplitude (µV)", fontsize=ft)

        # plt.legend()
        # plt.tight_layout()
        # # Save the figure instead of showing
        # output_path = os.path.join(output_dir, f"{participant}_max_and_min_unfiltered.png")
        # plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #
        # plt.clf()  # close the figure so it doesn't stack in memory

    # if DO_PLOTS:
    #     plt.figure(figsize=(8, 4))
    #     plt.plot(time[post_stimulus_indices], average_uv[post_stimulus_indices], color='blue')
    #
    #     # Plot vertical lines at minima
    #     ymin_global, ymax_global = plt.ylim()
    #     for i, t in enumerate(min_time_negative):
    #         # Find y-value on curve at this x (interpolation in case t not exactly in time array)
    #         if (t > 0) & (t<12):
    #             y_val = np.interp(t, time[post_stimulus_indices], average_uv[post_stimulus_indices])
    #             plt.vlines(x=t, ymin=y_val, ymax=ymax_global+0.04,
    #
    #                        color='blue', linestyle='--')
    #
    #     # Green lines: from curve to top
    #     for i, t in enumerate(max_time_positive):
    #         if (t > 0) & (t<12):
    #             y_val = np.interp(t, time[post_stimulus_indices], average_uv[post_stimulus_indices])
    #             plt.vlines(x=t, ymin=ymin_global-0.04, ymax=y_val,
    #                        color='blue', linestyle='--')
    #     plt.ylim(ymin_global-0.04, ymax_global+0.04)
    #
    #     plt.plot([], [], color='red', linestyle='--', label='Minima')
    #     plt.plot([], [], color='green', linestyle='--', label='Maxima')
    #
    #     plt.title("ABR")
    #     plt.xlabel("Time (ms)")
    #     plt.ylabel("Amplitude (µV)")
    #
    #     plt.legend()
    #     plt.tight_layout()
    #     # Save the figure instead of showing
    #     output_path = os.path.join(output_dir, f"{participant}_max_and_min_subset_unfiltered.png")
    #     plt.savefig(output_path, dpi=300, bbox_inches='tight')
    #
    #     plt.clf() # close the figure so it doesn't stack in memory

    # print(min_time_negative)
    max_time_positive = filter_with_nearby_fixed(max_time_positive, identified_wave_latencies)
    true_max_times = filter_with_nearby_fixed(true_max_times, identified_wave_latencies)
    min_time_negative = filter_with_nearby_fixed(min_time_negative, identified_wave_latencies_negative)
    true_min_times = filter_with_nearby_fixed(true_min_times, identified_wave_latencies_negative)
    # print(min_time_negative)
    negative_gmm = np.zeros_like(time, dtype=np.float64)  # Ensure float64 type

    if DO_PLOTS:
        # plt.figure(figsize=(8, 4))
        # plt.plot(time[post_stimulus_indices], average_uv[post_stimulus_indices], color='blue')

        roman_numerals_from_positive_gaussians = find_closest_within_threshold(true_max_times,
                                                                               identified_wave_latencies)
        roman_numerals_from_negative_gaussians = find_closest_within_threshold_negative(true_min_times,
                                                                                        identified_wave_latencies_negative)
        for i in range(len(true_min_times)):

            # --- Marker + Label logic ---
            t_val = float(true_min_times[i])  # x position (mean of Gaussian)
            closest_idx = np.argmin(np.abs(time - t_val))  # index closest to mean
            y_val = average_uv[closest_idx]  # corresponding y value

            label = roman_numerals_from_negative_gaussians[i] if i < len(
                roman_numerals_from_negative_gaussians) else ''
            plt.xticks(fontsize=ft)
            plt.yticks(fontsize=ft)
            plt.plot(t_val, y_val, 'o', color='black')  # point marker
            plt.text(t_val, y_val - 0.03, label, ha='center', va='top', fontsize=12)
            # note: va='top' so text is placed below marker (since curve is negative)
        # Plot Positive GMM
        for i in range(len(true_max_times)):

            # Plot marker and label at the peak
            t_val = float(true_max_times[i])
            closest_idx = np.argmin(np.abs(time - t_val))
            y_val = average_uv[closest_idx]

            label = roman_numerals_from_positive_gaussians[i] if i < len(
                roman_numerals_from_positive_gaussians) else ''
            plt.plot(t_val, y_val, 'o', color='black')  # point marker
            plt.text(t_val, y_val + 0.02, label, ha='center', va='bottom', fontsize=12)

        # ymin_global, ymax_global = plt.ylim()
        for i, t in enumerate(true_min_times):
            # Find y-value on curve at this x (interpolation in case t not exactly in time array)
            if (t > 0) & (t<12):
                y_val = np.interp(t, time[post_stimulus_indices], average_uv[post_stimulus_indices])
                plt.vlines(x=t, ymin=y_val, ymax=ymax_global+0.04,

                           color='green', linestyle='--')

        # Green lines: from curve to top
        for i, t in enumerate(true_max_times):
            if (t > 0) & (t<12):
                y_val = np.interp(t, time[post_stimulus_indices], average_uv[post_stimulus_indices])
                plt.vlines(x=t, ymin=ymin_global-0.04, ymax=y_val,
                           color='green', linestyle='--')
        # plt.title("ABR")
        # plt.xlabel("Time (ms)")
        # plt.ylabel("Amplitude (µV)")
        plt.plot([], [], color='green', linestyle='--', label='Analyzed Numerically')

        # plt.legend()
        # plt.ylim(ymin_global-0.04, ymax_global+0.04)
        # plt.tight_layout()
        # # Save the figure instead of showing
        # output_path = os.path.join(output_dir, f"{participant}_max_and_min.png")
        # plt.savefig(output_path, dpi=300, bbox_inches='tight')
        #
        # plt.clf()  # close the figure so it doesn't stack in memory

    if DO_PLOTS:
        # plt.figure(figsize=(8, 4))

        roman_numerals_from_positive_gaussians = find_closest_within_threshold(max_time_positive,
                                                                               identified_wave_latencies)
        roman_numerals_from_negative_gaussians = find_closest_within_threshold_negative(min_time_negative,
                                                                                        identified_wave_latencies_negative)


        # plt.plot(time[post_stimulus_indices], average_uv[post_stimulus_indices], color='blue')
        for i in range(len(min_time_negative)):

            # --- Marker + Label logic ---
            t_val = float(min_time_negative[i])  # x position (mean of Gaussian)
            closest_idx = np.argmin(np.abs(time - t_val))  # index closest to mean
            y_val = average_uv[closest_idx]  # corresponding y value

            label = roman_numerals_from_negative_gaussians[i] if i < len(
                roman_numerals_from_negative_gaussians) else ''

            plt.plot(t_val, y_val, 'o', color='black')  # point marker
            plt.text(t_val, y_val - 0.03, label, ha='center', va='top', fontsize=12)
            # note: va='top' so text is placed below marker (since curve is negative)
        # Plot Positive GMM
        for i in range(len(max_time_positive)):

            # Plot marker and label at the peak
            t_val = float(max_time_positive[i])
            closest_idx = np.argmin(np.abs(time - t_val))
            y_val = average_uv[closest_idx]

            label = roman_numerals_from_positive_gaussians[i] if i < len(
                roman_numerals_from_positive_gaussians) else ''
            plt.plot(t_val, y_val, 'o', color='black')  # point marker
            plt.text(t_val, y_val + 0.02, label, ha='center', va='bottom', fontsize=12)

        # Plot vertical lines at minima
        # Red lines: from bottom to the curve
        # ymin_global, ymax_global = plt.ylim()
        for i, t in enumerate(min_time_negative):
            # Find y-value on curve at this x (interpolation in case t not exactly in time array)
            y_val = np.interp(t, time[post_stimulus_indices], average_uv[post_stimulus_indices])
            plt.vlines(x=t, ymin=y_val, ymax=ymax_global+0.04,

                       color='blue', linestyle='--')

        # Green lines: from curve to top
        for i, t in enumerate(max_time_positive):
            y_val = np.interp(t, time[post_stimulus_indices], average_uv[post_stimulus_indices])
            plt.vlines(x=t, ymin=ymin_global-0.04, ymax=y_val,
                       color='blue', linestyle='--')
        plt.ylim(ymin_global-0.04, ymax_global+0.04)
        plt.plot([], [], color='blue', linestyle='--', label='Fit with Gaussian')

        plt.title("ABR", fontsize=ft)
        plt.xlabel("Time (ms)", fontsize=ft)
        plt.ylabel("Amplitude (µV)", fontsize=ft)
        plt.xticks(fontsize=ft)
        plt.yticks(fontsize=ft)
        plt.legend(fontsize=ft-4)
        # Save the figure instead of showing
        output_path = os.path.join(output_dir, f"{participant}_max_and_min_subset.png")
        plt.gcf().supylabel("B", fontsize=16, fontweight='bold', x=-0.08, rotation=0)
        plt.savefig(output_path, dpi=300, bbox_inches="tight")
        plt.clf()  # close the figure so it doesn't stack in memory

    for i in range(len(min_time_negative)):
        negative_gmm += np.real(average_uv[min_indices[i]]) * norm.pdf(time, loc=min_time_negative[i], scale=0.5)

    positive_gmm = np.zeros_like(time, dtype=np.float64)  # Ensure float64 type

    for i in range(len(max_time_positive)):
        positive_gmm += np.real(average_uv[max_indices[i]]) * norm.pdf(time, loc=max_time_positive[i], scale=0.5)

    # Create a 2x2 subplot figure

    # fig, axs = plt.subplots(2, 2, figsize=(12, 8))  # 2 rows, 2 columns

    # Ensure max_time and min_time are arrays or tensors already
    max_time_tensor = torch.tensor(max_time_positive, dtype=torch.float32, requires_grad=False)
    min_time_tensor = torch.tensor(min_time_negative, dtype=torch.float32, requires_grad=False)

    # Initialize learnable parameters as leaf tensors
    max_scale = torch.ones_like(max_time_tensor, requires_grad=True)
    max_amp = torch.ones_like(max_time_tensor, requires_grad=True)
    min_scale = torch.ones_like(min_time_tensor, requires_grad=True)
    min_amp = torch.ones_like(min_time_tensor, requires_grad=True)

    # Define separate optimizers for each parameter tensor
    optimizer_max_scale = optim.Adam([max_scale], lr=0.15)
    optimizer_max_amp = optim.Adam([max_amp], lr=0.025)
    optimizer_min_scale = optim.Adam([min_scale], lr=0.15)
    optimizer_min_amp = optim.Adam([min_amp], lr=0.025)
    roman_numerals = find_closest_within_threshold(max_time_positive, identified_wave_latencies)
    print(roman_numerals)
    if 'V' in roman_numerals:
        V_index = np.where(roman_numerals == 'V')[0][0]
        wave_5_latency = max_time_positive[V_index]

        # Parameters with gradients
        wave_4_pre_time = torch.tensor([0.3], requires_grad=True)
        wave_4_amp = torch.tensor([0.0], requires_grad=True)
        wave_4_sigma = torch.tensor([1.0], requires_grad=True)

        # Individual optimizers
        opt_wave_4_pre_time = torch.optim.Adam([wave_4_pre_time], lr=0.05)
        opt_wave_4_amp = torch.optim.Adam([wave_4_amp], lr=0.05)
        opt_wave_4_sigma = torch.optim.Adam([wave_4_sigma], lr=0.05)


        # Put parameters in a list
        params = [wave_4_pre_time,  wave_4_amp, wave_4_sigma]

        # Define optimizer
        optimizer = torch.optim.Adam(params, lr=0.01)

    # plt.clf()

    num_epochs = 200
    time_tensor = torch.tensor(time, dtype=torch.float32)
    scaler = 0.5
    average_uv_tensor = torch.tensor(average_uv, dtype=torch.float32)

    closest_idx_neg = np.abs(time[:, None] - min_time_negative).argmin(axis=0)
    y_vals_negative = average_uv[closest_idx_neg]

    closest_idx_pos = np.abs(time[:, None] - max_time_positive).argmin(axis=0)
    y_vals_positive = average_uv[closest_idx_pos]

    for epoch in range(num_epochs):
        # Reset gradients
        optimizer_max_scale.zero_grad()
        optimizer_max_amp.zero_grad()
        optimizer_min_scale.zero_grad()
        optimizer_min_amp.zero_grad()

        pos_gmm = torch.zeros_like(torch.tensor(time), dtype=torch.float32)  # Ensure tensor type
        neg_gmm = torch.zeros_like(torch.tensor(time), dtype=torch.float32)  # Ensure tensor type
        # Compute Gaussian mixture model (GMM) contribution

        for i in range(len(min_time_negative)):
            mu = min_time_tensor[i]
            sigma = scaler * torch.sigmoid(min_scale[i])  # σ > 0
            amp = y_vals_negative[i]  # observed value, target peak

            pdf = Normal(mu, sigma).log_prob(torch.tensor(time)).exp()
            pdf_scaled = amp * pdf / pdf.max()

            neg_gmm += pdf_scaled
        for i in range(len(max_time_positive)):  # Fixed: using max_time instead of min_time
            mu = max_time_tensor[i]
            sigma = scaler * torch.sigmoid(max_scale[i])  # σ > 0
            amp = y_vals_positive[i]  # observed value, target peak

            pdf = Normal(mu, sigma).log_prob(torch.tensor(time)).exp()
            pdf_scaled = amp * pdf / pdf.max()

            pos_gmm += pdf_scaled


        gmm = neg_gmm + pos_gmm
        # Compute loss
        loss = torch.norm(gmm - torch.tensor(average_uv))

        # Backpropagation for each optimizer
        loss.backward()

        optimizer_max_scale.step()
        optimizer_max_amp.step()
        optimizer_min_scale.step()
        optimizer_min_amp.step()

        # print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.6f}")
        if 'V' in roman_numerals:
            # Convert input data to tensors once

            gmm_tensor = torch.tensor(gmm)

            opt_wave_4_pre_time.zero_grad()
            opt_wave_4_amp.zero_grad()
            opt_wave_4_sigma.zero_grad()

            wave_4_gaussian = 0.1*scaler * torch.nn.functional.sigmoid(wave_4_amp) * Normal(
                wave_5_latency - torch.nn.functional.sigmoid(wave_4_pre_time),
                scaler * torch.nn.functional.sigmoid(wave_4_sigma)
            ).log_prob(time_tensor).exp()

            average_uv_tensor = torch.tensor(average_uv, dtype=torch.float32)

            loss = torch.norm(gmm_tensor + wave_4_gaussian - average_uv_tensor)

            # Backpropagation
            loss.backward()
            # Step each optimizer
            opt_wave_4_pre_time.step()
            opt_wave_4_amp.step()
            opt_wave_4_sigma.step()




        gmm = gmm.detach().numpy()
        pos_gmm = pos_gmm.detach().numpy()
        neg_gmm = neg_gmm.detach().numpy()

        fixed_gmm = gmm.copy()
        colormap2 = plt.cm.hot  # You can choose other colormaps like 'plasma', 'inferno', etc.
        colormap = plt.cm.viridis

        roman_numerals_from_positive_gaussians = find_closest_within_threshold(max_time_positive,
                                                                               identified_wave_latencies)
        roman_numerals_from_negative_gaussians = find_closest_within_threshold_negative(min_time_negative,
                                                                                        identified_wave_latencies_negative)

        if epoch == 0 or epoch == (num_epochs - 1):
            positive_time_mask = (time > 0) & (time < 12)
            positive_time = time[positive_time_mask]

            smaller_window_mask = (time > identified_wave_latencies[0]) & (time < identified_wave_latencies[-1])
            small_window_time = time[smaller_window_mask]
            # Plot the points and Roman numerals
            target_indices = [np.argmin(np.abs(positive_time - t)) for t in max_time_positive]
            if DO_PLOTS:
                fig, axs = plt.subplots(2, 1, figsize=(8, 8))


                for i in range(len(min_time_negative)):
                    color = colormap(((1 + i) / (len(min_time_negative)+0.5)))

                    mu = min_time_tensor[i]
                    sigma = scaler * torch.sigmoid(min_scale[i])  # σ > 0
                    amp = y_vals_negative[i]  # observed value, target peak

                    pdf = Normal(mu, sigma).log_prob(torch.tensor(time)).exp()
                    pdf_scaled = amp * pdf / pdf.max()


                    # Calculate the Gaussian curve (negative)
                    gaussian = pdf_scaled.detach().numpy()

                    # Plot the Gaussian
                    axs[0].plot(positive_time, gaussian[positive_time_mask], color=color)

                    label = roman_numerals_from_negative_gaussians[i] if i < len(
                        roman_numerals_from_negative_gaussians) else ''

                    axs[0].plot(mu, amp, 'o', markerfacecolor='none', markeredgecolor='black')
                    axs[0].text(mu, amp - 0.03, label, ha='center', va='top', fontsize=12)
                    # note: va='top' so text is placed below marker (since curve is negative)
                # Plot Positive GMM
                for i in range(len(max_time_positive)):
                    color = colormap2((i / len(max_time_positive)))
                    mu = max_time_tensor[i]
                    sigma = scaler * torch.sigmoid(max_scale[i])  # σ > 0
                    amp = y_vals_positive[i]  # observed value, target peak

                    pdf = Normal(mu, sigma).log_prob(torch.tensor(time)).exp()
                    pdf_scaled = amp * pdf / pdf.max()

                    # Calculate the Gaussian curve
                    gaussian = pdf_scaled.detach().numpy()

                    # Plot the Gaussian
                    axs[0].plot(positive_time, gaussian[positive_time_mask], color=color)


                    label = roman_numerals_from_positive_gaussians[i] if i < len(
                        roman_numerals_from_positive_gaussians) else ''
                    axs[0].plot(mu, amp, 'o', color='black')  # point marker
                    axs[0].text(mu, amp + 0.02, label, ha='center', va='bottom', fontsize=12)

                axs[0].set_title("Decomposed GMM-FET", fontsize = ft)
                axs[0].set_ylabel("Amplitude (uV)", fontsize = ft)


            # Third graph
            if 'V' in roman_numerals:
                if DO_PLOTS:
                    axs[0].plot(positive_time, wave_4_gaussian.detach().numpy()[positive_time_mask], color='red')
                    t_val = np.array([(wave_5_latency - torch.nn.functional.sigmoid(wave_4_pre_time)).item()])
                    y_val = np.array([wave_4_gaussian.max().item()])
                    axs[0].plot(t_val, y_val, 'o', color='black')
                    axs[0].text(t_val, y_val + 0.02, 'IV', ha='center', va='bottom', fontsize=12)



                    axs[1].plot(positive_time,
                                gmm[positive_time_mask] + wave_4_gaussian.detach().numpy()[positive_time_mask],
                                label="GMM-FET")
                # MSE_list.append(np.linalg.norm(
                #     gmm[positive_time_mask] + wave_4_gaussian.detach().numpy()[positive_time_mask] +
                #     wave_6_gaussian.detach().numpy()[positive_time_mask] - average_uv[positive_time_mask]) ** 2 / (
                #                     gmm[positive_time_mask].size))
                print(np.linalg.norm(gmm[smaller_window_mask] + wave_4_gaussian.detach().numpy()[smaller_window_mask] - average_uv[
                                         smaller_window_mask]) ** 2 / (gmm[smaller_window_mask].size))

            else:
                if DO_PLOTS:
                    axs[1].plot(positive_time, gmm[positive_time_mask], label="GMM-FET")
                # print((gmm[smaller_window_mask] - average_uv[smaller_window_mask]) ** 2 / (gmm[smaller_window_mask].size))




            if DO_PLOTS:
                axs[0].axhline(y=0, color='black')
                axs[1].plot(positive_time, average_uv[positive_time_mask], label="ABR")

                axs[1].set_title("ABR vs GMM-FET", fontsize=ft)
                axs[1].set_xlabel("Time (ms)", fontsize=ft)
                axs[1].set_ylabel("Amplitude (uV)", fontsize=ft)
                axs[1].legend(fontsize=ft-4)
                axs[1].tick_params(axis='both', labelsize=ft)
                axs[0].tick_params(axis='both', labelsize=ft)
                # Match y-limits
                ymin, ymax = axs[1].get_ylim()
                axs[0].set_ylim(ymin - 0.04, ymax + 0.04)
                axs[1].set_ylim(ymin - 0.04, ymax + 0.04)

                output_path = os.path.join(output_dir, f"{participant}_epochs_{epoch}.png")
                plt.gcf().supylabel("C", fontsize=16, fontweight='bold', x=-0.08, rotation=0)
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.clf()

            if DO_PLOTS:
                fig, axs = plt.subplots(1, 1, figsize=(4.75, 3.5))

                # Third graph
            if 'V' in roman_numerals:
                if DO_PLOTS:

                    axs.plot(positive_time,
                                gmm[positive_time_mask] + wave_4_gaussian.detach().numpy()[positive_time_mask],
                                label="GMM-FET")

            else:
                if DO_PLOTS:
                    axs.plot(positive_time, gmm[positive_time_mask], label="GMM-FET")
                # print((gmm[smaller_window_mask] - average_uv[smaller_window_mask]) ** 2 / (gmm[smaller_window_mask].size))

            if DO_PLOTS:
                axs.plot(positive_time, average_uv[positive_time_mask], label="ABR")
                if stimulus == 'tb4':
                    sti = '4 kHz'
                elif stimulus == 'tb5':
                    sti = '500 Hz'
                elif stimulus == 'da':
                    sti = '/da/'
                else:
                    sti = stimulus
                axs.set_title(f"{sti} {decibel}", fontsize=ft)
                axs.set_xlabel("Time (ms)", fontsize=ft)
                # axs.set_ylabel("Amplitude (uV)", fontsize=ft)
                # axs.legend(fontsize=ft - 4)
                axs.tick_params(axis='both', labelsize=ft)
                # Match y-limits
                ymin, ymax = axs.get_ylim()
                axs.set_ylim(ymin - 0.04, ymax + 0.04)

                output_path = os.path.join(output_dir, f"{participant}_epochs_self_{epoch}.png")
                plt.savefig(output_path, dpi=300, bbox_inches="tight")
                plt.clf()

    fixed_gmm = gmm.copy()
    colormap = plt.cm.viridis  # You can choose other colormaps like 'plasma', 'inferno', etc.
    colormap2 = plt.cm.inferno

    roman_numerals_from_positive_gaussians = find_closest_within_threshold(max_time_positive, identified_wave_latencies)
    print(roman_numerals)
    # plt.plot(time, gmm)
    # plt.plot(time, average_uv)
    # plt.show()

    roman_numerals = find_closest_within_threshold(true_max_times, identified_wave_latencies)

    print(max_time_positive)
    print(true_max_times)
    print(roman_numerals)
    pos_gmm = torch.zeros_like(torch.tensor(time), dtype=torch.float32)  # Ensure tensor type
    neg_gmm = torch.zeros_like(torch.tensor(time), dtype=torch.float32)  # Ensure tensor type

    error = gmm[positive_time_mask] - average_uv[positive_time_mask]
    MSE = np.mean(error ** 2)
    RMSE = np.sqrt(MSE)
    normalized_RMSE = RMSE / (np.max(average_uv[positive_time_mask]) - np.min(average_uv[positive_time_mask]))

    MSE_list.append(normalized_RMSE)

    error = gmm[smaller_window_mask] - average_uv[smaller_window_mask]
    MSE = np.mean(error ** 2)
    RMSE = np.sqrt(MSE)
    normalized_RMSE = RMSE / (np.max(average_uv[smaller_window_mask]) - np.min(average_uv[smaller_window_mask]))
    MSE_list_smaller_window.append(normalized_RMSE)



    if 'I' in roman_numerals:
        I_index = np.where(roman_numerals == 'I')[0][0]
        wave_I_latency_list.append(true_max_times[I_index])
        index = np.argmin(np.abs(time - true_max_times[I_index]))
        height = gmm[index]

        if 'I' not in roman_numerals_from_positive_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_max_times[I_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
            height = gmm[index]
            MSE_Wave_I.append(None)

        else:
            I_index = np.where(roman_numerals_from_positive_gaussians == 'I')[0][0]
            curvature = (-torch.nn.functional.sigmoid(max_amp[I_index]) / (
                    scaler * torch.nn.functional.sigmoid(max_scale[I_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(max_amp[I_index]) * torch.nn.functional.sigmoid(
                max_scale[I_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(max_scale[I_index])).detach().item()
            pos_gmm += scaler * torch.nn.functional.sigmoid(max_amp[I_index]) * Normal(max_time_tensor[I_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           max_scale[
                                                                                               I_index])).log_prob(
                torch.tensor(time)).exp()
            height = (torch.nn.functional.sigmoid(max_amp[I_index]) / (
                    torch.nn.functional.sigmoid(max_scale[I_index]) * np.sqrt(2 * np.pi))).detach().item()
            wave_I_latency = true_max_times[I_index]

            wave_I_mask = (
                    (time >= wave_I_latency - 2 * sigma) &
                    (time <= wave_I_latency + 2 * sigma)
            )
            error_I = gmm[wave_I_mask] - average_uv[wave_I_mask]
            MSE = np.mean(error_I ** 2)
            RMSE = np.sqrt(MSE)
            normalized_RMSE = RMSE / (np.max(average_uv[wave_I_mask]) - np.min(average_uv[wave_I_mask]))
            MSE_Wave_I.append(normalized_RMSE)
        wave_I_curvature_list.append(curvature)
        wave_I_sigma_list.append(sigma)
        wave_I_autc_list.append(autc)
        wave_I_height_list.append(height)

    else:
        wave_I_latency_list.append(identified_wave_latencies[0])
        wave_I_height_list.append(0.0)
        wave_I_curvature_list.append(-1.0)
        wave_I_sigma_list.append(1.0)
        wave_I_autc_list.append(1.0)
        MSE_Wave_I.append(None)

    if 'II' in roman_numerals:
        II_index = np.where(roman_numerals == 'II')[0][0]
        wave_II_latency_list.append(true_max_times[II_index])
        index = np.argmin(np.abs(time - true_max_times[II_index]))
        height = gmm[index]

        if 'II' not in roman_numerals_from_positive_gaussians:
            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_max_times[II_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
            height = gmm[index]

        else:
            II_index = np.where(roman_numerals_from_positive_gaussians == 'II')[0][0]
            curvature = (-torch.nn.functional.sigmoid(max_amp[II_index]) / (
                    scaler * torch.nn.functional.sigmoid(max_scale[II_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(max_amp[II_index]) * torch.nn.functional.sigmoid(
                max_scale[II_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(max_scale[II_index])).detach().item()
            pos_gmm += scaler * torch.nn.functional.sigmoid(max_amp[II_index]) * Normal(max_time_tensor[II_index],
                                                                                        scaler * torch.nn.functional.sigmoid(
                                                                                            max_scale[
                                                                                                II_index])).log_prob(
                torch.tensor(time)).exp()
            height = (torch.nn.functional.sigmoid(max_amp[II_index]) / (
                    torch.nn.functional.sigmoid(max_scale[II_index]) * np.sqrt(2 * np.pi))).detach().item()
        wave_II_height_list.append(height)
        wave_II_curvature_list.append(curvature)
        wave_II_sigma_list.append(sigma)
        wave_II_autc_list.append(autc)
    else:
        wave_II_latency_list.append(identified_wave_latencies[1])
        wave_II_height_list.append(0.0)
        wave_II_curvature_list.append(-1.0)
        wave_II_sigma_list.append(1.0)
        wave_II_autc_list.append(1.0)

    if 'III' in roman_numerals:
        III_index = np.where(roman_numerals == 'III')[0][0]
        wave_III_latency_list.append(true_max_times[III_index])
        index = np.argmin(np.abs(time - true_max_times[III_index]))
        height = gmm[index]

        if 'III' not in roman_numerals_from_positive_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_max_times[III_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            height = gmm[index]
            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
            MSE_Wave_III.append(None)
        else:
            III_index = np.where(roman_numerals_from_positive_gaussians == 'III')[0][0]
            curvature = (-torch.nn.functional.sigmoid(max_amp[III_index]) / (
                    scaler * torch.nn.functional.sigmoid(max_scale[III_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(max_amp[III_index]) * torch.nn.functional.sigmoid(
                max_scale[III_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(max_scale[III_index])).detach().item()
            pos_gmm += scaler * torch.nn.functional.sigmoid(max_amp[III_index]) * Normal(max_time_tensor[III_index],
                                                                                         scaler * torch.nn.functional.sigmoid(
                                                                                             max_scale[
                                                                                                 III_index])).log_prob(
                torch.tensor(time)).exp()
            height = (torch.nn.functional.sigmoid(max_amp[III_index]) / (
                    torch.nn.functional.sigmoid(max_scale[III_index]) * np.sqrt(2 * np.pi))).detach().item()
            wave_III_latency = true_max_times[III_index]

            wave_III_mask = (
                    (time >= wave_III_latency - 2 * sigma) &
                    (time <= wave_III_latency + 2 * sigma)
            )
            error_III = gmm[wave_III_mask] - average_uv[wave_III_mask]
            MSE = np.mean(error_III ** 2)
            RMSE = np.sqrt(MSE)
            normalized_RMSE = RMSE / (np.max(average_uv[wave_III_mask]) - np.min(average_uv[wave_III_mask]))
            MSE_Wave_III.append(normalized_RMSE)
        wave_III_height_list.append(height)
        wave_III_curvature_list.append(curvature)
        wave_III_sigma_list.append(sigma)
        wave_III_autc_list.append(autc)
    else:
        wave_III_latency_list.append(identified_wave_latencies[2])
        wave_III_height_list.append(0.0)
        wave_III_curvature_list.append(-1.0)
        wave_III_sigma_list.append(1.0)
        wave_III_autc_list.append(1.0)
        MSE_Wave_III.append(None)

    if 'V' in roman_numerals:
        V_index = np.where(roman_numerals == 'V')[0][0]
        wave_V_latency_list.append(true_max_times[V_index])
        index = np.argmin(np.abs(time - true_max_times[V_index]))
        height = gmm[index]



        if 'V' not in roman_numerals_from_positive_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_max_times[V_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
            height = gmm[index]

            wave_IV_latency_list.append(identified_wave_latencies[4])
            wave_IV_height_list.append(0.0)
            wave_IV_curvature_list.append(-1.0)
            wave_IV_sigma_list.append(1.0)
            wave_IV_autc_list.append(1.0)
            MSE_Wave_V.append(None)
        else:
            V_index = np.where(roman_numerals_from_positive_gaussians == 'V')[0][0]
            curvature = (-torch.nn.functional.sigmoid(max_amp[V_index]) / (
                    scaler * torch.nn.functional.sigmoid(max_scale[V_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(max_amp[V_index]) * torch.nn.functional.sigmoid(
                max_scale[V_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(max_scale[V_index])).detach().item()
            pos_gmm += scaler * torch.nn.functional.sigmoid(max_amp[V_index]) * Normal(max_time_tensor[V_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           max_scale[
                                                                                               V_index])).log_prob(
                torch.tensor(time)).exp()
            height = (torch.nn.functional.sigmoid(max_amp[V_index]) / (
                    torch.nn.functional.sigmoid(max_scale[V_index]) * np.sqrt(2 * np.pi))).detach().item()
            wave_IV_height_list.append(
                (torch.sigmoid(wave_4_amp) / (np.sqrt(2 * np.pi) * torch.sigmoid(wave_4_sigma))).detach().item())
            wave_IV_latency_list.append((wave_5_latency - torch.nn.functional.sigmoid(wave_4_pre_time)).detach().item())
            wave_IV_curvature_list.append((-torch.nn.functional.sigmoid(wave_4_amp) / (
                    scaler * torch.nn.functional.sigmoid(wave_4_sigma) ** 2)).detach().item())
            wave_IV_sigma_list.append(torch.nn.functional.sigmoid(wave_4_sigma).detach().item())
            wave_IV_autc_list.append((scaler ** 2 * torch.nn.functional.sigmoid(
                wave_4_amp) * torch.nn.functional.sigmoid(wave_4_sigma) * np.sqrt(2 * np.pi)).detach().item())
            wave_V_latency = true_max_times[V_index]

            wave_V_mask = (
                    (time >= wave_V_latency - 2 * sigma) &
                    (time <= wave_V_latency + 2 * sigma)
            )
            error_V = gmm[wave_V_mask] - average_uv[wave_V_mask]
            MSE = np.mean(error_V ** 2)
            RMSE = np.sqrt(MSE)
            normalized_RMSE = RMSE / (np.max(average_uv[wave_V_mask]) - np.min(average_uv[wave_V_mask]))
            MSE_Wave_V.append(normalized_RMSE)
        wave_V_curvature_list.append(curvature)
        wave_V_sigma_list.append(sigma)
        wave_V_autc_list.append(autc)
        wave_V_height_list.append(height)
    else:
        wave_V_latency_list.append(identified_wave_latencies[4])
        wave_V_height_list.append(0.0)
        wave_V_curvature_list.append(-1.0)
        wave_V_sigma_list.append(1.0)
        wave_V_autc_list.append(1.0)
        wave_IV_latency_list.append(identified_wave_latencies[4])
        wave_IV_height_list.append(0.0)
        wave_IV_curvature_list.append(-1.0)
        wave_IV_sigma_list.append(1.0)
        wave_IV_autc_list.append(1.0)
        MSE_Wave_V.append(None)

    if 'VI' in roman_numerals:
        VI_index = np.where(roman_numerals == 'VI')[0][0]
        wave_VI_latency_list.append(true_max_times[VI_index])
        index = np.argmin(np.abs(time - true_max_times[VI_index]))
        height = gmm[index]

        if 'VI' not in roman_numerals_from_positive_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_max_times[VI_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
            height = gmm[index]
        else:
            VI_index = np.where(roman_numerals_from_positive_gaussians == 'VI')[0][0]
            curvature = (-torch.nn.functional.sigmoid(max_amp[VI_index]) / (
                    scaler * torch.nn.functional.sigmoid(max_scale[VI_index] ** 2))).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(max_amp[VI_index]) * torch.nn.functional.sigmoid(
                max_scale[VI_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(max_scale[VI_index])).detach().item()
            pos_gmm += scaler * torch.nn.functional.sigmoid(max_amp[VI_index]) * Normal(max_time_tensor[VI_index],
                                                                                        scaler * torch.nn.functional.sigmoid(
                                                                                            max_scale[
                                                                                                VI_index])).log_prob(
                torch.tensor(time)).exp()
            height = (torch.nn.functional.sigmoid(max_amp[VI_index]) / (
                    torch.nn.functional.sigmoid(max_scale[VI_index]) * np.sqrt(2 * np.pi))).detach().item()

        wave_VI_curvature_list.append(curvature)
        wave_VI_sigma_list.append(sigma)
        wave_VI_autc_list.append(autc)
        wave_VI_height_list.append(height)
    else:
        wave_VI_latency_list.append(identified_wave_latencies[6])
        wave_VI_height_list.append(0.0)
        wave_VI_curvature_list.append(-1.0)
        wave_VI_sigma_list.append(1.0)
        wave_VI_autc_list.append(1.0)

    if 'VII' in roman_numerals:
        VII_index = np.where(roman_numerals == 'VII')[0][0]
        wave_VII_latency_list.append(true_max_times[VII_index])
        index = np.argmin(np.abs(time - true_max_times[VII_index]))
        height = gmm[index]

        if 'VII' not in roman_numerals_from_positive_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_max_times[VII_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
            height = gmm[index]
        else:
            VII_index = np.where(roman_numerals_from_positive_gaussians == 'VII')[0][0]
            curvature = (-torch.nn.functional.sigmoid(max_amp[VII_index]) / (
                    scaler * torch.nn.functional.sigmoid(max_scale[VII_index] ** 2))).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(max_amp[VII_index]) * torch.nn.functional.sigmoid(
                max_scale[VII_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(max_scale[VII_index])).detach().item()
            pos_gmm += scaler * torch.nn.functional.sigmoid(max_amp[VII_index]) * Normal(max_time_tensor[VII_index],
                                                                                         scaler * torch.nn.functional.sigmoid(
                                                                                             max_scale[
                                                                                                 VII_index])).log_prob(
                torch.tensor(time)).exp()
            height = (torch.nn.functional.sigmoid(max_amp[VII_index]) / (
                    torch.nn.functional.sigmoid(max_scale[VII_index]) * np.sqrt(2 * np.pi))).detach().item()

        wave_VII_curvature_list.append(curvature)
        wave_VII_sigma_list.append(sigma)
        wave_VII_autc_list.append(autc)
        wave_VII_height_list.append(height)
    else:
        wave_VII_latency_list.append(identified_wave_latencies[6])
        wave_VII_height_list.append(0.0)
        wave_VII_curvature_list.append(-1.0)
        wave_VII_sigma_list.append(1.0)
        wave_VII_autc_list.append(1.0)

    roman_numerals_from_negative_gaussians = find_closest_within_threshold_negative(min_time_negative,
                                                                                    identified_wave_latencies_negative)

    roman_numerals = find_closest_within_threshold_negative(true_min_times, identified_wave_latencies_negative)
    print(min_time_negative)
    print(true_min_times)
    print(roman_numerals)
    if 'nI' in roman_numerals:
        A_index = np.where(roman_numerals == 'nI')[0][0]
        wave_A_latency_list.append(true_min_times[A_index])
        index = np.argmin(np.abs(time - true_min_times[A_index]))
        height = gmm[index]
        wave_A_height_list.append(height)

        if 'nI' not in roman_numerals_from_negative_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_min_times[A_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
        else:
            A_index = np.where(roman_numerals_from_negative_gaussians == 'nI')[0][0]
            curvature = (-torch.nn.functional.sigmoid(min_amp[A_index]) / (
                    scaler * torch.nn.functional.sigmoid(min_scale[A_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(min_amp[A_index]) * torch.nn.functional.sigmoid(
                min_scale[A_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(min_scale[A_index])).detach().item()
            neg_gmm -= scaler * torch.nn.functional.sigmoid(min_amp[A_index]) * Normal(min_time_tensor[A_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           min_scale[
                                                                                               A_index])).log_prob(
                torch.tensor(time)).exp()
        wave_A_curvature_list.append(curvature)
        wave_A_sigma_list.append(sigma)
        wave_A_autc_list.append(autc)

    else:
        wave_A_latency_list.append(identified_wave_latencies[0])
        wave_A_height_list.append(0.0)
        wave_A_curvature_list.append(1.0)
        wave_A_sigma_list.append(1.0)
        wave_A_autc_list.append(1.0)

    if 'nII' in roman_numerals:
        B_index = np.where(roman_numerals == 'nII')[0][0]
        wave_B_latency_list.append(true_min_times[B_index])
        index = np.argmin(np.abs(time - true_min_times[B_index]))
        height = gmm[index]
        wave_B_height_list.append(height)

        if 'nII' not in roman_numerals_from_negative_gaussians:
            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_min_times[B_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
        else:
            B_index = np.where(roman_numerals_from_negative_gaussians == 'nII')[0][0]
            curvature = (-torch.nn.functional.sigmoid(min_amp[B_index]) / (
                    scaler * torch.nn.functional.sigmoid(min_scale[B_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(min_amp[B_index]) * torch.nn.functional.sigmoid(
                min_scale[B_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(min_scale[B_index])).detach().item()
            neg_gmm -= scaler * torch.nn.functional.sigmoid(min_amp[B_index]) * Normal(min_time_tensor[B_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           min_scale[
                                                                                               B_index])).log_prob(
                torch.tensor(time)).exp()
        wave_B_curvature_list.append(curvature)
        wave_B_sigma_list.append(sigma)
        wave_B_autc_list.append(autc)
    else:
        wave_B_latency_list.append(identified_wave_latencies[1])
        wave_B_height_list.append(0.0)
        wave_B_curvature_list.append(1.0)
        wave_B_sigma_list.append(1.0)
        wave_B_autc_list.append(1.0)

    if 'nIII' in roman_numerals:
        C_index = np.where(roman_numerals == 'nIII')[0][0]
        wave_C_latency_list.append(true_min_times[C_index])
        index = np.argmin(np.abs(time - true_min_times[C_index]))
        height = gmm[index]
        wave_C_height_list.append(gmm[index])

        if 'nIII' not in roman_numerals_from_negative_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_min_times[C_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
        else:
            C_index = np.where(roman_numerals_from_negative_gaussians == 'nIII')[0][0]
            curvature = (-torch.nn.functional.sigmoid(min_amp[C_index]) / (
                    scaler * torch.nn.functional.sigmoid(min_scale[C_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(min_amp[C_index]) * torch.nn.functional.sigmoid(
                min_scale[C_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(min_scale[C_index])).detach().item()
            neg_gmm -= scaler * torch.nn.functional.sigmoid(min_amp[C_index]) * Normal(min_time_tensor[C_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           min_scale[
                                                                                               C_index])).log_prob(
                torch.tensor(time)).exp()
        wave_C_curvature_list.append(curvature)
        wave_C_sigma_list.append(sigma)
        wave_C_autc_list.append(autc)
    else:
        wave_C_latency_list.append(identified_wave_latencies[2])
        wave_C_height_list.append(0.0)
        wave_C_curvature_list.append(1.0)
        wave_C_sigma_list.append(1.0)
        wave_C_autc_list.append(1.0)

    if 'nIV' in roman_numerals:
        D_index = np.where(roman_numerals == 'nIV')[0][0]
        wave_D_latency_list.append(true_min_times[D_index])
        index = np.argmin(np.abs(time - true_min_times[D_index]))
        height = gmm[index]
        wave_D_height_list.append(gmm[index])

        if 'nIV' not in roman_numerals_from_negative_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_min_times[D_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
        else:
            D_index = np.where(roman_numerals_from_negative_gaussians == 'nIV')[0][0]
            curvature = (-torch.nn.functional.sigmoid(min_amp[D_index]) / (
                    scaler * torch.nn.functional.sigmoid(min_scale[D_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(min_amp[D_index]) * torch.nn.functional.sigmoid(
                min_scale[D_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(min_scale[D_index])).detach().item()
            neg_gmm -= scaler * torch.nn.functional.sigmoid(min_amp[D_index]) * Normal(min_time_tensor[D_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           min_scale[
                                                                                               D_index])).log_prob(
                torch.tensor(time)).exp()
        wave_D_curvature_list.append(curvature)
        wave_D_sigma_list.append(sigma)
        wave_D_autc_list.append(autc)
    else:
        wave_D_latency_list.append(identified_wave_latencies[2])
        wave_D_height_list.append(0.0)
        wave_D_curvature_list.append(1.0)
        wave_D_sigma_list.append(1.0)
        wave_D_autc_list.append(1.0)

    if 'nV' in roman_numerals:
        E_index = np.where(roman_numerals == 'nV')[0][0]
        wave_E_latency_list.append(true_min_times[E_index])
        index = np.argmin(np.abs(time - true_min_times[E_index]))
        height = gmm[index]
        wave_E_height_list.append(gmm[index])

        if 'nV' not in roman_numerals_from_negative_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_min_times[E_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
        else:
            E_index = np.where(roman_numerals_from_negative_gaussians == 'nV')[0][0]
            curvature = (-torch.nn.functional.sigmoid(min_amp[E_index]) / (
                    scaler * torch.nn.functional.sigmoid(min_scale[E_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(min_amp[E_index]) * torch.nn.functional.sigmoid(
                min_scale[E_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(min_scale[E_index])).detach().item()
            neg_gmm -= scaler * torch.nn.functional.sigmoid(min_amp[E_index]) * Normal(min_time_tensor[E_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           min_scale[
                                                                                               E_index])).log_prob(
                torch.tensor(time)).exp()
        wave_E_curvature_list.append(curvature)
        wave_E_sigma_list.append(sigma)
        wave_E_autc_list.append(autc)
    else:
        wave_E_latency_list.append(identified_wave_latencies[4])
        wave_E_height_list.append(0.0)
        wave_E_curvature_list.append(1.0)
        wave_E_sigma_list.append(1.0)
        wave_E_autc_list.append(1.0)

    if 'nVI' in roman_numerals:
        F_index = np.where(roman_numerals == 'nVI')[0][0]
        wave_F_latency_list.append(true_min_times[F_index])
        index = np.argmin(np.abs(time - true_min_times[F_index]))
        height = gmm[index]
        wave_F_height_list.append(gmm[index])

        if 'nVI' not in roman_numerals_from_negative_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_min_times[F_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
        else:
            F_index = np.where(roman_numerals_from_negative_gaussians == 'nVI')[0][0]
            curvature = (-torch.nn.functional.sigmoid(min_amp[F_index]) / (
                    scaler * torch.nn.functional.sigmoid(min_scale[F_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * torch.nn.functional.sigmoid(min_amp[F_index]) * torch.nn.functional.sigmoid(
                min_scale[F_index]) * np.sqrt(2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(min_scale[F_index])).detach().item()
            neg_gmm -= scaler * torch.nn.functional.sigmoid(min_amp[F_index]) * Normal(min_time_tensor[F_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           min_scale[
                                                                                               F_index])).log_prob(
                torch.tensor(time)).exp()
        wave_F_curvature_list.append(curvature)
        wave_F_sigma_list.append(sigma)
        wave_F_autc_list.append(autc)
    else:
        wave_F_latency_list.append(identified_wave_latencies[2])
        wave_F_height_list.append(0.0)
        wave_F_curvature_list.append(1.0)
        wave_F_sigma_list.append(1.0)
        wave_F_autc_list.append(1.0)

    if 'nVII' in roman_numerals:
        G_index = np.where(roman_numerals == 'nVII')[0][0]
        wave_G_latency_list.append(true_min_times[G_index])
        index = np.argmin(np.abs(time - true_min_times[G_index]))
        wave_G_height_list.append(gmm[index])

        if 'nVII' not in roman_numerals_from_negative_gaussians:

            # Step 1: Compute absolute differences
            diffs = np.abs(time - true_min_times[G_index])

            # Step 2: Get indices of the 3 smallest differences
            nearest_indices = np.argpartition(diffs, 3)[:3]

            # Optional: sort the indices by actual closeness
            nearest_indices = nearest_indices[np.argsort(diffs[nearest_indices])]
            y_values = np.abs(average_uv[nearest_indices])

            curvature = (y_values[0] - 2 * y_values[1] + y_values[2]) / delta_t ** 2
            sigma = np.sqrt(np.abs(height / curvature))
            autc = height * sigma * np.sqrt(2 * np.pi)
        else:
            G_index = np.where(roman_numerals_from_negative_gaussians == 'nVII')[0][0]
            curvature = (-torch.nn.functional.sigmoid(min_amp[G_index]) / (
                    scaler * torch.nn.functional.sigmoid(min_scale[G_index]) ** 2)).detach().item()
            autc = (scaler ** 2 * min_amp[G_index] * torch.nn.functional.sigmoid(min_scale[G_index]) * np.sqrt(
                2 * np.pi)).detach().item()
            sigma = (scaler * torch.nn.functional.sigmoid(min_scale[G_index])).detach().item()
            neg_gmm -= scaler * torch.nn.functional.sigmoid(min_amp[G_index]) * Normal(min_time_tensor[G_index],
                                                                                       scaler * torch.nn.functional.sigmoid(
                                                                                           min_scale[
                                                                                               G_index])).log_prob(
                torch.tensor(time)).exp()
        wave_G_curvature_list.append(curvature)
        wave_G_sigma_list.append(sigma)
        wave_G_autc_list.append(autc)
    else:
        wave_G_latency_list.append(identified_wave_latencies[6])
        wave_G_height_list.append(0.0)
        wave_G_curvature_list.append(1.0)
        wave_G_sigma_list.append(1.0)
        wave_G_autc_list.append(1.0)
    plt.close('all')
    # plt.plot(time, (neg_gmm+pos_gmm).detach().numpy(), color='blue')
    # plt.plot(time, gmm, color='orange')
    # plt.plot(time, average_uv, color='red')
    # plt.show()

data_allen_project_Wave_latency = {
    "Participant": participant_list,
    "Decibel": decibel_list,
    "Visit": visit_list,
    "Run": run_list,
    "Stimulus": stimulus_list,
    "MSE_first_wave_to_last_wave": MSE_list_smaller_window,
    "NRMSE_wave_V": MSE_Wave_V,
    "NRMSE_wave_I": MSE_Wave_I,
    "NRMSE_wave_III": MSE_Wave_III,

    # Latencies
    "WaveIlatency": wave_I_latency_list,
    "WaveIIlatency": wave_II_latency_list,
    "WaveIIIlatency": wave_III_latency_list,
    "WaveIVlatency": wave_IV_latency_list,
    "WaveVlatency": wave_V_latency_list,
    "WaveVIlatency": wave_VI_latency_list,
    "WaveVIIlatency": wave_VII_latency_list,

    # Heights
    "WaveIheight": wave_I_height_list,
    "WaveIIheight": wave_II_height_list,
    "WaveIIIheight": wave_III_height_list,
    "WaveIVheight": wave_IV_height_list,
    "WaveVheight": wave_V_height_list,
    "WaveVIheight": wave_VI_height_list,
    "WaveVIIheight": wave_VII_height_list,

    # Sigmas (Standard Deviation)
    "WaveIsigma": wave_I_sigma_list,
    "WaveIIsigma": wave_II_sigma_list,
    "WaveIIIsigma": wave_III_sigma_list,
    "WaveIVsigma": wave_IV_sigma_list,
    "WaveVsigma": wave_V_sigma_list,
    "WaveVIsigma": wave_VI_sigma_list,
    "WaveVIIsigma": wave_VII_sigma_list,

    # AUTC (Area Under the Curve)
    "WaveIautc": wave_I_autc_list,
    "WaveIIautc": wave_II_autc_list,
    "WaveIIIautc": wave_III_autc_list,
    "WaveIVautc": wave_IV_autc_list,
    "WaveVautc": wave_V_autc_list,
    "WaveVIautc": wave_VI_autc_list,
    "WaveVIIautc": wave_VII_autc_list,

    # Curvature
    "WaveIcurvature": wave_I_curvature_list,
    "WaveIIcurvature": wave_II_curvature_list,
    "WaveIIIcurvature": wave_III_curvature_list,
    "WaveIVcurvature": wave_IV_curvature_list,
    "WaveVcurvature": wave_V_curvature_list,
    "WaveVIcurvature": wave_VI_curvature_list,
    "WaveVIIcurvature": wave_VII_curvature_list,

    # Latencies
    "WaveAlatency": wave_A_latency_list,
    "WaveBlatency": wave_B_latency_list,
    "WaveClatency": wave_C_latency_list,
    "WaveDlatency": wave_D_latency_list,
    "WaveElatency": wave_E_latency_list,
    "WaveFlatency": wave_F_latency_list,
    "WaveGlatency": wave_G_latency_list,

    # Heights
    "WaveAheight": wave_A_height_list,
    "WaveBheight": wave_B_height_list,
    "WaveCheight": wave_C_height_list,
    "WaveDheight": wave_D_height_list,
    "WaveEheight": wave_E_height_list,
    "WaveFheight": wave_F_height_list,
    "WaveGheight": wave_G_height_list,

    # Sigmas (Standard Deviation)
    "WaveAsigma": wave_A_sigma_list,
    "WaveBsigma": wave_B_sigma_list,
    "WaveCsigma": wave_C_sigma_list,
    "WaveDsigma": wave_D_sigma_list,
    "WaveEsigma": wave_E_sigma_list,
    "WaveFsigma": wave_F_sigma_list,
    "WaveGsigma": wave_G_sigma_list,

    # AUTC (Area Under the Curve)
    "WaveAautc": wave_A_autc_list,
    "WaveBautc": wave_B_autc_list,
    "WaveCautc": wave_C_autc_list,
    "WaveDautc": wave_D_autc_list,
    "WaveEautc": wave_E_autc_list,
    "WaveFautc": wave_F_autc_list,
    "WaveGautc": wave_G_autc_list,

    # Curvature
    "WaveAcurvature": wave_A_curvature_list,
    "WaveBcurvature": wave_B_curvature_list,
    "WaveCcurvature": wave_C_curvature_list,
    "WaveDcurvature": wave_D_curvature_list,
    "WaveEcurvature": wave_E_curvature_list,
    "WaveFcurvature": wave_F_curvature_list,
    "WaveGcurvature": wave_G_curvature_list,
}
for key, value in data_allen_project_Wave_latency.items():
    print(f"{key}: {len(value)}")
df = pd.DataFrame(data_allen_project_Wave_latency)

# def unwrap_single_element_array(val):
#     if isinstance(val, np.ndarray) or isinstance(val, list):
#         return float(val.item())
#     return val
#
#
# # Apply the unwrapping to each list in the dictionary
# for key, value in data_allen_project_Wave_latency.items():
#     data_allen_project_Wave_latency[key] = [unwrap_single_element_array(v) for v in value]

df.to_csv("explore_exponential.csv", index=False)