import numpy as np
from scipy.signal import detrend
import matplotlib.pyplot as plt
import os
# Importing NewmarkIntegation and BaselineCorrection from pylib_gm_proc
# pylib_gm_proc is part of the GM-GANO project available at:
# https://github.com/yzshi5/GM-GANO/blob/main/Python_libs/pylib_gm_proc.py
from pylib_gm_proc import NewmarkIntegation, BaselineCorrection
# Set the directory containing the extracted files
extracted_files_dir = 'Untitled Folder 2'

if not os.path.exists(extracted_files_dir):
    raise FileNotFoundError(f"The directory {extracted_files_dir} does not exist.")

# Get all TXT files in the directory
txt_files = [os.path.join(root, file)
             for root, dirs, files in os.walk(extracted_files_dir)
             for file in files if file.endswith('.txt')]

if not txt_files:
    raise FileNotFoundError("No .txt files found in the specified directory.")

# Baseline correction function
def baseline_correction(time, waveform):
    """
    Baseline correction method using a 7th-order polynomial to fit the data,
    ensuring that motion (acceleration, velocity, displacement) stops at the end.
    """
    try:
        _, _, vel_nm, disp_nm = NewmarkIntegation(time, waveform, int_type='midle point')
        _, acc_bs, vel_bs, disp_bs = BaselineCorrection(time, vel_nm, disp_nm, n=7, f_taper_beg=0.05, f_taper_end=0.05)
        return acc_bs
    except Exception as e:
        raise RuntimeError(f"Error during baseline correction: {e}")

# Initialize a counter for processed files
file_count = 0

# Process each TXT file
for data_file in txt_files:
    try:
        # Load data from the file, read the second column
        data = np.loadtxt(data_file, usecols=1)
    except Exception as e:
        print(f"Error loading data from {data_file}: {e}")
        continue

    waveform = data  # Seismic waveform data

    ndim = len(waveform)  # Get the length of the data
    time_step = 0.01  # Assume a sampling rate of 100 Hz
    times = np.arange(ndim) * time_step

    try:
        # Detrend the waveform
        waveform_detrended = detrend(waveform, type='constant')  # Use constant detrend

        # Apply baseline correction
        waveform_corrected = baseline_correction(times, waveform_detrended)

        # Save the corrected data to a new file with "_corrected1" suffix
        base_name = os.path.basename(data_file)
        name, ext = os.path.splitext(base_name)
        output_file = os.path.join(os.path.dirname(data_file), f'{name}_corrected1{ext}')
        np.savetxt(output_file, waveform_corrected, fmt='%.6f')
        print(f"Corrected waveform saved to {output_file}")

        # Increment the file counter
        file_count += 1
    except Exception as e:
        print(f"Error processing {data_file}: {e}")
        continue

# Print the total number of processed files
print(f"Total number of files processed: {file_count}")
