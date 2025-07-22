import time
import tkinter as tk
from tkinter import filedialog
import os
import numpy as np
from pathlib import Path
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
from matplotlib import colormaps, gridspec
from matplotlib.ticker import LogLocator
import matplotlib.image as mpimg
from matplotlib.lines import Line2D
import matplotlib.cm as cm
from matplotlib.offsetbox import AnchoredText
from pylibCZIrw import czi as pyczi
import fastddm as fddm
import fastddm.mask as mask
from fastddm.fit import simple_structure_function, fit
from fastddm.intermediatescatteringfunction import ISFReader, azavg2isf_estimate
from fastddm.fit import fit_multik
from fastddm.fit_models import simple_exponential_model as model 
from lmfit import Model 
import lmfit
import warnings
import csv
from skimage.color import rgb2gray
from skimage.io import imread

# Set pixel size, frame rate, and test q values
pixel_size = 1  # µm/px
frame_rate = 1  # frames/s 
test_k = np.array([3, 4, 5, 6, 7], dtype=np.int64) # Specify the test q values you would like to analyze

def select_folder():
    """Prompt the user to select a folder containing TIF files."""
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    folder_path = filedialog.askdirectory(title="Select the folder containing the TIF files")
    if not folder_path:
        raise ValueError("No folder selected.")
    parent_folder = os.path.basename(folder_path)
    return folder_path, parent_folder

def load_images(folder_path):
    """Load and normalize all grayscale or RGB .tif images from the given folder."""
    folder_path = Path(folder_path)
    image_list = []

    for im in sorted(folder_path.iterdir()):
        if im.name.endswith(".tif"):
            img = imread(str(im))

            if img.ndim == 4:  # Multi-frame RGB (unlikely, but possible)
                img = np.mean(img, axis=-1)  # Convert to grayscale
            elif img.ndim == 3:
                if img.shape[-1] == 3:  # RGB image
                    img = rgb2gray(img)
                elif img.shape[0] > 10 and img.shape[-1] != 3:  # Likely multi-frame grayscale
                    for i in range(img.shape[0]):
                        image_list.append(img[i])
                    continue  # Skip appending entire stack
                # Else: single grayscale frame (H, W)

            image_list.append(img)

    images = np.array(image_list)
    images = np.ascontiguousarray(images)
    print("Loaded images shape:", images.shape)
    return images

def create_results_folder(folder_path):
    """Create a 'Results' folder inside the selected folder."""
    results_folder = os.path.join(folder_path, "Results")
    os.makedirs(results_folder, exist_ok=True)
    return results_folder

def process_tif_files_in_folder(folder_path, results_folder):
    """Process all tif files in the given folder and calculate or load azimuthal averages."""
    print(f"Processing folder: {folder_path}")
    dqt_list = []
    aa_list = []
    base_filenames = []

    # Look for _average.aa.ddm files first (only one expected per subfolder)
    average_aa_files = sorted([f for f in os.listdir(folder_path) if f.endswith('_averaged.aa.ddm')])

    # If we find an _averaged.aa.ddm file, load it and skip other files
    if average_aa_files:
        for aa_file in average_aa_files:
            aa_file_path = os.path.join(folder_path, aa_file)
            base_filename = os.path.splitext(aa_file)[0]
            print(f"Loading precomputed average azimuthal average from {aa_file}")
            aa = fddm.load(aa_file_path)
            aa_list.append(aa)
            base_filenames.append(base_filename)
        return dqt_list, aa_list, base_filenames 

    # If no _averaged.aa.ddm, check for .aa.ddm files (multiple expected per subfolder)
    aa_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.aa.ddm') and not f.endswith('_averaged.aa.ddm')])

    if aa_files:
        for aa_file in aa_files:
            aa_file_path = os.path.join(folder_path, aa_file)
            base_filename = os.path.splitext(aa_file)[0]
            print(f"Loading precomputed azimuthal average from {aa_file}")
            aa = fddm.load(aa_file_path)
            aa_list.append(aa)
            base_filenames.append(base_filename)
            
        # Resample and average the azimuthal averages if multiple .aa.ddm files are found
        if len(aa_list) > 1:
            print(f"Resampling and averaging azimuthal averages in {folder_path}...")
            aa_list = average_azimuthal_averages(aa_list, os.path.basename(folder_path), folder_path)

        return dqt_list, aa_list, base_filenames  

    # If no .aa.ddm, check for .sf.ddm files (multiple expected per subfolder)
    sf_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.sf.ddm')])

    if sf_files:
        for sf_file in sf_files:
            sf_file_path = os.path.join(folder_path, sf_file)
            base_filename = os.path.splitext(sf_file)[0]
            print(f"Loading precomputed structure function from {sf_file}")
            dqt = fddm.load(sf_file_path)
            bin_size = dqt.ky[1] - dqt.ky[0]
            bins = np.arange(int(dqt.ky[-1] / bin_size) + 1) * bin_size
            cc_mask = mask.central_cross_mask(dqt.shape[1:])
            aa = fddm.azimuthal_average(dqt, bins=bins, mask=cc_mask)
            aa_list.append(aa)

            # Save the azimuthal average
            aa_file_path = os.path.join(folder_path, f"{base_filename}.aa.ddm")
            aa.save(aa_file_path)

            # Append to dqt_list and base_filenames
            dqt_list.append(dqt)
            base_filenames.append(base_filename)

        return dqt_list, aa_list, base_filenames  

    # If no .sf.ddm, process .tif files (multiple expected per subfolder)
    tif_files = sorted([f for f in os.listdir(folder_path) if f.endswith('.tif')])

    if not tif_files:
        print(f"No files found for processing in folder: {folder_path}")
        return dqt_list, aa_list, base_filenames

    for n, tif_file in enumerate(tif_files):
        tif_file_path = os.path.join(folder_path, tif_file)
        base_filename = os.path.splitext(tif_file)[0]
        sf_file_path = os.path.join(folder_path, f"{base_filename}.sf.ddm")
        aa_file_path = os.path.join(folder_path, f"{base_filename}.aa.ddm")

        # Check if the azimuthal average (.aa.ddm) file already exists
        if os.path.exists(aa_file_path):
            print(f"Loading precomputed azimuthal average for {tif_file}")
            aa = fddm.load(aa_file_path)
            aa_list.append(aa)
            base_filenames.append(base_filename)
            continue  # Skip further processing since .aa.ddm is loaded

        # Check if the structure function (.sf.ddm) file already exists
        if os.path.exists(sf_file_path):
            print(f"Loading precomputed structure function for {tif_file}")
            dqt = fddm.load(sf_file_path)
        else:
            # Calculate the structure function (.sf.ddm) if not already available
            print(f"Processing file {n+1}/{len(tif_files)}: {tif_file}")

            images = load_images(folder_path)    
            start_time = time.perf_counter()
            dqt = fddm.ddm(images, range(1, len(images)), core="py",mode='fft')
            print(f"Time taken to compute structure function for {tif_file}: {time.perf_counter() - start_time:.2f} seconds")
            dqt.pixel_size = float(pixel_size)
            dqt.set_frame_rate(frame_rate)
                
            # Save the calculated structure function to .sf.ddm file
            dqt.save(sf_file_path)

        # Calculate the azimuthal average based on the dqt
        bin_size = dqt.ky[1] - dqt.ky[0]
        bins = np.arange(int(dqt.ky[-1] / bin_size) + 1) * bin_size
        cc_mask = mask.central_cross_mask(dqt.shape[1:])
        aa = fddm.azimuthal_average(dqt, bins=bins, mask=cc_mask)
        aa_list.append(aa)

        # Save the azimuthal average
        aa.save(aa_file_path)

        # Append to dqt_list and base_filenames only if dqt exists
        dqt_list.append(dqt)
        base_filenames.append(base_filename)

    # Always call average_azimuthal_averages, even if there is only one dataset
    print("Resampling and (if necessary) averaging azimuthal averages...")
    base_filename = os.path.basename(folder_path)  # Use folder name as base filename
    aa_res = average_azimuthal_averages(aa_list, base_filename, folder_path)

    return dqt_list, aa_res, base_filenames

def average_azimuthal_averages(aa_list, base_filename, folder_path):
    """Resamples (and averages, if necessary) azimuthal averages from aa_list."""
    num_replicates = len(aa_list)
    print(f"Number of replicates: {num_replicates}")
    
    if num_replicates == 0:
        print("No replicates found.")
        return []
    
    # Initialize the result (whether single or multiple datasets)
    aa_res = None

    for n, a in enumerate(aa_list):
        if aa_res is None:
            # If this is the first dataset, initialize the result container
            aa_res = a
            aa_res._err **= 2  # Square the errors for combining
        else:
            # Add the next resampled data to the existing result
            tmp = a
            aa_res._data += tmp._data
            aa_res._err += tmp._err**2  # Add squared errors

    # Finalize averaging (if there were multiple datasets)
    if num_replicates > 1:
        aa_res._data /= num_replicates
        aa_res._err = np.sqrt(aa_res._err / num_replicates)

    # Save the (resampled and possibly averaged) result
    avg_aa_file_path = os.path.join(folder_path, f"{base_filename}_averaged.aa.ddm")
    aa_res.save(avg_aa_file_path)
    print(f"Saved resampled/averaged azimuthal average to {avg_aa_file_path}")

    return [aa_res]

def color_space( 
    length: int,
    colormap: cm.colors.LinearSegmentedColormap = cm.viridis,
    vmin: float = 0.2, 
    vmax: float = 1.0
) -> np.ndarray:
    return colormap(np.linspace(vmin, vmax, length))

# noise estimation
def estimate_noise(aa_res):
    B_est = {}
    B_err = {}

    # min method
    B_est["min"] = []
    B_err["min"] = []
    for n, a in enumerate(aa_res):
        best, berr = fddm.noise_est.estimate_camera_noise(a, mode="min")
        B_est["min"].append(best)
        B_err["min"].append(berr)
        
    # high_q method
    B_est["high_q"] = []
    B_err["high_q"] = []
    for n, a in enumerate(aa_res):
        best, berr = fddm.noise_est.estimate_camera_noise(a, mode="high_q", k_min=a.k[-7], k_max=a.k[-1])
        B_est["high_q"].append(best)
        B_err["high_q"].append(berr)
        
    # power_spec method
    B_est["power_spec"] = []
    B_err["power_spec"] = []
    for n, a in enumerate(aa_res):
        best, berr = fddm.noise_est.estimate_camera_noise(a, mode="power_spec", k_min=a.k[-7], k_max=a.k[-1])
        B_est["power_spec"].append(best)
        B_err["power_spec"].append(berr)
        
    # var method
    B_est["var"] = []
    B_err["var"] = []
    for n, a in enumerate(aa_res):
        best, berr = fddm.noise_est.estimate_camera_noise(a, mode="var", k_min=a.k[-7], k_max=a.k[-1])
        B_est["var"].append(best)
        B_err["var"].append(berr)

    # polyfit method
    B_est["polyfit"] = []
    B_err["polyfit"] = []
    for n, a in enumerate(aa_res):
        best, berr = fddm.noise_est.estimate_camera_noise(a, mode="polyfit", num_points=10)
        B_est["polyfit"].append(best)
        B_err["polyfit"].append(berr)
    
    return B_est, B_err

def exponential_fit(tau, a, b):
    return a * np.exp(-b * tau)

def fit_azimuthal_average(aa_res, B_est, test_k, results_folder, base_filenames, subfolder_names):
    """Fit azimuthal average data and save plots to the results folder."""

    # Ensure there is data to process
    if not aa_res or not B_est or not base_filenames:
        print("No data available for fitting and plotting.")
        return None, None

    fit_results = {}  # Dictionary to store fit results for each k
    chi_squared_results = {}  # Dictionary to store chi squared results for each k

    # subfolder_map should match the length of averaged azimuthal data
    subfolder_map = subfolder_names[:len(aa_res)]  # Match length with aa_res

    if len(subfolder_map) != len(aa_res):
        print(f"Error: subfolder_map length ({len(subfolder_map)}) does not match azimuthal average length ({len(aa_res)}).")
        return None, None

    for k in test_k:
        fig, ax = plt.subplots(figsize=(10, 6))

        chi_squared_sum = 0  # Initialize chi-squared sum for current k

        for n, a in enumerate(aa_res):
            # Check if k is a valid index
            if k >= len(a.data):
                print(f"Invalid k index: {k}. Skipping this plot.")
                continue

            x = a.tau
            y = 1 - (a.data[k] - B_est["polyfit"][n][k]) / (2.0 * a.var[k] - B_est["polyfit"][n][k]) # y = f(q,Δt)

            # Fit the data with an exponential
            try:
                popt, pcov = curve_fit(exponential_fit, x, y, maxfev=10000)
                fit_data = exponential_fit(x, *popt)

                # Calculate the chi-squared for the fit
                residuals = y - fit_data
                chi_squared = np.sum(residuals ** 2 / fit_data)
                chi_squared_sum += chi_squared

                # Plot the data and the fit, using the mapped subfolder name as the label
                subfolder_name = subfolder_map[n]
                ax.plot(x, y, '.', label=f'{subfolder_name}, k_ref={k}')
                ax.plot(x, fit_data, '-', label=f'Fit {subfolder_name}, k_ref={k}')

                # Store the fit results
                fit_results.setdefault(k, []).append((popt, np.sqrt(np.diag(pcov))))

            except Exception as e:
                print(f"Error fitting data for subfolder {subfolder_map[n]}, k_ref={k}: {e}")
                continue

        # Store the chi-squared results
        chi_squared_results[k] = chi_squared_sum / len(aa_res)

        ax.axhline(y=np.exp(-1), color='gray', linestyle='--', label='exp(-1)')
        ax.set_xscale('log')
        ax.set_title(f"Results for K = {k}", fontweight='bold', fontsize=12)
        ax.set_xlabel(r'$\Delta t$ (s)', fontweight='bold', fontsize=12)
        ax.set_ylabel(r'$g_1$', fontweight='bold')
        ax.legend()
        fig.tight_layout()

    # Determine the best k based on the lowest chi-squared
    if chi_squared_results:
        best_k = min(chi_squared_results, key=chi_squared_results.get)
        best_fit_parameters = np.mean([popt for popt, _ in fit_results[best_k]], axis=0)
        print(f"Best k_ref: {best_k}, with parameters: {best_fit_parameters}")
        return best_k, best_fit_parameters
    else:
        print("No valid fit results.")
        return None, None

def model_fit_best_k(aa_res, best_k, results_folder, base_filenames):
    fit_res = []
    model_res = []

    for n, a in enumerate(aa_res):
        y = 1 - (a.data[best_k] - B_est["polyfit"][n][best_k]) / (2.0 * a.var[best_k] - B_est["polyfit"][n][best_k])  # y-axis
        tau = a.tau[np.argmin(np.abs(y - np.exp(-1)))]  # x-axis

        # Set model parameter hints
        model.set_param_hint("B", value=B_est["polyfit"][n][best_k])
        model.set_param_hint("A", value=2.0 * a.var[best_k] - B_est["polyfit"][n][best_k])
        model.set_param_hint("Gamma", value=1 / tau)

        # Weight data points with 1/sqrt(t)
        weights = 1 / np.sqrt(a.tau)

        # Fit
        res, mres = fit_multik(a, model, best_k,
                       use_err=False,
                       return_model_results=True,
                       weights=weights)
                       #method='leastsq',               # or method='least_squares'
                       #fit_kws={'calc_covar': True,"max_nfev": 2000})   
        
        Gamma_errs = []
        A_errs = []

        for fit in mres:
            if fit is None:
                Gamma_errs.append(np.nan)
                A_errs.append(np.nan)
                continue  # skip to next fit

            gamma_p = fit.params.get("Gamma")
            a_p = fit.params.get("A")

            Gamma_err = gamma_p.stderr if gamma_p is not None and gamma_p.stderr is not None else np.nan
            A_err = a_p.stderr if a_p is not None and a_p.stderr is not None else np.nan

            Gamma_errs.append(Gamma_err)
            A_errs.append(A_err)

        res["Gamma_err"] = np.array(Gamma_errs)
        res["A_err"] = np.array(A_errs)

        fit_res.append(res)
        model_res.append(mres)

    return fit_res, model_res 

def compute_and_plot_msd(fit_res, aa_res, B_est, subfolder_names, conc, ax1, ax2):
    """
    For each dataset (indexed by n):
    — Computes MSD = 4/k^2 * log(A / (A + B - Dqt)) for all valid k
    — Averages over k-values to get MSD_ave and standard error
    — Plots MSD vs tau for each k (on ax1)
    — Plots MSD_ave ± MSD_err vs tau (on ax2)
    """
    # for n, (fr, a) in enumerate(zip(fit_res, aa_res)):
    #     k_vals = fr["k"]
    #     A_vals = fr["A"]
    #     A_err_vals = fr["A_err"]
    #     B_vals = B_est["polyfit"][n]
    #     Dqt_all = a._data  # shape: (10, len(a.tau))
    #     tau_vals = a.tau

    #     msd_matrix = []

    #     for i, k in enumerate(k_vals):
    #         A = A_vals[i]
    #         A_err = A_err_vals[i]
    #         B = B_vals[i]
            
    #         Dqt = Dqt_all[i]
    #         log_term = np.log(A / (A + B - Dqt))
    #         msd = 4 / (k ** 2) * log_term
            
    #         msd_matrix.append(msd)

    #         # if (A == 0 or np.isnan(A) or np.isnan(B) or 
    #         # A_err is None or np.isnan(A_err) or A_err / A >= 0.025):
    #         #     continue

    #         # Dqt = Dqt_all[i]

    #         # # --- 1. trim Dqt and tau so they match ---------------------------------
    #         # len_common = min(len(tau_vals), len(Dqt))
    #         # Dqt = Dqt[:len_common]
    #         # tau_use = tau_vals[:len_common]
    #         # # -----------------------------------------------------------------------

    #         # with np.errstate(divide='ignore', invalid='ignore'):
    #         #     log_term = np.log(A / (A + B - Dqt))
    #         #     msd = 4 / (k ** 2) * log_term
    #         #     msd[np.isnan(msd) | np.isinf(msd)] = np.nan

    #         # msd_matrix.append(msd)

    #         # ax1.plot(tau_use, msd, label=f"k={k:.2f}", alpha=0.4)

    #         # # Export MSD vs tau for this k
    #         # k_str = f"{k:.4f}".replace('.', 'p')  # for safe filename
    #         # csv_filename = f"{results_folder}/{subfolder_names[n]}_MSD_q_{k_str}.csv"
    #         # with open(csv_filename, "w", newline="") as csvfile:
    #         #     writer = csv.writer(csvfile)
    #         #     writer.writerow(["tau", f"MSD (k={k:.4f})"])
    #         #     for t_val, msd_val in zip(tau_use, msd):
    #         #         writer.writerow([t_val, msd_val])

    #     msd_matrix = np.array(msd_matrix)

    #     MSD_ave = np.nanmean(msd_matrix, axis=0)
    #     MSD_err = np.nanstd(msd_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(msd_matrix), axis=0))

    #     # Plot MSD_ave with error bars
    #     #ax2.errorbar(tau_use, MSD_ave, yerr=MSD_err, fmt="o-", label=conc[n])

    # # Format both plots
    # for ax in [ax1, ax2]:
    #     ax.set_xscale("log")
    #     ax.set_yscale("log")
    #     ax.set_xlabel(r"$\Delta t$ (s)", fontsize=16)
    #     ax.set_ylabel(r"MSD ($\mu m^2$)", fontsize=16)
    #     ax.grid(True, which='both', linestyle='--', alpha=0.3)
    #     ax.legend()

    # ax1.set_title("MSD vs τ at all k", fontsize=16)
    # ax2.set_title("Average MSD ± error", fontsize=16)

    # Save individual *_MSD.CSV files for each subfolder
    for n, (fr, a, subfolder_name) in enumerate(zip(fit_res, aa_res, subfolder_names)):
        k_vals = fr["k"]
        A_vals = fr["A"]
        A_err_vals = fr["A_err"]
        B_vals = B_est["polyfit"][n]
        Dqt_all = a._data
        tau_vals = a.tau

        msd_matrix = []

        for i, k in enumerate(k_vals[1:]):
            A = A_vals[i]
            A_err = A_err_vals[i]
            B = B_vals[i]

            # if (A == 0 or np.isnan(A) or np.isnan(B) or 
            # A_err is None or np.isnan(A_err) or A_err / A >= 0.025):
            #     continue

            Dqt = Dqt_all[i]

            #len_common = min(len(tau_vals), len(Dqt))
            #tau_use = tau_vals[:len_common]
            #Dqt = Dqt[:len_common]

            #with np.errstate(divide='ignore', invalid='ignore'):

            log_term = np.log(A / (A + B - Dqt))
            msd = 4 / (k ** 2) * log_term
            msd[np.isnan(msd) | np.isinf(msd)] = np.nan

            msd_matrix.append(msd)

        if not msd_matrix:
            continue  # no usable MSD data

        msd_matrix = np.array(msd_matrix)
        MSD_ave = np.nanmean(msd_matrix, axis=0)
        MSD_err = np.nanstd(msd_matrix, axis=0) / np.sqrt(np.sum(~np.isnan(msd_matrix), axis=0))

        # Save CSV
        csv_filename = f"{results_folder}/{subfolder_name}_MSD.csv"
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["tau", "MSD_ave", "MSD_err"])
            for t, ave, err in zip(tau_vals, MSD_ave, MSD_err):
                writer.writerow([t, ave, err])

def fit_a_b_gamma(fit_res, model_res, results_folder, selected_folder_name, B_est, conc):
    print("Determining A, B, and Gamma...")
    show_transparency = False
    fig = plt.figure(figsize=(12, 12))
    gs = gridspec.GridSpec(1, 2)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    fig.suptitle(f'{selected_folder_name}', fontsize=20)

    # Define k ranges for each subfolder k_min and k_max
    k_min = [0.1] * len(fit_res)
    k_max = [2.0] * len(fit_res)
    k_mask = []

    for n, fr in enumerate(fit_res):
        k_mask.append((fr["k"] > k_min[n]) & (fr["k"] <= k_max[n]))

    # Plot Gamma data
    for n, fr in enumerate(fit_res):
        ax1.plot(fr["k"][k_mask[n]], fr["Gamma"][k_mask[n]], f"C{n}o", markerfacecolor="none", label=conc[n])
        if show_transparency: 
            ax1.plot(fr["k"][~k_mask[n]], fr["Gamma"][~k_mask[n]], f"C{n}0", markerfacecolor="none", alpha=0.2)

    # Save individual *_Gamma.CSV files for each subfolder
    for n, (fr, subfolder_name) in enumerate(zip(fit_res, subfolder_names)):
        csv_filename = f"{results_folder}/{subfolder_name}_Gamma.csv"
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            # Write header
            writer.writerow(["k", "Gamma", "Gamma_err"])
            for i, k_value in enumerate(fr["k"]):
                Gamma_value = fr["Gamma"][i]
                Gamma_err_value = fr["Gamma_err"][i] if "Gamma_err" in fr else "N/A"
                writer.writerow([k_value, Gamma_value, Gamma_err_value])

    # Plot settings for ax1
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.xaxis.label.set_fontweight("bold")
    ax1.yaxis.label.set_fontweight("bold")
    ax1.xaxis.label.set_fontsize(30)
    ax1.yaxis.label.set_fontsize(30)
    ax1.xaxis.label.set_fontname("Arial")
    ax1.yaxis.label.set_fontname("Arial")
    ax1.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax1.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * .1))
    ax1.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax1.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * .1))
    ax1.grid(True, which='both', linestyle='-', color='black', alpha=0.25, zorder=0) 
    ax1.set_xlabel(r"$k$ ($\mu m^{-1}$)", fontsize=20, fontname="Arial")
    ax1.set_ylabel(r"$\Gamma$ ($s^{-1}$)", fontsize=20, fontname="Arial")
    ax1.legend()
    ax1.grid()
    
    # Plot A and B
    custom_lines = [Line2D([0], [0], marker="o", linestyle="none", color="black", markerfacecolor="none", label=r"$A$"),
                    Line2D([0], [0], linestyle="--", color="black", label=r"$B$")]

    for n, fr in enumerate(fit_res):
        ax2.plot(fr["k"][k_mask[n]], fr["A"][k_mask[n]], f"C{n}o", markerfacecolor="none")
        ax2.plot(fr["k"][k_mask[n]], B_est["var"][n][k_mask[n]], f"C{n}--")
        if show_transparency:
            ax2.plot(fr["k"][~k_mask[n]], fr["A"][~k_mask[n]], f"C{n}o", markerfacecolor="none", alpha=0.2)
            ax2.plot(fr["k"][~k_mask[n]], B_est["var"][n][~k_mask[n]], f"C{n}--", alpha=0.2)
    
    # Save individual *_A_B.CSV files for each subfolder
    for n, (fr, subfolder_name) in enumerate(zip(fit_res, subfolder_names)):
        csv_filename = f"{results_folder}/{subfolder_name}_A_B.csv"
        with open(csv_filename, "w", newline="") as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(["k", "A", "A_err", "B", "B_err"])
            for i, k_value in enumerate(fr["k"]):
                A_value = fr["A"][i]
                A_err_value = fr["A_err"][i] if "A_err" in fr else "N/A"
                B_value = B_est["var"][n][i]
                B_err_value = B_err["var"][n][i]
                writer.writerow([k_value, A_value, A_err_value, B_value, B_err_value])

    # Plot settings for ax2
    ax2.fill_betweenx(y=[50,1e7], x1=0.4, x2=0.7, color='gray', alpha=0.1)
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ax2.set_ylim([1e2, 1e7])
    ax2.xaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax2.xaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * .1))
    ax2.yaxis.set_major_locator(LogLocator(base=10.0, subs=[1.0]))
    ax2.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * .1))
    ax2.grid(True, which='both', linestyle='-', color='black', alpha=0.25, zorder=0)
    ax2.xaxis.label.set_fontweight("bold")
    ax2.yaxis.label.set_fontweight("bold")
    ax2.xaxis.label.set_fontsize(30)
    ax2.yaxis.label.set_fontsize(30)
    ax2.xaxis.label.set_fontname("Arial")
    ax2.yaxis.label.set_fontname("Arial")
    ax2.set_xlabel(r"$q$ ($\mu m^{-1}$)", fontsize=20, fontname="Arial")
    ax2.set_ylabel(r"$A / B$", fontsize=20, fontname="Arial")
    ax2.legend(handles=custom_lines, labelspacing=0.4)

    # Save figure
    try:
        save_path = os.path.join(results_folder, f"Fit_Results.png")
        fig.tight_layout()
        fig.savefig(save_path)
        print(f"Figure saved successfully to: {save_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")

    plt.close(fig)

    # Save D(q,Δt) in CSV files for each subfolder
    for n, (a, subfolder_name) in enumerate(zip(aa_res, subfolder_names)):
        for k_index in range(10):
            csv_filename = f"{results_folder}/{subfolder_name}_Dqt_{k_index}.csv"
            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["dt (s)", "Dqt"])
                for dt, Dqt_value in zip(a.tau, a._data[k_index]):
                    writer.writerow([dt, Dqt_value])       

    # Save f(q,Δt) in CSV files for each subfolder
    for n, (a, subfolder_name) in enumerate(zip(aa_res, subfolder_names)):
        for k_index in range(10):
            fqt = 1 - (a.data[k_index] - B_est["polyfit"][n][k_index]) / (2.0 * a.var[k_index] - B_est["polyfit"][n][k_index])
            csv_filename = f"{results_folder}/{subfolder_name}_fqt_{k_index}.csv"
            with open(csv_filename, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(["dt (s)", "fqt"])
                for dt, fqt_value in zip(a.tau, fqt):
                    writer.writerow([dt, fqt_value]) 
    # ============================
    # Plot MSD and MSD_ave vs tau
    # ============================
    fig2, (msd_ax1, msd_ax2) = plt.subplots(1, 2, figsize=(12, 6))
    compute_and_plot_msd(fit_res, aa_res, B_est, subfolder_names, conc, msd_ax1, msd_ax2)

    try:
        save_path = os.path.join(results_folder, f"MSD_Results.png")
        fig2.tight_layout()
        fig2.savefig(save_path)
        print(f"MSD figure saved to: {save_path}")
    except Exception as e:
        print(f"Error saving MSD figure: {e}")

    plt.close(fig2)

    return None

def show_fit_results(aa_res, k_min, k_max, fit_res, model_res, results_folder, subfolder_names):
    '''Display the fits for each azimuthal average f(q, ∆t)'''
    print("Displaying fit results...")

    # If only one azimuthal average, create a single subplot
    if len(aa_res) == 1:
        fig, ax = plt.subplots(figsize=(12, 12))
        axs = [ax]  # Make axs a list to keep indexing consistent
    else:
        fig, axs = plt.subplots(len(aa_res), figsize=(12, 12))

    fig.tight_layout()
    gs = fig.add_gridspec(len(aa_res), hspace=0)

    k_min = [0.1] * len(aa_res)
    k_max = [2.0] * len(aa_res)

    k_idx = []
    cspace = []

    for n, (k, K) in enumerate(zip(k_min, k_max)):
        idx_min = np.argwhere(aa_res[n].k >= k)[0, 0]
        idx_max = np.argwhere(aa_res[n].k <= K)[-1, 0]
        k_list = np.linspace(idx_min, idx_max, num=5, dtype=int)
        k_idx.append(k_list)
        cspace.append(color_space(len(k_idx[-1])))

    for i, a in enumerate(aa_res):
        ax = axs[i] if len(aa_res) > 1 else axs[0]  # Handle single vs. multiple subplots
        for n, k in enumerate(k_idx[i]):
            y = 1 - (a.data[k] - fit_res[i]['B'][k]) / fit_res[i]['A'][k]
            y_fit = 1 - (model_res[i][k].best_fit - fit_res[i]['B'][k]) / fit_res[i]['A'][k]

            klabel = a.k[k]
            ax.plot(a.tau, y, '.', color=cspace[i][n], label=f"k={klabel:.2f} μm$^{-1}$")
            ax.plot(a.tau, y_fit, '-', color=cspace[i][n])

        ax.set_xscale('log')
        ax.xaxis.label.set_fontweight("bold")
        ax.yaxis.label.set_fontweight("bold")
        ax.xaxis.label.set_fontsize(20)
        ax.yaxis.label.set_fontsize(20)
        ax.xaxis.label.set_fontname("Arial")
        ax.yaxis.label.set_fontname("Arial")
        ax.set_ylim(-0.25,1)
        ax.set_xlabel(r'$\Delta t$ (s)')
        ax.set_ylabel(r'$f(q, \Delta t)$')
        ax.legend()

        # Add subfolder name as an annotation in the lower left corner of each subplot
        subfolder_name = subfolder_names[i]
        at = AnchoredText(f"{subfolder_name}", prop=dict(size=20), frameon=True, loc='lower left')
        ax.add_artist(at)

    # Save figure
    try:
        save_path = os.path.join(results_folder, f"ISF.png")
        fig.tight_layout()
        fig.savefig(save_path)
        print(f"Figure saved successfully to: {save_path}")
    except Exception as e:
        print(f"Error saving figure: {e}")

    plt.close(fig)
    return None

def raeFunction(dqt_list):
    af_q = np.zeros((0,2))
    for i in np.arange(dqt_list[0].data.shape[0]):
        sf = dqt_list[0].data[i]
        sf_rot = np.fliplr(sf)
        sf_rot = np.flipud(sf_rot)
        sfFull = np.hstack((sf_rot,sf))
        
        af_q_eachLagTime = find_alignment_factor_one_lagtime(sfFull, orientation_axis=0,
                                                  remove_vert_line=True, remove_hor_line=True)
        lagTime = i+1
        af_q_eachLagTime = np.hstack((af_q_eachLagTime[:,np.newaxis],np.ones((af_q_eachLagTime.shape[0],1))*lagTime))
        af_q = np.vstack((af_q,af_q_eachLagTime))
    return af_q

def process_all_subfolders(main_folder, results_folder):
    """Process all subfolders containing TIF file replicates."""
    all_base_filenames = {}
    all_aa_res = []
    subfolder_names = []  # Keep track of subfolder names

    for subfolder in os.listdir(main_folder):
        subfolder_path = os.path.join(main_folder, subfolder)
        if os.path.isdir(subfolder_path) and subfolder != "Results":  # Skip the Results folder
            print(f"Processing subfolder: {subfolder_path}")
            dqt_list, aa_list, base_filenames = process_tif_files_in_folder(subfolder_path, results_folder)
            af_qq = raeFunction(dqt_list)
            #np.savetxt(results_folder+'/'+subfolder+'_AlignmentFactor.txt',af_qq)
            np.savetxt(results_folder+'/'+subfolder+'_AlignmentFactor.csv', np.abs(af_qq), delimiter=',')
            print(f"Number of azimuthal averages: {len(aa_list)}")
            all_base_filenames[subfolder] = base_filenames
            all_aa_res.extend(aa_list)
            subfolder_names.append(subfolder)  # Add the subfolder name to the list

    return all_base_filenames, all_aa_res, subfolder_names  # Return subfolder names

def find_alignment_factor_one_lagtime(ddmmatrix2d, orientation_axis=0,
                                          remove_vert_line=True, remove_hor_line=True):
        r"""
        Parameters
        ----------
        orientation_axis : TYPE, optional
            DESCRIPTION. The default is np.pi/4.
        Returns
        -------
        None.
        """
        ddm_matrix_at_lagtime = ddmmatrix2d.copy()
        nx,ny = ddm_matrix_at_lagtime.shape
        if remove_vert_line:
            ddm_matrix_at_lagtime[:,int(ny/2)]=0
        if remove_hor_line:
            ddm_matrix_at_lagtime[int(nx/2),:]=0
        x = np.arange(-1*ny/2, ny/2, 1)
        y = np.arange(-1*nx/2, nx/2, 1)
        xx,yy = np.meshgrid(x,y)
        with np.errstate(divide='ignore', invalid='ignore'):
            cos2theta = np.cos(2*np.arctan(1.0*xx/yy) + orientation_axis)
        cos2theta[int(nx/2),int(ny/2)]=0
        dists = np.sqrt(np.arange(-1*nx/2, nx/2)[:,None]**2 + np.arange(-1*ny/2, ny/2)[None,:]**2)
        bins = np.arange(max(nx,ny)/2+1)
        histo_of_bins = np.histogram(dists, bins)[0]
        af_numerator = np.histogram(dists, bins, weights=ddm_matrix_at_lagtime*cos2theta)[0]
        af_denominator = np.histogram(dists, bins, weights=ddm_matrix_at_lagtime)[0]
        with np.errstate(divide='ignore', invalid='ignore'):
            af = af_numerator / af_denominator
        return af

def power_fit(x, y, y_err):

    #log_x = np.log10(x)
    #log_y = np.log10(y)
    #log_yerr = np.log10(y_err)
    
    def power_law(x, a, b):
        return a * x ** b
    
    coeffs, cov = curve_fit(power_law, x, y, sigma=y_err)
    coeff = coeffs[0]
    exp = coeffs[1]
    #slope_err = np.sqrt(cov[0,0]) #standard deviation error on the slope "a" parameter
    
    #r_val = np.corrcoef(log_x, log_y)[0,1]
    #r_squared = r_val**2
    
    return exp, coeff

def avgAligmentFactor(file_folder,file_name,fractionCalculate):
    alignmentFactor = np.loadtxt(file_folder+file_name,delimiter=',')
    lagTime,numQ = np.unique(alignmentFactor[:,1].astype(int),return_counts=True)
    numQ = np.unique(numQ)
    #plt.figure()
    af_lagtime = np.zeros((lagTime.shape[0],2))
    for i,lt in enumerate(lagTime):
        af_oneLagTime = alignmentFactor[alignmentFactor[:,1]==lt,0]
        af_oneLagTime = af_oneLagTime[np.arange(1,np.round(af_oneLagTime.shape[0]/2)).astype(int)]
        af_oneLagTime = np.abs(af_oneLagTime)
        af_lagtime[i,:] = [lt,np.average(af_oneLagTime)]    
    np.savetxt(file_folder+file_name[:-4]+'_lagTime.csv',af_lagtime,delimiter=',')
    fig = plt.figure()
    plt.plot(af_lagtime[:,0],af_lagtime[:,1])
    plt.ylabel('Af')
    plt.xlabel('lag time [s]')
    plt.savefig(file_folder+file_name[:-4]+'.png',dpi=200,bbox_inches='tight')
    fig.clear()
    plt.close(fig)    
    return np.average(af_lagtime[np.arange(np.round(af_lagtime.shape[0]*fractionCalculate)).astype(int),1])
    
def MSD_Fit(file_folder,file_name,fractionCalculate):
    data = np.loadtxt(file_folder+file_name, delimiter=',', skiprows=1)
    data = data[0:int(data.shape[0]*fractionCalculate),:]
    exponent, transportCoefficient = power_fit(data[:,0], data[:,1], data[:,2])
    
    fig = plt.figure()
    plt.errorbar(data[:,0], data[:,1], data[:,2], fmt='o')
    fit_y = transportCoefficient*data[:,0]**exponent
    plt.errorbar(data[:,0], fit_y, fmt='k-')
    plt.ylabel('MSD [$\\mu$m$^2$]')
    plt.xlabel('$\\Delta$t [s]')
    plt.xscale('log')
    plt.yscale('log')
    plt.savefig(file_folder+file_name[:-4]+'.png',bbox_inches='tight',dpi=200)
    
    return transportCoefficient,exponent

if __name__ == "__main__":
    # Prompt the user for concentrations before folder selection
    conc = input("Enter numerical names of folders (comma-separated): ")
    conc = [float(c) for c in conc.split(",")]

    # Proceed to select folder and run the rest of the process
    folder_path, selected_folder_name = select_folder()  # Unpack the tuple with the correct parent folder name
    results_folder = create_results_folder(folder_path)  # Pass only folder_path
    base_filenames, aa_res, subfolder_names = process_all_subfolders(folder_path, results_folder)
    #base_filenames, aa_res, subfolder_names = process_all_subfolders(folder_path, results_folder)

    print("Base filenames: ", base_filenames)
    print(f"Number of azimuthal averages: {len(aa_res)}")

    # Estimate noise for the azimuthal averages
    B_est, B_err = estimate_noise(aa_res)

    # Analyze each subfolder
    for subfolder, filenames in base_filenames.items():
        subfolder_index = list(base_filenames.keys()).index(subfolder)  # Get the index for subfolder
        subfolder_name = subfolder_names[subfolder_index]  # Use the correct subfolder name for labeling

        # Pass the entire subfolder_names list instead of [subfolder_name]
        # best_k, best_fit_params = fit_azimuthal_average(aa_res, B_est, test_k, results_folder, filenames, subfolder_names)
        best_k, best_fit_params = fit_azimuthal_average(aa_res, B_est, test_k, results_folder, base_filenames, subfolder_names)

        print(f"Best k_ref for {subfolder}: {best_k} with parameters: {best_fit_params}")
        
        # Call the fit function to generate fit_res and model_res
        fit_res, model_res = model_fit_best_k(aa_res, best_k, results_folder, filenames)
        
        # Pass the correct parent folder name and concentrations for the plot's suptitle
        fit_a_b_gamma(fit_res, model_res, results_folder, selected_folder_name, B_est, conc)

        # Display the fit results
        show_fit_results(aa_res, test_k, test_k+2, fit_res, model_res, results_folder, subfolder_names)

        fractionCalculate = 0.25
        af = avgAligmentFactor(results_folder+'/',subfolder+'_AlignmentFactor.csv',fractionCalculate)
        coeff, exponent = MSD_Fit(results_folder+'/',subfolder+'_MSD.csv',fractionCalculate)
        
        BARCODE_Metrics = np.savetxt(results_folder+'/'+subfolder+'_BARCODE_Metrics.txt', [af, exponent, coeff])
        