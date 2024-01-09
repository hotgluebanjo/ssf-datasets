"""
SSF criteria:
  - Quantum efficiency included
  - Testing illuminant excluded
  - Increment <= 10nm for best precision
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

SSFS = "cameras/arri_alexa.txt"
OUTPUT = "res.txt"

CUBE_SIZE = 5
SWEEP_MIN = -5.0
SWEEP_MAX = 5.0
SWEEP_INCREMENT = 1
PLOT = True

# Working spectrum.
SPECTRUM_MIN = 380
SPECTRUM_MAX = 780

def logc_encode(x):
    return np.where(
        x > 0.010591,
        0.247190 * np.log10(5.555556 * x + 0.052272) + 0.385537,
        5.367655 * x + 0.092809,
    )

def load_ssfs(file):
    with open(file, 'r') as file:
        first_line = file.readline()
        delimiter = ',' if ',' in first_line else ' '

        try:
            float(first_line.split(delimiter)[0])
            file.seek(0)
            lines = file.readlines()
        except ValueError:
            lines = file.readlines()

    wavelengths = []
    values = []

    for line in lines:
        parts = line.strip().split(delimiter)
        wavelengths.append(float(parts[0]))
        values.append(np.array([float(parts[1]), float(parts[2]), float(parts[3])]))

    pchip = PchipInterpolator(
        wavelengths,
        values,
        axis=0,
        extrapolate=False,
    )

    spectrum = np.arange(SPECTRUM_MIN, SPECTRUM_MAX + 1, 1)
    spectrum_values = np.nan_to_num(pchip(spectrum), nan=0.0)
    camera = dict(zip(spectrum, spectrum_values))

    return camera

def load_spectrum(file):
    spectrum_data = []
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            spectrum_data.append([float(p) for p in parts])
    return np.array(spectrum_data)

def process_spectral_data(file_path):
    with open(file_path, 'r') as file:
        first_line = file.readline()
        delimiter = ',' if ',' in first_line else ' '
        file.seek(0)
        data = np.array([line.strip().split(delimiter) for line in file], dtype=float)

    wavelengths, values = data[:, 0], data[:, 1:].T # transpose to make the values column-wise

    if wavelengths[0] > 380:
        # extend and lerp wavelengths to 380nm by 50% of each inital value
        extended_wl = np.arange(380, wavelengths[0])
        initial_val = values[:, 0] * 0.5
        slope = (values[:, 0] - initial_val) / (wavelengths[0] - 380)
        extended_val = initial_val[:, np.newaxis] + slope[:, np.newaxis] * (extended_wl - 380)

        wavelengths = np.concatenate([extended_wl, wavelengths])
        values = np.concatenate([extended_val, values], axis=1)

    spectrum = np.arange(380, wavelengths[-1] + 1, 1)
    interp_val = np.array([np.nan_to_num(PchipInterpolator(wavelengths, v, extrapolate=False)(spectrum), nan=0.0)
                                    for v in values])

    spectral_data = {i: interp_val[i] for i in range(interp_val.shape[0])}

    return spectral_data

def dataset_from_skypanel_lattice(camera):
    def gaussian(x, center, size):
        return np.exp(-np.power((x - center) / size, 2.0))

    # Relative SPD of approximated SkyPanel. RGB lights only, not RGBW.
    # Really more like the transmission.
    # https://www.desmos.com/calculator/gvyahvtfam
    def arri_skypanel(wavelength, r, g, b):
        return np.sum([
            r * gaussian(wavelength, 628.0, 15.0),
            g * gaussian(wavelength, 523.0, 25.0),
            b * gaussian(wavelength, 453.0, 15.0),
        ])
    
    gray_patch = np.zeros((3))
    for wavelength in range(SPECTRUM_MIN, SPECTRUM_MAX + 1):
        gray_patch += camera[wavelength] * 0.18 * arri_skypanel(wavelength, 1.0, 1.0, 1.0)
    exp_wb_coeff = 0.18 / gray_patch

    grid = np.linspace(0.0, 1.0, CUBE_SIZE)
    sweeps = np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_INCREMENT, SWEEP_INCREMENT)
    dataset = []

    for b in grid:
        for g in grid:
            for r in grid:
                ts = np.zeros((3))
                for wavelength in range(SPECTRUM_MIN, SPECTRUM_MAX + 1):
                    ts += camera[wavelength] * arri_skypanel(wavelength, r, g, b)
                for stop in sweeps:
                    res = logc_encode(ts * exp_wb_coeff * np.power(2.0, stop))
                    dataset.append(res)
    return dataset

def plot_3d(data, bg="#222222"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg)
    ax.grid(False)
    ax.axis('off')
    max_val = np.amax(data)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_zlim(0, max_val)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout(pad=0)

    grid_x = np.linspace(0, 1, num=10)
    grid_y = np.linspace(0, 1, num=10)

    for gx in grid_x:
        ax.plot([gx, gx], [0, 1], 0, color='white', linestyle='-', linewidth=0.5, alpha=0.2)
    for gy in grid_y:
        ax.plot([0, 1], [gy, gy], 0, color='white', linestyle='-', linewidth=0.5, alpha=0.2)
    for z in [0, 1]:
        for start in [0, 1]:
            ax.plot([start, start], [0, 1], z, color='white', linestyle='-', linewidth=0.5, alpha=0.2)
            ax.plot([0, 1], [start, start], z, color='white', linestyle='-', linewidth=0.5, alpha=0.2)
    for x in [0, 1]:
        for y in [0, 1]:
            ax.plot([x, x], [y, y], [0, 1], color='white', linestyle='-', linewidth=0.5, alpha=0.2)

    ax.scatter(data[:, 2], data[:, 0], data[:, 1], c=np.clip(data, 0.0, 1.0), marker='o', s=3)
    plt.show()

def plot_ssfs(camera):
    ssfs = np.array(list(camera.values()))
    plt.plot(list(camera.keys()), ssfs[:, 0], c="r")
    plt.plot(list(camera.keys()), ssfs[:, 1], c="g")
    plt.plot(list(camera.keys()), ssfs[:, 2], c="b")
    plt.plot()

def main():
    illuminant = load_spectrum("illuminants/incandescent_abs.txt")
    reflectance = process_spectral_data("charts/sg_spectral.txt")
    camera = load_ssfs(SSFS)
    sweeps = np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_INCREMENT, SWEEP_INCREMENT)
    print(reflectance)

    gray_patch = np.zeros((3))
    for i, wavelength in enumerate(range(SPECTRUM_MIN, SPECTRUM_MAX + 1)):
        gray_patch += camera[wavelength] * 0.18 * illuminant[i, 1]
    exp_wb_coeff = 0.18 / gray_patch

    dataset = []

    for sds in reflectance:
        ts = np.zeros((3))
        for i, wavelength in enumerate(range(SPECTRUM_MIN, SPECTRUM_MAX + 1)):
            ts += camera[wavelength] * reflectance[sds][i] * illuminant[i, 1]
        for stop in sweeps:
            res = logc_encode(ts * exp_wb_coeff * np.power(2.0, stop))
            dataset.append(res)

    if PLOT:
        plot_ssfs(camera)
        plot_3d(np.array(dataset))

    if OUTPUT != "":
        with open(OUTPUT, 'w') as output:
            match OUTPUT.split('.')[-1]:
                case "csv":
                    delimiter = ','
                case "tsv":
                    delimiter = '\t'
                case _:
                    delimiter = ' '

            for point in dataset:
                output.write(f"{point[0]}{delimiter}{point[1]}{delimiter}{point[2]}\n")
    else:
        print(dataset)

if __name__ == "__main__":
    main()
