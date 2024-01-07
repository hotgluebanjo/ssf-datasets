"""
SSF criteria:
  - Quantum efficiency included
  - Testing illuminant excluded
  - Increment <= 10nm for best precision
"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import PchipInterpolator

SSFS = "cameras/sony_ilce_7.txt"
OUTPUT = "sony_dataset.txt"

CUBE_SIZE = 5
SWEEP_MIN = -5.0
SWEEP_MAX = 5.0
SWEEP_INCREMENT = 1
PLOT_POINTS = True

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

def logc_encode(x):
    return np.where(
        x > 0.010591,
        0.247190 * np.log10(5.555556 * x + 0.052272) + 0.385537,
        5.367655 * x + 0.092809,
    )

def load_ssfs(file):
    camera = {}

    with open(file, 'r') as file:
        first_line = file.readline()
        delimiter = ',' if ',' in first_line else ' '

        try:
            float(first_line.split(delimiter)[0])
            file.seek(0)
            lines = file.readlines()
        except ValueError:
            lines = file.readlines()

    for line in lines:
        parts = line.strip().split(delimiter)
        wavelength = float(parts[0])
        value = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
        camera[wavelength] = value

    # PCHIP if necessary
    unique_wavelengths = np.array(sorted(camera.keys()))
    if np.any(np.diff(unique_wavelengths) > 1):
        pchip = PchipInterpolator(unique_wavelengths, np.array(list(camera.values())), axis=0, extrapolate=True)
        spectrum = np.arange(380, 780+1, 1)
        interpolated_values = pchip(spectrum)
        camera = dict(zip(spectrum, interpolated_values))

    return camera

def scatter_plot(data, bg="#222222"):
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

def main():
    camera = load_ssfs(SSFS)

    # TODO: Ignore check or linearly extrapolate.
    assert list(camera.keys())[0] == 380 and list(camera.keys())[-1] == 780

    gray_patch = np.zeros((3))
    for wavelength in range(380, 780+1):
        gray_patch += camera[wavelength] * 0.18 * arri_skypanel(wavelength, 1.0, 1.0, 1.0)
    exp_wb_coeff = 0.18 / gray_patch

    grid = np.linspace(0.0, 1.0, CUBE_SIZE)
    sweeps = np.arange(SWEEP_MIN, SWEEP_MAX + SWEEP_INCREMENT, SWEEP_INCREMENT)
    dataset = []

    for b in grid:
        for g in grid:
            for r in grid:
                ts = np.zeros((3))
                for wavelength in range(380, 780+1):
                    ts += camera[wavelength] * arri_skypanel(wavelength, r, g, b)
                for stop in sweeps:
                    res = logc_encode(ts * exp_wb_coeff * np.power(2.0, stop))
                    dataset.append(res)

    if PLOT_POINTS:
        scatter_plot(np.array(dataset))

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

if __name__ == "__main__":
    main()
