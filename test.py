# SSF criteria:
#   - Quantum efficiency included
#   - Testing illuminant excluded
#
# SDs suggestions:
#   - Increment <= 10nm for best precision

import colour
import matplotlib.pyplot as plt
import numpy as np
import os
import scipy

SPECTRUM_MIN = 380
SPECTRUM_MAX = 780

# Great alias.
T = None

def into_float(x):
    try:
        v = float(x)
        return (v, True)
    except:
        return (0.0, False)

def make_vec(x, n):
    return [x] * n

def gaussian(x, center, size):
    return np.exp(-np.power((x - center) / size, 2.0))

# TODO: How much overlap with Colour?
class Sds:
    wavelengths: T
    values: T

    def __init__(sds, wavelengths, values):
        assert len(wavelengths) == len(values)

        # Always interpolating for now. Much faster to precompute
        # than evaluate the spline.
        interpolant = scipy.interpolate.PchipInterpolator(
            wavelengths,
            values,
            extrapolate=False,
        )
        sds.wavelengths = np.arange(SPECTRUM_MIN, SPECTRUM_MAX + 1, 1)
        sds.values = np.nan_to_num(interpolant(sds.wavelengths), nan=0.0)

    def sample(sds, wavelength, extrapolate=True):
        if wavelength < SPECTRUM_MIN:
            return sds.values[0]
        if wavelength > SPECTRUM_MAX:
            return sds.values[-1]

        i = wavelength - SPECTRUM_MIN
        return sds.values[i]

    # Separate N distribution columns into an array of Sds.
    def separate(sds):
        n_dist = sds.values.shape[1]
        res = make_vec(Sds, n_dist)
        for i in range(0, n_dist):
            res[i] = Sds(sds.wavelengths, sds.values[:, i])
        return res

# Loads N distributions column-wise. First column is assumed
# to be wavelengths and N columns after to be individual
# distributions.
def sds_from_file(filename, delim=None, comment='#'):
    # Assume there's junk at the top.
    skip_lines = 0
    with open(filename, "r") as file:
        while True:
            line = file.readline().strip()
            if delim == None and line[0] != comment:
                # Fallthrough certainty.
                if ' ' in line:
                    delim = ' '
                if '\t' in line:
                    delim = '\t'
                if ',' in line:
                    delim = ','
                if ';' in line:
                    delim = ';'
            _, ok = into_float(line.split(delim)[0])
            if ok:
                break
            skip_lines += 1

    data = np.loadtxt(
        filename,
        delimiter=delim,
        dtype=float,
        comments=comment,
        skiprows=skip_lines,
    )
    assert data.shape[1] >= 2

    return Sds(data[:, 0], data[:, 1:])

def dataset_from_skypanel_lattice(opts):
    # Relative SPD of approximated SkyPanel. RGB lights only, not RGBW.
    # Really more like the transmission.
    # https://www.desmos.com/calculator/gvyahvtfam
    def arri_skypanel(wavelength, r, g, b):
        return np.sum([
            r * gaussian(wavelength, 628.0, 15.0),
            g * gaussian(wavelength, 523.0, 25.0),
            b * gaussian(wavelength, 453.0, 15.0),
        ])
    
    camera = sds_from_file(opts.camera)

    gray_patch = np.zeros((3))
    for wavelength in range(SPECTRUM_MIN, SPECTRUM_MAX + 1):
        gray_patch += camera.sample(wavelength) * 0.18 * arri_skypanel(wavelength, 1.0, 1.0, 1.0)
    exp_wb_coeff = 0.18 / gray_patch

    grid = np.linspace(0.0, 1.0, SKYPANEL_LATTICE_SIZE)
    sweeps = np.arange(opts.sweep_min, opts.sweep_max + opts.sweep_increment, opts.sweep_increment)
    dataset = []

    for b in grid:
        for g in grid:
            for r in grid:
                ts = np.zeros((3))
                for wavelength in range(SPECTRUM_MIN, SPECTRUM_MAX + 1):
                    ts += camera.sample(wavelength) * arri_skypanel(wavelength, r, g, b)
                for stop in sweeps:
                    res = opts.transfer_function(ts * exp_wb_coeff * np.power(2.0, stop))
                    dataset.append(res)

    return np.array(dataset)

def dataset_from_chart(opts):
    illuminant = sds_from_file(opts.illuminant)
    chart = sds_from_file(opts.chart).separate()
    camera = sds_from_file(opts.camera)

    gray_patch = np.zeros((3))
    for wavelength in range(SPECTRUM_MIN, SPECTRUM_MAX + 1):
        gray_patch += camera.sample(wavelength) * 0.18 * illuminant.sample(wavelength)
    exp_wb_coeff = 0.18 / gray_patch

    sweeps = np.arange(opts.sweep_min, opts.sweep_max + opts.sweep_increment, opts.sweep_increment)
    dataset = []

    for patch in chart:
        ts = np.zeros((3))
        for wavelength in range(SPECTRUM_MIN, SPECTRUM_MAX + 1):
            ts += camera.sample(wavelength) * patch.sample(wavelength) * illuminant.sample(wavelength)
        for stop in sweeps:
            res = opts.transfer_function(ts * exp_wb_coeff * np.power(2.0, stop))
            dataset.append(res)

    return np.array(dataset)

def plot_3d(data, bg="#222222"):
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_facecolor(bg)
    ax.grid(False)
    ax.axis("off")
    max_val = np.amax(data)
    ax.set_xlim(0, max_val)
    ax.set_ylim(0, max_val)
    ax.set_zlim(0, max_val)
    ax.set_box_aspect([1, 1, 1])
    fig.tight_layout(pad=0)

    grid_x = np.linspace(0, 1, num=10)
    grid_y = np.linspace(0, 1, num=10)

    for gx in grid_x:
        ax.plot([gx, gx], [0, 1], 0, color="white", linestyle='-', linewidth=0.5, alpha=0.2)
    for gy in grid_y:
        ax.plot([0, 1], [gy, gy], 0, color="white", linestyle='-', linewidth=0.5, alpha=0.2)
    for z in [0, 1]:
        for start in [0, 1]:
            ax.plot([start, start], [0, 1], z, color="white", linestyle='-', linewidth=0.5, alpha=0.2)
            ax.plot([0, 1], [start, start], z, color="white", linestyle='-', linewidth=0.5, alpha=0.2)
    for x in [0, 1]:
        for y in [0, 1]:
            ax.plot([x, x], [y, y], [0, 1], color="white", linestyle='-', linewidth=0.5, alpha=0.2)

    ax.scatter(data[:, 2], data[:, 0], data[:, 1], c=np.clip(data, 0.0, 1.0), marker='o', s=3)
    plt.show()

def plot_ssfs(camera):
    plt.plot(camera.wavelengths, camera.values[:, 0], c="r")
    plt.plot(camera.wavelengths, camera.values[:, 1], c="g")
    plt.plot(camera.wavelengths, camera.values[:, 2], c="b")
    plt.plot()

class Mode:
    CHART = 0
    SKYPANEL = 1

class Config:
    illuminant = "illuminants/incandescent_abs.csv"
    chart = "charts/SG_140.csv"
    camera = "cameras/arri_alexa.txt"
    output = "alexa_dataset_out.csv"
    mode = Mode.CHART
    sweep_min = -5.0
    sweep_max = 5.0
    sweep_increment = 1
    skypanel_lattice_size = 5
    transfer_function = colour.models.log_encoding_ARRILogC3
    plot = True

def main():
    opts = Config

    match opts.mode:
        case Mode.CHART:
            dataset = dataset_from_chart(opts)
        case Mode.SKYPANEL:
            dataset = dataset_from_skypanel_lattice(opts)

    if opts.plot:
        # plot_ssfs(sds_from_file(opts.camera))
        plot_3d(dataset)

    if opts.output != "":
        with open(opts.output, 'w') as output:
            match opts.output.split('.')[-1]:
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
