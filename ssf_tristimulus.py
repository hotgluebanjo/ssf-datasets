import numpy as np
import matplotlib.pyplot as plt

SSFS = "cameras/sony_ilce_7.txt"
CUBE_SIZE = 5
DELIMITER = ' '
PLOT_POINTS = True

def gaussian(x, center, size):
    return np.exp(-np.power((x - center) / size, 2.0))

# Relative SPD of approximate RGB SkyPanel. Not RGBW.
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
    camera = {}
    with open(SSFS, 'r') as file:
        for line in file:
            parts = line.split(DELIMITER)
            wavelength = int(parts[0])
            values = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            camera[wavelength] = values

    gray_patch = np.zeros((3))
    for wavelength in range(400, 700+1, 10):
        gray_patch += camera[wavelength] * 0.18 * arri_skypanel(wavelength, 1.0, 1.0, 1.0)
    exp_wb_coeff = 0.18 / gray_patch

    plot_data = []

    output = open(SSFS.split('/')[-1].split('.')[0] + "_out.txt", 'w')
    grid = np.linspace(0.0, 1.0, CUBE_SIZE)
    for b in grid:
        for g in grid:
            for r in grid:
                ts = np.zeros((3))
                for wavelength in range(400, 700+1, 10):
                    ts += camera[wavelength] * arri_skypanel(wavelength, r, g, b)
                for stop in range(-5, 5+1):
                    scaled = ts * exp_wb_coeff * np.power(2.0, stop)
                    res = logc_encode(scaled)
                    output.write(f"{res[0]}{DELIMITER}{res[1]}{DELIMITER}{res[2]}\n")
                    plot_data.append(res)
    output.close()

    if PLOT_POINTS:
        plot_data = np.array(plot_data)
        scatter_plot(plot_data)

if __name__ == "__main__":
    main()
