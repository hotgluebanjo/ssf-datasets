import numpy as np
import matplotlib.pyplot as plt

SSFS = "cameras/arri_alexa.txt"
CUBE_SIZE = 5
DELIMITER = ' '
MAGIC_EXPOSURE_CONSTANT = 12.5
PLOT_POINTS = True

def gaussian(x, center, size):
    return np.exp(-np.power((x - center) / size, 2.0))

def wavelength_to_skypanel(w):
    # https://www.desmos.com/calculator/gvyahvtfam
    return np.array([
        gaussian(w, 628.0, 15.0),
        gaussian(w, 523.0, 25.0),
        gaussian(w, 453.0, 15.0)])

def logc_encode(v):
    return np.where(
        v > 0.010591,
        0.247190 * np.log10(5.555556 * v + 0.052272) + 0.385537,
        5.367655 * v + 0.092809)

def logc_decode(v):
    return np.where(
        v > 0.1496582,
        (np.power(10.0, (v - 0.385537) / 0.2471896) - 0.052272) / 5.555556,
        (v - 0.092809) / 5.367655)

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

    max_v = np.array([camera[400][0], camera[400][1], camera[400][2]])
    for v in camera.values():
        if v[0] > max_v[0]: max_v[0] = v[0]
        if v[1] > max_v[1]: max_v[1] = v[1]
        if v[2] > max_v[2]: max_v[2] = v[2]

    for w in camera.keys():
        camera[w] /= max_v

    plot_data = []

    output = open(SSFS.split('/')[-1].split('.')[0] + "_out.txt", 'w')
    grid = np.linspace(0.0, 1.0, CUBE_SIZE)
    for b in grid:
        for g in grid:
            for r in grid:
                ts = np.zeros((3))
                for wavelength in range(400, 700+10, 10):
                    stimulus = np.sum(wavelength_to_skypanel(wavelength) * np.array([r, g, b]))
                    ts += stimulus * camera[wavelength]
                for stop in range(-5, 5+1):
                    scaled = (ts * np.power(2.0, stop)) # / logc_decode(1.0) * MAGIC_EXPOSURE_CONSTANT
                    res = logc_encode(scaled)
                    output.write(f"{res[0]}{DELIMITER}{res[1]}{DELIMITER}{res[2]}\n")
                    plot_data.append(res)
    output.close()

    if PLOT_POINTS:
        plot_data = np.array(plot_data)
        scatter_plot(plot_data)

if __name__ == "__main__":
    main()
