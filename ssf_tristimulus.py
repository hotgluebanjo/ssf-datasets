import numpy as np

SSFS = "cameras/arri_alexa.txt"
CUBE_SIZE = 5
DELIMITER = ' '

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

def main():
    camera = {}
    with open(SSFS, 'r') as file:
        for line in file:
            parts = line.split(DELIMITER)
            wavelength = int(parts[0])
            values = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
            camera[wavelength] = values

    grid = np.linspace(0.0, 1.0, CUBE_SIZE)
    with open(SSFS.split('/')[-1], 'w') as output:
        for b in grid:
            for g in grid:
                for r in grid:
                    ts = np.zeros((3))
                    for wavelength in range(400, 700 + 10, 10):
                        ts += np.sum(wavelength_to_skypanel(wavelength) * np.array([r, g, b])) * camera[wavelength]
                    for stop in range(-5, 6):
                        scaled = ts * np.power(2.0, float(stop))
                        res = logc_encode(scaled)
                        output.write(f"{res[0]}{DELIMITER}{res[1]}{DELIMITER}{res[2]}\n")

if __name__ == "__main__":
    main()
