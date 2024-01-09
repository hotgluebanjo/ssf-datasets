import colour
import numpy as np
import os
import scipy

# --------- TEMP CONFIG -----------
#
# SSF criteria:
#   - Quantum efficiency included
#   - Testing illuminant excluded
#   - Increment <= 10nm for best precision

TRANSFER_FUNCTION = colour.models.log_encoding_ARRILogC3

SSFS = "cameras/arri_alexa.txt"
OUTPUT = "res.txt"

CUBE_SIZE = 5
SWEEP_MIN = -5.0
SWEEP_MAX = 5.0
SWEEP_INCREMENT = 1

PLOT = True

# --------------------------------

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

def lerp(a, b, t):
    return (1.0 - t) * a + t * b
mix = lerp

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

data = sds_from_file("illuminants/incandescent_abs.txt")
print(data.values)
