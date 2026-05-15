import numpy as np
import pywt
import matplotlib.pyplot as plt
import nibabel as nib

####################################################
# Part I functions
def scale_magnitude(arr):
    """Min-max scale array to [0, 1] for display."""
    mag = np.abs(arr).astype(float)
    lo, hi = mag.min(), mag.max()
    if hi == lo:
        return np.zeros_like(mag)
    return (mag - lo) / (hi - lo)


def load_volume(path):
    """Load a NIfTI file and return its data array as float32."""
    img = nib.load(str(path))
    data = np.array(img.dataobj, dtype=np.float32)
    print(f"Loaded {path.name}  shape={data.shape}  dtype={img.get_data_dtype()}")
    return data

def get_center_slices(vol):
    """Return the center index along each spatial axis."""
    return (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)


def extract_slices(vol, ix, iy, iz):
    """
    Return sagittal, coronal, and axial slices.

    NIfTI standard RAS orientation:
      axis 0  ->  Left-Right         (sagittal: fix axis 0)
      axis 1  ->  Posterior-Anterior (coronal:  fix axis 1)
      axis 2  ->  Inferior-Superior  (axial:    fix axis 2)
    """
    sagittal = vol[ix, :, :]
    coronal  = vol[:, iy, :]
    axial    = vol[:, :, iz]
    return sagittal, coronal, axial



####################################################
# Part II functions
def simple_compression(arr, s):
    """
    Takes an array and a number s (0-100).
    Sets the lowest-magnitude s% of values to 0 (simple compression).
    """
    arr_flat = arr.flatten()
    threshold = np.percentile(np.abs(arr_flat), s)
    compressed = np.where(np.abs(arr) <= threshold, 0, arr)
    return compressed


def mse(arr, ground_truth):
    """
    Computes the mean square error between two arrays,
    where ground_truth is the reference.
    """
    return np.mean((np.array(arr) - np.array(ground_truth)) ** 2)


def plot_wavelet(arr, wavelet_name, levels=2):
    """
    Takes an array (2D image) and a wavelet domain name.
    Plots the approximation and detail images for the given
    wavelet transform at two levels, concatenated on one set of axes.
    """
    coeffs = pywt.wavedec2(arr, wavelet=wavelet_name, level=levels)

    # coeffs[0] = approximation at coarsest level
    # coeffs[1..n] = (cH, cV, cD) detail tuples per level (finest last)
    titles = []
    images = []

    # Approximation (coarsest)
    images.append(coeffs[0])
    titles.append(f"Approx (Level {levels})")

    # Details from coarsest to finest
    for lvl_idx, (cH, cV, cD) in enumerate(coeffs[1:], start=1):
        level_num = levels - lvl_idx + 1
        images += [cH, cV, cD]
        titles += [
            f"Horizontal Detail (L{level_num})",
            f"Vertical Detail (L{level_num})",
            f"Diagonal Detail (L{level_num})",
        ]

    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img, cmap="gray", aspect="auto")
        ax.set_title(title, fontsize=9)
        ax.axis("off")

    plt.suptitle(f"Wavelet: {wavelet_name} — {levels} Levels", fontsize=12)
    plt.tight_layout()
    plt.show()