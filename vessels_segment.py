import os

import math
import numpy as np
import matplotlib.pyplot as plt
import cv2

from skimage import io, exposure, img_as_ubyte
from skimage.color import rgb2gray
from skimage.morphology.footprints import disk
from sklearn.metrics import accuracy_score, ConfusionMatrixDisplay, jaccard_score
from scipy.ndimage import gaussian_laplace


def normalise(image, out_low, out_up):
    """
    Normalising an image to a specific range from out_low to out_up.

    Parameters
    ----------
    image : array
        An image array.
    out_low : int
        Lower boundary of the image to be normalised to.
    out_up : int
        Upper boundary of the image to be normalised to.

    Returns
    -------
    out : array
        A normalised image.
    """
    in_low = np.min(image)
    in_up = np.max(image)
    return (image - in_low) * ((out_up - out_low) / (in_up - in_low)) + out_low


def sub2ind(y, x):
    """
    Reformat the indexes to be compatible with numpy.

    Parameters
    ----------
    y : np.ndarray
        Row indexes.
    x : np.ndarray
        Column indexes.

    Returns
    -------
    res : tuple or x
        Returns (y, x) or x depends on input dimension.
    """
    y = y.astype(np.int8) - 1
    x = x.astype(np.int8) - 1
    res = (y, x) if np.any(y) else x
    return res


def line(line_len, degrees=0):
    """
    Constructs and returns a line structuring element.

    Parameters
    ----------
    line_len : int
        Length of the line structuring element.
    degrees : int
        Degree of orientation of the line. (Default 0)

    Returns
    -------
    line : ndarray
        Line structuring element.
    """
    deg90 = degrees % 90
    if deg90 > 45:
        alpha = math.pi * (90 - deg90) / 180
    else:
        alpha = math.pi * deg90 / 180
    ray = (line_len - 1) / 2

    c = round(ray * math.cos(alpha)) + 1
    r = round(ray * math.sin(alpha)) + 1

    # Line rasterisation
    shape = (r, c) if r > 1 else c
    basic_line = np.zeros(shape)

    m = math.tan(alpha)
    x = np.arange(1, c + 1)
    y = r - np.floor(m * (x - 0.5))
    indexes = sub2ind(y, x)

    basic_line[indexes] = 1

    # Preparing blocks
    line_strip = basic_line[:(c - 1)] if r == 1 else basic_line[0, :(c - 1)]
    line_rest = np.array([]) if r == 1 else basic_line[1, :(c - 1)] if r == 2 else basic_line[1:r, :(c - 1)]

    shape = (r - 1, c) if r > 2 else c
    z = np.zeros(shape)

    def multiRow(arr):
        return len(arr.shape) > 1 and arr.shape[0] > 1

    # Assembling blocks
    h1 = np.hstack((z, line_rest[::-1, ::-1] if multiRow(line_rest) else line_rest[::-1]))
    h2 = np.hstack((line_strip, [1], line_strip[::-1, ::-1] if multiRow(line_strip) else line_strip[::-1]))
    h3 = np.hstack((line_rest, z[::-1, ::-1] if multiRow(z) else z[::-1]))

    def stackable(a1, a2, a3):
        """
        Determines if the arrays are stackable.
        Parameters
        ----------
        a1 : array
            A 1-D array.
        a2 : ndarray
            A 1-D array.
        a3 : ndarray
            A 1-D array.

        Returns
        -------
        out : bool
            If all the arrays are stackable vertically.
        """
        s1 = a1.shape
        s2 = a2.shape
        s3 = a3.shape
        am1 = np.argmax(s1)
        am2 = np.argmax(s2)
        am3 = np.argmax(s3)

        return (am1 >= am2 and am2 <= am3) and (s1[am1] == s2[am2] and s2[am2] == s3[am3])

    res = np.vstack((h1, h2, h3)).astype(np.uint8) if stackable(h1, h2, h3) else np.array([h2.astype(np.uint8)])

    # Rotate/transpose/flip
    sect = math.floor((degrees % 180) / 45)
    if sect == 1:
        res = np.transpose([res] if res.ndim < 2 else res)
    elif sect == 2:
        res = np.rot90([res] if res.ndim < 2 else res, 1)
    elif res.ndim >= 2 and sect == 3:
        res = np.fliplr([res] if res.ndim < 2 else res)
    if res.ndim == 2 and np.min(res.shape) == 1:
        mx = np.max(res.shape) // 2
        dm = np.argmax(res.shape)
        wd1, wd2 = (mx, mx), (0, 0)
        res = np.pad(res, (wd1, wd2) if dm == 1 else (wd2, wd1))
    return res


def plot_image(title, axes, image, fig, nbins=128, hist=True):
    """
    Plot images and histogram with given titles.

    Parameters
    ----------
    title : str
        Title of the image plotted.
    axes : tuple or Axes
        Tuple of Axes or single Axes object.
    image : ndarray
        Image to be plotted.
    fig : Fig
        Figure to be plot on.
    nbins : int, optional
        Number of bins for the histogram. (Default 128)
    hist : bool, optional
        Flag to indicate if histogram needs to be plotted. (Default True)

    Returns
    -------
    ax_img, ax_hist : tuple(Axes)
        Axes of plotted image and histogram.
    """

    not_3D = len(image.shape) < 3

    v = image.flatten() if not_3D else np.reshape(image, (-1, image.shape[-1]))

    ax_img, ax_hist = (None, None)
    if hist:
        ax_img, ax_hist = axes
    else:
        ax_img = axes

    ax_img.set_title(title)
    imret = ax_img.imshow(image, cmap="gray")
    fig.colorbar(imret, orientation="horizontal",
                 ax=ax_img)  # Place a colour bar to know what is the current colour range.

    if hist:
        immin = np.min(image)
        immax = np.max(image)

        nml = (immin >= 0 and 0.1 <= immax <= 1)
        is_eightbit = (immin >= 0 and 10 <= immax <= 255)
        bins = np.linspace(0 if (nml or is_eightbit) else immin, 1 if nml else (255 if is_eightbit else immax),
                           nbins)

        if not_3D:
            ax_hist.hist(v, bins, color='k', histtype='step')
        else:
            ax_hist.hist(v, bins, color=['r', 'g', 'b'], histtype='step')

        ax_hist.set_xlabel('Intensity')
        ax_hist.set_ylabel('Frequency')

    return ax_img, ax_hist


def extract(im_path, gt_path):
    """
    Segment and extract vessels from retinal images.

    Parameters
    ----------
    im_path : str
        The path to a retina image
    gt_path : str
        The path to the ground truth of a retina image
    """
    im = io.imread(im_path)
    im_gray = rgb2gray(im)
    mask = np.zeros(im_gray.shape)
    mask[im_gray >= 0.07] = 1

    # Extract the green channel.
    im_green = im[..., 1]

    # Contrast enhancing by histogram equalisation
    equalised = exposure.equalize_adapthist(im_green)

    equalised = equalised * mask

    # Edge detection
    edge = gaussian_laplace(equalised, sigma=4, mode='reflect')

    # Thresholding

    # Clip value minimum value to 0 for local adaptive thresholding.
    cv_edge = np.array(edge)
    cv_edge[cv_edge < 0] = 0
    cv_edge = img_as_ubyte(normalise(cv_edge, 0, 255).astype(np.ubyte))

    # Segments image using adaptive threshold method.
    segmented = cv_edge > cv2.adaptiveThreshold(cv_edge, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 877, 0)

    image = segmented.astype(np.float64) * mask

    # Initialise a temporary array to store results of opening using each line orientation

    """ Title: The Multiscale Bowler-Hat Transform for Blood Vessel Enhancement in Retinal Images
        Author: Çiğdem Sazak and Carl J. Nelson and Boguslaw Obara
        Date: 2008
        Code version: 1.0
        Availability: https://github.com/CigdemSazak/bowler-hat-2d """
    holder = np.zeros((image.shape[0], image.shape[1], 12))
    for i in range(12):
        holder[:, :, i] = cv2.morphologyEx(image, cv2.MORPH_OPEN, line(45, i * 15), iterations=1)

    combined = np.max(holder, axis=2)
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, disk(5), iterations=1)

    y_test = img_as_ubyte(io.imread(gt_path) // 255)

    filename = os.path.basename(gt_path).split('.')[0]

    y_pred = closed.flat
    y_test_f = y_test.flat
    j_score = jaccard_score(y_test_f, y_pred)

    fig, (ax_IM, ax_GT, ax_segmented) = plt.subplots(1, 3, figsize=(18, 6))

    plot_image("Original Image", ax_IM, im, fig, hist=False)
    plot_image("Ground Truth", ax_GT, y_test, fig, hist=False)
    plot_image("Segmented, Jaccard score = {:.2f}".format(j_score), ax_segmented, closed, fig, hist=False)

    fig.suptitle(filename, fontsize=20, fontweight="bold", y=0.93)

    plt.show()

    return j_score


if __name__ == "__main__":
    IM_dir = "./IM/"
    GT_dir = "./GT/"

    IM_entries = os.listdir(IM_dir)
    GT_entries = os.listdir(GT_dir)

    scores = []
    for IM_file, GT_file in zip(IM_entries, GT_entries):
        scores.append(extract(os.path.join(IM_dir, IM_file), os.path.join(GT_dir, GT_file)))

    print("Scores Summary\n-----------------")
    print("Maximum : {:.2f}".format(np.max(scores)))
    print("Minimum : {:.2f}".format(np.min(scores)))
    print("Mean : {:.2f}".format(np.mean(scores)))
    print("Median : {:.2f}".format(np.median(scores)))
