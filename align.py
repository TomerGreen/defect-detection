import numpy as np
import matplotlib.pyplot as plt
import cv2
from params import *


def find_translation(img1, img2):
    """
    finds the the translation that aligns the two images
    """

    # Compute the 2D Discrete Fourier Transform (DFT) of both images
    f_img1 = np.fft.fft2(img1)
    f_img2 = np.fft.fft2(img2)

    # Compute the cross-power spectrum
    cross_power_spectrum = f_img1 * np.conj(f_img2)

    # Compute the phase correlation by taking the inverse Fourier transform
    phase_correlation_map = np.fft.ifft2(cross_power_spectrum)

    # Find the peak value (maximum correlation) in the correlation map
    peak = np.unravel_index(np.argmax(phase_correlation_map), phase_correlation_map.shape)

    # Determine the translation vector
    translation = np.array(peak)
    translation[0] -= 0 if translation[0] < (img1.shape[0] / 2) else img1.shape[0]
    translation[1] -= 0 if translation[1] < (img1.shape[1] / 2) else img1.shape[1]

    return translation


def align(img1, img2):
    """Takes two unaligned images and returns crops (img1_crop, img2_crop) that are aligned.

    Args:
        img1 (ndarray): image 1
        img2 (ndarray): image 2

    Returns:
        ndarray, ndarray, translation: the two aligned patches and the (x,y) of the translation
    """
    translation = find_translation(img1, img2)
    img1_crop = img1[max(translation[0], 0): img1.shape[0] + min(translation[0], 0),
                     max(translation[1], 0): img1.shape[1] + min(translation[1], 0)]
    img2_crop = img2[max(-translation[0], 0): img1.shape[0] + min(-translation[0], 0),
                     max(-translation[1], 0): img1.shape[1] + min(-translation[1], 0)]
    # assert(img1_crop.shape == img2_crop.shape, "Aligned patches differ in shape")
    return img1_crop, img2_crop, translation


if __name__ == "__main__":
    pass
