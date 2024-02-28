import cv2
import numpy as np
import matplotlib.pyplot as plt
from params import *
from align import align

def histogram_matching(source_img, target_img):
    """
    Transforms the historgram of the source image such that it is similar to that of the target image.
    """
    # Compute histograms
    source_hist = cv2.calcHist([source_img], [0], None, [256], [0,256])
    target_hist = cv2.calcHist([target_img], [0], None, [256], [0,256])

    # Normalize histograms
    source_hist /= source_img.size
    target_hist /= target_img.size

    # Compute cumulative distribution functions (CDFs)
    source_cdf = source_hist.cumsum()
    target_cdf = target_hist.cumsum()

    # Map intensities
    mapping = np.zeros(256)
    for i in range(256):
        mapping[i] = np.argmin(np.abs(target_cdf - source_cdf[i]))

    # Apply mapping to source image
    matched_img = mapping[source_img].astype(np.uint8)

    return matched_img


def edge_proximity_suppression(img):
    """
    Produces a mask for edge proximity suppression. The function takes the reference image and applies
    Canny edge detection. It then produces a suppression map such that each pixel's detection score
    is suppressed according to its proximity to an edge. This is later used to suppress falsely detected
    defects that are due to small variation in the chip edge location.

    Args:
        img (ndarray): a grayscale image

    Returns:
        ndaray: the suppression mask.
    """

    # Apply Canny edge detection
    edges = cv2.Canny(img, CANNY_FIRST_THRESHOLD, CANNY_SECOND_THRESHOLD)

    # Efficiently measure L2 disatance from nearest edge
    edge_proximity = cv2.distanceTransform(cv2.bitwise_not(edges), cv2.DIST_L2, 5)

    # Creates suppression mask using parameters
    suppression_mask = 1 - (MAX_SUPPRESSION / ((edge_proximity / SUPPRESSION_DECAY) + 1))

    return suppression_mask


def detect_defects(inspect_img, ref_img, demo=False):
    """Detects defects in a chip image by comparing it to a reference image.

    Args:
        inspect_img (ndarray): the chip image
        ref_img (ndarray): the reference chip image
        demo (bool, optional): Whether 
    """
    inspect_crop, ref_crop = align(inspect_img, ref_img)
    inspect_crop = histogram_matching(inspect_crop, ref_crop)
    if demo:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(inspect_crop, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Inspect Patch')
        axes[1].imshow(ref_crop, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Reference Patch')
        plt.tight_layout()
        plt.show()
    smooth_inspect_crop = cv2.bilateralFilter(inspect_crop,
                                              d=SMOOTH_KER_SIZE,
                                              sigmaColor=SIGMA_COLOR,
                                              sigmaSpace=SIGMA_SPACE)
    smooth_ref_crop = cv2.bilateralFilter(ref_crop,
                                          d=SMOOTH_KER_SIZE,
                                          sigmaColor=SIGMA_COLOR,
                                          sigmaSpace=SIGMA_SPACE)
    if demo:

        plt.subplot(2, 2, 1)  # First subplot
        plt.imshow(inspect_crop, vmin=np.min(ref_crop), vmax=np.max(ref_crop), cmap='gray')
        plt.axis('off')
        plt.title('Original Inspect Patch')

        plt.subplot(2, 2, 2)  # Second subplot
        plt.imshow(ref_crop, vmin=np.min(ref_crop), vmax=np.max(ref_crop), cmap='gray')
        plt.axis('off')
        plt.title('Original Reference Patch')

        plt.subplot(2, 2, 3)  # Third subplot
        plt.imshow(smooth_inspect_crop, vmin=np.min(ref_crop), vmax=np.max(ref_crop), cmap='gray')
        plt.axis('off')
        plt.title('Preprocessed Inspect Patch')

        plt.subplot(2, 2, 4)  # Fourth subplot
        plt.imshow(smooth_ref_crop, vmin=np.min(ref_crop), vmax=np.max(ref_crop), cmap='gray')
        plt.axis('off')
        plt.title('Preprocessed Reference Patch')

        plt.tight_layout()
        plt.show()

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(smooth_inspect_crop, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title('Inspect After Preprocessing')
        axes[1].imshow(smooth_ref_crop, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title('Reference After Preprocessing')
        plt.tight_layout()
        plt.show()
    
    diff = np.abs(np.float32(smooth_inspect_crop) - np.float32(smooth_ref_crop))
    # diff = cv2.bilateralFilter(diff,
    #                             d=DIFF_SMOOTH_KER_SIZE,
    #                             sigmaColor=DIFF_SIGMA_COLOR,
    #                             sigmaSpace=DIFF_SIGMA_SPACE)
    # diff = cv2.GaussianBlur(diff, (DIFF_SMOOTH_KER_SIZE, DIFF_SMOOTH_KER_SIZE), 0)
    suppressed_diff =  diff * edge_proximity_suppression(ref_crop)
    _, detection = cv2.threshold(suppressed_diff, THRESHOLD, 255, cv2.THRESH_BINARY)
    kernel = np.array([[0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0]], dtype=np.uint8)
    detection = cv2.dilate(detection, kernel, iterations=2)
    detection = cv2.erode(detection, kernel, iterations=2)

    fig, axes = plt.subplots(1, 2)
    if demo:
        axes[0].imshow(diff, cmap='plasma')
        axes[0].axis('off')
        axes[0].set_title('diff')
        axes[1].imshow(suppressed_diff, cmap='plasma', vmin=np.min(diff), vmax=np.max(diff))
        axes[1].axis('off')
        axes[1].set_title('suppressed diff')
        plt.tight_layout()
        plt.show()
        plt.imshow(detection, cmap='gray')
        plt.show()
