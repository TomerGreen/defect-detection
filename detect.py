import cv2
import numpy as np
import matplotlib.pyplot as plt
from params import *
from align import align

def histogram_matching(source_img, target_img):
    """
    Transforms the colors of the source image such that the histogram is similar to that of the target
    image.
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
        demo (bool, optional): whether to present plots demonstrating the process

    Returns:
        ndaray: a binary mask representing the detection image. The mask fits the inspected image.
    """

    # Extracted aligned patches
    inspect_crop, ref_crop, translation = align(inspect_img, ref_img)

    # Match the histogram of the inspected image to that of the reference image
    matched_insepct_crop = histogram_matching(inspect_crop, ref_crop)

    # Smooth both images using a color-aware filter
    smooth_inspect_crop = cv2.bilateralFilter(matched_insepct_crop,
                                              d=SMOOTH_KER_SIZE,
                                              sigmaColor=SIGMA_COLOR,
                                              sigmaSpace=SIGMA_SPACE)
    smooth_ref_crop = cv2.bilateralFilter(ref_crop,
                                          d=SMOOTH_KER_SIZE,
                                          sigmaColor=SIGMA_COLOR,
                                          sigmaSpace=SIGMA_SPACE)
    
    # Present the patches and their preprocessed versions
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
    
    # Get the detection signal by taking the absolute difference between the images
    diff = np.abs(np.float32(smooth_inspect_crop) - np.float32(smooth_ref_crop))

    # Suppress the detection signal close to edges 
    suppressed_diff =  diff * edge_proximity_suppression(ref_crop)

    # Binarize the signal using the threshold
    _, detection = cv2.threshold(suppressed_diff, THRESHOLD, 255, cv2.THRESH_BINARY)

    # Use dialation and erosion to connect defect components
    kernel = np.array([[0, 1, 1, 1, 0],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [1, 1, 1, 1, 1],
                        [0, 1, 1, 1, 0]], dtype=np.uint8)
    detection = cv2.dilate(detection, kernel, iterations=2)
    detection = cv2.erode(detection, kernel, iterations=2)
    inplace_detection = np.zeros_like(inspect_img)
    tr_y, tr_x = max(0, translation[0]), max(0, translation[1])
    inplace_detection[tr_y:tr_y + detection.shape[0], tr_x:tr_x + detection.shape[1]] = detection
    inplace_detection = np.uint8(inplace_detection)

    # Show the detection signal before and after edge proximity suppression, and the detection mask.
    if demo:
        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(diff, cmap='plasma', vmin=np.min(suppressed_diff), vmax=np.max(suppressed_diff))
        axes[0].axis('off')
        axes[0].set_title('Simple Diff')
        axes[1].imshow(suppressed_diff, cmap='plasma')
        axes[1].axis('off')
        axes[1].set_title('Suppressed Diff')
        plt.tight_layout()
        plt.show()

        plt.imshow(detection, cmap='gray')
        plt.title('Detection Mask')
        plt.show()
    
    return inplace_detection
