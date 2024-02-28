from align import find_translation, align
from detect import histogram_matching, detect_defects
import cv2
import matplotlib.pyplot as plt

def test_alignment(img1, img2):

    fig, axes = plt.subplots(1, )
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Inspected')
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Reference')
    plt.tight_layout()
    plt.show()

    img1_crop, img2_crop = align(img1, img2)
    print("crop shape: ", img1_crop.shape)

    fig1, axes1 = plt.subplots(1, 2)
    axes1[0].imshow(img1_crop)
    axes1[0].axis('off')
    axes1[0].set_title('Inspected Crop')
    axes1[1].imshow(img2_crop)
    axes1[1].axis('off')
    axes1[1].set_title('Reference Crop')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    img1 = cv2.imread('./home exercise/non_defective_examples/case3_inspected_image.tif', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./home exercise/non_defective_examples/case3_reference_image.tif', cv2.IMREAD_GRAYSCALE)
    # test_alignment(img1, img2)
    detect_defects(img1, img2, demo=True)

    img1 = cv2.imread('./home exercise/defective_examples/case1_inspected_image.tif', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./home exercise/defective_examples/case1_reference_image.tif', cv2.IMREAD_GRAYSCALE)
    # test_alignment(img1, img2)
    detect_defects(img1, img2, demo=True)

    img1 = cv2.imread('./home exercise/defective_examples/case2_inspected_image.tif', cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread('./home exercise/defective_examples/case2_reference_image.tif', cv2.IMREAD_GRAYSCALE)
    # test_alignment(img1, img2)
    detect_defects(img1, img2, demo=True)

