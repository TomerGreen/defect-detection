import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

def main():
    img1 = cv2.imread('./home exercise/non_defective_examples/case3_inspected_image.tif')
    img2 = cv2.imread('./home exercise/non_defective_examples/case3_reference_image.tif')
    print(img1.shape)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Non defective inspected')
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Non defective reference')

    img1 = cv2.imread('./home exercise/defective_examples/case1_inspected_image.tif')
    img2 = cv2.imread('./home exercise/defective_examples/case1_reference_image.tif')
    print(img1.shape)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Non defective inspected')
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Non defective reference')

    img1 = cv2.imread('./home exercise/defective_examples/case2_inspected_image.tif')
    img2 = cv2.imread('./home exercise/defective_examples/case2_reference_image.tif')
    print(img1.shape)
    fig, axes = plt.subplots(1, 2)
    axes[0].imshow(img1)
    axes[0].axis('off')
    axes[0].set_title('Non defective inspected')
    axes[1].imshow(img2)
    axes[1].axis('off')
    axes[1].set_title('Non defective reference')

    # Adjust layout and display images
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":

    main()