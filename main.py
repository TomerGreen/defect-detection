import os
import cv2
import argparse
from detect import detect_defects
import matplotlib.pyplot as plt


def main():
    """
    Runs defect detection
    """

    # Create ArgumentParser
    parser = argparse.ArgumentParser(description='Detects defects in a chip image by comparing to a reference image')

    # Add arguments
    parser.add_argument('inspect_impath', type=str, help='path to the inspected chip image')
    parser.add_argument('ref_impath', type=str, help='path to reference chip image')
    parser.add_argument('--demo', action='store_true', help='whether to present plots along the process')

    # Parse the arguments
    args = parser.parse_args()

    # Checks the files exist
    if not os.path.exists(args.inspect_impath):
        raise FileNotFoundError(f"The file '{args.inspect_impath}' does not exist.")
    if not os.path.exists(args.ref_impath):
        raise FileNotFoundError(f"The file '{args.ref_impath}' does not exist.")

    # Get the images
    inspect_img = cv2.imread(args.inspect_impath, cv2.IMREAD_GRAYSCALE)
    ref_img = cv2.imread(args.ref_impath, cv2.IMREAD_GRAYSCALE)

    # Run detection
    detection = detect_defects(inspect_img, ref_img, demo=args.demo)
    
    # Save detection image
    filepath, extension = os.path.splitext(args.inspect_impath)
    detection_impath = os.path.join(filepath + '_detection' + extension)
    cv2.imwrite(detection_impath, detection)
    print(detection_impath)

    # Make sure it worked
    if os.path.exists(detection_impath):
        print(f"Detection mask saved in path '{detection_impath}'")
        # det_img = cv2.imread(detection_impath, cv2.IMREAD_GRAYSCALE)
        # plt.imshow(det_img, cmap='gray')
        # plt.axis('off')
        # plt.tight_layout()
        # plt.show()

    else:
        print("Something went wrong. Detection image not saved.")


if __name__ == "__main__":
    main()