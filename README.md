# Defect Detection

This project detects defects in chip images

## Installation

1. Clone this repository: git clone https://github.com/TomerGreen/defect-detection.git
2. Install the requirements: pip install -r requirements.txt

## Usage

To run the project, navigate to the project directory and run:
python main.py path/to/insepected_image.tif path/to/reference_image.tif
The defect detection mask will be saved as path/to/insepected_image_detection.tif
If you want to present matplotlib plots that show the detection process, add the flag --demo when running main.py
