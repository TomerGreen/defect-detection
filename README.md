# Defect Detection

This project detects defects in chip images

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/TomerGreen/defect-detection.git
   ```
3. Navigate to the project and install the requirements:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the project, navigate to the project directory and run:
```bash
python main.py path/to/insepected_image.tif path/to/reference_image.tif
```
The defect detection mask will be saved as path/to/insepected_image_detection.tif
To also show matplotlib plots that present the detection process, run
```bash
python main.py path/to/insepected_image.tif path/to/reference_image.tif --demo
```
