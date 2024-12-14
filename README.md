# Segmentation of Blood Vessels in Eye Fundus Images

## Overview
This project implements a segmentation pipeline to detect and highlight blood vessels in eye fundus images. The tool can assist in medical image analysis, particularly for diagnosing and monitoring retinal conditions such as diabetic retinopathy or glaucoma.

## Features
- Efficient segmentation of blood vessels from eye fundus images.
- Test script included for validating the segmentation process.
- Input dataset support with example images provided in the `EyeFundus_input` folder.

## Project Structure
```
segmentation_of_blood_vessels-main
├── segmentation_blood_vessels.py        # Main script for segmentation
├── segmentation_blood_vessels_test.py  # Test script for validation
├── EyeFundus_input/                    # Input folder with sample fundus images
├── .gitignore                          # Git ignore file
└── README.md                           # Project documentation
```

## Prerequisites
- Python 3.x
- Required libraries (install via pip):
  - `numpy`
  - `opencv-python`
  - `matplotlib`

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/segmentation_of_blood_vessels.git
   cd segmentation_of_blood_vessels
   ```

2. Install dependencies:
   ```bash
   pip install --user opencv-python-headless numpy matplotlib
   ```

## Usage
1. Place your input eye fundus images in the `EyeFundus_input` folder.
2. Run the segmentation script:
   ```bash
   python segmentation_blood_vessels.py
   ```
3. Outputs will be saved or displayed with segmented blood vessels highlighted.

## Testing
To run the tests and verify the pipeline:
```bash
python segmentation_blood_vessels_test.py
```

## Contributing
Contributions are welcome! Please fork the repository and submit a pull request for any enhancements or fixes.


## Acknowledgements
Special thanks to the open-source community and datasets used for developing and testing this tool.
