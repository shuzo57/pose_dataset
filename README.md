# YOLO Format Conversion and Dataset Creation Tool
This tool consists of two primary functions:

1. convert_to_yolo_format(): This function is used to convert COCO format JSON files to YOLO format.
2. create_dataset(): This function is used to create a dataset for training, by dividing the available data into training and validation sets.

## Installation
No specific installation is required, but Python 3.7 or above is needed to run these scripts. Make sure you have all the necessary libraries installed. If not, you can install them using pip:

```pip install -r requirements.txt```

Note: There is no requirements.txt provided. If you face any ImportError, please install the required modules.

## Usage
### Convert COCO format JSON to YOLO format
To run the conversion script, navigate to the directory containing the script and use the following command:

```python3 convert_to_yolo_format.py --json_path [path_to_COCO_JSON_file] --output_dir [output_directory]```

This script will read in a COCO formatted JSON file and output YOLO formatted text files to the specified directory.

- json_path: This is the path to the input JSON file in COCO format.
- output_dir: This is the directory where the converted files will be saved.
### Create a dataset for training
To run the dataset creation script, navigate to the directory containing the script and use the following command:


```python3 create_dataset.py --images_dir [images_directory] --labels_dir [labels_directory] --output_dir [output_directory] --train_ratio [train_ratio]```

This script divides the data into training and validation sets according to the specified ratio, and copies the files to the appropriate directories.

- images_dir: This is the directory containing the image files.
- labels_dir: This is the directory containing the label files in YOLO format.
- output_dir: This is the directory where the created dataset will be saved.
- train_ratio: This is the proportion of the total data to be used for training. The remaining data will be used for validation.
# Contributing
Contributions are welcome. Please fork the repository and create a pull request with your changes.

# License
This project is licensed under the terms of the MIT license.