# Automated Product Returns Classification System

## Project Context
This project was developed as part of the "From Model to Production Environment" module in the Bachelor of Science in Data Science program at IU International University of Applied Sciences. The module focuses on implementing machine learning solutions in production environments, bridging the gap between theoretical model development and practical application.

## Problem Statement
An e-commerce platform specializing in sustainable products faces increasing challenges with returns processing as their business grows. The manual categorization of returned items has become a significant bottleneck, requiring unsustainable workforce expansion. This project implements a machine learning solution that automatically classifies returned products into 21 distinct categories using image analysis.
The system features:
- Automated image classification using Random Forest with 94% accuracy
- RESTful API for both single-item and batch processing
- Overnight batch processing capability for efficient returns handling
- Confidence threshold filtering for reliable categorization
- Comprehensive error handling and logging

This solution aims to streamline the returns process, reduce manual sorting requirements, and enable efficient daily categorization of incoming returned items.

## Approach
This project implements an automated image classification system for processing returned items in an e-commerce context. The system architecture follows a modular design with four main components: Data Ingestion, Processing, ML Pipeline, and Outputs Handling. New product images are collected in a designated directory and processed automatically through a scheduled cronjob. The processing pipeline includes batch handling and feature extraction, where important image characteristics such as color distributions, shapes, and textures are computed. These features are then fed into a trained Random Forest model for classification. The system outputs include timestamped CSV files containing classification results and an organized archive of processed images. This implementation emphasizes automation, scalability, and robust data management, making it suitable for production environments where consistent processing of returned items is required.

Key features:
- Automated batch processing through cronjob scheduling
- Comprehensive feature extraction for image classification
- Systematic data organization with separate directories for new, processed, and archived items
- Detailed logging and error handling for production reliability

![image](https://github.com/user-attachments/assets/bd1f354a-8be0-452f-b722-2c74c9cebb08)

- Dataset used : [Retail Products Classification](https://www.kaggle.com/competitions/retail-products-classification/overview) from Kaggle
- Packages to install: refer to [requirements.txt](requirements.txt) 

## Instructions
- Create a virtual environment
```python
python3 -m venv venv
source venv/bin/activate
```

- Install packages
```python
source venv/bin/activate
pip install -r requirements.txt
```

- Download the dataset, unzip it and save the content in a folder called Dataset using the names and structure below.
- Images stored in train_images will be used to train the model. Images from test_images folder can be used later on to test batch processing.
```
Dataset/
├── test_images/
├── train_images/
├── test_styles.csv
└── styles.csv
```
## 1. Explore dataset
See instructions [here](notebooks/01_data_exploration.ipynb)

## 2. Prepare dataset
- Run [prepare_dataset.py](prepare_dataset.py)
- More informations are available in the notebook [here](notebooks/02_dataset_preparation.ipynb)

## 3. Feature extraction
All the necessary functions are explained [here](notebooks/03_feature_extraction.ipynb).
Run [prepare_dataset.py](prepare_dataset.py).

_Optional:_ analyze features
[analyze_features.py](analyze_features.py) provides insights into how Random Forest classifier makes its decisions by examining which features contribute most significantly to the classification process

## 4. Train model
Run [model_training.py](model_training.py) and see this [notebook](notebooks/04_model_training.ipynb) for more details.
The model is saved as random_forest_model.joblib.

## 5. Model Deployment and API Implementation
An overview of the process in available in [05_model_deployment.ipynb](notebooks/05_model_deployment.ipynb) notebook
1. Run api.py and wait for api to be active
```python
 * Serving Flask app 'api'
 * Debug mode: off
```
2. Update [test_api.py](test_api.py) line 11 with a picture name
```python
image_path = "Dataset/train_images/image_name.jpg"
```
3. Run [test_api.py](test_api.py) to get predictions

## 6. Set up batch processing
Create a cron job and automate batch processing by following the steps explained in [06_batch_processing.ipynb](notebooks/06_batch_processing.ipynb)
