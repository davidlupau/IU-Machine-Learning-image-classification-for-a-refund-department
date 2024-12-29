# IU Machine Learning image classification for a refund department
 University project that involves image classification for a refund department

- Dataset used : Retail Products Classification from Kaggle  https://www.kaggle.com/competitions/retail-products-classification/overview
- Packages to install: refer to [requirements.txt](requirements.txt) and run the below code to install all of them
```python
source venv/bin/activate
pip install -r requirements.txt
```

## Intructions
Download the dataset, unzip it and save the content in a folder called Dataset using the names below
```
Dataset/
├── test_images/
├── train_images/
├── test_styles.csv
└── styles.csv
```
## 1. Explore dataset.
See instructions [here](notebooks/01_data_exploration.ipynb)

## 2. Prepare dataset
Run [prepare_dataset.py](prepare_dataset.py)
More informations [here](notebooks/02_dataset_preparation.ipynb)

## 3. Feature extraction
All the necessary functions are explained [here](notebooks/03_feature_extraction.ipynb).
The function is used in [prepare_dataset.py](prepare_dataset.py) file.

_Optional:_ analyze features
[analyze_features.py](analyze_features.py) provides insights into how Random Forest classifier makes its decisions by examining which features contribute most significantly to the classification process

## 4. Train model
Run [model_training.py](model_training.py) and see this [notebook](notebooks/04_model_training.ipynb) for more details.
The model is saved as random_forest_model.joblib

## 5. Model Deployment and API Implementation
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