<p align="center">
    <br>
    <a href="https://github.com/KriszgruberL" target="_blank"> <img alt="Made with Frogs" src="./assets/made-with-🐸.svg" style="border-radius:0.5rem"></a>
    <br>
    <br><br>
    <a><img src="./assets/logo-modified.png" width="350"  /></a>
    <h2 align="center">Using:
    <br>
    <br>
    <a href="https://www.python.org/downloads/release/python-3120/" target="_blank"><img alt="Python 3.12" src="https://img.shields.io/badge/Python%203.12-python?style=for-the-badge&logo=python&logoColor=F8E71C&labelColor=427EC4&color=2680D1" style="border-radius:0.5rem"></a>
    <a href="https://scikit-learn.org/stable/user_guide.html" target="_blank"><img alt="Sklearn" src="https://img.shields.io/badge/sklearn%20-%20sklearn?style=for-the-badge&logo=sklearn&color=blue" style="border-radius:0.5rem"></a>
    <a href="https://pandas.pydata.org/docs/" target="_blank"><img alt="Pandas" src="https://img.shields.io/badge/Pandas-Pandas?style=for-the-badge&logo=pandas&color=61B3DD" style="border-radius:0.5rem"></a>
    <br>
</p>

## BeCode red line project - Immo_Eliza 3/4

1. [Scrapping](https://github.com/KriszgruberL/Immo_Eliza)
2. [Data Analysis](https://github.com/KriszgruberL/Immo_Eliza_Data_Analysis)
3. [Preprocessing and Machine Learning](https://github.com/KriszgruberL/Immo_eliza_ML)
4. [API and Deployment](https://github.com/KriszgruberL/Immo_Eliza_front)
   
## 📚 Overview

This project focuses on preprocessing and analyzing real estate data to predict property prices. The preprocessing pipeline includes data filtering, encoding, imputations, and scaling to prepare the data for machine learning models. The main model used is a RandomForestRegressor, but the pipeline is designed to be flexible for other regressors as well.

## 📁 Link to .pkl

<a href="https://drive.google.com/drive/folders/1P9cWejusu_b2_qOeUjrPpoTJ-sfiN_W0?usp=sharing" target="_blank"> <img alt="GoogleDrive" src="https://img.shields.io/badge/Google_Drive%20-%20Google_Drive?style=for-the-badge&logo=googledrive&labelColor=%23ADD8E6%09&color=%236495ED%09" style="border-radius:0.5rem"></a>

## 🚧 Project Structure
```
nom_projet/
├── .vscode/
│   ├── settings.json
├── data/
│   ├── final_dataset.json
│   ├── zipcode-belgium.json
│   ├── preprocessed_df.pkl
│   ├── model.pkl
├── preprocessing/
│   ├── modeling.py
│   ├── preprocessing.py
│   ├── draft2.ipynb
├── README.md
└── requirements.txt
└── Immo_Eliza.pptx
```

## ⚒️ Setup

1. **Clone the repository**:
    ```sh
    https://github.com/KriszgruberL/Immo_eliza_ML.git
    cd Immo_eliza_ML
    ```

2. **Create a virtual environment**:
    ```sh
    python -m venv venv
    source venv/bin/activate  
    # On Windows use `venv\Scripts\activate`
    ```

3. **Install the required packages**:
    ```sh
    pip install -r requirements.txt
    ```

## ⚙️ Usage

1. To run the preprocessing, execute the `preprocessing.py` script:
    ```sh
    python preprocessing.py
    ```

1. To run the modeling, execute the `modeling.py` script:
    ```sh
    python modeling.py
    ```


### 👀 Classes Overview

---
#### **modeling.py**
Contain the modeling features

**Functions:**
- `load_preprocessed_data`: This function loads the preprocessed data from a pickle file. 

- `prepare_data`: This function separates the features and the target variable from the DataFrame. 

- `print_score`: This function prints the performance metrics of the model. 
- `main`: This function is the main entry point of the script. It loads the data, prepares it, trains the model, and evaluates it. 

#### **preprocessing.py**
Contain the preprocessing features

**Functions:**
- ``filter_data``: This function filters the data based on specific conditions, such as price, construction year, garden area, and counts of showers and toilets.

- ``encode_binary_value``: This function encodes binary values for specific columns. 

- ``impute_and_encode``: This function imputes missing values and encodes categorical variables. 

- ``map_values``: This function maps values for specific columns to binary values. 

- ``limit_number_of_facades``: This function limits the number of facades to a maximum of 4. 

- ``fill_na``: This function fills missing values for specific columns with default values. 

- ``knn_impute``: This function imputes missing values using KNN imputation. 

- ``drop_categorical``: This function drops categorical columns. 

- `preprocess_pipeline`: This function creates a preprocessing pipeline. 

- `main`: This function is the main entry point of the script. It loads the data, applies preprocessing, and saves the preprocessed data. 


#### **data/**
Contains data files used and generated by the scripts.

- `final_dataset.json`: Data scrapped from ImmoWeb.be
- `zipcode-belgium.json`: Valid zip code in Belgium
- `preprocessed_df.pkl`: Preprocessing saved as a pickle file
- `preprocessed_df.pkl`: Model saved as a pickle file

---
#### **Other Files**

- `.gitignore`: Specifies which files and directories to ignore in Git.
- `README.md`: Provides an overview and instructions for the project.
- `requirements.txt`: Lists required Python packages and their versions.
- `Immo_Eliza_pp.pptx`: Short powerpoint

---

## 🎯 Requirements

- `pandas==2.2.2`
- `numpy==2.0.0`
- `seaborn==0.11.2`
- `scikit-learn`
- `setuptools`
