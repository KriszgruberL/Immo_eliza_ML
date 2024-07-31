import datetime
import json
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
import pickle

def filter_data(df : pd.DataFrame) -> pd.DataFrame:
    """
    Filter the data based on specific conditions.

    Parameters:
    df (pd.DataFrame): The input data containing real estate information.

    Returns:
    pd.DataFrame: The filtered data.
    """
    year_threshold = datetime.datetime.today().year + 10
    df = df.loc[
        (df["Price"] < 15000000) &
        ((df["ConstructionYear"] <= year_threshold) | pd.isna(df["ConstructionYear"])) 
    ]
    
    df = df.drop(["Fireplace", "Furnished", "PropertyId","Country", "SubtypeOfProperty", "Url", "MonthlyCharges", "RoomCount"], axis=1)
    df = df.dropna(subset=['Locality', 'District', "StateOfBuilding", "LivingArea"], how='all')
    df = df.dropna(subset=['TypeOfSale', "TypeOfProperty", "Province", "Region"])
    df = df.drop_duplicates()
    exclude_annuity = ["annuity_monthly_amount", "annuity_without_lump_sum", "annuity_lump_sum", "homes_to_build"]
    df = df[~df["TypeOfSale"].isin(exclude_annuity)]
    return df

def encode_binary_value(df : pd.DataFrame) -> pd.DataFrame:
    """
    Encode binary values for specific columns.

    Parameters:
    df (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The data with binary values encoded.
    """
    df = df.loc[df['TypeOfSale'].isin(['residential_sale', 'residential_monthly_rent'])].copy()
    df.loc[:, "TypeOfSale"] = df["TypeOfSale"].apply(lambda x: 0 if x == "residential_sale" else 1)
    df.loc[:, "TypeOfProperty"] = df["TypeOfProperty"].apply(lambda x: 0 if x == 1 else 1)
    df["TypeOfSale"] = df["TypeOfSale"].astype(float)
    return df

def impute_and_encode(df : pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values and encode categorical variables.

    Parameters:
    df (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The data with imputed and encoded values.
    """
    valid_peb_values = ['D', 'B', 'F', 'E', 'C', 'A', 'G', 'A++', 'A+', None]
    df = df.loc[df["PEB"].isin(valid_peb_values)].copy()

    simple_imputer = SimpleImputer(missing_values=None, strategy='constant', fill_value='Unknown')
    df.loc[:, 'PEB'] = simple_imputer.fit_transform(df[['PEB']])
    ordinal_encoder = OrdinalEncoder(categories=[['Unknown', 'A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']])
    df.loc[:, "PEB"] = ordinal_encoder.fit_transform(df[["PEB"]])

    simple_imputer = SimpleImputer(missing_values=None, strategy='constant', fill_value='Unknown')
    df.loc[:, 'StateOfBuilding'] = simple_imputer.fit_transform(df[['StateOfBuilding']])
    ordinal_encoder = OrdinalEncoder(categories=[['Unknown', 'AS_NEW', 'JUST_RENOVATED', 'GOOD', 'TO_RESTORE', 'TO_RENOVATE', 'TO_BE_DONE_UP']])
    df.loc[:, "StateOfBuilding"] = ordinal_encoder.fit_transform(df[["StateOfBuilding"]])
    
    return df

def map_values(df : pd.DataFrame) -> pd.DataFrame:
    """
    Map values for specific columns to binary values.

    Parameters:
    df (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The data with mapped values.
    """
    binary_kitchen_mapping = {None: 0, 'USA_HYPER_EQUIPPED': 1, 'NOT_INSTALLED': 0, 'USA_UNINSTALLED': 0, 'SEMI_EQUIPPED': 1, 'USA_SEMI_EQUIPPED': 1, 'INSTALLED': 1, 'USA_INSTALLED': 1, 'HYPER_EQUIPPED': 1}
    df.loc[:, 'Kitchen'] = df['Kitchen'].map(binary_kitchen_mapping)

    binary_flooding_mapping = {None: 0, 'NON_FLOOD_ZONE': 0, 'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE': 0, 'RECOGNIZED_FLOOD_ZONE': 1, 'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 1, 'CIRCUMSCRIBED_FLOOD_ZONE': 1, 'CIRCUMSCRIBED_WATERSIDE_ZONE': 1, 'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 1, 'POSSIBLE_FLOOD_ZONE': 1, 'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 1}
    df.loc[:, 'FloodingZone'] = df['FloodingZone'].map(binary_flooding_mapping)
    return df

def limit_number_of_facades(df : pd.DataFrame) -> pd.DataFrame:
    """
    Limit the number of facades to a maximum of 4.

    Parameters:
    df (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The data with limited number of facades.
    """
    df.loc[df['NumberOfFacades'] > 4, 'NumberOfFacades'] = 4
    return df

def fill_na(df : pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing values for specific columns with default values.

    Parameters:
    df (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The data with filled missing values.
    """
    df.fillna({"Garden": 0, "SwimmingPool": 0, "Terrace": 0}, inplace=True)
    df.loc[(df["TypeOfProperty"] == 1) & (df["SurfaceOfPlot"].isna()), "SurfaceOfPlot"] = 0
    df.loc[(df["Garden"] == 0) & (df["GardenArea"].isna()), "GardenArea"] = 0
    return df

def hot_encode(df: pd.DataFrame) -> pd.DataFrame:
    """
    One-hot encodes the 'Province' and 'Region' columns of the given DataFrame.

    Parameters:
        df (pd.DataFrame): The DataFrame to be one-hot encoded.

    Returns:
        pd.DataFrame: The DataFrame with the 'Province' and 'Region' columns one-hot encoded.
    """

    encoder = OneHotEncoder(sparse_output=False)
    encoded_array = encoder.fit_transform(df[['Region', 'Province']])
    df_encoded = pd.DataFrame(encoded_array, columns=encoder.get_feature_names_out(['Region', 'Province']))
    df_encoded = df_encoded.astype(int)
    df = pd.concat([df.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

    df = df.drop(['Region', 'Province'], axis=1)
    return df

def knn_impute(df : pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values using KNN imputation.

    Parameters:
    df (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The data with imputed values.
    """
    imputer = KNNImputer(n_neighbors=2)
    columns_to_impute = ["StateOfBuilding", "FloodingZone", "Kitchen", "PEB", "SurfaceOfPlot", "ConstructionYear", "NumberOfFacades"]
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    for col in columns_to_impute: 
        df[col] = round(df[col])
    return df

def drop_categorical(df : pd.DataFrame) -> pd.DataFrame:
    """
    Drop categorical columns.

    Parameters:
    df (pd.DataFrame): The input data.

    Returns:
    pd.DataFrame: The data without categorical columns.
    """
    df.drop(["Locality", "District"], axis=1, inplace=True)
    return df


def preprocess_pipeline() -> Pipeline:
    """
    Create a preprocessing pipeline.

    Returns:
    sklearn.pipeline.Pipeline: The preprocessing pipeline.
    """
    return Pipeline([
        ('filter_data', FunctionTransformer(filter_data)),
        ('map_values', FunctionTransformer(map_values)),
        ('encode_binary_value', FunctionTransformer(encode_binary_value)),
        ('impute_and_encode', FunctionTransformer(impute_and_encode)),  # Handles specific imputations and encodings
        ('fill_na', FunctionTransformer(fill_na)),  # General missing value handling
        ('limit_number_of_facades', FunctionTransformer(limit_number_of_facades)),  # Specific transformation
        ('knn_impute', FunctionTransformer(knn_impute)),  # Further imputation
        ('hot_encode', FunctionTransformer(hot_encode)),  # One-hot encoding
        ('drop_categorical', FunctionTransformer(drop_categorical)),  # Remove original categorical columns
    ])

def main() -> None:
    """
    The main function to load data, apply preprocessing, and save the preprocessed data.
    """
    start_time = datetime.datetime.now()

    # Load the data
    filename = "data/final_dataset.json"
    df = pd.read_json(filename)
    df = df.copy()

    # Apply preprocessing
    pipeline = preprocess_pipeline()
    preprocessed_df = pipeline.fit_transform(df)
    
    preprocessed_df.to_csv("data/preprocessed_df.csv", index=False)

    print(preprocessed_df.dtypes)
    
    # Save the preprocessed data
    with open("data/preprocessed_df.pkl", "wb") as f:
        pickle.dump(preprocessed_df, f)
        
    end_time = datetime.datetime.now()

    print('Duration: {}'.format(end_time - start_time))

if __name__ == "__main__":
    main()
