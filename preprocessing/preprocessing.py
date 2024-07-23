import datetime
import pandas as pd
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
import pickle

# Define preprocessing functions
def filter_data(df):
    
    year_threshold = datetime.datetime.today().year + 10
    df = df.loc[
        (df["Price"] < 15000000) &
        ((df["ConstructionYear"] <= year_threshold) | pd.isna(df["ConstructionYear"])) &
        ~((df["GardenArea"] > 0) & (df["Garden"] == 0)) &
        (df["ShowerCount"] < 30) &
        (df["ToiletCount"] < 50)
    ]
    
    df = df.drop(["Fireplace", "Furnished","PropertyId","Region", "Country", "SubtypeOfProperty", "Url", "MonthlyCharges", "RoomCount"], axis = 1)
    df = df.dropna(subset=['Locality', 'District', "StateOfBuilding", "LivingArea"], how='all')
    df = df.drop_duplicates()
    
    exclude_annuity = ["annuity_monthly_amount", "annuity_without_lump_sum", "annuity_lump_sum", "homes_to_build"]
    df = df[~df["TypeOfSale"].isin(exclude_annuity)]
    
    return df

def encode_binary_value(df):
    df = df.loc[df['TypeOfSale'].isin(['residential_sale', 'residential_monthly_rent'])].copy()
    df.loc[:, "TypeOfSale"] = df["TypeOfSale"].apply(lambda x: 0 if x == "residential_sale" else 1)
    df.loc[:, "TypeOfProperty"] = df["TypeOfProperty"].apply(lambda x: 0 if x == 1 else 1)
    
    return df


def impute_and_encode(df):
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


def map_values(df):
    binary_kitchen_mapping = {None: 0, 'USA_HYPER_EQUIPPED': 1, 'NOT_INSTALLED': 0, 'USA_UNINSTALLED': 0, 'SEMI_EQUIPPED': 1, 'USA_SEMI_EQUIPPED': 1, 'INSTALLED': 1, 'USA_INSTALLED': 1, 'HYPER_EQUIPPED': 1}
    df.loc[:, 'Kitchen'] = df['Kitchen'].map(binary_kitchen_mapping)
    
    binary_flooding_mapping = {None: 0, 'NON_FLOOD_ZONE': 0, 'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE': 0, 'RECOGNIZED_FLOOD_ZONE': 1, 'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 1, 'CIRCUMSCRIBED_FLOOD_ZONE': 1, 'CIRCUMSCRIBED_WATERSIDE_ZONE': 1, 'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 1, 'POSSIBLE_FLOOD_ZONE': 1, 'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 1}
    df.loc[:, 'FloodingZone'] = df['FloodingZone'].map(binary_flooding_mapping)
    return df


def limit_number_of_facades(df):
    df.loc[df['NumberOfFacades'] > 4, 'NumberOfFacades'] = 4
    return df

def fill_na(df):
    df.fillna({"Garden": 0, "SwimmingPool": 0, "Terrace": 0}, inplace=True)
    df.loc[(df["TypeOfProperty"] == 1) & (df["SurfaceOfPlot"].isna()), "SurfaceOfPlot"] = 0
    df.loc[(df["Garden"] == 0) & (df["GardenArea"].isna()), "GardenArea"] = 0
    return df

def knn_impute(df):
    imputer = KNNImputer(n_neighbors=2)
    columns_to_impute = ["StateOfBuilding", "FloodingZone", "Kitchen", "PEB", "SurfaceOfPlot", "ConstructionYear", "NumberOfFacades"]
    df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
    for col in columns_to_impute: 
        df[col] = round(df[col])
    return df

def drop_categorical(df) : 
    df.drop(["Locality", "District", "Province"], axis = 1, inplace= True)
    return df 

def preprocess_pipeline():
    return Pipeline([
        ('filter_data', FunctionTransformer(filter_data)),
        ('encode_binary_value', FunctionTransformer(encode_binary_value)),
        ('impute_and_encode', FunctionTransformer(impute_and_encode)),
        ('map_values', FunctionTransformer(map_values)),
        ('limit_number_of_facades', FunctionTransformer(limit_number_of_facades)),
        ('fill_na', FunctionTransformer(fill_na)),
        ('knn_impute', FunctionTransformer(knn_impute)),
        ('drop_categorical', FunctionTransformer(drop_categorical))
    ])

def main():
    
    from datetime import datetime
    start_time = datetime.now()

    filename = "../data/final_dataset.json"
    df = pd.read_json(filename)
    df = df.copy()

    # Apply preprocessing
    pipeline = preprocess_pipeline()
    preprocessed_df = pipeline.fit_transform(df)
    
    # Save the preprocessed data
    with open("../data/preprocessed_df.pkl", "wb") as f:
        pickle.dump(preprocessed_df, f)
        
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

if __name__ == "__main__":
    main()
