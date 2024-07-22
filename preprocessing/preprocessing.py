from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import KNNImputer, SimpleImputer
import datetime

filename = "../data/final_dataset.json"
df = pd.read_json(filename)
df.head()

year_threshold = datetime.datetime.today().year + 10
# Keep rows where the condition is true 
df = df.loc[
    (df["Price"] < 15000000) &
    (df["ConstructionYear"] <= year_threshold) | (pd.isna(df["ConstructionYear"]))&
    ~((df["GardenArea"] > 0) & (df["Garden"] == 0)) &
    (df["ShowerCount"] < 30) &
    (df["ToiletCount"] < 50)     
]

df.drop(["Fireplace", "Furnished","PropertyId","Region", "Country", "SubtypeOfProperty", "Url", "MonthlyCharges", "RoomCount", "Locality", "District", "Province"], axis = 1, inplace=True)
df.dropna(subset=["StateOfBuilding", "LivingArea"], how='all', inplace=True)
df.drop_duplicates(inplace= True)

exclude_annuity = ["annuity_monthly_amount", "annuity_without_lump_sum", "annuity_lump_sum", "homes_to_build"]
df = df[~df["TypeOfSale"].isin(exclude_annuity)]


if df["TypeOfSale"].dtype == 'O' : 
    df = df.loc[df['TypeOfSale'].isin(['residential_sale', 'residential_monthly_rent'])]
    df["TypeOfSale"] = df["TypeOfSale"].apply(lambda x : 0 if x == "residential_sale" else 1)

df["TypeOfProperty"] = df["TypeOfProperty"].apply(lambda x : 0 if x == 1 else 1)

# Subset the plausible value 
df = df.loc[df["PEB"].isin(['D', 'B', 'F', 'E', 'C', 'A', 'G', 'A++', 'A+', None])]


columns_to_impute = ['PEB', 'StateOfBuilding']
imputer = SimpleImputer(missing_values=None, strategy='constant', fill_value='Unknown')
df[columns_to_impute] = pd.DataFrame(imputer.fit_transform(df[columns_to_impute]), columns=columns_to_impute)

ordinal_encoders = {
    'PEB': OrdinalEncoder(categories=[['Unknown','A++', 'A+', 'A', 'B', 'C', 'D', 'E', 'F', 'G']]),
    'StateOfBuilding': OrdinalEncoder(categories=[['Unknown','AS_NEW','JUST_RENOVATED','GOOD','TO_RESTORE','TO_RENOVATE','TO_BE_DONE_UP']])
}

for column, encoder in ordinal_encoders.items():
    df[column] = encoder.fit_transform(df[[column]])



binary_kitchen_mapping = {None : 0 , 'USA_HYPER_EQUIPPED': 1,'NOT_INSTALLED': 0,'USA_UNINSTALLED': 0,'SEMI_EQUIPPED': 1,'USA_SEMI_EQUIPPED': 1,'INSTALLED': 1,'USA_INSTALLED': 1,'HYPER_EQUIPPED': 1}
df['Kitchen'] = df['Kitchen'].map(binary_kitchen_mapping) # Apply the mapping to create the new binary column


binary_flooding_mapping = {None : 0 ,'NON_FLOOD_ZONE': 0,'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE': 0,'RECOGNIZED_FLOOD_ZONE': 1,'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 1,'CIRCUMSCRIBED_FLOOD_ZONE': 1,'CIRCUMSCRIBED_WATERSIDE_ZONE': 1,'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 1,'POSSIBLE_FLOOD_ZONE': 1,'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 1}
df['FloodingZone'] = df['FloodingZone'].map(binary_flooding_mapping) # Apply the mapping to create the new binary column

df.loc[df['NumberOfFacades'] > 4, 'NumberOfFacades'] = 4

df.fillna({"Garden": 0, "SwimmingPool": 0, "Terrace": 0}, inplace=True)
df.loc[(df["TypeOfProperty"] == 1) & (df["SurfaceOfPlot"].isna()), "SurfaceOfPlot"] = 0
df.loc[(df["Garden"] == 0) & (df["GardenArea"].isna()), "GardenArea"] = 0

imputer = KNNImputer(n_neighbors=2)
columns_to_impute = ["StateOfBuilding", "FloodingZone", "Kitchen", "PEB", "SurfaceOfPlot", "ConstructionYear", "NumberOfFacades" ]

# Fit and transform the data
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])
for col in columns_to_impute : 
    df[col] = round(df[col])

# Prepare the data
X = df.drop(columns=["Price"], axis=1)
y = df["Price"]

# Initialize the regressor
regressor = RandomForestRegressor(random_state=0, n_jobs=-1)

# Perform cross-validation
cross_val_scores = cross_val_score(regressor, X, y, cv=10)
print("Cross-validation scores for each fold:", cross_val_scores)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
score_train = regressor.score(X_train, y_train)

print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Score train: {score_train}')