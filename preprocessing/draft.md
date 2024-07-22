```python
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns
import setuptools
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import KNNImputer
```


```python
filename = "../data/final_dataset.json"
df = pd.read_json(filename)
df.head()
```


```python
import datetime

year_threshold = datetime.datetime.today().year + 10
# Keep rows where the condition is true 
df = df.loc[
    (df["Price"] < 15000000) &
    (df["ConstructionYear"] <= year_threshold) | (pd.isna(df["ConstructionYear"]))&
    ~((df["GardenArea"] > 0) & (df["Garden"] == 0)) &
    ~((df["GardenArea"] > 0) & (df["Garden"] == 0)) &
    (df["ShowerCount"] < 30) &
    (df["ToiletCount"] < 50) &
    (df["LivingArea"] >= 9) & (df["LivingArea"] <= 2000 ) & 
    ((df["NumberOfFacades"] == 0) | ((df["NumberOfFacades"] >= 2) & (df["NumberOfFacades"] <= 10))) 
    
]
```


```python
df.describe().T

```


    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

## Drop useless


```python
df.drop(["Fireplace", "Furnished","PropertyId","Region", "Country", "SubtypeOfProperty", "Url", "MonthlyCharges"], axis = 1, inplace=True)
df.dropna(subset = ["Locality", "District", "Province", "StateOfBuilding", "LivingArea"], inplace=True)
df.drop_duplicates(inplace= True)

exclude_annuity = ["annuity_monthly_amount", "annuity_without_lump_sum", "annuity_lump_sum", "homes_to_build"]
df = df[~df["TypeOfSale"].isin(exclude_annuity)]


print("There are {} rows of data".format(len(df)))

# if count < 1 : 
#     df.drop(["Fireplace", "Furnished","PropertyId","Region", "Country", "SubtypeOfProperty", "Url", "RoomCount", "ShowerCount"], axis = 1, inplace=True)
#     df.dropna(subset = ["Locality", "District", "Province", "StateOfBuilding", "LivingArea"], inplace=True)
#     df.drop_duplicates(inplace= True)

#     exclude_annuity = ["annuity_monthly_amount", "annuity_without_lump_sum", "annuity_lump_sum", "homes_to_build"]
#     df = df[~df["TypeOfSale"].isin(exclude_annuity)]


# print("There are {} rows of data".format(len(df)))
```

    There are 53058 rows of data
    

### Check outlier 


```python
df.describe().T
```

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }

```python
columns = ["ToiletCount", "SurfaceOfPlot", "ShowerCount", "RoomCount", "NumberOfFacades", "LivingArea", "GardenArea",  "BedroomCount", "BathroomCount" ]
for col in columns : 
    if df[col].dtype != 'O' : 
        # Computing IQR
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1

        # Define bounds for filtering
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Filter the dataframe
        df[col] = df[col].apply(lambda x: x if lower_bound <= x <= upper_bound else None)


```


```python
len(df)

```

```python
df.describe().T

```


```python
df.isna().sum()
```

```python
df.nunique()[df.nunique() > 100]
df["StateOfBuilding"].isna().sum()

```



## Clean locality


```python

pattern = r'\d+|\(\d+\)|\([a-zA-Z]+\)'
regex = re.compile(pattern)
df["Locality"].unique().tolist()
cleaned_localities = [regex.sub('', locality).strip().lower() for locality in df["Locality"]]
df["Cleaned_Localities"] = cleaned_localities
df[["Cleaned_Localities", "Locality"]].isnull()
df.drop(columns= "Locality", inplace= True)
```


```python
df.describe(include="all").T
```


## Preprocessing 
### TypeOfSale and TypeOfProperty


```python
if df["TypeOfSale"].dtype == 'O' : 
    df = df.loc[df['TypeOfSale'].isin(['residential_sale', 'residential_monthly_rent'])]
    df["TypeOfSale"] = df["TypeOfSale"].apply(lambda x : 0 if x == "residential_sale" else 1)

df["TypeOfProperty"] = df["TypeOfProperty"].apply(lambda x : 0 if x == 1 else 1)
```

###  def check_drop_zip_code(self) : 


```python
df.columns
```

```python
# Check if postal code is in Belgium 
postal_code = pd.read_json("../data/zipcode-belgium.json")
valid = set(postal_code["zip"])

# Create a new column 'PostalCodeValid' with True if 'PostalCode' is in 'valid', else False
df["PostalCodeValid"] = df["PostalCode"].apply(lambda x: x in valid)
df = df[df["PostalCodeValid"] == True]
df.drop(columns=["PostalCodeValid"], inplace=True)

```

### Mapping

```python
# Remove the weird values from the PEB 
valid_peb_values = ['D', 'B', 'F', 'E', 'C', 'A', 'G', 'A++', 'A+', None]
df = df.loc[df["PEB"].isin(valid_peb_values)]

# print(df["StateOfBuilding"].unique())

peb_mapping = {'A++': 9, 'A+': 8, 'A': 7, 'B': 6, 'C': 5, 'D': 4, 'E': 3, 'F': 2, 'G': 1}
state_mapping = {'TO_BE_DONE_UP': 0, 'TO_RENOVATE': 1, 'TO_RESTORE': 2, 'GOOD': 3, 'JUST_RENOVATED': 4, 'AS_NEW': 5}
flooding_mapping = {'RECOGNIZED_N_CIRCUMSCRIBED_FLOOD_ZONE': 0, 'RECOGNIZED_FLOOD_ZONE': 1, 'RECOGNIZED_N_CIRCUMSCRIBED_WATERSIDE_FLOOD_ZONE': 2, 'CIRCUMSCRIBED_FLOOD_ZONE': 3, 'CIRCUMSCRIBED_WATERSIDE_ZONE': 4, 'POSSIBLE_N_CIRCUMSCRIBED_FLOOD_ZONE': 5, 'POSSIBLE_FLOOD_ZONE': 6, 'POSSIBLE_N_CIRCUMSCRIBED_WATERSIDE_ZONE': 7, 'NON_FLOOD_ZONE': 8}
kitchen_mapping = {'NOT_INSTALLED': 0, 'USA_UNINSTALLED': 1, 'SEMI_EQUIPPED': 2, 'USA_SEMI_EQUIPPED': 3, 'INSTALLED': 4, 'USA_INSTALLED': 5, 'HYPER_EQUIPPED': 6, 'USA_HYPER_EQUIPPED': 7}

df["PEB"] = df["PEB"].map(peb_mapping)
df["StateOfBuilding"] = df["StateOfBuilding"].map(state_mapping)
df["FloodingZone"] = df["FloodingZone"].map(flooding_mapping)
df["Kitchen"] = df["Kitchen"].map(kitchen_mapping)

df[["PEB", "StateOfBuilding", "FloodingZone","Kitchen" ]]


```



### Fill NaN values 


```python
df.fillna({"Garden" : 0}, inplace=True)
df.fillna({"SwimmingPool" : 0}, inplace=True)
df.fillna({"Terrace" : 0}, inplace=True)

df.loc[(df["Garden"] == 0) & (df["GardenArea"].isna()), "GardenArea"] = 0
df.loc[(df["TypeOfProperty"] == 1) & (df["SurfaceOfPlot"].isna()), "SurfaceOfPlot"] = 0
df.fillna({"ConstructionYear" : 0}, inplace= True)
df.fillna({"MonthlyCharges" : 0}, inplace= True) 

df.isna().sum().to_dict()
```



```python
# Apply KNN Imputer to fill missing values
imputer = KNNImputer(n_neighbors=2)

```


```python
# List of columns to be imputed
columns_to_impute = ["PEB", "StateOfBuilding", "FloodingZone", "Kitchen"]

# Fit and transform the data
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Display the transformed columns
print(df[columns_to_impute].head())
```

```python
# List of columns to be imputed
columns_to_impute = ["BathroomCount", "BedroomCount"]

# Fit and transform the data
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Display the transformed columns
# Fill empty ShowerCount values with BathroomCount values if ShowerCount or ToiletCount is null
df.loc[df["ShowerCount"].isnull(), "ShowerCount"] = df["BathroomCount"]
df.loc[df["ToiletCount"].isnull(), "ToiletCount"] = df["BathroomCount"]
df[columns_to_impute] = round(df[columns_to_impute])
print(df[columns_to_impute].head())
```

       BathroomCount  BedroomCount
    0            1.0           1.0
    1            1.0           2.0
    2            2.0           4.0
    3            0.0           2.0
    4            1.0           1.0
    


```python
# List of columns to be imputed
columns_to_impute = ["SurfaceOfPlot","NumberOfFacades", "RoomCount"]
# columns_to_impute = ["SurfaceOfPlot","NumberOfFacades", "RoomCount"]

# Fit and transform the data
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Display the transformed columns
print(df[columns_to_impute].head())
```

        SurfaceOfPlot  NumberOfFacades  RoomCount
    2             0.0              2.5        1.0
    6           130.0              3.0        5.5
    8             0.0              2.0        1.0
    11            0.0              2.5        1.0
    14            0.0              2.0        1.0
    


```python
# # Ensure all numeric columns are properly typed
# numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()


# # Apply interpolation to fill missing values in numeric columns
# df[numeric_columns] = df[numeric_columns].interpolate(method='linear', axis=0)

df.dtypes
```




    BathroomCount         float64
    BedroomCount          float64
    ConstructionYear      float64
    District               object
    FloodingZone          float64
    Garden                float64
    GardenArea            float64
    Kitchen               float64
    LivingArea            float64
    NumberOfFacades       float64
    PEB                   float64
    PostalCode              int64
    Price                   int64
    Province               object
    RoomCount             float64
    ShowerCount           float64
    StateOfBuilding       float64
    SurfaceOfPlot         float64
    SwimmingPool          float64
    Terrace               float64
    ToiletCount           float64
    TypeOfProperty          int64
    TypeOfSale              int64
    Cleaned_Localities     object
    dtype: object




```python
# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# iterative_imputer = IterativeImputer()
# df[numeric_columns] = iterative_imputer.fit_transform(df[numeric_columns])
```


```python
df["Cleaned_Localities"]
```




    2            zeebrugge
    6              tournai
    8         blankenberge
    11             hasselt
    14          schaerbeek
                  ...     
    181710         lebbeke
    181713         lebbeke
    181783     middelkerke
    181784        mouscron
    181786          wellen
    Name: Cleaned_Localities, Length: 53027, dtype: object




```python
encoder = OneHotEncoder(drop = 'first',sparse_output=False)

df.drop(["Cleaned_Localities"],axis = 1, inplace = True)
df.dtypes

cat_column = df.select_dtypes(include=['object']).columns.tolist()  
print(df.dtypes)

# cat_column = df.select_dtypes(include=['object']).columns.drop(["Url", "Cleaned_Localities", "District", "Province", "SubtypeOfProperty"]).tolist()  

encoded_data = encoder.fit_transform(df[cat_column])

one_hot_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_column))
df = pd.concat([df.reset_index(drop=True), one_hot_df.reset_index(drop=True)], axis=1)
df = df.drop(cat_column, axis=1)

# Drop columns that are entirely zero
df = df.loc[:, (df != 0).any(axis=0)]

df.columns
df.shape
df.head()
```

    BathroomCount       float64
    BedroomCount        float64
    ConstructionYear    float64
    District             object
    FloodingZone        float64
    Garden              float64
    GardenArea          float64
    Kitchen             float64
    LivingArea          float64
    NumberOfFacades     float64
    PEB                 float64
    PostalCode            int64
    Price                 int64
    Province             object
    RoomCount           float64
    ShowerCount         float64
    StateOfBuilding     float64
    SurfaceOfPlot       float64
    SwimmingPool        float64
    Terrace             float64
    ToiletCount         float64
    TypeOfProperty        int64
    TypeOfSale            int64
    dtype: object
    




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>BathroomCount</th>
      <th>BedroomCount</th>
      <th>ConstructionYear</th>
      <th>FloodingZone</th>
      <th>Garden</th>
      <th>GardenArea</th>
      <th>Kitchen</th>
      <th>LivingArea</th>
      <th>NumberOfFacades</th>
      <th>PEB</th>
      <th>...</th>
      <th>Province_Brussels</th>
      <th>Province_East Flanders</th>
      <th>Province_Flemish Brabant</th>
      <th>Province_Hainaut</th>
      <th>Province_Limburg</th>
      <th>Province_Liège</th>
      <th>Province_Luxembourg</th>
      <th>Province_Namur</th>
      <th>Province_Walloon Brabant</th>
      <th>Province_West Flanders</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1969.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>29.0</td>
      <td>2.5</td>
      <td>7.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.0</td>
      <td>NaN</td>
      <td>1920.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>2008.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>111.0</td>
      <td>2.0</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1972.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>92.0</td>
      <td>2.5</td>
      <td>6.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1994.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>50.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 73 columns</p>
</div>




```python
df.isnull().sum()
```




    BathroomCount               0
    BedroomCount                0
    ConstructionYear            0
    FloodingZone                0
    Garden                      0
                               ..
    Province_Liège              0
    Province_Luxembourg         0
    Province_Namur              0
    Province_Walloon Brabant    0
    Province_West Flanders      0
    Length: 73, dtype: int64




```python
# sampled = df.sample(n=500, random_state=42)
# features = sampled.drop(columns = "Price")

# sns.pairplot(features.assign(target=sampled['Price']), diag_kind='kde', corner=True)
# plt.show()
```

### Model


```python

df = df[np.isfinite(df['Price'])]
len(df)
```




    53027




```python
df.isna().sum().to_dict()
```




    {'BathroomCount': 0,
     'BedroomCount': 0,
     'ConstructionYear': 0,
     'FloodingZone': 0,
     'Garden': 0,
     'GardenArea': 1300,
     'Kitchen': 0,
     'LivingArea': 3109,
     'NumberOfFacades': 0,
     'PEB': 0,
     'PostalCode': 0,
     'Price': 0,
     'RoomCount': 0,
     'ShowerCount': 0,
     'StateOfBuilding': 0,
     'SurfaceOfPlot': 0,
     'SwimmingPool': 0,
     'Terrace': 0,
     'ToiletCount': 0,
     'TypeOfProperty': 0,
     'TypeOfSale': 0,
     'District_Antwerp': 0,
     'District_Arlon': 0,
     'District_Ath': 0,
     'District_Bastogne': 0,
     'District_Brugge': 0,
     'District_Brussels': 0,
     'District_Charleroi': 0,
     'District_Dendermonde': 0,
     'District_Diksmuide': 0,
     'District_Dinant': 0,
     'District_Eeklo': 0,
     'District_Gent': 0,
     'District_Halle-Vilvoorde': 0,
     'District_Hasselt': 0,
     'District_Huy': 0,
     'District_Ieper': 0,
     'District_Kortrijk': 0,
     'District_Leuven': 0,
     'District_Liège': 0,
     'District_Maaseik': 0,
     'District_Marche-en-Famenne': 0,
     'District_Mechelen': 0,
     'District_Mons': 0,
     'District_Mouscron': 0,
     'District_Namur': 0,
     'District_Neufchâteau': 0,
     'District_Nivelles': 0,
     'District_Oostend': 0,
     'District_Oudenaarde': 0,
     'District_Philippeville': 0,
     'District_Roeselare': 0,
     'District_Sint-Niklaas': 0,
     'District_Soignies': 0,
     'District_Thuin': 0,
     'District_Tielt': 0,
     'District_Tongeren': 0,
     'District_Tournai': 0,
     'District_Turnhout': 0,
     'District_Verviers': 0,
     'District_Veurne': 0,
     'District_Virton': 0,
     'District_Waremme': 0,
     'Province_Brussels': 0,
     'Province_East Flanders': 0,
     'Province_Flemish Brabant': 0,
     'Province_Hainaut': 0,
     'Province_Limburg': 0,
     'Province_Liège': 0,
     'Province_Luxembourg': 0,
     'Province_Namur': 0,
     'Province_Walloon Brabant': 0,
     'Province_West Flanders': 0}




```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

categorical_attributes = list(df.select_dtypes(include=['object']).columns)
numerical_attributes = list(df.select_dtypes(include=['float64', 'int64']).columns)


```


```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

print(len(df))

X = df.drop(columns=["Price"])
y = df["Price"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

```

    53027
    


```python
regressor = xgb.XGBRegressor(random_state=0)
cross_val_scores = cross_val_score(regressor, X, y, cv=10)
# Print the cross-validation scores for each fold
print("Cross-validation scores for each fold:", cross_val_scores)
# Print the mean cross-validation score
print("Mean cross-validation score:", cross_val_scores.mean())
```

    Cross-validation scores for each fold: [0.75733119 0.74350047 0.73682666 0.72959375 0.75114763 0.77140415
     0.76415175 0.76278454 0.70763826 0.6963166 ]
    Mean cross-validation score: 0.7420695006847382
    


```python
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)


print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Score train: {score_train}')
```

    Mean Absolute Error: 102624.53558546423
    R-squared: 0.7337589859962463
    Score train: 0.9143015146255493
    


```python
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeRegressor

X = df.drop(columns=["Price"])
y = df["Price"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

regressor = DecisionTreeRegressor()
cross_val_scores = cross_val_score(regressor, X, y, cv=10)
# Print the cross-validation scores for each fold
print("Cross-validation scores for each fold:", cross_val_scores)
# Print the mean cross-validation score
print("Mean cross-validation score:", cross_val_scores.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
score_train = regressor.score(X_train, y_train)


print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Score train: {score_train}')

```

    Cross-validation scores for each fold: [0.58791425 0.3551257  0.23920004 0.32503521 0.23961431 0.57930098
     0.4939387  0.43654594 0.54572198 0.60040951]
    Mean cross-validation score: 0.4402806621568908
    Mean Absolute Error: 116731.15134929343
    R-squared: 0.5881185492262159
    Score train: 0.9987872733579325
    


```python

from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

print(len(df))
X = df.drop(columns=["Price"])
y = df["Price"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

regressor = RandomForestRegressor(random_state=0,  n_jobs=-1)
cross_val_scores = cross_val_score(regressor, X, y, cv=10)
# Print the cross-validation scores for each fold
print("Cross-validation scores for each fold:", cross_val_scores)
# Print the mean cross-validation score
print("Mean cross-validation score:", cross_val_scores.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)


print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Score train: {score_train}')
```


```python
from sklearn.ensemble import GradientBoostingRegressor

X = df.drop(columns=["Price"])
y = df["Price"]

regressor = GradientBoostingRegressor(random_state=0)
cross_val_scores = cross_val_score(regressor, X, y, cv=10)
# Print the cross-validation scores for each fold
print("Cross-validation scores for each fold:", cross_val_scores)
# Print the mean cross-validation score
print("Mean cross-validation score:", cross_val_scores.mean())

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
regressor.fit(X_train, y_train)
# Predict and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
score_train = regressor.score(X_train, y_train)
score_test = regressor.score(X_test, y_test)

print(f'Mean Squared Error: {mse}')
print(f'Mean Error: {mse ** 0.5}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Score train: {score_train}')
print(f'Score test: {score_test}')
```
