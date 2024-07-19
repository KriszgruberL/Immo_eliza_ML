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

count = 0
```


```python
import datetime

year_threshold = datetime.datetime.today().year + 10
# Keep rows where the condition is true 
df = df.loc[
    (df["ConstructionYear"] <= year_threshold) | (pd.isna(df["ConstructionYear"]))&
    ~((df["GardenArea"] > 0) & (df["Garden"] == 0)) &
    ~((df["GardenArea"] > 0) & (df["Garden"] == 0)) &
    (df["ShowerCount"] < 30) &
    (df["ToiletCount"] < 50) &
    (df["LivingArea"] >= 9) & (df["LivingArea"] <= 2000 ) & 
    ((df["NumberOfFacades"] == 0) | ((df["NumberOfFacades"] >= 2) & (df["NumberOfFacades"] <= 10)))
]
```

## Drop useless


```python


df.drop(["Fireplace", "Furnished","PropertyId","Region", "Country", "SubtypeOfProperty", "Url"], axis = 1, inplace=True)
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

    There are 53061 rows of data
    


```python
df.isna().sum()
```




    BathroomCount        1885
    BedroomCount            0
    ConstructionYear     8034
    District                0
    FloodingZone        20946
    Garden              40318
    GardenArea          40318
    Kitchen             16369
    LivingArea              0
    Locality                0
    MonthlyCharges      47355
    NumberOfFacades     12439
    PEB                  8822
    PostalCode              0
    Price                   0
    Province                0
    RoomCount           38013
    ShowerCount         21662
    StateOfBuilding         0
    SurfaceOfPlot       25916
    SwimmingPool        30934
    Terrace             19048
    ToiletCount          4540
    TypeOfProperty          0
    TypeOfSale              0
    dtype: int64




```python
df.nunique()[df.nunique() > 100]
df["StateOfBuilding"].isna().sum()

```




    np.int64(0)



## Clean locality


```python
if count < 1 : 
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
      <th>count</th>
      <th>unique</th>
      <th>top</th>
      <th>freq</th>
      <th>mean</th>
      <th>std</th>
      <th>min</th>
      <th>25%</th>
      <th>50%</th>
      <th>75%</th>
      <th>max</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>BathroomCount</th>
      <td>51176.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.318157</td>
      <td>0.976977</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>58.0</td>
    </tr>
    <tr>
      <th>BedroomCount</th>
      <td>53061.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.833644</td>
      <td>1.672839</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>3.0</td>
      <td>100.0</td>
    </tr>
    <tr>
      <th>ConstructionYear</th>
      <td>45027.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1981.879717</td>
      <td>41.827213</td>
      <td>1753.0</td>
      <td>1960.0</td>
      <td>1990.0</td>
      <td>2021.0</td>
      <td>2027.0</td>
    </tr>
    <tr>
      <th>District</th>
      <td>53061</td>
      <td>43</td>
      <td>Brussels</td>
      <td>7833</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>FloodingZone</th>
      <td>32115</td>
      <td>9</td>
      <td>NON_FLOOD_ZONE</td>
      <td>30751</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Garden</th>
      <td>12743.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>GardenArea</th>
      <td>12743.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>724.348113</td>
      <td>3462.557692</td>
      <td>1.0</td>
      <td>60.0</td>
      <td>180.0</td>
      <td>575.0</td>
      <td>150000.0</td>
    </tr>
    <tr>
      <th>Kitchen</th>
      <td>36692</td>
      <td>8</td>
      <td>INSTALLED</td>
      <td>16563</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>LivingArea</th>
      <td>53061.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>172.88174</td>
      <td>1553.820799</td>
      <td>10.0</td>
      <td>91.0</td>
      <td>130.0</td>
      <td>195.0</td>
      <td>355500.0</td>
    </tr>
    <tr>
      <th>MonthlyCharges</th>
      <td>5706.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>86.902383</td>
      <td>120.641954</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>50.0</td>
      <td>130.0</td>
      <td>1500.0</td>
    </tr>
    <tr>
      <th>NumberOfFacades</th>
      <td>40622.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>2.789572</td>
      <td>0.87442</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>15.0</td>
    </tr>
    <tr>
      <th>PEB</th>
      <td>44239</td>
      <td>18</td>
      <td>B</td>
      <td>10123</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>PostalCode</th>
      <td>53061.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>4878.002224</td>
      <td>3113.103042</td>
      <td>1000.0</td>
      <td>2018.0</td>
      <td>4000.0</td>
      <td>8310.0</td>
      <td>9992.0</td>
    </tr>
    <tr>
      <th>Price</th>
      <td>53061.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>418873.072181</td>
      <td>509889.696006</td>
      <td>30.0</td>
      <td>210000.0</td>
      <td>319000.0</td>
      <td>469000.0</td>
      <td>15000000.0</td>
    </tr>
    <tr>
      <th>Province</th>
      <td>53061</td>
      <td>11</td>
      <td>West Flanders</td>
      <td>9111</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>RoomCount</th>
      <td>15048.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>6.597887</td>
      <td>5.92604</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>6.0</td>
      <td>11.0</td>
      <td>68.0</td>
    </tr>
    <tr>
      <th>ShowerCount</th>
      <td>31399.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.797764</td>
      <td>4.22061</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>439.0</td>
    </tr>
    <tr>
      <th>StateOfBuilding</th>
      <td>53061</td>
      <td>6</td>
      <td>GOOD</td>
      <td>27052</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>SurfaceOfPlot</th>
      <td>27145.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1305.812894</td>
      <td>9214.311465</td>
      <td>0.0</td>
      <td>170.0</td>
      <td>405.0</td>
      <td>931.0</td>
      <td>950774.0</td>
    </tr>
    <tr>
      <th>SwimmingPool</th>
      <td>22127.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>0.062774</td>
      <td>0.242561</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>Terrace</th>
      <td>34013.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>ToiletCount</th>
      <td>48521.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>80.544877</td>
      <td>17398.099807</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3832365.0</td>
    </tr>
    <tr>
      <th>TypeOfProperty</th>
      <td>53061.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>1.488419</td>
      <td>0.499871</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>TypeOfSale</th>
      <td>53061</td>
      <td>2</td>
      <td>residential_sale</td>
      <td>47355</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>Cleaned_Localities</th>
      <td>53061</td>
      <td>2826</td>
      <td>antwerpen</td>
      <td>1544</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



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
# Check if postal code is in Belgium 
postal_code = pd.read_json("../zipcode-belgium.json")
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


if count < 1 : 
    df["PEB"] = df["PEB"].map(peb_mapping)
    df["StateOfBuilding"] = df["StateOfBuilding"].map(state_mapping)
    df["FloodingZone"] = df["FloodingZone"].map(flooding_mapping)
    df["Kitchen"] = df["Kitchen"].map(kitchen_mapping)

df[["PEB", "StateOfBuilding", "FloodingZone","Kitchen" ]]


```




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
      <th>PEB</th>
      <th>StateOfBuilding</th>
      <th>FloodingZone</th>
      <th>Kitchen</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>2</th>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>6</th>
      <td>4.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>8</th>
      <td>6.0</td>
      <td>3</td>
      <td>8.0</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>11</th>
      <td>6.0</td>
      <td>5</td>
      <td>8.0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>14</th>
      <td>3.0</td>
      <td>5</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>181710</th>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>181713</th>
      <td>NaN</td>
      <td>3</td>
      <td>NaN</td>
      <td>6.0</td>
    </tr>
    <tr>
      <th>181783</th>
      <td>5.0</td>
      <td>4</td>
      <td>NaN</td>
      <td>4.0</td>
    </tr>
    <tr>
      <th>181784</th>
      <td>4.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
    <tr>
      <th>181786</th>
      <td>7.0</td>
      <td>3</td>
      <td>NaN</td>
      <td>2.0</td>
    </tr>
  </tbody>
</table>
<p>53030 rows × 4 columns</p>
</div>



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




    {'BathroomCount': 1885,
     'BedroomCount': 0,
     'ConstructionYear': 0,
     'District': 0,
     'FloodingZone': 20916,
     'Garden': 0,
     'GardenArea': 0,
     'Kitchen': 16365,
     'LivingArea': 0,
     'MonthlyCharges': 0,
     'NumberOfFacades': 12421,
     'PEB': 8822,
     'PostalCode': 0,
     'Price': 0,
     'Province': 0,
     'RoomCount': 37982,
     'ShowerCount': 21646,
     'StateOfBuilding': 0,
     'SurfaceOfPlot': 0,
     'SwimmingPool': 0,
     'Terrace': 0,
     'ToiletCount': 4540,
     'TypeOfProperty': 0,
     'TypeOfSale': 0,
     'Cleaned_Localities': 0}




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

        PEB  StateOfBuilding  FloodingZone  Kitchen
    2   5.5              3.0           8.0      4.0
    6   4.0              3.0           8.0      6.0
    8   6.0              3.0           8.0      4.0
    11  6.0              5.0           8.0      4.0
    14  3.0              5.0           8.0      6.0
    


```python
# List of columns to be imputed
columns_to_impute = ["BathroomCount"]

# Fit and transform the data
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Display the transformed columns
# Fill empty ShowerCount values with BathroomCount values if ShowerCount or ToiletCount is null
# df.loc[df["ShowerCount"].isnull(), "ShowerCount"] = df["BathroomCount"]
df.loc[df["ToiletCount"].isnull(), "ToiletCount"] = df["BathroomCount"]

print(df[columns_to_impute].head())
```

        BathroomCount
    2             1.0
    6             6.0
    8             2.0
    11            0.0
    14            1.0
    


```python
# List of columns to be imputed
columns_to_impute = ["SurfaceOfPlot","NumberOfFacades"]
# columns_to_impute = ["SurfaceOfPlot","NumberOfFacades", "RoomCount"]

# Fit and transform the data
df[columns_to_impute] = imputer.fit_transform(df[columns_to_impute])

# Display the transformed columns
print(df[columns_to_impute].head())
```

        SurfaceOfPlot  NumberOfFacades
    2             0.0              3.5
    6           130.0              3.0
    8             0.0              2.0
    11            0.0              3.5
    14            0.0              2.0
    


```python
# # Ensure all numeric columns are properly typed
# numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()


# # Apply interpolation to fill missing values in numeric columns
# df[numeric_columns] = df[numeric_columns].interpolate(method='linear', axis=0)

df.dtypes
```




    BathroomCount         float64
    BedroomCount            int64
    ConstructionYear      float64
    District               object
    FloodingZone          float64
    Garden                float64
    GardenArea            float64
    Kitchen               float64
    LivingArea            float64
    MonthlyCharges        float64
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
    Name: Cleaned_Localities, Length: 53030, dtype: object




```python
encoder = OneHotEncoder(drop='first',sparse_output=False)
if count < 1 : 
    df.drop(["Cleaned_Localities"],axis = 1, inplace = True)
    df.dtypes

cat_column = df.select_dtypes(include=['object']).columns.tolist()  
print(df.dtypes)

# cat_column = df.select_dtypes(include=['object']).columns.drop(["Url", "Cleaned_Localities", "District", "Province", "SubtypeOfProperty"]).tolist()  

encoded_data = encoder.fit_transform(df[cat_column])

one_hot_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out(cat_column))
df = pd.concat([df, one_hot_df], axis=1)
df = df.drop(cat_column, axis=1)

df.columns
df.shape
df.head()
```

    BathroomCount       float64
    BedroomCount          int64
    ConstructionYear    float64
    District             object
    FloodingZone        float64
    Garden              float64
    GardenArea          float64
    Kitchen             float64
    LivingArea          float64
    MonthlyCharges      float64
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
      <th>MonthlyCharges</th>
      <th>NumberOfFacades</th>
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
      <th>2</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1969.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>29.0</td>
      <td>0.0</td>
      <td>3.5</td>
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
      <th>6</th>
      <td>6.0</td>
      <td>13.0</td>
      <td>1920.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>391.0</td>
      <td>0.0</td>
      <td>3.0</td>
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
      <th>8</th>
      <td>2.0</td>
      <td>4.0</td>
      <td>2008.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>111.0</td>
      <td>0.0</td>
      <td>2.0</td>
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
      <th>11</th>
      <td>0.0</td>
      <td>2.0</td>
      <td>1972.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>4.0</td>
      <td>92.0</td>
      <td>0.0</td>
      <td>3.5</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>14</th>
      <td>1.0</td>
      <td>1.0</td>
      <td>1994.0</td>
      <td>8.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>6.0</td>
      <td>50.0</td>
      <td>0.0</td>
      <td>2.0</td>
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
<p>5 rows × 74 columns</p>
</div>




```python
df.isnull().sum()
```




    BathroomCount               37260
    BedroomCount                37260
    ConstructionYear            37260
    FloodingZone                37260
    Garden                      37260
                                ...  
    Province_Liège              37260
    Province_Luxembourg         37260
    Province_Namur              37260
    Province_Walloon Brabant    37260
    Province_West Flanders      37260
    Length: 74, dtype: int64




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




    53030




```python
df.isna().sum().to_dict()
```




    {'BathroomCount': 0,
     'BedroomCount': 0,
     'ConstructionYear': 0,
     'FloodingZone': 0,
     'Garden': 0,
     'GardenArea': 0,
     'Kitchen': 0,
     'LivingArea': 0,
     'MonthlyCharges': 0,
     'NumberOfFacades': 0,
     'PEB': 0,
     'PostalCode': 0,
     'Price': 0,
     'RoomCount': 37982,
     'ShowerCount': 21646,
     'StateOfBuilding': 0,
     'SurfaceOfPlot': 0,
     'SwimmingPool': 0,
     'Terrace': 0,
     'ToiletCount': 0,
     'TypeOfProperty': 0,
     'TypeOfSale': 0,
     'District_Antwerp': 37260,
     'District_Arlon': 37260,
     'District_Ath': 37260,
     'District_Bastogne': 37260,
     'District_Brugge': 37260,
     'District_Brussels': 37260,
     'District_Charleroi': 37260,
     'District_Dendermonde': 37260,
     'District_Diksmuide': 37260,
     'District_Dinant': 37260,
     'District_Eeklo': 37260,
     'District_Gent': 37260,
     'District_Halle-Vilvoorde': 37260,
     'District_Hasselt': 37260,
     'District_Huy': 37260,
     'District_Ieper': 37260,
     'District_Kortrijk': 37260,
     'District_Leuven': 37260,
     'District_Liège': 37260,
     'District_Maaseik': 37260,
     'District_Marche-en-Famenne': 37260,
     'District_Mechelen': 37260,
     'District_Mons': 37260,
     'District_Mouscron': 37260,
     'District_Namur': 37260,
     'District_Neufchâteau': 37260,
     'District_Nivelles': 37260,
     'District_Oostend': 37260,
     'District_Oudenaarde': 37260,
     'District_Philippeville': 37260,
     'District_Roeselare': 37260,
     'District_Sint-Niklaas': 37260,
     'District_Soignies': 37260,
     'District_Thuin': 37260,
     'District_Tielt': 37260,
     'District_Tongeren': 37260,
     'District_Tournai': 37260,
     'District_Turnhout': 37260,
     'District_Verviers': 37260,
     'District_Veurne': 37260,
     'District_Virton': 37260,
     'District_Waremme': 37260,
     'Province_Brussels': 37260,
     'Province_East Flanders': 37260,
     'Province_Flemish Brabant': 37260,
     'Province_Hainaut': 37260,
     'Province_Limburg': 37260,
     'Province_Liège': 37260,
     'Province_Luxembourg': 37260,
     'Province_Namur': 37260,
     'Province_Walloon Brabant': 37260,
     'Province_West Flanders': 37260}




```python
# import xgboost as xgb

# print(len(df))

# X = df.drop(columns=["Price"])
# y = df["Price"]

# # Standardize features
# scaler = StandardScaler()
# X = scaler.fit_transform(X)

# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, random_state=42)
# reg = xgb.XGBRegressor(random_state=0)
# reg.fit(X_train, y_train)

# # Predict and evaluate
# y_pred = regressor.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# mae = mean_absolute_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)
# score_train = regressor.score(X_train, y_train)
# score_test = regressor.score(X_test, y_test)

# print(f'Mean Squared Error: {mse}')
# print(f'Mean Error: {mse ** 0.5}')
# print(f'Mean Absolute Error: {mae}')
# print(f'R-squared: {r2}')
# print(f'Score train: {score_train}')

```


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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
regressor = DecisionTreeRegressor(random_state=0)
cross_val_score(regressor, X, y, cv=10)
regressor.fit(X_train, y_train)

# Predict and evaluate
y_pred = regressor.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
score_train = regressor.score(X_train, y_train)

print(f'Mean Squared Error: {mse}')
print(f'Mean Error: {mse ** 0.5}')
print(f'Mean Absolute Error: {mae}')
print(f'R-squared: {r2}')
print(f'Score train: {score_train}')

```

    Mean Squared Error: 96173535741.82512
    Mean Error: 310118.5833545373
    Mean Absolute Error: 116410.19591658209
    R-squared: 0.6722229587601489
    Score train: 0.9945524588638446
    


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

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create and train the model
regressor = RandomForestRegressor(random_state=0,  n_jobs=-1)
cross_val_score(regressor, X, y, cv=10)
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

    53030
    Mean Squared Error: 54849586404.5866
    Mean Error: 234199.88557765482
    Mean Absolute Error: 88245.94845758898
    R-squared: 0.8130625539941932
    Score train: 0.9688455014523405
    Score test: 0.8130625539941932
    


```python
from sklearn.ensemble import GradientBoostingRegressor

X = df.drop(columns=["Price"])
y = df["Price"]

# Standardize features
scaler = StandardScaler()
X = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42)
reg = GradientBoostingRegressor(random_state=0)
reg.fit(X_train, y_train)

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


    ---------------------------------------------------------------------------

    ValueError                                Traceback (most recent call last)

    Cell In[1238], line 13
         10 X_train, X_test, y_train, y_test = train_test_split(
         11     X, y, random_state=42)
         12 reg = GradientBoostingRegressor(random_state=0)
    ---> 13 reg.fit(X_train, y_train)
         15 # Predict and evaluate
         16 y_pred = regressor.predict(X_test)
    

    File c:\Users\lkris\OneDrive\Bureau\BeCode\becode_projects\Immo_eliza_Regression\.venv\Lib\site-packages\sklearn\base.py:1473, in _fit_context.<locals>.decorator.<locals>.wrapper(estimator, *args, **kwargs)
       1466     estimator._validate_params()
       1468 with config_context(
       1469     skip_parameter_validation=(
       1470         prefer_skip_nested_validation or global_skip_validation
       1471     )
       1472 ):
    -> 1473     return fit_method(estimator, *args, **kwargs)
    

    File c:\Users\lkris\OneDrive\Bureau\BeCode\becode_projects\Immo_eliza_Regression\.venv\Lib\site-packages\sklearn\ensemble\_gb.py:659, in BaseGradientBoosting.fit(self, X, y, sample_weight, monitor)
        653     self._clear_state()
        655 # Check input
        656 # Since check_array converts both X and y to the same dtype, but the
        657 # trees use different types for X and y, checking them separately.
    --> 659 X, y = self._validate_data(
        660     X, y, accept_sparse=["csr", "csc", "coo"], dtype=DTYPE, multi_output=True
        661 )
        662 sample_weight_is_none = sample_weight is None
        663 sample_weight = _check_sample_weight(sample_weight, X)
    

    File c:\Users\lkris\OneDrive\Bureau\BeCode\becode_projects\Immo_eliza_Regression\.venv\Lib\site-packages\sklearn\base.py:650, in BaseEstimator._validate_data(self, X, y, reset, validate_separately, cast_to_ndarray, **check_params)
        648         y = check_array(y, input_name="y", **check_y_params)
        649     else:
    --> 650         X, y = check_X_y(X, y, **check_params)
        651     out = X, y
        653 if not no_val_X and check_params.get("ensure_2d", True):
    

    File c:\Users\lkris\OneDrive\Bureau\BeCode\becode_projects\Immo_eliza_Regression\.venv\Lib\site-packages\sklearn\utils\validation.py:1301, in check_X_y(X, y, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_2d, allow_nd, multi_output, ensure_min_samples, ensure_min_features, y_numeric, estimator)
       1296         estimator_name = _check_estimator_name(estimator)
       1297     raise ValueError(
       1298         f"{estimator_name} requires y to be passed, but the target y is None"
       1299     )
    -> 1301 X = check_array(
       1302     X,
       1303     accept_sparse=accept_sparse,
       1304     accept_large_sparse=accept_large_sparse,
       1305     dtype=dtype,
       1306     order=order,
       1307     copy=copy,
       1308     force_writeable=force_writeable,
       1309     force_all_finite=force_all_finite,
       1310     ensure_2d=ensure_2d,
       1311     allow_nd=allow_nd,
       1312     ensure_min_samples=ensure_min_samples,
       1313     ensure_min_features=ensure_min_features,
       1314     estimator=estimator,
       1315     input_name="X",
       1316 )
       1318 y = _check_y(y, multi_output=multi_output, y_numeric=y_numeric, estimator=estimator)
       1320 check_consistent_length(X, y)
    

    File c:\Users\lkris\OneDrive\Bureau\BeCode\becode_projects\Immo_eliza_Regression\.venv\Lib\site-packages\sklearn\utils\validation.py:1064, in check_array(array, accept_sparse, accept_large_sparse, dtype, order, copy, force_writeable, force_all_finite, ensure_2d, allow_nd, ensure_min_samples, ensure_min_features, estimator, input_name)
       1058     raise ValueError(
       1059         "Found array with dim %d. %s expected <= 2."
       1060         % (array.ndim, estimator_name)
       1061     )
       1063 if force_all_finite:
    -> 1064     _assert_all_finite(
       1065         array,
       1066         input_name=input_name,
       1067         estimator_name=estimator_name,
       1068         allow_nan=force_all_finite == "allow-nan",
       1069     )
       1071 if copy:
       1072     if _is_numpy_namespace(xp):
       1073         # only make a copy if `array` and `array_orig` may share memory`
    

    File c:\Users\lkris\OneDrive\Bureau\BeCode\becode_projects\Immo_eliza_Regression\.venv\Lib\site-packages\sklearn\utils\validation.py:123, in _assert_all_finite(X, allow_nan, msg_dtype, estimator_name, input_name)
        120 if first_pass_isfinite:
        121     return
    --> 123 _assert_all_finite_element_wise(
        124     X,
        125     xp=xp,
        126     allow_nan=allow_nan,
        127     msg_dtype=msg_dtype,
        128     estimator_name=estimator_name,
        129     input_name=input_name,
        130 )
    

    File c:\Users\lkris\OneDrive\Bureau\BeCode\becode_projects\Immo_eliza_Regression\.venv\Lib\site-packages\sklearn\utils\validation.py:172, in _assert_all_finite_element_wise(X, xp, allow_nan, msg_dtype, estimator_name, input_name)
        155 if estimator_name and input_name == "X" and has_nan_error:
        156     # Improve the error message on how to handle missing values in
        157     # scikit-learn.
        158     msg_err += (
        159         f"\n{estimator_name} does not accept missing values"
        160         " encoded as NaN natively. For supervised learning, you might want"
       (...)
        170         "#estimators-that-handle-nan-values"
        171     )
    --> 172 raise ValueError(msg_err)
    

    ValueError: Input X contains NaN.
    GradientBoostingRegressor does not accept missing values encoded as NaN natively. For supervised learning, you might want to consider sklearn.ensemble.HistGradientBoostingClassifier and Regressor which accept missing values encoded as NaNs natively. Alternatively, it is possible to preprocess the data, for instance by using an imputer transformer in a pipeline or drop samples with missing values. See https://scikit-learn.org/stable/modules/impute.html You can find a list of all estimators that handle NaN values at the following page: https://scikit-learn.org/stable/modules/impute.html#estimators-that-handle-nan-values



```python
count += 1 
```
