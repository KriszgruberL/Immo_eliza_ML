import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

from preprocessing.categorical_to_num_dict import CategoricalNumDict
from utils.save_read import save_df_to_csv

class Preprocessing : 
    def __init__(self, df) -> None:
        self.df_to_return = df.copy()
    
    def transform_categorical(self, df : pd.DataFrame, dict_key : str, dicts : CategoricalNumDict) -> pd.DataFrame:
        mapping_dict = dicts.get_dict(dict_key)
        
        if mapping_dict is None:
            raise ValueError(f"No dictionary found for key: {dict_key}")
    
        self.df_to_return.loc[:, f'{dict_key}'] = self.df_to_return[f'{dict_key}'].map(mapping_dict)
        self.df_to_return = self.df_to_return.dropna(subset=[f'{dict_key}'])
        
        # save_df_to_csv(df_to_return, f"data/{dict_key}.csv")
    
        return self.df_to_return
    
    def transform_all_categorical(self, df: pd.DataFrame,  dicts : CategoricalNumDict) -> pd.DataFrame:
        for key in dicts.dictionaries.keys():
            try:
                self.df_to_return = self.transform_categorical(self.df_to_return, key, dicts)
            except KeyError:
                print(f"Column {key} not found in the dataframe.")
        
        save_df_to_csv(self.df_to_return, "data/numerical_data.csv")
        return self.df_to_return
    
    def get_numerical_data(self) : 
        return self.df_to_return

