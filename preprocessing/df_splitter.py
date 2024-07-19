
from typing import Dict, List, Tuple
import pandas as pd

from utils.save_read import save_df_to_csv

class DataFrameSplitter : 
    """
    A class used to split a DataFrame into sub-DataFrames using various methods.
    """
    
    def __init__(self) -> None:
        """
        Initializes the DataFrameSplitter with default conditions and save paths.
        
        conditions (Dict[str, str]): A dictionary where keys are the names of the sub-DataFrames 
                                     and values are the conditions to split the DataFrame by.
        save_paths (Dict[str, str]): A dictionary where keys are the names of the sub-DataFrames 
                                     and values are the paths to save the CSV files.
        
        """
        self.conditions = {
            "house_sale": 'TypeOfProperty == 1 and TypeOfSale == "residential_sale"',
            "house_rent": 'TypeOfProperty == 1 and TypeOfSale == "residential_monthly_rent"',
            "apartment_sale": 'TypeOfProperty == 2 and TypeOfSale == "residential_sale"',
            "apartment_rent": 'TypeOfProperty == 2 and TypeOfSale == "residential_monthly_rent"',
            "apartment" : "TypeOfProperty == 2",
            "rent" : "TypeOfSale == residential_monthly_rent",
            "sale" : "TypeOfSale == residential_monthly_rent"
            
        }

        self.save_paths = {
            "house_sale": "data/house_sale.csv",
            "house_rent": "data/house_rent.csv",
            "apartment_sale": "data/apartment_sale.csv",
            "apartment_rent": "data/apartment_rent.csv",
            "apartment": "data/apartment.csv",
        }

    
    def split_and_save(self, df: pd.DataFrame) -> None:
        """
        Splits the DataFrame into sub-DataFrames based on multiple conditions and saves them to CSV files.

        Parameters:
        df (pd.DataFrame): The DataFrame to split.
        """
        
        # Verify column values
        # print("Unique values in 'TypeOfProperty':", df["TypeOfProperty"].unique())
        # print("Unique values in 'TypeOfSale':", df["TypeOfSale"].unique())
        
        for key, condition in self.conditions.items():
            sub_df = df.query(condition)
            save_df_to_csv(sub_df, self.save_paths[key])
            # print(sub_df.head(), "\n")
            
        # Print the results for verification
        # for key, sub_df in sub_dfs.items():
        #     print(f"Processed Data for {key.replace('_', ' ').title()}:")
        #     print(sub_df.head(), "\n")
