import datetime
from typing import Dict, List
import pandas as pd
import numpy as np

from utils.save_read import read_to_df, save_df_to_csv


class CleanData:
    """
    A class used to process a dataset with various cleaning and preprocessing steps.
    """

    def __init__(self, data_path: str, zip_path : str, save_path : str) -> None:
        """
        Initializes the DataProcessor with data and DataFrame attributes set to None.
        """
        self.data_path = data_path
        self.zip_path = zip_path
        self.save_path = save_path
        
        self.data = None
        self.df = None
        
        self.default_values = {"numeric": 0, "string": "null"}
        self.drop_columns = ["Country", "Fireplace", "Locality"]
        self.exclude_columns = ["PostalCode", "Price", "PropertyId", "TypeOfProperty", "TypeOfSale"]
        self.exclude_annuity = ["annuity_monthly_amount", "annuity_without_lump_sum", "annuity_lump_sum", "homes_to_build"]
        
        

    def process(self) -> None:
        """
        Main method to execute all processing steps in sequence.
        """
        try:
            self.df = read_to_df(self.data_path, "json")
            if self.df is not None:
                print("START: ", self.df.shape)
                self.fill_empty()
                self.strip_blank()
                self.check_coherence()
                self.drop_unusable()
                print("END: ", self.df.shape)
                save_df_to_csv(self.df, self.save_path)
            else:
                print("DataFrame is None, skipping processing.")
        except Exception as e:
            print(f"An error occurred during processing: {e}")


    def fill_empty(self) -> None:
        """
        Fills empty values in the DataFrame with appropriate default values.
        """
        # Update specific columns with appropriate empty values
        numeric_columns = [
            "BathroomCount",
            "BedroomCount",
            "ConstructionYear",
            "GardenArea",
            "LivingArea",
            "MonthlyCharges",
            "NumberOfFacades",
            "RoomCount",
            "ShowerCount",
            "SurfaceOfPlot",
            "SwimmingPool",
            "Terrace",
            "ToiletCount",
            "Furnished",
            "Garden",
        ]
        fill_values = {
            col: self.default_values["numeric"] if col in numeric_columns else self.default_values["string"]
            for col in self.df.columns
        }
        
        self.df.fillna(value=fill_values, inplace=True)
        
    def strip_blank(self)  -> None: 
        """
        Strips leading and trailing whitespace from all string columns.
        """
        if "PEB" in self.df.columns and self.df["PEB"].dtype == "object":
            self.df["PEB"] = self.df["PEB"].str.strip().str.upper().str.replace("_", "/")

        for col in self.df.columns:
            if self.df[col].dtype == "object" and col != "PEB":
                self.df[col] = self.df[col].str.strip().str.lower().str.replace(" ", "_")

    def drop_unusable(self) -> None:
        """
        Drops duplicate rows and unnecessary columns, and removes rows with null values in critical columns.
        """
        self.df.drop_duplicates(inplace=True)
        self.df = self.df.drop(columns = self.drop_columns)  # Don't need those

        # List of columns where a missing value is not acceptable
        self.df.dropna(subset=self.exclude_columns, inplace=True)  # Drop rows where any of the exclude columns have null values

        # Drop rows where TypeOfSale is in the exclude list
        self.df = self.df[~self.df["TypeOfSale"].isin(self.exclude_annuity)]
        self.check_drop_zip_code()

    def check_drop_zip_code(self) : 
        # Check if postal code is in Belgium 
        self.postal_code = pd.read_json("data/zipcode-belgium.json")
        valid = set(self.postal_code["zip"])
        
        # Create a new column 'PostalCodeValid' with True if 'PostalCode' is in 'valid', else False
        self.df["PostalCodeValid"] = self.df["PostalCode"].apply(lambda x: x in valid)
        self.df = self.df[self.df["PostalCodeValid"] == True]
        self.df.drop(columns=["PostalCodeValid"], inplace=True)
        

    def check_coherence(self) -> None:
        """
        Ensures data coherence by checking and adjusting certain columns' values.
        """
        year_threshold = datetime.datetime.today().year + 10

        # Keep rows where the condition is true 
        self.df = self.df.loc[
            ((self.df["ConstructionYear"] == "null") | (self.df["ConstructionYear"] <= year_threshold)) &
            ~((self.df["GardenArea"] > 0) & (self.df["Garden"] == 0)) &
            ~((self.df["GardenArea"] > 0) & (self.df["Garden"] == 0)) &
            (self.df["ShowerCount"] < 30) &
            (self.df["ToiletCount"] < 50) &
            (self.df["LivingArea"] >= 9) & (self.df["LivingArea"] <= 2000 ) & 
            ((self.df["NumberOfFacades"] == 0) | ((self.df["NumberOfFacades"] >= 2) & (self.df["NumberOfFacades"] <= 10)))
        ]

    def get_summary_stats(self) -> pd.DataFrame:
        """
        Returns summary statistics of the DataFrame.
        """
        return self.df.describe()

    def get_data(self) -> pd.DataFrame:
        """
        Returns the processed DataFrame.
        """
        return self.df

    def get_column(self) -> pd.Index:
        """
        Returns the columns of the DataFrame.
        """
        return self.df.columns


