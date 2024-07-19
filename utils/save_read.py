import pandas as pd


def read_to_df(file_path: str, extension : str) -> pd.DataFrame:
    """
    Reads the dataset from a JSON file and initializes a DataFrame.

    Parameters:
    file_path (str): The path to the JSON file.

    Returns:
    pd.DataFrame: The DataFrame containing the data from the JSON file.
    """
    try:
        if extension == "csv" : 
            data = pd.read_csv(file_path)
        elif extension == "json" : 
            data = pd.read_json(file_path) 
        else : 
            print("Wrong extension! Check again!")
             
        df = pd.DataFrame(data)
        return df
    except FileNotFoundError as e:
        print(f"File not found: {e}")
        raise
    except Exception as e:
        print(f"An error occurred while reading the data: {e}")
        raise


def save_df_to_csv(df: pd.DataFrame, file_path: str) -> None:
    """
    Saves the DataFrame to a CSV file.

    Parameters:
    df (pd.DataFrame): The DataFrame to save.
    file_path (str): The path to the CSV file.
    """
    try:
        df.to_csv(file_path, index=False)
        print(f"Data saved to {file_path}.")
    except Exception as e:
        print(f"An error occurred while saving the data: {e}")
        raise
    
    