
from preprocessing.categorical_to_num_dict import CategoricalNumDict
from preprocessing.preprocessing import Preprocessing
from preprocessing.df_splitter import DataFrameSplitter
from preprocessing.clean_data import CleanData
from utils.save_read import read_to_df

def main():
    """
    Main function to create an instance of DataProcessor and execute the processing steps.
    """
    data_path = "data/final_dataset.json"
    zip_path = "data/zipcode-belgium.json"
    save_path = "data/clean_dataset.csv"
    
    data_cleaner = CleanData(data_path, zip_path, save_path)
    data_cleaner.process() 

    # Get the processed data
    processed_data = data_cleaner.get_data()
    
    splitter = DataFrameSplitter()
    # splitter.split_and_save(processed_data)
    
    pre = Preprocessing(processed_data)
    dicts = CategoricalNumDict()

    pre.transform_all_categorical(processed_data, dicts)
    num_data = pre.get_numerical_data()


    # print(p.get_data().head())
    # print(p.get_column())
    # print(p.get_summary_stats())
    

if __name__ == "__main__":
    main()
