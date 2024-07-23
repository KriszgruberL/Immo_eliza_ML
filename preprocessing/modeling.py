import pickle
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from datetime import datetime

def load_preprocessed_data(filepath):
    """
    Load preprocessed data from a pickle file.

    Parameters:
    filepath (str): The path to the pickle file containing the preprocessed data.

    Returns:
    pd.DataFrame: The preprocessed data.
    """
    with open(filepath, "rb") as file:
        return pickle.load(file)

def prepare_data(df):
    """
    Prepare the data for training by separating features and target variable.

    Parameters:
    df (pd.DataFrame): The input data containing features and target variable.

    Returns:
    tuple: A tuple containing the features (X) and the target variable (y).
    """
    X = df.drop(columns=["Price"], axis=1)
    y = df["Price"]
    return X, y

def print_score(y_test, y_pred, model_name):
    """
    Print the performance scores of the model.

    Parameters:
    y_test (array-like): The true values of the target variable.
    y_pred (array-like): The predicted values of the target variable.
    model_name (str): The name of the model.
    """
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)

    print(f"""
          +------------Score for {model_name}------------+
            Mean Absolute Error: {mae}
            R-squared: {r2}
          """)

def main():
    """
    The main function to load data, prepare data, train the model, and evaluate the model.
    """
    start_time = datetime.now()
    
    filepath = "../data/preprocessed_df.pkl"
    preprocessed_df = load_preprocessed_data(filepath)

    X, y = prepare_data(preprocessed_df)

    regressor = RandomForestRegressor(random_state=0, n_jobs=-1)

    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = regressor.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = regressor.predict(X_test)
    print_score(y_test, y_pred, "RandomForestRegressor")
    
    with open("../data/model.pkl", "wb") as f:
        pickle.dump(model, f)
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

if __name__ == "__main__":
    main()
