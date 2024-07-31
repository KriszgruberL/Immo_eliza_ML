import json
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

def print_score(y_test, y_pred, model_name, target_scaler):
    """
    Print the performance scores of the model after inverse transforming the predictions and test values.

    Parameters:
    y_test (array-like): The true values of the target variable (scaled).
    y_pred (array-like): The predicted values of the target variable (scaled).
    model_name (str): The name of the model.
    target_scaler (StandardScaler): The scaler used for the target variable during training.
    """
    # Inverse transform the scaled predictions and test values to original scale
    y_pred_original = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_test_original = target_scaler.inverse_transform(y_test)

    # Calculate the Mean Absolute Error (MAE) and R-squared (R²) on the original scale
    mae = mean_absolute_error(y_test_original, y_pred_original)
    r2 = r2_score(y_test_original, y_pred_original)

    print(f"""
          +------------Score for {model_name}------------+
            Mean Absolute Error (Original Scale): {mae}
            R-squared (Original Scale): {r2}
          """)

def main():
    """
    The main function to load data, prepare data, train the model, and evaluate the model.
    """
    start_time = datetime.now()
    
    filepath = "./data/preprocessed_df.pkl"
    preprocessed_df = load_preprocessed_data(filepath)
    
    print(preprocessed_df.dtypes)

    X, y = prepare_data(preprocessed_df)
    
    feature_names = X.columns.tolist()
    # Saving feature names for future reference
    with open("./data/feature_names.json", "w") as f:
        json.dump(feature_names, f)

    regressor = RandomForestRegressor(random_state=0, n_jobs=-1)

    # Separate scalers for features and target
    feature_scaler = StandardScaler()
    target_scaler = StandardScaler()

    # Standardize features
    X = feature_scaler.fit_transform(X)

    # # Reshape y to 2D for scaling
    y = y.values.reshape(-1, 1)
    y = target_scaler.fit_transform(y)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = regressor.fit(X_train, y_train)

    # Predict and evaluate
    y_pred_scaled = model.predict(X_test)
    print_score(y_test, y_pred_scaled, "RandomForestRegressor", target_scaler)
    
    with open("./data/model.pkl", "wb") as f:
        pickle.dump(model, f)
        
    with open("./data/feature_scaler.pkl", "wb") as scaler_file:
        pickle.dump(feature_scaler, scaler_file)    
        
    with open("./data/target_scaler.pkl", "wb") as target_scaler_file:
        pickle.dump(target_scaler, target_scaler_file)
    
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

if __name__ == "__main__":
    main()
