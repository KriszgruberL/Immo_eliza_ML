import pickle
import pandas as pd
from sklearn.discriminant_analysis import StandardScaler
from sklearn.model_selection import GridSearchCV, RepeatedKFold, cross_val_score, train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

def load_preprocessed_data(filepath):
    with open(filepath, "rb") as file:
        return pickle.load(file)

def prepare_data(df):
    X = df.drop(columns=["Price"], axis=1)
    y = df["Price"]
    return X, y

def print_score(y_test, y_pred, model_name):
    mae = mean_absolute_error(y_true=y_test, y_pred=y_pred)
    r2 = r2_score(y_true=y_test, y_pred=y_pred)

    print(f"""
          +------------Score for {model_name}------------+
            Mean Absolute Error: {mae}
            R-squared: {r2}
          """)

def main():
    from datetime import datetime
    from xgboost import XGBRFRegressor
    import xgboost as xgb
    
    start_time = datetime.now()
    
    filepath = "../data/preprocessed_df.pkl"
    preprocessed_df = load_preprocessed_data(filepath)

    # Prepare the data
    X, y = prepare_data(preprocessed_df)

    # Initialize the regressor
    regressor = RandomForestRegressor(random_state=0, n_jobs=-1)

    # # Perform cross-validation
    # cross_val_scores = cross_val_score(regressor, X, y, cv=10)
    # print("Cross-validation scores for each fold:", cross_val_scores)
    
    # Standardize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    regressor.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = regressor.predict(X_test)
    print_score(y_test, y_pred, "RandomForestRegressor")
    
    end_time = datetime.now()
    print('Duration: {}'.format(end_time - start_time))

if __name__ == "__main__":
    main()

