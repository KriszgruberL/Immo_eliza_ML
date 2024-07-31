import json
import time
import pickle
import numpy as np
import streamlit as st
import pandas as pd

# Title of the app
st.title("House Price Prediction")

# Initialize session state variables

if "submitted" not in st.session_state:
    st.session_state.submitted = False


# # Callback function to handle submission
def handle_submit():
    st.session_state.submitted = True
    st.session_state.show_inputs = False


peb_string = ["A++", "A+", "A", "B", "C", "D", "E", "F", "G"]
state_string = [
    "As new",
    "Just renovated",
    "Good",
    "To restore",
    "To renovate",
    "To be done up",
]

def peb_stringify(i: int = 0) -> str:
    return peb_string[i - 1]

def state_stringify(i: int = 0) -> str:
    return state_string[i - 1]

import time

def progress_bar():
    """
    Displays a progress bar with a text indicating the operation in progress.
    """
    # Progress bar
    progress_text = "Operation in progress. Please wait."
    my_bar = st.progress(0, text=progress_text)
    for percent_complete in range(100):
        time.sleep(0.01)
        my_bar.progress(percent_complete + 1, text=progress_text)
    my_bar.empty()
    
def load_model_and_scalers():
    """
    Load the pre-trained model, scalers, and feature names from the specified files.

    Returns:
        model (object): The pre-trained model.
        feature_scaler (object): The scaler used for feature normalization.
        target_scaler (object): The scaler used for target normalization.
        feature_names (list): The names of the features used by the model.
    """
    # Load the pre-trained model and scalers
    with open("data/model.pkl", "rb") as file:
        model = pickle.load(file)
    with open("data/scaler.pkl", "rb") as scaler_file:
        feature_scaler = pickle.load(scaler_file)
    with open("data/target_scaler.pkl", "rb") as target_scaler_file:
        target_scaler = pickle.load(target_scaler_file)
    with open("data/feature_names.json", "r") as f:
        feature_names = json.load(f)    
    
    return model, feature_scaler, target_scaler, feature_names

def prepare_data_for_prediction(region, province, feature_names, data):
    """
    Prepare the data for prediction by performing one-hot encoding on the given features.

    Args:
        region (str): The region value.
        province (str): The province value.
        feature_names (list): A list of feature names.
        data (dict): The data dictionary containing the feature values.

    Returns:
        dict: The updated data dictionary with one-hot encoded features.

    """
    # Initialize all features to zero
    prepared_data = {feature: 0 for feature in feature_names}

    # Update with provided data values
    for key, value in data.items():
        if key in prepared_data:
            prepared_data[key] = value

    # Set the one-hot encoded fields for province and region
    province_column = f"Province_{province}"
    region_column = f"Region_{region}"

    if province_column in prepared_data:
        prepared_data[province_column] = 1

    if region_column in prepared_data:
        prepared_data[region_column] = 1

    return prepared_data

# Form container
with st.container():
    with st.container(border=True):
        st.write("Property Details")
        col1, col2 = st.columns(2)
        with col1:
            region = st.selectbox(
                "Select the region",
                options=["Brussels", "Flanders", "Wallonia"],
                index=None,
                key="region",
            )
        with col2:
            if region == "Brussels":
                province = st.selectbox(
                    "Select the province",
                    options=["Brussels"],
                    index=0,
                    key="province",
                )
            elif region == "Flanders":
                province = st.selectbox(
                    "Select the province",
                    options=[
                        "Antwerp",
                        "Flemish Brabant",
                        "Limburg",
                        "East Flanders",
                        "West Flanders",
                    ],
                    index=None,
                    key="province",
                )
            elif region == "Wallonia":
                province = st.selectbox(
                    "Select the province",
                    options=[
                        "Walloon Brabant",
                        "Hainaut",
                        "Liege",
                        "Luxembourg",
                        "Namur",
                    ],
                    index=None,
                    key="province",
                )
        st.divider()

        col1, col2 = st.columns(2)
        with col1:
            property_type = st.radio(
                "Select the type of property",
                ["House", "Apartment"],
                index=None,
                key="type_property",
                horizontal=True,
            )

            year = st.number_input(
                "Enter the construction year (1600 - 2040)",
                min_value=1600,
                max_value=2040,
                value=2000,
                format="%d",
                help="Enter a value between 1600 - 2040",
            )

            nb_facade = st.number_input(
                "Enter the number of facades (0 - 4)",
                min_value=0,
                max_value=4,
                value=1,
                format="%d",
                help="Enter a value between 0 and 4",
            )

        with col2:
            sale_type = st.radio(
                "Select the type of sale",
                ["For Sale", "For Rent"],
                index=None,
                key="type_sale",
                horizontal=True,
            )
            zip_code = st.number_input(
                "Enter the postal code",
                min_value=1000,
                max_value=9992,
                value=5000,
                format="%d",
                help="Enter a value between 1000 and 9992 (Belgium zip codes)",
            )
            state_building = st.selectbox(
                "Select the state of the building",
                options=range(1, len(state_string) + 1),
                index=None,
                key="state_building",
                format_func=state_stringify,
            )

        living_area = st.number_input(
            "Enter the living area (in m²)",
            min_value=1,
            value=1,
            format="%d",
            help="Enter a value greater than 0",
        )
        plot_area = st.number_input(
            "Enter the plot area (in m²)",
            min_value=0,
            value=1,
            format="%d",
            help="Enter a numerical value",
        )
        peb = st.select_slider(
            "Select the PEB value",
            options=range(1, len(peb_string) + 1),
            key="peb",
            value=5,
            format_func=peb_stringify,
        )

        flood = st.radio(
            "Is it in a flooding zone?",
            options=["Yes", "No"],
            index=None,
            key="flood",
            horizontal=True,
        )

    with st.container(border=True):
        st.write("Inside Details")
        col1, col2 = st.columns(2)
        with col1:
            bed = st.number_input(
                "Enter the number of bedrooms (0 - 200)",
                min_value=0,
                max_value=200,
                step=1,
                value=0,
                help="Enter a numerical value between 0 and 200",
            )
            kitchen = st.radio(
                "Is the kitchen installed?",
                ["Yes", "No"],
                index=None,
                key="kitchen",
                horizontal=True,
            )
        with col2:
            bath = st.number_input(
                "Enter the number of bathrooms",
                min_value=0,
                value=0,
                format="%d",
                help="Enter a numerical value",
            )
            shower = st.number_input(
                "Enter the number of showers",
                min_value=0,
                value=0,
                format="%d",
                help="Enter a numerical value",
            )
            toilet = st.number_input(
                "Enter the number of toilets ",
                min_value=0,
                value=0,
                format="%d",
                help="Enter a numerical value",
            )

    with st.container(border=True):
        st.write("Outside Details")
        garden = st.radio(
            "Does the house have a garden?",
            ["Yes", "No"],
            index=None,
            key="garden",
            horizontal=True,
        )
        # Dynamic garden area input
        if garden == "Yes":
            garden_area = st.number_input(
                "Garden Area (in m²)", min_value=1, format="%d", value=None
            )
        else:
            garden_area = 0

        pool = st.radio(
            "Does the house have a swimming pool?",
            ["Yes", "No"],
            index=None,
            key="pool",
            horizontal=True,
        )
        terrace = st.radio(
            "Does the house have a terrace?",
            ["Yes", "No"],
            index=None,
            key="terrace",
            horizontal=True,
        )


# Button for submitting the form
if st.button("Predict !"):
    st.session_state.submitted = True
    progress_bar()
    # Load the pre-trained model and scalers
    model, feature_scaler, target_scaler, feature_names = load_model_and_scalers()
    
    # Prepare the input data for model prediction
    data = {
        "TypeOfProperty": 0 if property_type == "House" else 1,
        "ConstructionYear": year,
        "NumberOfFacades": nb_facade,
        "TypeOfSale": 0 if sale_type == "For Sale" else 1,
        "PostalCode": zip_code,
        "StateOfBuilding": state_building,
        "LivingArea": living_area,
        "SurfaceOfPlot": plot_area,
        "PEB": peb,
        "FloodingZone": 1 if flood == "Yes" else 0,
        "Kitchen": 1 if kitchen == "Yes" else 0,
        "BathroomCount": bath,
        "BedroomCount": bed,
        "ShowerCount": shower,
        "ToiletCount": toilet,
        "Garden": 1 if garden == "Yes" else 0,
        "GardenArea": garden_area,
        "SwimmingPool": 1 if pool == "Yes" else 0,
        "Terrace": 1 if terrace == "Yes" else 0,
    }

    data = prepare_data_for_prediction(region, province, feature_names, data)
    
    st.write(data)
    # Create a DataFrame for the input data
    df = pd.DataFrame([data], columns=feature_names)

    # Ensure all columns are of the correct type
    df = df.astype(
        {
            "TypeOfProperty": "int",
            "ConstructionYear": "float",
            "NumberOfFacades": "float",
            "TypeOfSale": "float",
            "PostalCode": "int",
            "StateOfBuilding": "float",
            "LivingArea": "float",
            "SurfaceOfPlot": "float",
            "PEB": "float",
            "FloodingZone": "float",
            "Kitchen": "float",
            "BathroomCount": "float",
            "BedroomCount": "int",
            "ShowerCount": "float",
            "ToiletCount": "float",
            "Garden": "float",
            "GardenArea": "float",
            "SwimmingPool": "float",
            "Terrace": "float",
            "Region_Brussels": "int",
            "Region_Flanders": "int",
            "Region_Wallonie": "int",
            "Province_Antwerp": "int",
            "Province_Brussels": "int",
            "Province_East Flanders": "int",
            "Province_Flemish Brabant": "int",
            "Province_Hainaut": "int",
            "Province_Limburg": "int",
            "Province_Liège": "int",
            "Province_Luxembourg": "int",
            "Province_Namur": "int",
            "Province_Walloon Brabant": "int",
            "Province_West Flanders": "int",
        }
    )
    ### DEBUG
    st.write("FUCK THIS SHIT:",df)

    # Scale the features
    X = feature_scaler.transform(df)

    # Predict using the model
    predictions_scaled = model.predict(X)

    # Debug: Print the scaled predictions
    st.write("Scaled Predictions:")
    st.write(predictions_scaled)

    predictions_original = target_scaler.inverse_transform(
        predictions_scaled.reshape(-1, 1)
    )

    # Display the predicted price
    st.write(f"Predicted price: {predictions_original.flatten()[0]:.2f} €")
        


