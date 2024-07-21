

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import NearestNeighbors
import streamlit as st

# Function to get brand name
def get_brand_name(car_name):
    car_name = car_name.split(' ')[0]
    return car_name.strip()

# Function to clean data
def clean_data(value):
    if isinstance(value, str):
        value = value.split(' ')[0]
        value = value.strip()
        if value == '':
            value = 0
    return float(value)

# Load the dataset

def load_data():
    cars_data = pd.read_csv('modified_car_dataset.csv')
    cars_data.dropna(inplace=True)
    cars_data.drop_duplicates(inplace=True)
    cars_data['Brand'] = cars_data['Brand'].apply(get_brand_name)
    cars_data['Mileage (kmpl)'] = cars_data['Mileage (kmpl)'].apply(clean_data)
    return cars_data

# Load and preprocess data
cars_data = load_data()

# Encode categorical variables
le_brand = LabelEncoder()
le_model = LabelEncoder()
le_fuel = LabelEncoder()
le_owner = LabelEncoder()
le_transmission = LabelEncoder()
le_registration_state = LabelEncoder()

cars_data['Brand'] = le_brand.fit_transform(cars_data['Brand'])
cars_data['Model'] = le_model.fit_transform(cars_data['Model'])
cars_data['Fuel Type'] = le_fuel.fit_transform(cars_data['Fuel Type'])
cars_data['Owner'] = le_owner.fit_transform(cars_data['Owner'])
cars_data['Transmission'] = le_transmission.fit_transform(cars_data['Transmission'])
cars_data['Registration State'] = le_registration_state.fit_transform(cars_data['Registration State'])

# Create reverse mappings
brand_mapping = {index: label for index, label in enumerate(le_brand.classes_)}
model_mapping = {index: label for index, label in enumerate(le_model.classes_)}
fuel_mapping = {index: label for index, label in enumerate(le_fuel.classes_)}
owner_mapping = {index: label for index, label in enumerate(le_owner.classes_)}
transmission_mapping = {index: label for index, label in enumerate(le_transmission.classes_)}
registration_state_mapping = {index: label for index, label in enumerate(le_registration_state.classes_)}

# Streamlit code
st.header('Car Price Prediction ML Model')

brand = st.selectbox('Select Car Brand', le_brand.classes_)
filtered_cars = cars_data[cars_data['Brand'] == le_brand.transform([brand])[0]]
car_model = st.selectbox('Select Car Model', le_model.inverse_transform(filtered_cars['Model'].unique()))
year = st.selectbox('Car Manufactured Year', list(range(2000, 2024)))
km_driven = st.selectbox('No of kms Driven', ['20,000-50,000', '50,000-80,000', '80,000-110,000', '110,000-140,000', '140,000-170,000', '170,000-200,000', '200,000-230,000', '230,000-260,000', '260,000-290,000', '290,000-320,000', '320,000-350,000', '350,000-380,000', '380,000-400,000'])
fuel = st.selectbox('Fuel type', le_fuel.classes_)
owner = st.selectbox('Owner type', le_owner.classes_)
mileage = st.selectbox('Car Mileage (kmpl)', ['10-20', '20-30', '30-40'])
transmission = st.selectbox('Transmission type', le_transmission.classes_)
registration_state = st.selectbox('Registration State', le_registration_state.classes_)  # Input from user, but not used for prediction

# Prepare input data model
km_driven_values = {
    '20,000-50,000': 35000, '50,000-80,000': 65000, '80,000-110,000': 95000,
    '110,000-140,000': 125000, '140,000-170,000': 155000, '170,000-200,000': 185000,
    '200,000-230,000': 215000, '230,000-260,000': 245000, '260,000-290,000': 275000,
    '290,000-320,000': 305000, '320,000-350,000': 335000, '350,000-380,000': 365000,
    '380,000-400,000': 390000
}
mileage_values = {'10-20': 15, '20-30': 25, '30-40': 35}

input_data_model = pd.DataFrame(
    [[brand, car_model, year, km_driven_values[km_driven], fuel, owner, mileage_values[mileage], transmission]],
    columns=['Brand', 'Model', 'Manufacturing Year', 'KM Driven', 'Fuel Type', 'Owner', 'Mileage (kmpl)', 'Transmission']
)

# Encode the categorical values
input_data_model['Brand'] = le_brand.transform(input_data_model['Brand'])
input_data_model['Model'] = le_model.transform(input_data_model['Model'])
input_data_model['Fuel Type'] = le_fuel.transform(input_data_model['Fuel Type'])
input_data_model['Owner'] = le_owner.transform(input_data_model['Owner'])
input_data_model['Transmission'] = le_transmission.transform(input_data_model['Transmission'])

if st.button("Predict"):
    # Filter the dataset to include only relevant rows for prediction
    filtered_model_data = filtered_cars[filtered_cars['Model'] == le_model.transform([car_model])[0]].copy()  # Use .copy()

    # Train the model on the filtered data, excluding 'Registration State'
    input_data_filtered = filtered_model_data.drop(columns=['Selling Price', 'Registration State'])  # Exclude 'Registration State'
    output_data_filtered = filtered_model_data['Selling Price']

    # If using a consistent split, you can store the trained model in a session state or outside the button
    if 'trained_model' not in st.session_state:
        x_train_filtered, x_test_filtered, y_train_filtered, y_test_filtered = train_test_split(input_data_filtered, output_data_filtered, test_size=0.2, random_state=42)
        reg_model_filtered = LinearRegression()
        reg_model_filtered.fit(x_train_filtered, y_train_filtered)
        st.session_state.trained_model = reg_model_filtered

    car_price = st.session_state.trained_model.predict(input_data_model[input_data_filtered.columns])[0]

    rounded_price = int(round(car_price, -4))
    price_range = f'{rounded_price - 5000:,} - {rounded_price + 5000:,}'

    st.markdown(f'Car Price is going to be {price_range}')

    # Prepare data for KNN search
    knn_features = input_data_model.values  # Use the input data as the basis for finding similar cars

    # Fit KNN on the filtered model data
    knn = NearestNeighbors(n_neighbors=5)
    knn.fit(filtered_model_data.drop(columns=['Selling Price', 'Registration State']))  # Exclude 'Registration State'

    # Find similar cars
    distances, indices = knn.kneighbors(knn_features)

    # Get similar cars
    similar_cars = filtered_model_data.iloc[indices[0]].copy()  # Use .copy()

    # Map the encoded values back to original categories using .loc
    similar_cars.loc[:, 'Brand'] = similar_cars['Brand'].map(brand_mapping)
    similar_cars.loc[:, 'Model'] = similar_cars['Model'].map(model_mapping)
    similar_cars.loc[:, 'Fuel Type'] = similar_cars['Fuel Type'].map(fuel_mapping)
    similar_cars.loc[:, 'Owner'] = similar_cars['Owner'].map(owner_mapping)
    similar_cars.loc[:, 'Transmission'] = similar_cars['Transmission'].map(transmission_mapping)
    similar_cars.loc[:, 'Registration State'] = similar_cars['Registration State'].map(registration_state_mapping)

    # Display similar cars
    st.subheader("Similar Cars:")
    st.write(similar_cars[['Brand', 'Model', 'Manufacturing Year', 'KM Driven', 'Fuel Type', 'Owner', 'Mileage (kmpl)', 'Transmission', 'Registration State', 'Selling Price']])
