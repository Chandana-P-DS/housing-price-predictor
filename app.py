from flask import Flask, render_template, request
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

app = Flask(__name__)

# Load dataset
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

csv_path = os.path.join(BASE_DIR, 'housing.csv')

housing = pd.read_csv(csv_path)

# Features and target
X = housing[[
    'ocean_proximity',
    'population',
    'housing_median_age',
    'total_bedrooms',
    'total_rooms',
    'latitude',
    'longitude',
    'households'
]]

y = housing['median_house_value']

# Numerical columns
num_cols = [
    'population',
    'housing_median_age',
    'total_bedrooms',
    'total_rooms',
    'latitude',
    'longitude',
    'households'
]

# Categorical columns
cat_cols = ['ocean_proximity']

# Numerical pipeline
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Preprocessor
preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_cols),
    ('cat', cat_pipeline, cat_cols)
])

# Final model pipeline
model = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', RandomForestRegressor(
        n_estimators=100,
        random_state=42
    ))
])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model.fit(X_train, y_train)

# Prediction route
@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():

    ocean_proximity = request.form['ocean_proximity']
    population = float(request.form['population'])
    housing_median_age = float(request.form['housing_median_age'])
    total_bedrooms = float(request.form['total_bedrooms'])
    total_rooms = float(request.form['total_rooms'])
    latitude = float(request.form['latitude'])
    longitude = float(request.form['longitude'])
    households = float(request.form['households'])

    input_data = pd.DataFrame({
        'ocean_proximity': [ocean_proximity],
        'population': [population],
        'housing_median_age': [housing_median_age],
        'total_bedrooms': [total_bedrooms],
        'total_rooms': [total_rooms],
        'latitude': [latitude],
        'longitude': [longitude],
        'households': [households]
    })

    prediction = model.predict(input_data)[0]

    return render_template(
        'index.html',
        prediction_text=f'Predicted House Price: ${round(prediction, 2)}'
    )


if __name__ == '__main__':
    app.run(host='0.0.0.0')