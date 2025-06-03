import pandas as pd
import numpy as np
import streamlit as st
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# Load data directly from CSV in repo
df = pd.read_csv("data.csv")

# Clean column names
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# Parse dates
df['sale_date'] = pd.to_datetime(df['sale_date'], errors='coerce')
df['seller'] = df['seller'].astype(str)

# Convert numeric columns
df['unit_size'] = pd.to_numeric(df['unit_size'], errors='coerce')
df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
df['zip_code'] = df['zip_code'].astype(str)

# Drop rows with missing values
df = df.dropna(subset=['unit_size', 'sale_price', 'zip_code'])

# Remove outliers
df = df[df['sale_price'] < df['sale_price'].quantile(0.95)]

# Feature engineering
df['log_price'] = np.log1p(df['sale_price'])

# Define features and target
features = ['city', 'product_style', 'unit_size', 'subdivision', 'seller', 'zip_code']
target = 'log_price'

X = df[features]
y = df[target]

# Categorical and numerical columns
categorical_features = ['city', 'product_style', 'subdivision', 'seller', 'zip_code']
numerical_features = ['unit_size']

# Preprocessor
preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, verbosity=0))
])

# Simplified hyperparameters
param_grid = {
    'regressor__n_estimators': [200],
    'regressor__max_depth': [6],
    'regressor__learning_rate': [0.1],
    'regressor__subsample': [0.8]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=0)
grid_search.fit(X, y)

model = grid_search.best_estimator_

# Calculate residual std deviation
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
residuals = np.expm1(y_test) - np.expm1(y_pred)
residual_std = np.std(residuals)

# Streamlit App UI
st.title("Real Estate Pricing Model")

# Build dropdowns dynamically from dataset
city_options = sorted(df['city'].dropna().unique())
product_style_options = sorted(df['product_style'].dropna().unique())
seller_options = sorted(df['seller'].dropna().unique())

# User inputs
city = st.selectbox("City:", city_options)
product_style = st.selectbox("Product Style:", product_style_options)
unit_size = st.number_input("Unit Size (sqft):", 500, 10000, 2500)
subdivision = st.text_input("Subdivision:", "")
seller = st.selectbox("Seller:", seller_options)
zip_code = st.text_input("Zip Code:", "34787")

if st.button("Predict Sale Price"):
    input_df = pd.DataFrame([{
        'city': city,
        'product_style': product_style,
        'unit_size': unit_size,
        'subdivision': subdivision,
        'seller': seller,
        'zip_code': zip_code
    }])
    
    predicted_log_price = model.predict(input_df)[0]
    predicted_price = np.expm1(predicted_log_price)
    lower_bound = predicted_price - 1.96 * residual_std
    upper_bound = predicted_price + 1.96 * residual_std

    st.subheader(f"Estimated Sale Price: ${predicted_price:,.2f}")
    st.write(f"Confidence Interval (95%): ${lower_bound:,.2f} - ${upper_bound:,.2f}")





