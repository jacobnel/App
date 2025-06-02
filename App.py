
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import joblib

# Load data
sheet_id = "1asq49oPhhMBS_7yIdGuaaneP6Kjv64mD0BzIMTqiYgE"
sheet_name = "Sheet1"
url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/gviz/tq?tqx=out:csv&sheet={sheet_name}"
df = pd.read_csv(url)

# Clean columns
df.columns = df.columns.str.strip().str.replace(' ', '_').str.lower()

# Convert data types
df['unit_size'] = pd.to_numeric(df['unit_size'], errors='coerce')
df['sale_price'] = pd.to_numeric(df['sale_price'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')
df['zip_code'] = df['zip_code'].astype(str)
df['seller'] = df['seller'].astype(str)

# Drop missing rows
df = df.dropna(subset=['unit_size', 'sale_price', 'latitude', 'longitude'])

# Remove outliers
df = df[df['sale_price'] < df['sale_price'].quantile(0.95)]

# Target variable
df['log_price'] = np.log1p(df['sale_price'])

# Features and target
features = ['city', 'product_style', 'unit_size', 'subdivision', 'seller', 'zip_code', 'latitude', 'longitude']
target = 'log_price'

X = df[features]
y = df[target]

# Preprocessing
categorical_features = ['city', 'product_style', 'subdivision', 'seller', 'zip_code']
numerical_features = ['unit_size', 'latitude', 'longitude']

preprocessor = ColumnTransformer([
    ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
], remainder='passthrough')

# Model pipeline
pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('regressor', XGBRegressor(random_state=42, verbosity=0))
])

# Simplified hyperparameters for now
param_grid = {
    'regressor__n_estimators': [200],
    'regressor__max_depth': [6],
    'regressor__learning_rate': [0.1],
    'regressor__subsample': [0.8]
}

grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='neg_root_mean_squared_error', verbose=1)
grid_search.fit(X, y)

model = grid_search.best_estimator_

# Evaluate model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(np.expm1(y_test), np.expm1(y_pred)))
print(f"Model RMSE: ${rmse:,.2f}")

# Save model
joblib.dump(model, 'pricing_model.pkl')

# Download model file (Colab)
from google.colab import files
files.download('pricing_model.pkl')


