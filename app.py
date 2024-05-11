import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from joblib import dump, load

# Load the trained model
loaded_model = load('housing_price_prediction_model.joblib')

# Create a pipeline with the same preprocessing steps as before
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('std_scaler', StandardScaler())
])

# Load your training data
housing = pd.read_csv("data.csv")
X_train = housing[['ZN', 'RM']]

# Fit the pipeline with the training data
my_pipeline.fit(X_train)

# Now, you can transform new data using the fitted pipeline
new_data = pd.DataFrame({'ZN': [20], 'RM': [6]})
new_data_preprocessed = my_pipeline.transform(new_data)

# Use the preprocessed new data to make predictions
predicted_price = loaded_model.predict(new_data_preprocessed)
print("Predicted Price:", predicted_price)
