import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
# Load dataset
data = pd.read_csv('house_prices.csv')
# Features and target
X = data[['Size', 'Bedrooms', 'Age']] # Independent variables
y = data['Price'] # Dependent variable
# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Train model
model = LinearRegression()
model.fit(X_train, y_train)
# Predict
y_pred = model.predict(X_test)
# Evaluate
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared Score: {r2}")
# Display coefficients
print("Intercept:", model.intercept_)
print("Coefficients:", model.coef_)
