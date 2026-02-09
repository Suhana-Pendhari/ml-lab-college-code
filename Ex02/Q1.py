import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# 1. Load the experience dataset
data = pd.read_csv('experience_salary.csv')

# 2. Features (X) and Target (y)
# We use 'YearsExperience' to predict 'Salary'
X = data[['YearsExperience']]
y = data['Salary']

# 3. Split data (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# 5. Predict salaries for the test set
y_pred = model.predict(X_test)

# 6. Evaluate the accuracy using RMSE
# Note: RMSE is in the same units as the target (Dollars)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"Root Mean Squared Error: ${rmse:.2f}")

# 7. Visualize the results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='darkblue', label='Actual Salary')
plt.plot(X_test, y_pred, color='orange', linewidth=2, label='Regression Line')
plt.title('Salary vs Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary ($)')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.show()
