import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Create sample dataset
data = {
    'Age': [22, 25, 30, 28, np.nan],
    'Salary': [30000, 35000, 50000, 45000, 40000],
    'Experience': [1, 2, 5, 3, 4],
    'Purchased': ['No', 'No', 'Yes', 'Yes', 'No']
}
df = pd.DataFrame(data)

# 2. Handling missing values
df['Age'] = df['Age'].fillna(df['Age'].mean())

# 3. Encoding categorical data
df['Purchased'] = df['Purchased'].map({'No': 0, 'Yes': 1})

# 4. Feature Scaling
scaler = StandardScaler()
df[['Age', 'Salary', 'Experience']] = scaler.fit_transform(
    df[['Age', 'Salary', 'Experience']]
)

# 5. Splitting Features and Target
X = df[['Age', 'Salary', 'Experience']]
y = df['Purchased']

# 6. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Display Results
print("--- Processed Dataframe ---")
print(df)
print("\n--- Split Sizes ---")
print(f"Training data size: {X_train.shape}")
print(f"Testing data size: {X_test.shape}")
