import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. Create sample dataset (Patient Health Records)
data = {
    'BloodPressure': [120, 140, 130, np.nan, 150],
    'Cholesterol': [200, 240, 210, 230, 250],
    'HeartRate': [72, 85, 78, 90, 88],
    'HighRisk': ['No', 'Yes', 'No', 'Yes', 'Yes']
}
df = pd.DataFrame(data)

# 2. Handling missing values
# Replacing the missing BloodPressure with the average (mean)
df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())

# 3. Encoding categorical data
# Converting "Yes/No" labels into machine-readable 1s and 0s
df['HighRisk'] = df['HighRisk'].map({'No': 0, 'Yes': 1})

# 4. Feature Scaling
# We scale the data so that 150 (BP) doesn't overshadow 72 (HeartRate)
scaler = StandardScaler()
df[['BloodPressure', 'Cholesterol', 'HeartRate']] = scaler.fit_transform(
    df[['BloodPressure', 'Cholesterol', 'HeartRate']]
)

# 5. Splitting Features (X) and Target (y)
X = df[['BloodPressure', 'Cholesterol', 'HeartRate']]
y = df['HighRisk']

# 6. Train-Test Split (80% Training, 20% Testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# --- Display Results ---
print("--- Processed Medical Data (Scaled) ---")
print(df)
print("\n--- Training & Testing Summary ---")
print(f"Features for Training (X_train):\n{X_train.shape}")
print(f"Target for Training (y_train):\n{y_train.shape}")
