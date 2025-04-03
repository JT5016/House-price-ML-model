import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import joblib  # For saving the model

# -----------------------------------------------------------
# 1. Load Data and Initial Exploration
# -----------------------------------------------------------
# Load the training dataset from the ML folder
data = pd.read_csv('ML/train.csv')

# Visualize the distribution of SalePrice
plt.hist(data['SalePrice'], bins=30, edgecolor='black')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.title('Distribution of Sale Price')
plt.show()

# Check for missing data in the dataset
missing_data = data.isnull().sum().sort_values(ascending=False)
print("Missing Data (only showing columns with missing values):")
print(missing_data[missing_data > 0])

# -----------------------------------------------------------
# 2. Data Preprocessing
# -----------------------------------------------------------
# Define the numerical features to be used in the model.
# Note: We are no longer using any neighborhood-based features.
numerical_features = ['LotArea', 'GrLivArea', 'GarageCars', 'FullBath', 'OverallQual']

# Extract features (X) and target variable (y).
# We apply a log transformation to SalePrice to reduce skewness.
x = data[numerical_features]
y = np.log(data['SalePrice'])

# Verify that each selected feature exists in the DataFrame
for feature in numerical_features:
    if feature not in data.columns:
        print(f"Warning: {feature} not found in data.")

# -----------------------------------------------------------
# 3. Model Training and Evaluation
# -----------------------------------------------------------
# Split the data into training and testing sets (80% train, 20% test)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Initialize and train a Linear Regression model
model = LinearRegression()
model.fit(x_train, y_train)

# Predict on the test set (predictions are in log space)
y_pred_log = model.predict(x_test)

# Evaluate model performance in log space
mae_log = mean_absolute_error(y_test, y_pred_log)
mse_log = mean_squared_error(y_test, y_pred_log)
r2_log = r2_score(y_test, y_pred_log)
print("Evaluation in log space:")
print(f"Mean Absolute Error (MAE): {mae_log:.2f}")
print(f"Mean Squared Error (MSE): {mse_log:.2f}")
print(f"R² Score: {r2_log:.2f}")

# Convert predictions and actual values back to original sale price scale
y_pred = np.exp(y_pred_log)
y_test_orig = np.exp(y_test)

# Evaluate model performance in the original sale price scale
mae_orig = mean_absolute_error(y_test_orig, y_pred)
mse_orig = mean_squared_error(y_test_orig, y_pred)
r2_orig = r2_score(y_test_orig, y_pred)
print("\nEvaluation in original sale price scale:")
print(f"Mean Absolute Error (MAE): {mae_orig:.2f}")
print(f"Mean Squared Error (MSE): {mse_orig:.2f}")
print(f"R² Score: {r2_orig:.2f}")

# -----------------------------------------------------------
# 4. Save the Trained Model
# -----------------------------------------------------------
# Ensure that the ML folder exists (if it doesn't, create it)
os.makedirs('ML', exist_ok=True)

# Save the trained model to a pickle file
joblib.dump(model, 'ML/house_price_model.pkl')
print("Trained model saved as 'ML/house_price_model.pkl'")
