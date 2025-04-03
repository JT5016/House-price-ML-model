import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv('ML/train.csv')

# Visualize SalePrice distribution
plt.hist(data['SalePrice'], bins=30, edgecolor='black')
plt.xlabel('Sale Price')
plt.ylabel('Frequency')
plt.show()

# Check for missing data
missing_data = data.isnull().sum().sort_values(ascending=False)
print(missing_data[missing_data > 0])

# Encode categorical variable 'Neighborhood'
data = pd.get_dummies(data, columns=['Neighborhood'], drop_first=True)

# Define numerical features and add dummy variables for Neighborhood
numerical_features = ['LotArea', 'GrLivArea', 'GarageCars', 'FullBath', 'OverallQual']
neighborhood_dummies = [col for col in data.columns if col.startswith('Neighborhood_')]
features = numerical_features + neighborhood_dummies

# Extract features and target
x = data[features]
y = np.log(data['SalePrice'])

# Verify that all selected features exist
for feature in features:
    if feature not in data.columns:
        print(f"Warning: {feature} not found in data.")
        
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state=42)
model = LinearRegression()
model.fit(x_train, y_train)
y_pred_log = model.predict(x_test)
mae_log = mean_absolute_error(y_test, y_pred_log)
mse_log = mean_squared_error(y_test, y_pred_log)
r2_log = r2_score(y_test, y_pred_log)
print("Evaluation in log space:")
print(f"Mean Absolute Error (MAE): {mae_log:.2f}")
print(f"Mean Squared Error (MSE): {mse_log:.2f}")
print(f"R² Score: {r2_log:.2f}")


y_pred = np.exp(y_pred_log)
y_test_orig = np.exp(y_test)

# Evaluate model performance in original space
mae_orig = mean_absolute_error(y_test_orig, y_pred)
mse_orig = mean_squared_error(y_test_orig, y_pred)
r2_orig = r2_score(y_test_orig, y_pred)
print("\nEvaluation in original sale price scale:")
print(f"Mean Absolute Error (MAE): {mae_orig:.2f}")
print(f"Mean Squared Error (MSE): {mse_orig:.2f}")
print(f"R² Score: {r2_orig:.2f}")
