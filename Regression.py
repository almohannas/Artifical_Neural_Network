# Import necessary libraries
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Load preprocessed data
data = pd.read_csv('../preprocessing/preprocessed_housing.csv')

# Split data into features and target variable
X = data.drop('MEDV', axis=1)
y = data['MEDV']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Instantiate random forest regressor
rf = RandomForestRegressor(n_estimators=100, random_state=42)

# Fit model on training data
rf.fit(X_train, y_train)

# Predict on test data
y_pred = rf.predict(X_test)

# Calculate root mean squared error (RMSE) of predictions
rmse = mean_squared_error(y_test, y_pred, squared=False)

# Print RMSE
print('RMSE:', rmse)

# Get feature importances
importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False)

# Print feature importances
print('Feature importances:\n', importances)
