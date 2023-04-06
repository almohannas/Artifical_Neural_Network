import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

# Load the data
df = pd.read_csv('data/Housing.csv')

# Check for missing values
print(df.isnull().sum())

# Split the dataset into features and target variable
X = df.drop('medv', axis=1) # features
y = df['medv'] # target variable

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Save the preprocessed data
preprocessed_df = pd.DataFrame(X_scaled, columns=X.columns)
preprocessed_df['medv'] = y
preprocessed_df.to_csv('data/preprocessed_Housing.csv', index=False)
