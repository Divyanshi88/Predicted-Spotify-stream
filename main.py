import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
try:
    df = pd.read_csv('spotify.csv', encoding='utf-8')
except UnicodeDecodeError:
    df = pd.read_csv('spotify.csv', encoding='ISO-8859-1')
    print(df.head())
    print(df.describe())
    print(df.isnull().sum())
     #Remove commas from numeric columns and convert them to floats
numeric_columns = ['Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach', 'YouTube Views', 
                   'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 'TikTok Views', 'YouTube Playlist Reach', 
                   'Apple Music Playlist Count', 'AirPlay Spins', 'SiriusXM Spins', 'Deezer Playlist Count', 
                   'Deezer Playlist Reach', 'Amazon Playlist Count', 'Pandora Streams', 'Pandora Track Stations', 
                   'Soundcloud Streams', 'Shazam Counts', 'TIDAL Popularity']
for col in numeric_columns:
    df[col] = df[col].astype(str).str.replace(',', '').astype(float)
    
df = df.dropna(axis=1, how='all')
    
    # Handle missing values
# For numerical features, we will use mean imputation
# For categorical features, we will use the most frequent value imputation
numerical_features = ['Track Score', 'Spotify Streams', 'Spotify Playlist Count', 'Spotify Playlist Reach', 
                      'Spotify Popularity', 'YouTube Views', 'YouTube Likes', 'TikTok Posts', 'TikTok Likes', 
                      'TikTok Views', 'YouTube Playlist Reach', 'Apple Music Playlist Count', 'AirPlay Spins', 
                      'SiriusXM Spins', 'Deezer Playlist Count', 'Deezer Playlist Reach', 'Amazon Playlist Count', 
                      'Pandora Streams', 'Pandora Track Stations', 'Soundcloud Streams', 'Shazam Counts']
categorical_features = ['Track', 'Album Name', 'Artist', 'ISRC', 'Explicit Track']

# Define the column transformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', SimpleImputer(strategy='mean'), numerical_features),
        ('cat', SimpleImputer(strategy='most_frequent'), categorical_features)
    ])
# Apply the transformations
df[numerical_features + categorical_features] = preprocessor.fit_transform(df[numerical_features + categorical_features])
print(df.isnull().sum())
# Encode categorical variables
encoder = OneHotEncoder(drop='first', sparse_output=False)
# Apply OneHotEncoder to the categorical features and convert to DataFrame
encoded_categories = pd.DataFrame(encoder.fit_transform(df[categorical_features]), columns=encoder.get_feature_names_out(categorical_features))
# Concatenate the encoded features with the original dataframe
df = pd.concat([df.drop(categorical_features, axis=1), encoded_categories], axis=1)
# Normalize numerical features
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
# Feature and target variables
X = df.drop(columns=['Spotify Streams'])  # Assuming 'Spotify Streams' is the target variable for popularity analysis
y = df['Spotify Streams']
# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Display the shape of the train and test sets
print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# Save the preprocessed data to CSV files for later use
X_train.to_csv('X_train_preprocessed.csv', index=False)
X_test.to_csv('X_test_preprocessed.csv', index=False)
y_train.to_csv('y_train_preprocessed.csv', index=False)
y_test.to_csv('y_test_preprocessed.csv', index=False)

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

# Load preprocessed data
X_train = pd.read_csv('X_train_preprocessed.csv')
X_test = pd.read_csv('X_test_preprocessed.csv')
y_train = pd.read_csv('y_train_preprocessed.csv')['Spotify Streams']
y_test = pd.read_csv('y_test_preprocessed.csv')['Spotify Streams']
# Drop non-numeric columns (if any)
X_train = X_train.select_dtypes(include=['number'])
X_test = X_test.select_dtypes(include=['number'])

# Check and handle categorical variables (if any)
# Example: One-hot encoding for categorical variables
X_train = pd.get_dummies(X_train)
X_test = pd.get_dummies(X_test)

# Initialize and train the Random Forest Regressor model
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Calculate evaluation metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print the evaluation metrics
print(f"Mean Absolute Error: {mae:.2f}")
print(f"Mean Squared Error: {mse:.2f}")
print(f"R-squared: {r2:.2f}")

# Visualize actual vs. predicted values
plt.scatter(y_test, y_pred, alpha=0.5)
plt.xlabel("Actual Spotify Streams")
plt.ylabel("Predicted Spotify Streams")
plt.title("Actual vs. Predicted Spotify Streams")
plt.show()

# Plot feature importances
importances = model.feature_importances_
indices = pd.Series(importances, index=X_train.columns).sort_values(ascending=False).head(10)
plt.barh(indices.index, indices.values)
plt.xlabel("Importance")
plt.ylabel("Feature")
plt.title("Top 10 Feature Importances")
plt.show()


from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Check for non-numeric columns and convert if necessary
X_train = X_train.apply(pd.to_numeric, errors='coerce')
X_test = X_test.apply(pd.to_numeric, errors='coerce')

# Check for NaN values and handle them
X_train.fillna(0, inplace=True)
X_test.fillna(0, inplace=True)

# Create a binary classification target based on the median value
y_train_class = (y_train > y_train.median()).astype(int)
y_test_class = (y_test > y_test.median()).astype(int)

# Train a RandomForestClassifier model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train_class)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Compute the confusion matrix
cm = confusion_matrix(y_test_class, y_pred)

# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot(cmap='Blues')
plt.title('Confusion Matrix')
plt.show()
