import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import RandomForestRegressor

print("Training started...")

# Load dataset
df = pd.read_csv("resale_data.csv")

print("Dataset loaded")

# Select required columns
df = df[['month','storey_range','floor_area_sqm','lease_commence_date','resale_price']]

# Convert month (YYYY-MM → MM)
df['month'] = df['month'].str[-2:].astype(int)

# Add year column
df['year'] = 2023

# Encode categorical column
encoder = OrdinalEncoder()
df[['storey_range']] = encoder.fit_transform(df[['storey_range']])

# Split data
X = df[['month','storey_range','floor_area_sqm','lease_commence_date','year']]
y = df['resale_price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestRegressor()
model.fit(X_train, y_train)

print("Model trained")

# Save files
pickle.dump(model, open("DTR_model.pkl", "wb"))
pickle.dump(X_test, open("X_test.pkl", "wb"))
pickle.dump(y_test, open("y_test.pkl", "wb"))

print("Training completed! Files saved.")