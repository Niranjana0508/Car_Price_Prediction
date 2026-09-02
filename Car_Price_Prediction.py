import pandas as pd

# Load the car dataset
df = pd.read_csv("car data.csv")

# Display the first 5 rows
print(df.head())

# Display the size of the dataset
print("\nDataset shape:")
print(df.shape)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())
# Display information about the dataset
print("\nDataset information:")
print(df.info())

# Display statistical summary
print("\nStatistical summary:")
print(df.describe())
# Convert categorical columns into numbers

df["Fuel_Type"] = df["Fuel_Type"].map({
    "Petrol": 0,
    "Diesel": 1,
    "CNG": 2
})

df["Selling_type"] = df["Selling_type"].map({
    "Dealer": 0,
    "Individual": 1
})

df["Transmission"] = df["Transmission"].map({
    "Manual": 0,
    "Automatic": 1
})

print("\nAfter converting categorical data:")
print(df.head())
# Separate input features and target

X = df.drop(["Selling_Price", "Car_Name"], axis=1)
y = df["Selling_Price"]

print("\nInput features:")
print(X.head())

print("\nTarget values:")
print(y.head())
# Split data
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nTraining data shape:")
print(X_train.shape)

print("\nTesting data shape:")
print(X_test.shape)


# Train Random Forest model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel training completed!")
import joblib

joblib.dump(model, "car_price_model.pkl")
print("Model saved successfully!")


# Make predictions
y_pred = model.predict(X_test)

print("\nPredicted prices:")
print(y_pred[:10])


# Evaluate model
from sklearn.metrics import r2_score, mean_absolute_error

r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

print("\nModel Evaluation:")
print("R2 Score:", r2)
print("Mean Absolute Error:", mae)

# Get car details from user
print("\nEnter car details for price prediction:")

year = int(input("Year: "))
present_price = float(input("Present Price: "))
driven_kms = int(input("Driven Kms: "))

print("\nFuel Type:")
print("0 = Petrol")
print("1 = Diesel")
print("2 = CNG")
fuel_type = int(input("Enter Fuel Type: "))

print("\nSelling Type:")
print("0 = Dealer")
print("1 = Individual")
selling_type = int(input("Enter Selling Type: "))

print("\nTransmission:")
print("0 = Manual")
print("1 = Automatic")
transmission = int(input("Enter Transmission: "))

owner = int(input("Number of Previous Owners: "))

# Create input data
car_data = [[
    year,
    present_price,
    driven_kms,
    fuel_type,
    selling_type,
    transmission,
    owner
]]

# Predict price
predicted_price = model.predict(car_data)

print("\nPredicted Selling Price:")
print("₹", round(predicted_price[0], 2), "Lakhs")