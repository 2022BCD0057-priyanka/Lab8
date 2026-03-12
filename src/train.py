import pandas as pd
import json
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load dataset
df = pd.read_csv("data/housing.csv")

# Drop rows with missing values
df = df.dropna()

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Convert ocean_proximity to numbers
X = pd.get_dummies(X)

# Train test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
pred = model.predict(X_test)

# Metrics
mse = mean_squared_error(y_test, pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, pred)

metrics = {
    "RMSE": rmse,
    "R2": r2,
    "dataset_size": len(df)
}

print(metrics)

# Save metrics
with open("metrics.json", "w") as f:
    json.dump(metrics, f)