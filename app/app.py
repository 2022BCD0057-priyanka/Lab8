from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
from sklearn.linear_model import LinearRegression

app = FastAPI()

# Load dataset
df = pd.read_csv("data/housing.csv")
df = df.dropna()

# Features and target
X = df.drop("median_house_value", axis=1)
y = df["median_house_value"]

# Convert categorical column
X = pd.get_dummies(X)

# Train model
model = LinearRegression()
model.fit(X, y)

# Input schema
class HouseData(BaseModel):
    longitude: float
    latitude: float
    housing_median_age: float
    total_rooms: float
    total_bedrooms: float
    population: float
    households: float
    median_income: float
    ocean_proximity: str


@app.get("/")
def home():
    return {"message": "California Housing Prediction API"}


@app.post("/predict")
def predict(data: HouseData):

    input_data = pd.DataFrame([data.dict()])

    # Convert categorical
    input_data = pd.get_dummies(input_data)

    # Match columns
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    prediction = model.predict(input_data)

    return {"predicted_house_price": float(prediction[0])}