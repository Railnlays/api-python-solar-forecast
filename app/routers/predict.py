from sklearn.ensemble import RandomForestRegressor
from fastapi import APIRouter, Body
import pickle
from pydantic import BaseModel, Field
import numpy as np


class Features(BaseModel):
    day_binary: int = Field(default=20)
    month_binary: int
    year_binary: int
    interval: int
    orientation: float
    med_azimut_angle: float
    med_elevation_angle: float
    des_azimut_angle: float
    des_elevation_angle: float


with open("model.pkl", mode="rb") as file:
    model: RandomForestRegressor = pickle.load(file)

router = APIRouter(prefix="/api/v1/predict", tags=["predict"])


@router.post("/predict")
async def post_predict(features: Features = Body()) -> float:
    input_data = np.array([[
        features.day_binary,
        features.month_binary,
        features.year_binary,
        features.interval,
        features.orientation,
        features.med_azimut_angle,
        features.med_elevation_angle,
        features.des_azimut_angle,
        features.des_elevation_angle,
    ]])

    prediction = model.predict(input_data)

    return float(prediction[0])

