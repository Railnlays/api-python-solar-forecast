from sklearn.ensemble import RandomForestRegressor
from fastapi import APIRouter, Body
import pickle
from pydantic import BaseModel


class Features(BaseModel):
    day_binary: int
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
    return 4.0
