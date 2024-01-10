from sklearn.ensemble import RandomForestRegressor
from fastapi import APIRouter
import pickle

with open("model.pkl", mode= "rb") as file:
    model: RandomForestRegressor = pickle.load(file)

router = APIRouter(prefix="/api/v1/hello-word", tags=["hello-world"])

@router.get("")
async def get_hello_word() -> str:
    return "Hello World!"