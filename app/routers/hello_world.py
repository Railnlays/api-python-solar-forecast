from sklearn.ensemble import RandomForestRegressor
import pickle

with open("model.pkl", mode= "rb") as file:
    m: list[str] = pickle.load(file)