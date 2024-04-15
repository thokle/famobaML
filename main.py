import sys
import pandas as pd
from fastapi import FastAPI
from Pipeline import get_username_prediction

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/predict/{useremail}/{coursename}")
async def prediect(useremail: str, coursename: str):
    return get_username_prediction(useremail, coursename)
