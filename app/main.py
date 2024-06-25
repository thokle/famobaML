
from fastapi import FastAPI

import PipelineClass
import Recommandations
app = FastAPI()
uri = "neo4j://65.108.80.255:7687"

@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/recommandation/{useremail}")
async def prediect(useremail):

    s =  Recommandations.Neo4jRecommendationSystem(uri)
    s.establish_connection()
    return s.get_recommendation(useremail)


@app.get("/start")
async def start():
    s = PipelineClass.PipelineClass()
    s.create_pipeline()

@app.get("/prediect/{email}/{groupname")
async def prediect(email: str, groupname: str):
    s = PipelineClass.PipelineClass()
    s.get_username_prediction(email, groupname)