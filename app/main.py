import json
from fastapi import FastAPI, Response
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from app.Recommandations import Neo4jRecommendationSystem
from app.Pipeline import Pipe

app = FastAPI()
uri = "bolt://65.108.80.255:7687"

@app.get("/")
async def root():
    return {"message": "Hello World"}
def set_default(obj):
    if isinstance(obj, set):
        return list(obj)
    raise TypeError

@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


@app.get("/recommandation/{useremail}")
async def prediect(useremail):
    s = Neo4jRecommendationSystem(uri, "neo4j", "famoba2024")
    headers = {'Content-Type': 'application/json', 'cache-control': 'no-cache',}
    s.establish_connection()
    res = s.get_recommendation(useremail)
    return Response(json.dumps(res, default=set_default), headers=headers)


@app.get("/start")
def start():
    s = Pipe()
    s.create_pipeline()
    s.create_model()


@app.get("/prediect/{email}/{groupname")
async def prediect(email: str, groupname: str):
    s = Pipe()
    s.get_username_prediction(email, groupname)