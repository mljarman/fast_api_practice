from fastapi import FastAPI
from enum import Enum

class ModelName(str, Enum):
    alexnet = "alexnet"
    resnet = "resnet"
    lenet = "lenet"

app = FastAPI()

@app.get("/")
def read_root():
  return {"message": "Hello World"}

@app.get("/operationexp")
async def op_root():
    return{"message":"Hi World"}

# item_id is a path parameter, q is a query
#In this case, the function parameter q will be optional, and will be None by default.
@app.get("/items/{item_id}")
async def read_item(item_id: str, q: str = None):
    if q:
        return {"item_id": item_id, "q":q}
    return {"item_id" : item_id}

@app.get("/boolitem/{item_id}")
async def read_item(item_id: str, q: str = None, short:bool = False):
    item = {"item_id": item_id}
    if q:
        item.update({"q":q})
    if not short:
        item.update({"description": "awesome item w/long description"})
    return item

@app.get("/model/{model_name}")
# use class from above to return different attributes using Enum
# model_name is a path parameter
async def get_model(model_name: ModelName):
    if model_name.value == "alexnet":
        return {"model_name": model_name, "message": "Deep Learning"}
    if model_name == ModelName.lenet:
        return {"model_name": model_name, "message": "LeCNN all images"}
    return {"model_name": model_name, "message": "Have some residuals"}
