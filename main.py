# Bring in dependencies
from fastapi import FastAPI
from pydantic import BaseModel
import keras
from numpy import asarray


class Item(BaseModel):
    ima: list


app = FastAPI()
categories = ["Chinese ID FRONT", "Chinese ID BACK", 'Not Chinese ID']
model = keras.models.load_model('32x0x3-ID_NonID_CNN.model')


@app.post('/')
async def endpoint(item: Item):
    yhat = model.predict([asarray(item.ima)]).tolist()
    return {'prediction': categories[yhat[0].index(1)]}
