# Bring in dependencies
from fastapi import FastAPI
from pydantic import BaseModel
from numpy import asarray
import keras


class CNN:
    def __init__(self):
        self.model = keras.models.load_model('32x0x3-ID_NonID_CNN.model')


class Item(BaseModel):
    ima: list


Classifier = CNN()
app = FastAPI()
categories = ["Chinese ID FRONT", "Chinese ID BACK", 'Not Chinese ID']


@app.post('/')
async def endpoint(item: Item):
    yhat = Classifier.model.predict([asarray(item.ima)]).tolist()
    return {'prediction': categories[yhat[0].index(1)]}
