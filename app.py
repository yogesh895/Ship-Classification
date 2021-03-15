import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from BankNotes import BankNote
import numpy as np
import pickle
import pandas as pd

app = FastAPI()
templates = Jinja2Templates(directory="templates")
pickle_in = open("classifier.pkl","rb")
classifier=pickle.load(pickle_in)


# @app.get('/')
# def index():
#      return {'message': 'Hello, World'}


@app.get("/")
def home(request: Request):
     return templates.TemplateResponse("index.html", {"request": request})


@app.get('/{name}')
def get_name(name: str):
    return {'Welcome to Fast API': f'{name}'}


@app.post('/predict')
def predict_banknote(data:BankNote):
    data = data.dict()
    variance=data['variance']
    skewness=data['skewness']
    curtosis=data['curtosis']
    entropy=data['entropy']
    prediction = classifier.predict([[variance,skewness,curtosis,entropy]])
    if(prediction[0]>0.5):
        prediction="Fake note"
    else:
        prediction="Its a Bank note"
    return {
        'prediction': prediction
    }


if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload