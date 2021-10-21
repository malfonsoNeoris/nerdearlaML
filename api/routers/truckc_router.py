from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response
from starlette.responses import JSONResponse
from fastapi import File, UploadFile, Form


import sys
from detectors.fastai_classifier import FastAiClassifier
from common.image_utils import *

router = APIRouter()
print("Initializing Classifier ...")
filename = 'models/truck_classifier/trucks.pkl'
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath
truck_class = FastAiClassifier(filename, use_gpu=True)
pathlib.PosixPath = temp
print("Done initializing.")
import json

@router.post('/detect')
async def detect_base64(data: dict):
    result = detect(data)
    return JSONResponse(result)

import time
@router.post('/detect_bytes',response_class=Response)
async def detect_bytes(data: Request):
    data_b = await data.body()
    result = detect(data_b)
    return JSONResponse(result)

@router.post('/detect2',response_class=Response)
async def detect2(data: UploadFile = File(...)  ):
    data_b = await data.read()
    result = await detect(data_b)
    return JSONResponse(result)
    
async def detect(data):
    if type(data) is dict:
        image_readable = readb64(data['img'])
    else:
        image_readable = np.frombuffer(data,dtype=np.uint8)
        image_readable = cv2.imdecode(image_readable, cv2.IMREAD_COLOR)    
    # replace with real predictor
    return truck_class.predict(image_readable)
