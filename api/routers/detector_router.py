from fastapi import APIRouter, datastructures
from starlette.requests import Request
from starlette.responses import Response
from starlette.responses import JSONResponse
from fastapi import File, UploadFile, Form

import sys
import numpy as np
from common.image_utils import *
from detectors.yolo_detector import YoloDetector

names_file = "models/detector/coco.names"
weights_file = "models/detector/y4t-custom.weights"
cfg_file = "models/detector/y4t-custom.cfg"



router = APIRouter()
print("Initializing TRUCK DETECTOR ...")
detector = YoloDetector(names_file,weights_file,cfg_file, use_gpu=False)
print("Done initializing.")
import json

@router.post('/detect')
async def detect_base64(data: dict):
    result = await detect(data)
    return JSONResponse(result)


import time
@router.post('/detect_bytes',response_class=Response)
async def detect_bytes(data: Request):
    data_b = await data.body()
    result = await detect(data_b)
    return JSONResponse(result)

@router.post('/detect2',response_class=Response)
async def detect2(conf:float, nms:float , data: UploadFile = File(...)  ):
    data_b = await data.read()
    result = await detect(data_b, conf, nms)
    return JSONResponse(result)

async def detect(data, conf=.6, nms_tresh = .5):
    if(type(data) is dict):
        image_readable = readb64(data['img'])
    else:
        image_readable = np.frombuffer(data,dtype=np.uint8)
        image_readable = cv2.imdecode(image_readable, cv2.IMREAD_COLOR)    
    # replace with real predictor
    classes, scores, boxes = detector.DetectOnImageBase(image_readable, conf=conf, nms_tresh=nms_tresh)   
    if boxes is None:
        classes, scores, boxes = [],[],[]
    return detector._encode_info(classes, scores, boxes)
