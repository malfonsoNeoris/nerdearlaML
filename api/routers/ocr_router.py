import logging
import json
import numpy as np
import cv2
import sys

from fastapi import APIRouter, File
from starlette.requests import Request
from starlette.responses import Response
from starlette.responses import JSONResponse
from fastapi import File, UploadFile, Form

from common.image_utils import *

# from ....base.tensorrt_yolov4.model import TrtYOLO_API
from paddleocr import PaddleOCR
logging.info("Initializing PADDLE OCR...")
paddle_args = {'use_angle_cls':False,
        'lang':'en',
        'use_gpu':True,
        'gpu_mem':500,
        'det_model_dir':'models/ocr/det/',
        'rec_model_dir':'models/ocr/rec/',
        'cls_model_dir':'models/ocr/cls/'}
    
ocr = PaddleOCR(**paddle_args) # need to run only once to download and load model into memory
logging.info("Done initializing.")

router = APIRouter()

@router.post('/detect')
async def detect(data: dict):
    result = __internal_detect(data)
    return JSONResponse(result)

@router.post('/detect_bytes',response_class=Response)
async def detect_bytes(data: Request):
    data_b = await data.body()
    result = __internal_detect(data_b)
    return JSONResponse(result)

@router.post('/detect2',response_class=Response)
async def detect2( data: UploadFile = File(...)  ):
    data_b = await data.read()
    result = __internal_detect(data_b)
    return JSONResponse(result)

def __internal_detect(data):
    if(type(data) is dict):
        image_readable = readb64(data['img'])
    else:
        image_readable = np.frombuffer(data,dtype=np.uint8)
        image_readable = cv2.imdecode(image_readable, cv2.IMREAD_COLOR)
    
    # replace with real predictor
    try:
        result = ocr.ocr(image_readable)
        results=[]
        for res in result:
            box = res[0]
            text = res[1][0]
            score = float(res[1][1])
            results.append({'box':box, 'ocr':text, 'score':score})
            if len(res)>2:
                logging.warning('WARNING!!!!!!', res[2:])
        return results
    except:
        logging.exception("Error ocurred while using OCR model")
        return []