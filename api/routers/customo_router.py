from fastapi import APIRouter
from starlette.requests import Request
from starlette.responses import Response
from starlette.responses import JSONResponse
import time
import sys

from torch._C import device
from common.image_utils import *
from detectors.plate_recognizer import PlateRecognizer
from detectors.platemaskdetector import PlateMaskDetector

router = APIRouter()
print("Initializing  CUSTOM PLATE RECOGNIZER ...")
weights_file = "models/plate_recognition/y4t-letters_final.weights"
names_file = "models/plate_recognition/obj.names"
cfg_file = "models/plate_recognition/y4t-letters.cfg"
plate_ocr = PlateRecognizer(names_file,weights_file,cfg_file,input_size = (320, 160))
plate_warper = PlateMaskDetector("models/plate_mask/patentes_mask.pth", device='cuda')
print("Done initializing.")
import json

@router.post('/detect')
async def detect_base64(data: dict):
    result = detect(data)
    return JSONResponse(result)


@router.post('/detect_bytes',response_class=Response)
async def detect_bytes(data: Request):
    data_b = await data.body()
    result = detect(data_b)
    return JSONResponse(result)

async def detect(data):
    if(type(data) is dict):
        image_readable = readb64(data['img'])
    else:
        image_readable = np.frombuffer(data,dtype=np.uint8)
        image_readable = cv2.imdecode(image_readable, cv2.IMREAD_COLOR)    
    # replace with real predictor

    outputs = plate_warper.Predict(img)
    boxes = plate_warper.getBoxes(outputs)

    # keep only one or do for all
    results = []
    for n, box in enumerate(boxes):
        if box.cnt is None:
            continue
        try:
            warped, pts = plate_warper.getWarped(img,box, pad=20)
            plates = plate_ocr.DetectAndGetPlates(warped, conf=.2, nms=.2)
            results.append({'box':box, 'pts':pts, 'plates':plates})
        except:
            pass

        
    return results

