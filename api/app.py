import logging
from common.logging_utils import configure_logging
configure_logging("logging_config/logging_debug_debug.yaml")
from fastapi import FastAPI
from starlette.responses import JSONResponse
from fastapi import APIRouter
from routers.truckd_router import router as truckd_router
from routers.truckc_router import router as truckc_router
from routers.propd_router import router as propd_router
from routers.ocr_router import router as paddleo_router
from routers.customo_router import router as customo_router

app = FastAPI()
app.include_router(truckd_router, prefix='/truckdetector')
app.include_router(truckc_router, prefix='/classifier')
app.include_router(propd_router, prefix='/propdetector')
app.include_router(paddleo_router, prefix='/ocr')
app.include_router(customo_router, prefix='/platerecognition')


@app.get('/', status_code=200)
async def healthcheck():
    logging.info("Hola INFO")
    logging.warning("Hola WARNING")
    logging.error("Hola ERROR")
    return {"msg": "MultiApi Ready"}


# sudo docker build -t vehicle-detector-api .
# sudo docker run --rm --runtime nvidia -p 5000:5000  vehicle-detector-api


