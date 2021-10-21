import logging
from common.logging_utils import configure_logging
configure_logging("logging_config/logging_debug_debug.yaml")
from fastapi import FastAPI
from starlette.responses import JSONResponse
from fastapi import APIRouter
from routers.detector_router import router as detector_router
from routers.classifier_router import router as classifier_router


app = FastAPI()
app.include_router(detector_router, prefix='/detector')
app.include_router(classifier_router, prefix='/classifier')


@app.get('/', status_code=200)
async def healthcheck():
    logging.info("Hola INFO")
    logging.warning("Hola WARNING")
    logging.error("Hola ERROR")
    return {"msg": "MultiApi Ready"}


# sudo docker build -t vehicle-detector-api .
# sudo docker run --rm --runtime nvidia -p 5000:5000  vehicle-detector-api


