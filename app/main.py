from fastapi import FastAPI, Request
from contextlib import asynccontextmanager
from pydantic import BaseModel
import joblib
from pathlib import Path
import numpy as np


MODEL_PATH = Path(__file__).parent.parent / "iris_log_reg.joblib"

@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.model = joblib.load(MODEL_PATH)
    try:
        yield
    finally:
        app.state.model = None

app = FastAPI(lifespan= lifespan)


class Flowers(BaseModel):
    sepal_length: float  # Pydantic's float accepts int and convert it to float
    sepal_width: float
    petal_length: float
    petal_width: float

CLASSES = {
    0 : "setosa",
    1 : "versicolor",
    2 : "virginica"
}


@app.post('/predict')
def predict(request: Request, item: Flowers):
    X = np.array([
        [item.sepal_length, item.sepal_width, item.petal_length, item.petal_width]
    ] , dtype=float)

    model = request.app.state.model
    pred = model.predict(X)[0]
    pred = pred.item()
    return {'class_idx' : pred, 'class_name' : CLASSES[pred]}


@app.get('/health')
def health():
    return {'status' : 'ok'}


