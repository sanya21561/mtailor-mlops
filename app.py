from fastapi import FastAPI, Request
from model import OnnxModelLoader
import numpy as np

app = FastAPI()
model = OnnxModelLoader("model.onnx")

@app.post("/")
async def predict(req: Request):
    data = await req.json()
    input_array = np.array(data["input"]).astype(np.float32)
    pred_class = model.predict(input_array)
    return {"class_id": pred_class}
