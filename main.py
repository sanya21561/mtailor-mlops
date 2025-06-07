import base64
import io
import numpy as np
from PIL import Image
from model import OnnxModelLoader, ImagePreprocessor

model = OnnxModelLoader("model.onnx")
preprocessor = ImagePreprocessor()

def run(base64_image: str, run_id=None):
    try:
        image_data = base64.b64decode(base64_image)
        img = Image.open(io.BytesIO(image_data)).convert("RGB")

        input_array = preprocessor.preprocess(img)
        class_id = model.predict(input_array)

        return {
            "class_id": int(class_id),
            "status_code": 200
        }

    except Exception as e:
        return {
            "error": str(e),
            "status_code": 500
        }

