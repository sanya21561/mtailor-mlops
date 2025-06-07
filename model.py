import numpy as np
from PIL import Image
import onnxruntime as ort
import torch
from torchvision import transforms


class ImagePreprocessor:
    def __init__(self, image_size=(224, 224)):
        self.transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225]),
        ])

    def preprocess(self, img: Image.Image) -> np.ndarray:
        if img.mode != "RGB":
            img = img.convert("RGB")
        tensor = self.transform(img)  #shape: [3, 224, 224]
        return tensor.unsqueeze(0).numpy()  #shape: [1, 3, 224, 224]


class OnnxModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.session = ort.InferenceSession(model_path)
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def predict(self, input_array: np.ndarray) -> int:
        outputs = self.session.run([self.output_name], {self.input_name: input_array})
        output_array = outputs[0]  # shape: [1, 1000]
        predicted_class = int(np.argmax(output_array, axis=1)[0])
        return predicted_class
