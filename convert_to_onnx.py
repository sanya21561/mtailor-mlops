import torch
from PIL import Image
from pytorch_model import BasicBlock

def convert_model_to_onnx(weights_path: str, image_path: str, output_path: str = "model.onnx"):
    model = BasicBlock()
    model.load_weights(weights_path)
    model.eval()

    img = Image.open(image_path).convert("RGB")
    input_tensor = model.preprocess_numpy(img)  
    input_tensor = input_tensor.unsqueeze(0)   

    torch.onnx.export(
        model,
        input_tensor,
        output_path,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        opset_version=11,
        do_constant_folding=True
    )

    print(f"ONNX model exported to: {output_path}")

if __name__ == "__main__":
    weights_file = "weights/pytorch_model_weights.pth"
    sample_image_path = "images/n01440764_tench.jpeg" 
    convert_model_to_onnx(weights_file, sample_image_path)
