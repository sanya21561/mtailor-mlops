import requests
from PIL import Image
from model import ImagePreprocessor

def test_local_server(image_path: str):
    url = "http://localhost:8080/"
    img = Image.open(image_path).convert("RGB")
    input_array = ImagePreprocessor().preprocess(img)

    payload = {
        "input": input_array.tolist()
    }

    headers = {
        "Content-Type": "application/json"
    }

    response = requests.post(url, json=payload, headers=headers)

    if response.status_code == 200:
        print(f"Success: {response.json()}")
    else:
        print(f"Failed with status {response.status_code}")
        print(response.text)


if __name__ == "__main__":
    test_local_server("images/n01440764_tench.jpeg")
