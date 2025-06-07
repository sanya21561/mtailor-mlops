from PIL import Image
from model import ImagePreprocessor, OnnxModelLoader


def run_test(image_path: str, expected_class: int, model_path="model.onnx"):
    print(f"Testing {image_path} (expected class: {expected_class})")

    preprocessor = ImagePreprocessor()
    model = OnnxModelLoader(model_path)

    img = Image.open(image_path)
    input_array = preprocessor.preprocess(img)
    predicted_class = model.predict(input_array)

    print(f"Predicted class ID: {predicted_class}")
    assert predicted_class == expected_class, f"Mismatch: expected {expected_class}, got {predicted_class}"
    print("Test passed!\n")


if __name__ == "__main__":
    try:
        run_test("images/n01440764_tench.jpeg", expected_class=0)
        run_test("images/n01667114_mud_turtle.JPEG", expected_class=35)
        print("All tests passed successfully.")
    except AssertionError as e:
        print(e)
        print("One or more tests failed.")

