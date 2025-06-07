import argparse
import requests
import base64

def image_to_base64(image_path):
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def call_cerebrium_api(api_url, api_key, image_path):
    base64_image = image_to_base64(image_path)

    payload = {
        "base64_image": base64_image
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    response = requests.post(api_url, json=payload, headers=headers)

    if response.status_code != 200:
        print(f"Error: {response.status_code}")
        print(response.text)
        return None

    try:
        json_resp = response.json()
        class_id = json_resp.get("class_id")
        print(f"Predicted class: {class_id}")
        return int(class_id)
    except Exception as e:
        print(f"Failed to parse response: {e}")
        return None


def run_cerebrium_tests(api_url, api_key):
    test_cases = [
        ("images/n01440764_tench.jpeg", 0),
        ("images/n01667114_mud_turtle.JPEG", 35),
    ]

    for image_path, expected in test_cases:
        print(f"\nTesting {image_path} → Expected: {expected}")
        result = call_cerebrium_api(api_url, api_key, image_path)
        if result is None:
            print("Inference failed.")
        elif result == expected:
            print("Test passed")
        else:
            print(f"Test failed: Expected {expected}, got {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to test image")
    parser.add_argument("--url", type=str, required=True, help="Cerebrium POST endpoint URL")
    parser.add_argument("--api_key", type=str, required=True, help="Your Cerebrium API Key")
    parser.add_argument("--test", action="store_true", help="Run all preset test cases")

    args = parser.parse_args()

    if args.test:
        run_cerebrium_tests(args.url, args.api_key)
    elif args.image:
        call_cerebrium_api(args.url, args.api_key, args.image)
    else:
        print("Please provide either --image or --test")
