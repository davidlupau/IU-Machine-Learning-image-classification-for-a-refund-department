import requests
import base64
from PIL import Image
import io
import os

def test_api():
    try:
        # First, verify the image exists
        image_path = "Dataset/images/1554.jpg"
        if not os.path.exists(image_path):
            print(f"Error: Image not found at {image_path}")
            return

        # Read and encode the image
        print("Reading image...")
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        print("Sending request to API...")
        response = requests.post(
            'http://localhost:5001/predict',  # Updated port
            json={'image': encoded_image}
        )

        # More detailed error handling
        print(f"Response status code: {response.status_code}")

        if response.status_code == 200:
            print("\nSuccessful prediction!")
            print("\nPrediction results:")
            for category, probability in response.json()['predictions'].items():
                print(f"{category}: {probability:.2%}")
        else:
            print("Error from API:")
            print(f"Status code: {response.status_code}")
            print(f"Response content: {response.text}")

    except requests.exceptions.ConnectionError:
        print("Connection failed - Is the API server running on port 5001?")
    except Exception as e:
        print(f"Test failed: {str(e)}")
        print(f"Error type: {type(e)}")

if __name__ == "__main__":
    test_api()