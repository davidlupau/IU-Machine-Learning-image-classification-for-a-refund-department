import requests
import base64
from PIL import Image
import io

def test_api():
    try:
        # Replace this with the path to one of your test images
        image_path = "Dataset/images/your_test_image.jpg"

        # Read and encode the image
        print("Reading image...")
        with open(image_path, 'rb') as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode('utf-8')

        # Send the request to our API
        print("Sending request to API...")
        response = requests.post(
            'http://localhost:5000/predict',
            json={'image': encoded_image}
        )

        # Check if request was successful
        if response.status_code == 200:
            print("\nSuccessful prediction!")
            print("\nPrediction results:")
            for category, probability in response.json()['predictions'].items():
                print(f"{category}: {probability:.2%}")
        else:
            print(f"Error: {response.json()['error']}")

    except Exception as e:
        print(f"Test failed: {str(e)}")

if __name__ == "__main__":
    test_api()