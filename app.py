from flask import Flask, request, jsonify
from PIL import Image
from src.pipeline.inference import Inference

app = Flask(__name__)

@app.route('/inference', methods=['POST'])
def inference():
    """
    This function is used to perform inference on the image
    Input: Image
    Output: JSON response
    """

    # Create an instance of the Inference class
    inference = Inference()
    # Get the image from the request
    image = request.files['image']
    # Check if the image is None
    if image is None:
        return jsonify({'error': 'Image not found'})
    
    image = Image.open(image)
    # Perform inference on the image
    result = inference.inference(image=image)
    
    return jsonify(result)