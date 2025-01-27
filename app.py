import requests
import os
import io
from src.utils.image_handler import image_to_bytes
from src.utils.common import generate_random_hash
from src.pipeline.textract_inference import Inference
from src.constants.db import store_in_db, get_latest_record
from datetime import datetime
from PIL import Image
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
from src.logging.logger import logging

load_dotenv()

app = Flask(__name__)
CORS(app, origins='*', allow_headers='*', methods='*')

inference = Inference()

access_token_cache = {'token': None, 'expires': None}
base_url = os.getenv("BASE_URL")

def test_inference(image: Image = None, image_path: str = './assets/img_001.jpg'):

    if image is None:
        image = Image.open(image_path)

    width, height = image.size

    print(f'The resolution of the image is: {width} x {height}')

    result = inference.inference(image=image)
    return result

@app.route(base_url+'/get-token', methods=['GET'])
def get_access_token():
    #global access_token_cache
    # Check if we already have a valid token
    if access_token_cache['token'] and access_token_cache['expires'] > datetime.now():
        return access_token_cache['token']

    url = os.environ.get("ACCESS_TOKEN_URL")
    payload = {
        'username': os.getenv("DMS_USERNAME"),
        'password': os.getenv("DMS_PASSWORD"),
        'client_id': os.getenv("DMS_CLIENT_ID"),
        'grant_type': 'password',
        'client_secret': os.getenv("DMS_CLIENT_SECRET")
    }
    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    print("Token payload",payload)
    response = requests.post(url, headers=headers, data=payload)

    headers = {
        'Content-Type': 'application/x-www-form-urlencoded'
    }
    response = requests.post(url, headers=headers, data=payload)

    if response.status_code == 200:
        return response.json().get("access_token")

    else:
        raise Exception("Failed to obtain access token")

@app.route(base_url+'/upload-infer', methods=['POST'])
def upload_infer():
    try:
        # First try to get the image
        if 'image' not in request.files:
            return jsonify({'error': 'No image found in request'}), 400
        
        image = request.files['image']

        # Open the image
        image = Image.open(image)
        # Converting the image to a byte stream
        byte_image = image_to_bytes(image)
        # Here first step would be uploading an image to dms.
        dms_upload = os.environ.get("DMS_URL")

        # Get the access token
        access_token = get_access_token()
        # Making headers
        headers = {"Authorization": "Bearer " + access_token}
        # Parameters
        params = {
            'documentTypeId': os.getenv("DMS_DOC_TYPE_ID"),
        }
        # Making the request
        # Here I want to send two params in body, 1 is my image and other is a destination path.

        # Storing the image in dms, and getting the document id in response.
        response = requests.post(dms_upload,headers=headers, params=params, files={'file': byte_image}, data={'documentPath': os.getenv('DMS_DOC_PATH')})
        
        if response.status_code != 200:
            return jsonify({'error': 'Failed to upload image to DMS'}), 400
        
        # Retreiving the document id.
        document_id = response.json()['data']['id']
        
        # Now I will make a request to the inference API to get the details of the medicine.
        results = test_inference(image=image)

        # Adding a key to the results dictionary
        results['document_id'] = document_id

        # Storing the results in the database
        hash_id = generate_random_hash()
        # Creating this as hash is not getting jsonified.
        results_copy = results.copy()
        results_copy['hash_id'] = hash_id
        store_in_db(results_copy)

        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route(base_url+'/download-infer', methods=['GET'])
def download_infer():
    try:
        # Here Ill get a document id from the request. Which has to be used to get the file from the dms
        document_id = request.form.get('document_id')

        # Now Once i have to document_id ill send a request to the dms to get the file.
        headers = {"Authorization": "Bearer " + get_access_token()}

        dms_download = os.getenv('DMS_URL') + '/download'
        # Here I will send a request to the dms to get the file.
        response = requests.get(dms_download, headers=headers, params={'id': document_id})
        # Here Im getting a byte stream of the image.
        byte_image = response.content
        # I have to convert this in to an image.
        image = Image.open(io.BytesIO(byte_image))

        # Now I will send this image to the inference API to get the details of the medicine.
        results = test_inference(image=image)
        
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route(base_url+'/fetch-latest', methods=['GET'])
def fetch_latest():
    try:
        # Here I will fetch the latest entry from the database, here we sort the entries by the timestamp and get the latest one.
        results = get_latest_record()
        return jsonify(results)
    except Exception as e:
        return jsonify({'error': str(e)}), 400
    
@app.route(base_url+'/predict', methods=['POST'])
def prediction():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400
    #image must be in PIL format
    image = request.files['image']

    try :
        image = Image.open(image)
        logging.info("Image sucessfully uploaded")
    except :
        logging.info("Image not uploaded")

    result = inference.inference(image=image)

    # result will be of this format
    '''
        {
            "<Medicine Name>":
                {
                    "Category:<string>
                    "Area": "<Area>",
                    "Count": "<Count>",
                    "Details": {
                        "Manufacturer": "<Manufacturer>",
                        "Dosage": "<Dosage>",
                        "Strength": "<Strength>",
                        "Expiry_Date": "<Expiry Date>",
                        "Manufacture_Date": "<Manufacture Date>",
                        "Formula": "<Formula>",
                        "Price": "<Price>" 
                               },
                    
                }, ...
        }
    '''
    results_copy = result.copy()
    store_in_db(results_copy)
    logging.info("reponse saved in DB")
    return jsonify(result)
@app.route(base_url+'/add-new-med', methods=['POST'])
def add_new_medicine():
    if 'image' not in request.files:
        return jsonify({'error': 'No image found in request'}), 400
    #image must be in PIL format
    image = request.files['image']

    try :
        image = Image.open(image)
        logging.info("Image sucessfully uploaded")
    except :
        logging.info("Image not uploaded")

    result = inference.inference(image=image,pred=True)

    # result will be of this format
    '''
        {
            "<Medicine Name>":
                {
                    "Area": "<Area>",
                    "Count": "<Count>",
                    "Details": {
                        "Manufacturer": "<Manufacturer>",
                        "Dosage": "<Dosage>",
                        "Strength": "<Strength>",
                        "Expiry_Date": "<Expiry Date>",
                        "Manufacture_Date": "<Manufacture Date>",
                        "Formula": "<Formula>",
                        "Price": "<Price>" 
                               },
                    
                }, ...
        }
    '''
    results_copy = result.copy()
    store_in_db(results_copy)
    logging.info("reponse saved in DB")
    return jsonify(result)




if __name__ == '__main__':
    print(f'Host: {os.getenv("HOST")} and Port: {os.getenv("PORT")}')
    app.run(host=os.getenv('HOST'), port=os.getenv('PORT'))