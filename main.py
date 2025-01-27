import requests
import os
from src.utils.image_handler import image_to_bytes
from src.utils.common import generate_random_hash
from src.pipeline.textract_inference import Inference
from src.constants.db import store_in_db
from PIL import Image
from flask import Flask, request, jsonify
from dotenv import load_dotenv

load_dotenv()

# app = Flask(__name__)

inference = Inference()

def test_inference(image: Image = None, image_path: str = './packaged-drug-detection-module/assets/test_images/cut_strip_1.jpg'):

    if image is None:
        image = Image.open(image_path)

    result = inference.inference(image=image)
    return result

# @app.route('/get-token', methods=['GET'])
# def get_access_token():
#     #global access_token_cache
#     # Check if we already have a valid token
#     #if access_token_cache['token'] and access_token_cache['expires'] > datetime.now():
#     #    return access_token_cache['token']

#     url = os.environ.get("ACCESS_TOKEN_URL")
#     payload = {
#         'username': os.getenv("DMS_USERNAME"),
#         'password': os.getenv("DMS_PASSWORD"),
#         'client_id': os.getenv("DMS_CLIENT_ID"),
#         'grant_type': 'password',
#         'client_secret': os.getenv("DMS_CLIENT_SECRET")
#     }
#     headers = {
#         'Content-Type': 'application/x-www-form-urlencoded'
#     }
#     print("Token payload",payload)
#     response = requests.post(url, headers=headers, data=payload)

#     headers = {
#         'Content-Type': 'application/x-www-form-urlencoded'
#     }
#     response = requests.post(url, headers=headers, data=payload)

#     if response.status_code == 200:
#         return response.json().get("access_token")

#     else:
#         raise Exception("Failed to obtain access token")

# @app.route('/upload-infer', methods=['POST'])
# def upload_infer():
#     try:
#         # First try to get the image
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image found in request'}), 400
        
#         image = request.files['image']

#         # Open the image
#         image = Image.open(image)
#         # Converting the image to a byte stream
#         byte_image = image_to_bytes(image)
#         # Here first step would be uploading an image to dms.
#         dms_upload = os.environ.get("DMS_URL")

#         # Get the access token
#         access_token = get_access_token()
#         # Making headers
#         headers = {"Authorization": "Bearer " + access_token}
#         # Parameters
#         params = {
#             'documentTypeId': os.getenv("DMS_DOC_TYPE_ID"),
#         }
#         # Making the request
#         # Here I want to send two params in body, 1 is my image and other is a destination path.

#         # Storing the image in dms, and getting the document id in response.
#         response = requests.post(dms_upload,headers=headers, params=params, files={'file': byte_image}, data={'documentPath': os.getenv('DMS_DOC_PATH')})
        
#         if response.status_code != 200:
#             return jsonify({'error': 'Failed to upload image to DMS'}), 400
        
#         # Retreiving the document id.
#         document_id = response.json()['data']['id']
        
#         print(f'Document Id: {document_id}')

#         # Now I will make a request to the inference API to get the details of the medicine.
#         results = test_inference(image=image)

#         # Adding a key to the results dictionary
#         results['document_id'] = document_id

#         # Storing the results in the database
#         hash_id = generate_random_hash()
#         # Creating this as hash is not getting jsonified.
#         results_copy = results.copy()
#         results_copy['hash_id'] = hash_id
#         store_in_db(results_copy)

#         return jsonify(results)
#     except Exception as e:
#         return jsonify({'error': str(e)}), 400

# @app.route('/predict', methods=['POST'])
# def prediction():
#     if 'image' not in request.files:
#         return jsonify({'error': 'No image found in request'}), 400

#     image = request.files['image']
#     image = Image.open(image)

#     result = inference.inference(image=image)

#     # result will be of this format
#     '''
#         {
#             "<Medicine Name>":
#                 {
#                     "Medicine_Name": "<Medicine Name>",
#                     "Formula/Ingredients": "<Ingredients>",
#                     "Manufacturer": "<Manufacturer>",
#                     "Dosage": "<Dosage>",
#                     "Strength": "<Strength>",
#                     "Expiry_Date": "<Expiry Date>",
#                     "Manufacture_Date": "<Manufacture Date>",
#                     "Price": "<Price>"
#                 }, ...
#         }
#     '''

#     return jsonify(result)


# if __name__ == '__main__':
#     app.run(host=os.getenv('HOST'), port=os.getenv('PORT'))



if __name__ == '__main__':

    #with open('results.txt', 'w') as f:
    #    for i in range(1, 113):
            
    #path = f'./final_data/test_img{i}.jpeg'
    path="./packaged-drug-detection-module/assets/test_images/cut_strip_1.jpg"
    res = test_inference(image_path=path)
    store_in_db(res)

    # t0 = time.time()
    # # This is for single image.
    # path = '/home/syednoor/Desktop/Datasets/arm_images/img_012.jpg'
    # medicine_data = test_inference(image_path=path)
    # print(medicine_data)
    # t1 = time.time()
    # print(f'Image processed in {t1 - t0:.2f} seconds')