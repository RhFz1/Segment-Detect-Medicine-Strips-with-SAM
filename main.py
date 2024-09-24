from src.pipeline.inference import Inference


def test_inference():
    inference = Inference()
    image_path = './assets/img_011.jpg'

    result = inference.inference(image_path=image_path)
    return result

if __name__ == '__main__':
    medicine_data = test_inference() 
    # Extracting the key (medicine name) and details (dictionary)
    
    # Here im getting a dictionary with the medicine name as the key and the details as the value
    # I can extract the details of each medicine by using the key
    for key, value in medicine_data.items():
        print('Medicine Name:', key)
        for k, v in value.items():
            print(f'{k}: {round(v, 2) if isinstance(v, float) else v}')
        
        print('-----------------------------------')