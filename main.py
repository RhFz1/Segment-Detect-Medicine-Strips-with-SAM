from src.pipeline.inference import Inference
from PIL import Image


inference = Inference()

def test_inference(image_path: str = './assets/img_006.jpg'):
    
    image = Image.open(image_path)

    result = inference.inference(image=image)
    return result

if __name__ == '__main__':

    with open('results.txt', 'w') as f:
        for i in range(10):
            path = f'./assets/img_00{i}.jpg'
            f.write(f'Image: img_00{i}.jpg\n')
            f.write('-----------------------------------\n')
            medicine_data = test_inference(path) 
            # Extracting the key (medicine name) and details (dictionary)
            # Here im getting a dictionary with the medicine name as the key and the details as the value
            # I can extract the details of each medicine by using the key
            for key, value in medicine_data.items():
                for k, v in value.items():
                    f.write(f'{k}: {round(v, 2) if isinstance(v, float) else v}\n')
                f.write('----\n')
            f.write('-----------------------------------\n\n')
    
    # This is for single image.
    # path = './assets/img_009.jpg'
    # medicine_data = test_inference(path)
    # print(medicine_data)
                
                