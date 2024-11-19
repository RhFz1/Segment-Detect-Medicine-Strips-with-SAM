from src.pipeline.textract_inference import Inference

model = Inference()
  
for i in range(28, 29):
    img_path = f'assets/img_0{i}.jpeg'

    results = model.inference(image_path=img_path)

    print(results)