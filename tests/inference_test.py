# src/pipeline/test_inference.py
import torch
import numpy as np
import pandas as pd
import unittest
import json
from unittest.mock import patch, MagicMock
from src.pipeline.inference import Inference


class TestInference(unittest.TestCase):

    @patch('src.pipeline.inference.ViT')
    @patch('src.pipeline.inference.SAM')
    @patch('src.pipeline.inference.GPT')
    @patch('src.pipeline.inference.easyocr.Reader')
    @patch('src.pipeline.inference.pd.read_csv')
    def setUp(self, mock_read_csv, MockReader, MockGPT, MockSAM, MockViT):
        # Mock the CSV reading
        mock_read_csv.return_value = pd.DataFrame({
            'Tablet Name': ['MedicineA'],
            'Area in cm2': [1.0],
            'Total Tablets': [10]
        })

        # Initialize the Inference object
        self.inference = Inference()

        # Mock the OCR reader
        self.mock_ocr = MockReader.return_value
        self.mock_ocr.readtext.return_value = ['Sample text']

        # Mock the GPT model
        self.mock_gpt = MockGPT.return_value
        self.mock_gpt.inference.return_value = json.dumps([{'Medicine_Name': 'MedicineA'}])

        # Mock the SAM model
        self.mock_sam = MockSAM.return_value
        self.mock_sam.inference.return_value = [{
            'area': 100,
            'point_coords': [(50, 50)],
            'crop_box': [0, 0, 100, 100],
            'segmentation': np.ones((100, 100)),
            'bbox': [0, 0, 100, 100]
        }]

        # Mock the ViT model
        self.mock_vit = MockViT.return_value
        self.mock_vit.inference.return_value = torch.tensor([[0.95]])

    def test_init(self):
        self.assertIsInstance(self.inference.vit, MagicMock)
        self.assertIsInstance(self.inference.sam, MagicMock)
        self.assertIsInstance(self.inference.gpt, MagicMock)
        self.assertIsInstance(self.inference.ocr, MagicMock)
        self.assertEqual(self.inference.num_maps, 8)
        self.assertEqual(self.inference.vit_threshold, 0.90)
        self.assertEqual(self.inference.dist_min, 1e9)
        self.assertEqual(self.inference.area_min, 1e9)
        self.assertEqual(self.inference.scale_factor, 0)
        self.assertEqual(self.inference.area_real, 5.72555)
        self.assertFalse(self.inference.strip_config.empty)
        self.assertEqual(self.inference.result, {})

    @patch('src.pipeline.inference.cv2.imread')
    @patch('src.pipeline.inference.cv2.cvtColor')
    def test_inference_with_image_path(self, mock_cvtColor, mock_imread):
        mock_imread.return_value = np.ones((100, 100, 3))
        mock_cvtColor.return_value = np.ones((100, 100, 3))

        result = self.inference.inference(image_path='dummy_path')

        self.assertIn('MedicineA', result)
        self.assertAlmostEqual(result['MedicineA']['count'], 57.2555)
        self.assertAlmostEqual(result['MedicineA']['Area'], 572.55)
        self.assertEqual(result['MedicineA']['Medicine_Name'], 'MedicineA')

    def test_inference_with_image(self):
        image = np.ones((100, 100, 3))

        result = self.inference.inference(image_path=None, image=image)

        self.assertIn('MedicineA', result)
        self.assertAlmostEqual(result['MedicineA']['count'], 57.2555)
        self.assertAlmostEqual(result['MedicineA']['Area'], 572.55)
        self.assertEqual(result['MedicineA']['Medicine_Name'], 'MedicineA')

if __name__ == '__main__':
    unittest.main()