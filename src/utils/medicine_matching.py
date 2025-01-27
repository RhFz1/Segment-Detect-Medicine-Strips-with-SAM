import pandas as pd
from rapidfuzz import fuzz
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to MongoDB
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client[os.getenv('MONGO_DB_NAME')]
collection = db[os.getenv('MONGO_METADATA_COLLECTION')]

class MedicineMatcher:
    def __init__(self):
        """
        Initialize the matcher with a list of known medicine names.
        """
        #names = pd.read_csv('./assets/Tablet_Config.csv')['Tablet Name'].values
        #self.medicine_list = list(names)
        #self.lowercase_medicines = [med.lower() for med in self.medicine_list]
        medicine_docs = collection.find({}, {'medicine_name': 1})  # Fetch only the medicine_name field
        self.medicine_list = [doc['medicine_name'] for doc in medicine_docs]
        self.lowercase_medicines = [med.lower() for med in self.medicine_list]

        
    def get_matches(self, ocr_text, threshold=0.6, top_n=3):
        """
        Get potential matches for OCR text using partial matching algorithm.
        
        Args:
            ocr_text (str): The text extracted from OCR
            threshold (float): Minimum similarity score to consider (0-1)
            top_n (int): Number of top matches to return
            
        Returns:
            list: List of tuples (medicine_name, confidence_score, algorithm)
        """
        matches = []
        ocr_text = ocr_text.lower()
        
        
        
        # 2. Partial Ratio (handles substrings)
        partial_scores = [(med, fuzz.partial_ratio(ocr_text, med.lower()) / 100, 'partial')
                       for med in self.medicine_list]
        
        # Filter by threshold and sort by score
        filtered_scores = [score for score in partial_scores if score[1] >= threshold]
        filtered_scores.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_scores[:top_n]
    
    def get_name(self, ocr_text, pred=False,threshold=0.6):
        """
        Get the best match for OCR text using partial matching algorithm.
        
        Args:
            ocr_text (str): The text extracted from OCR
            threshold (float): Minimum similarity score to consider (0-1)
            
        Returns:
            str: The best matching medicine name
        """
        if pred:
            return "Missing"
        matches = self.get_matches(ocr_text, threshold=threshold, top_n=1)
        if len(matches) > 0:
            return matches[0][0]
        return None