import os
import sys
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
from src.logging.logger import logging
from src.exceptions.custom_exceptions import CustomException
 
# Load environment variables from .env file
load_dotenv('.env')
 
# Initialize the OpenAI client with API key from environment variables
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))

# This block is to fetch the medicine names from the CSV file
def get_names():
    # Read the CSV file
    df = pd.read_csv('./assets/Tablet_Config.csv')
    # Get the medicine names
    names = ', '.join(df['Tablet Name'].to_list())
    # Return the medicine names
    return names

class GPT():
    def __init__(self):
        self.names = get_names()

    def inference(self, question):
        """
        Function to ask a question and get a response from GPT-4o-mini.
        
        :param question: The OCR output from the medicine strip image
        :return: Structured response from GPT in JSON format
        """
        try:
            response = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL'),  # Note: Replace with the correct model name if "gpt-4o-mini" is not available
                messages=[
                    {"role": "system", "content": "You are a medical expert and also a json expert, you will be given a text which is the OCR output from the medicine strip image."
                                                  f"These are the possible medicine names: {self.names} that you can expect in the OCR output."},
                    {"role": "user", "content": question}
                ],
                max_tokens=200,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
# Instructions:
# 1. Ensure that the OpenAI API key is set in the .env file.
# 2. Call the ask_gpt function with the OCR output as the question parameter.
# 3. The function returns a JSON-formatted string containing extracted information.
# 4. Parse the returned JSON string to access individual fields like medicine name, ingredients, etc.
# 5. Handle potential errors or missing information in the returned JSON.