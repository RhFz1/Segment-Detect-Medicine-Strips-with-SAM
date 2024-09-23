from openai import OpenAI
from dotenv import load_dotenv
import os
from src.logging.logger import logging
from src.exceptions.custom_exceptions import CustomException
import sys
 
# Load environment variables from .env file
load_dotenv('.env')
 
# Initialize the OpenAI client with API key from environment variables
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))


class GPT():

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
                    {"role": "system", "content": "You are a medical expert and you are given the OCR output of a medicine strip image."
                                                  "The medicine name is one out of these (medicine names : Obnyx,Ecosprin-75,Ezamed D 40mg,Sebifin,Combiflam,Calpol 650,Zofer,Shelcal-500,Azithral-500,Dolo-650)"
                                                  "Pease extract information and return it in a structured format like json dont return quotes just the dictionary string."
                                                  "Medicine_Name, formula/ingredients, strength, expiry date, manufacturer, manifacturing date, in the same key format. If not found return null"},
                    {"role": "user", "content": question}
                ],
                max_tokens=200
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