import os
import sys
import json
from openai import OpenAI
from dotenv import load_dotenv
from src.logging.logger import logging
from src.exceptions.custom_exceptions import CustomException

# Explicitly disable any default proxy configurations
os.environ.pop('HTTP_PROXY', None)
os.environ.pop('HTTPS_PROXY', None)
 
# Load environment variables from .env file
load_dotenv('.env')
 
# Initialize the OpenAI client with API key from environment variables
client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'), http_client=None)

class GPT():
    def __init__(self):
        self.prompt = open('./assets/prompt.txt', 'r').read()
        self.new_med_prompt=open('./assets/new_med_prompt.txt', 'r').read()
    def inference(self, question: str, new_med=False) -> dict:
        """
        Function to ask a question and get a response from GPT-4o-mini.
        
        Args:
        question (str): Question to ask the model

        Returns:
        dict: JSON-formatted response from the model
        """
        try:
            if not new_med:
                # adding a prompt to the question
                question = self.prompt + '\n\n' + question
                # Call the OpenAI API to get the response from GPT-4o-mini
            else:
                question = self.new_med_prompt + '\n\n' + question

            response = client.chat.completions.create(
                model=os.getenv('OPENAI_MODEL'),  # Note: Replace with the correct model name if "gpt-4o-mini" is not available
                messages=[
                    {"role": "system", "content": "You are a medical expert and also a json expert, you will be given a text which is the OCR output from the medicine strip image."
                                                  f"You are supposed to pull out the expiry date, manufacture date, ingredients, manufacturer, dosage, strength, and price from the text."},
                    {"role": "user", "content": question}
                ],
                max_tokens=300,
            )
            result = response.choices[0].message.content.strip()
            result = result.strip("```json").strip('```')
            # Parse the response to a JSON string
            result = json.loads(result)
            # Return the JSON-formatted string
            return result
        except Exception as e:
            logging.info(e)
            raise CustomException(e, sys)
# Instructions:
# 1. Ensure that the OpenAI API key is set in the .env file.
# 2. Call the ask_gpt function with the OCR output as the question parameter.
# 3. The function returns a JSON-formatted string containing extracted information.
# 4. Parse the returned JSON string to access individual fields like medicine name, ingredients, etc.
# 5. Handle potential errors or missing information in the returned JSON.