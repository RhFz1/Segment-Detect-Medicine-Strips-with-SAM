import os
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

# Connect to MongoDB
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client[os.getenv('MONGO_DB_NAME')]
collection = db[os.getenv('MONGO_COLLECTION_NAME')]

def store_in_db(data):
    try:
        collection.insert_one(data)
        print("Data stored in MongoDB successfully")
    except Exception as e:
        print(f"An error occurred while storing data in MongoDB: {e}")
        
def get_latest_record():
    try:
        latest_record = collection.find_one(sort=[('_id', -1)])  # Sort by _id in descending order

        if '_id' in latest_record:
            latest_record['_id'] = str(latest_record['_id'])
        return latest_record
    except Exception as e:
        print(f"An error occurred while retrieving the latest record: {e}")



if __name__ == '__main__':
    print(get_latest_record())