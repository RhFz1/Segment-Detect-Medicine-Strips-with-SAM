import numpy as np
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv
import os

load_dotenv()

# Connect to MongoDB
mongo_client = MongoClient(os.getenv('MONGO_URI'))
db = mongo_client[os.getenv('MONGO_DB_NAME')]
collection = db[os.getenv('MONGO_METADATA_COLLECTION')]



class GetCountApproximation():


 
    def __init__(self):
        # Fetch medicine configuration from MongoDB
        self.medicine_config = pd.DataFrame(list(collection.find({}, {'medicine_name': 1, 'attributes': 1,'category':1})))

    def get_count(self, medicine_name, estimated_strip_area, strip):
        # Get the medicine details from the configuration
        #medicine_details = self.medicine_config[self.medicine_config['medicine_name'] == medicine_name and self.medicine_config['category'] =='strip' if strip else 'box']
        medicine_details = self.medicine_config[
        (self.medicine_config['medicine_name'] == medicine_name) & 
        (self.medicine_config['category'] == 'strip' if strip else self.medicine_config['category'] == 'box')
         ]
        # Check if the medicine is in the configuration
        if medicine_details.empty:
            return {'Count': None, 'Area': estimated_strip_area}

        # Access the attributes
        attributes = medicine_details['attributes'].values[0]
        real_strip_area = attributes['area_of_strip']
        tablet_count = attributes['total_tablets']

        count = GetCountApproximation.estimate_tablet_count(estimated_strip_area, real_strip_area, tablet_count)

        return {'Count': count, 'Area': estimated_strip_area}
        
    @staticmethod
    def estimate_tablet_count(estimated_strip_area: float, real_strip_area: float, tablet_count: int, tolerance=0.05, strip=True):
        """
        estimated_strip_area: The estimated area of the medicine strip in real units (e.g., cm^2).
        real_strip_area: The actual area of the medicine strip in real units (e.g., cm^2).
        tablet_count: The known number of tablets in the strip.
        tolerance: Allowed margin of error as a fraction of the real area (default: 5%).
        """

        if not strip:
            return tablet_count

        # Calculate the average area per tablet based on the real strip area and known tablet count
        average_area_per_tablet = real_strip_area / tablet_count

        # Calculate the estimated number of tablets using the estimated strip area
        estimated_count = estimated_strip_area / average_area_per_tablet

        # Cap the estimated count at the actual number of tablets to avoid over-counting
        if estimated_count > tablet_count:
            return tablet_count

        # Calculate upper and lower bounds based on the tolerance
        lower_bound = estimated_count - (estimated_count * tolerance)
        upper_bound = estimated_count + (estimated_count * tolerance)

        # Round the estimate to the nearest integer, but check the bounds for tolerance
        lower_bound_floor = int(np.floor(lower_bound))
        upper_bound_ceil = int(np.ceil(upper_bound))

        # Use the bound which is closer to the estimated count
        if abs(estimated_count - lower_bound_floor) <= abs(estimated_count - upper_bound_ceil):
            final_count = lower_bound_floor
        else:
            final_count = upper_bound_ceil

        # Ensure the count is at least 1
        final_count = max(final_count, 1)

        # Cap the final count at the actual number of tablets
        final_count = min(final_count, tablet_count)
        return final_count



if __name__ == "__main__":
    # Assume we have the following inputs
    estimated_strip_area = 71  # Example estimated area in cm^2 (greater than real)
    real_strip_area = 69       # Example real area in cm^2
    tablet_count = 15            # Known number of tablets in the strip
    tolerance = 0.05             # Allow 5% error margin

    # Estimate the tablet count
    GetCountApproximation.estimate_tablet_count(estimated_strip_area, real_strip_area, tablet_count, tolerance)