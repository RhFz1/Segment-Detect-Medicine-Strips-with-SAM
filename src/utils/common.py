import hashlib
import random
import string

def generate_random_hash(length=12):
    random_string = ''.join(random.choices(string.ascii_letters + string.digits, k=length))
    return str(hashlib.sha256(random_string.encode()).hexdigest())


def abridge_results(medicine_name, gpt_result, count_area, result,strip):
    if count_area['Count'] is None:
        result[medicine_name] = {
                'Category':('strip' if strip else 'box'),
                'Count': 'None',
                'Details': gpt_result,
                'Area': round(count_area['Area'], 2)
            }
        
    else:
        if medicine_name not in result:
            result[medicine_name] = {
                'Category':('strip' if strip else 'box'),
                'Count': float(count_area['Count']),
                'Details': gpt_result,
                'Area': round(count_area['Area'], 2)
            }
        else:
            result[medicine_name]['Count'] += count_area['Count']

    return result

