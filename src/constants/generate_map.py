import numpy as np


def generate_map(sam, image,coordinates):
    """
    Generate a map of the coordinates using the SAM model.

    Args:
        sam (SAM): The SAM model.
        image ([np.ndarray]): The image.
        coordinates ([np.ndarray]): The list of coordinates.

    Returns:
        [np.ndarray]: The list of maps.
    """
    
    masks, _, _ = sam.predict(
        point_coords=None,
        point_labels=None,
        box=np.array([coordinates]),
        multimask_output=False
    )
    # Retreive the mask
    mask = masks[0]
    # Getting the pixel area of the mask.
    area = np.sum(mask)


    return mask, area

