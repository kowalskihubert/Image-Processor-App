import numpy as np

def apply_negative(image):
    """
    Apply negative transformation to an image using efficient methods.

    Args:
        image: numpy array (H, W, C) in either uint8/uint16 or float format

    Returns:
        Negative image with same shape and dtype

    Raises:
        ValueError: for invalid input dimensions
    """
    # Validate input dimensions
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Input must be a 3D numpy array (H, W, C)")

    original_dtype = image.dtype

    if np.issubdtype(original_dtype, np.integer):
        # Integer image processing with LUT
        max_val = np.iinfo(original_dtype).max
        # Create reversed lookup table (e.g., 255->0, 254->1, ..., 0->255)
        lut = np.arange(max_val + 1, dtype=original_dtype)[::-1]
        return lut[image]
    else:
        # Float image processing (assumed 0.0-1.0 range)
        inverted = 1.0 - image
        return np.clip(inverted, 0.0, 1.0).astype(original_dtype)