import numpy as np

def adjust_brightness(image, delta):
    """
    Adjust image brightness using an additive parameter scaled to the data type range.

    Args:
        image: numpy array (H, W, C) in any color space
        delta: brightness adjustment (-100 to 100)
               -100 = completely black, 100 = completely white

    Returns:
        Adjusted image with same shape and dtype
    """
    # Validate input dimensions
    if not isinstance(image, np.ndarray) or image.ndim != 3:
        raise ValueError("Input must be a 3D numpy array (H, W, C)")

    # Clip delta to valid range
    delta = np.clip(delta, -100, 100)

    # Convert to float32 for calculations
    original_dtype = image.dtype
    working_image = image.astype(np.float32)

    # Determine value range based on data type
    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
        min_val = 0
    else:  # Float type (assumed 0.0-1.0)
        max_val = 1.0
        min_val = 0.0

    # Calculate actual additive value based on data type range
    scaled_delta = (delta / 100) * (max_val - min_val)

    # Apply delta to all channels
    working_image += scaled_delta

    # Clip values to valid range
    working_image = np.clip(working_image, min_val, max_val)

    # Round and convert back to original dtype
    if np.issubdtype(original_dtype, np.integer):
        working_image = np.round(working_image)

    return working_image.astype(original_dtype)


def adjust_gamma(image, gamma):
    """
    Apply gamma correction using a lookup table for efficient processing.

    Args:
        image: numpy array (H, W, C) in either uint8 (0-255) or float (0.0-1.0)
        gamma: float > 0, where:
               gamma < 1.0 brightens the image
               gamma > 1.0 darkens the image

    Returns:
        Gamma-corrected image with same shape and dtype
    """
    if gamma <= 0:
        raise ValueError("Gamma must be greater than 0")

    if np.issubdtype(image.dtype, np.integer):
        # Integer image processing with LUT
        max_val = np.iinfo(image.dtype).max
        # Create lookup table using vectorized operations
        lut = ((np.arange(max_val + 1, dtype=np.float32) / max_val) ** gamma)
        lut = np.round(lut * max_val).clip(0, max_val).astype(image.dtype)
        return lut[image]
    else:
        # Float image processing (assumed 0.0-1.0 range)
        return np.clip(image ** gamma, 0.0, 1.0).astype(image.dtype)