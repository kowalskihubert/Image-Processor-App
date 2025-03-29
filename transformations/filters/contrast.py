import numpy as np

def adjust_contrast(image, value):
    """
    Adjust image contrast using a value from 0.0 to 5.0
    0.0 = minimum contrast (uniform gray)
    1.0 = no change
    5.0 = maximum contrast

    Args:
        image: numpy array (H, W, 3) in any color space
        value: contrast adjustment value (0.0 to 5.0)

    Returns:
        Contrast-adjusted image with same shape and dtype
    """
    # Validate input
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel image in (H, W, 3) format")

    # Clip value to valid range
    value = np.clip(value, 0,5)

    original_dtype = image.dtype
    working_image = image.astype(np.float32)

    # Compute per-channel means
    channel_means = np.mean(working_image, axis=(0, 1), keepdims=True)

    # Apply contrast adjustment formula
    adjusted = (working_image - channel_means) * value + channel_means

    # Determine clipping range based on original dtype
    if np.issubdtype(original_dtype, np.integer):
        min_val = 0
        max_val = np.iinfo(original_dtype).max
    else:
        min_val = 0.0
        max_val = 1.0

    # Clip and convert back to original dtype
    adjusted = np.clip(adjusted, min_val, max_val)

    if np.issubdtype(original_dtype, np.integer):
        adjusted = np.round(adjusted)

    return adjusted.astype(original_dtype)