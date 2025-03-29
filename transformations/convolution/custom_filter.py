import numpy as np

def apply_custom_filter(image, custom_filter):
    """
    Apply a custom 3x3 filter to an image

    Args:
        image: numpy array (H, W, 3) in any data type
        custom_filter: list of 9 numbers representing the 3x3 kernel

    Returns:
        Filtered image with same shape and dtype

    Raises:
        ValueError: for invalid inputs
    """
    # Validate inputs
    if len(custom_filter) != 9:
        raise ValueError("Filter must contain exactly 9 elements")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel image in (H, W, 3) format")

    # Convert filter to numpy array and reshape
    kernel = np.array(custom_filter, dtype=np.float32).reshape(3, 3)

    pad = 1  # 3x3 kernel requires 1 pixel padding
    output = np.empty_like(image)

    # Process each channel independently
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        padded = np.pad(channel, pad, mode='reflect')

        # Create sliding window view
        windows = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))

        # Apply convolution
        convolved = np.sum(windows * kernel, axis=(-2, -1))

        # Handle integer types
        if np.issubdtype(image.dtype, np.integer):
            max_val = np.iinfo(image.dtype).max
            min_val = np.iinfo(image.dtype).min
            convolved = np.round(convolved).clip(min_val, max_val)
        else:
            convolved = np.clip(convolved, 0.0, 1.0)

        output[:, :, c] = convolved.astype(image.dtype)

    return output