import numpy as np

def apply_sharpen_image(image, size, alpha):
    """
    Sharpen an image using a custom convolution kernel.

    Args:
        image: numpy array (H, W, 3) in any data type
        size: odd integer specifying kernel size
        alpha: sharpening strength parameter

    Returns:
        Sharpened image with same shape and dtype

    Raises:
        ValueError: for invalid inputs
    """
    # Validate inputs
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel image in (H, W, 3) format")

    # Generate sharpening kernel
    kernel = np.full((size, size), -(alpha) / (size ** 2 - 1), dtype=np.float32)
    center = size // 2
    kernel[center, center] = 1 + alpha

    pad = size // 2
    output = np.empty_like(image)

    # Process each channel independently
    for c in range(3):
        channel = image[:, :, c].astype(np.float32)
        padded = np.pad(channel, pad, mode='reflect')

        # Create sliding window view
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))

        # Apply convolution
        convolved = np.sum(windows * kernel, axis=(-2, -1))

        # Handle integer types with rounding and clipping
        if np.issubdtype(image.dtype, np.integer):
            max_val = np.iinfo(image.dtype).max
            convolved = np.round(convolved).clip(0, max_val)
        else:
            convolved = np.clip(convolved, 0.0, 1.0)

        output[:, :, c] = convolved.astype(image.dtype)

    return output