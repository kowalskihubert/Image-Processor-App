import numpy as np

def apply_average_filter(image, size):
    """
    Apply averaging filter to an image using a square kernel.

    Args:
        image: numpy array (H, W, 3) in any data type
        size: odd integer specifying kernel size

    Returns:
        Filtered image with same shape and dtype

    Raises:
        ValueError: for even kernel size or invalid input dimensions
    """
    # Validate inputs
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel image in (H, W, 3) format")

    pad = (size - 1) // 2
    kernel = np.ones((size, size)) / (size ** 2)
    output = np.empty_like(image)

    # Process each channel independently
    for c in range(3):
        # Pad current channel
        channel = image[:, :, c]
        padded = np.pad(channel, pad, mode='reflect')

        # Create sliding window view
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))

        # Apply convolution
        convolved = np.sum(windows * kernel, axis=(-2, -1))

        # Handle integer types with rounding and clipping
        if np.issubdtype(image.dtype, np.integer):
            max_val = np.iinfo(image.dtype).max
            convolved = np.round(convolved).clip(0, max_val)

        output[:, :, c] = convolved.astype(image.dtype)

    return output


def apply_gaussian_blur(image, size, sigma):
    """
    Apply Gaussian blur to an image with a specified kernel size and sigma.

    Args:
        image: numpy array (H, W, 3) in any data type
        size: odd integer specifying kernel size
        sigma: standard deviation of Gaussian distribution (>0)

    Returns:
        Blurred image with same shape and dtype

    Raises:
        ValueError: for invalid inputs
    """
    # Validate inputs
    if size % 2 == 0:
        raise ValueError("Kernel size must be an odd number")
    if sigma <= 0:
        raise ValueError("Sigma must be a positive number")
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel image in (H, W, 3) format")

    # Generate Gaussian kernel
    ax = np.arange(size) - size // 2
    xx, yy = np.meshgrid(ax, ax)
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma ** 2))
    kernel /= kernel.sum()  # Normalize

    pad = size // 2
    output = np.empty_like(image)

    # Process each channel independently
    for c in range(3):
        channel = image[:, :, c]
        padded = np.pad(channel, pad, mode='reflect')

        # Create sliding window view
        windows = np.lib.stride_tricks.sliding_window_view(padded, (size, size))

        # Apply convolution
        convolved = np.sum(windows * kernel, axis=(-2, -1))

        # Handle integer types
        if np.issubdtype(image.dtype, np.integer):
            max_val = np.iinfo(image.dtype).max
            convolved = np.round(convolved).clip(0, max_val)

        output[:, :, c] = convolved.astype(image.dtype)

    return output



