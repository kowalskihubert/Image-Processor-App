import numpy as np


def roberts_edge_detection(image, threshold):
    """
    Perform Roberts Cross edge detection on a grayscale image (3 identical channels)

    Args:
        image: numpy array (H, W, 3) in any data type
        threshold: edge detection threshold (0-255) scaled to image's range

    Returns:
        Binary edge image with same shape and dtype
    """
    # Validate inputs
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel image in (H, W, 3) format")

    # Extract first channel (all channels assumed identical)
    gray = image[:, :, 0].astype(np.float32)

    # Define Roberts Cross kernels
    roberts_x = np.array([[1, 0],
                          [0, -1]], dtype=np.float32)
    roberts_y = np.array([[0, 1],
                          [-1, 0]], dtype=np.float32)

    # Pad image to maintain original size
    pad_width = ((0, 1), (0, 1))  # Add 1 pixel to bottom and right
    padded = np.pad(gray, pad_width, mode='reflect')

    # Create sliding window views
    windows = np.lib.stride_tricks.sliding_window_view(padded, (2, 2))

    # Compute gradients using both kernels
    gx = np.sum(windows * roberts_x, axis=(-2, -1))
    gy = np.sum(windows * roberts_y, axis=(-2, -1))

    # Calculate gradient magnitude
    magnitude = np.sqrt(gx ** 2 + gy ** 2)
    magnitude = (magnitude / np.max(magnitude)) * 255

    # Determine value range parameters
    original_dtype = image.dtype
    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
        scaled_threshold = threshold
    else:
        max_val = 1.0
        scaled_threshold = threshold / 255.0

    # Apply threshold and create binary image
    edges = np.where(magnitude >= scaled_threshold, max_val, 0)

    # Stack to 3 channels and cast to original dtype
    return np.stack([edges, edges, edges], axis=-1).astype(original_dtype)