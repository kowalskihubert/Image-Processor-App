import numpy as np

def sobel_edge_detection(image, threshold):
    """
    Perform Sobel edge detection on a grayscale image (3 identical channels)

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

    # Define Sobel kernels
    sobel_x = np.array([[-1, 0, 1],
                        [-2, 0, 2],
                        [-1, 0, 1]], dtype=np.float32)

    sobel_y = np.array([[-1, -2, -1],
                        [0, 0, 0],
                        [1, 2, 1]], dtype=np.float32)

    # Pad image for convolution
    pad = 1
    padded = np.pad(gray, pad, mode='reflect')

    # Create sliding window views
    windows_x = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))
    windows_y = np.lib.stride_tricks.sliding_window_view(padded, (3, 3))

    # Compute gradients
    gx = np.sum(windows_x * sobel_x, axis=(-2, -1))
    gy = np.sum(windows_y * sobel_y, axis=(-2, -1))

    # Calculate gradient magnitude
    magnitude = np.sqrt(gx ** 2 + gy ** 2)

    original_dtype = image.dtype
    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
    else:
        max_val = 1.0

    # Scale threshold to image's range
    if np.issubdtype(original_dtype, np.integer):
        scaled_threshold = threshold
    else:
        scaled_threshold = threshold / 255.0

    # Apply threshold and create binary image
    edges = np.where(magnitude >= scaled_threshold, max_val, 0)

    # Stack to 3 channels and cast to original dtype
    return np.stack([edges, edges, edges], axis=-1).astype(original_dtype)