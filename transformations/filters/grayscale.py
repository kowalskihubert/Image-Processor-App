import numpy as np

def validate_and_extract_channels(image):
    """
    Validate the input image shape and extract RGB channels.

    Args:
        image (np.ndarray): Input image as a 3D numpy array with shape (H, W, 3).

    Returns:
        tuple: Three 2D numpy arrays representing the R, G, and B channels.

    Raises:
        ValueError: If the input image is not a 3-channel RGB image in (H, W, 3) format.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel RGB image in (H, W, 3) format")

    r = image[..., 0].astype(np.float32)
    g = image[..., 1].astype(np.float32)
    b = image[..., 2].astype(np.float32)
    return r, g, b


def rgb_to_grayscale_custom(rgb_array, weights=(0.3, 0.6, 0.1)):
    """
    Convert an RGB image to grayscale using custom weights for the RGB channels.

    Args:
        rgb_array (np.ndarray): Input image as a 3D numpy array with shape (H, W, 3).
        weights (tuple): Weights for the R, G, and B channels respectively. Default is (0.3, 0.6, 0.1).

    Returns:
        np.ndarray: Grayscale image as a 3D numpy array with shape (H, W, 3).

    """

    r, g, b = validate_and_extract_channels(rgb_array)
    wr, wg, wb = weights
    wsum = np.sum(weights)

    # Custom weights (can be modified)
    gray_values = (r * wr + g * wg + b * wb) // wsum

    # Create 3-channel array for display
    gray_3d = np.stack([gray_values] * 3, axis=2).astype(np.uint8)
    return gray_3d


def rgb_to_grayscale_luminosity(image):
    """
    Convert RGB image to grayscale using luminosity method.
    """
    r, g, b = validate_and_extract_channels(image)

    # Luminosity formula: 0.2989 * R + 0.5870 * G + 0.1140 * B
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b

    # Handle different data types
    if np.issubdtype(image.dtype, np.integer):
        gray = np.clip(np.round(gray), 0, 255)

    # Convert back to original dtype and stack channels
    gray = gray.astype(image.dtype)
    return np.stack([gray, gray, gray], axis=-1)


def rgb_to_grayscale_average(image):
    """
    Convert RGB image to grayscale using average method.
    """
    r, g, b = validate_and_extract_channels(image)

    # Calculate average of RGB values
    gray = (r + g + b) / 3.0

    # Handle integer types with rounding and clipping
    if np.issubdtype(image.dtype, np.integer):
        gray = np.clip(np.round(gray), 0, 255)

    # Convert back to original data type
    gray = gray.astype(image.dtype)

    # Stack grayscale values into 3 identical channels
    return np.stack([gray, gray, gray], axis=-1)


def binarize(image, threshold=128):
    """
    Binarize a 3-channel grayscale image (H, W, 3) using a threshold.
    All channels are assumed identical in input, output will have identical channels.

    Args:
        image: numpy array (H, W, 3) - grayscale image with repeated channels
        threshold: integer value (0-255) to scale to the image's range

    Returns:
        Binary image with shape (H, W, 3) containing only min and max values
    """
    # Validate input shape
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel grayscale image in (H, W, 3) format")

    # Extract first channel (all channels assumed equal)
    gray_channel = image[..., 0]

    # Clip threshold to valid range
    threshold = np.clip(threshold, 0, 255)

    # Determine value range based on dtype
    if np.issubdtype(image.dtype, np.integer):
        max_val = np.iinfo(image.dtype).max
        min_val = 0
    else:  # Float type (assumed 0.0-1.0)
        max_val = 1.0
        min_val = 0.0

    # Scale threshold to image's range
    scaled_threshold = (threshold / 255) * (max_val - min_val) + min_val

    # Create binary mask from single channel
    binary_mask = np.where(gray_channel >= scaled_threshold, max_val, min_val)

    # Stack mask to 3 channels and cast to original dtype
    binary_image = np.stack([binary_mask] * 3, axis=-1)

    return binary_image.astype(image.dtype)

def is_grayscale(image):
    """
    Check if the given image (of type np.ndarray) is grayscale.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input image must be a 3-channel RGB image in (H, W, 3) format")

    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return np.array_equal(r, g) and np.array_equal(g, b)