import numpy as np
from sklearn.cluster import KMeans

def resize_image(image, new_size):
    """
    Custom image resizing using nearest-neighbor interpolation
    Args:
        image: numpy array of shape (H, W, C)
        new_size: tuple (new_height, new_width)
    Returns:
        Resized numpy array
    """
    # Validate input
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")

    h, w = image.shape[:2]
    new_h, new_w = new_size

    # Create grid of indices for new image
    row_indices = np.clip((np.arange(new_h) * (h / new_h)).astype(int), 0, h - 1)
    col_indices = np.clip((np.arange(new_w) * (w / new_w)).astype(int), 0, w - 1)

    # Use advanced indexing to sample pixels
    return image[row_indices[:, None], col_indices]


def get_dominant_color(image, k=5, resize_to=(100, 100)):
    """
    Find the most dominant color in a numpy image array
    Args:
        image: numpy array (H, W, C) in RGB format
        k: number of clusters for KMeans
        resize_to: target size for faster processing
    Returns:
        Hex string representing dominant color
    """
    # Validate input
    if not isinstance(image, np.ndarray):
        raise ValueError("Input must be a numpy array")

    # Handle grayscale and single-channel images
    if image.ndim == 2:  # Grayscale (H, W)
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 1:  # Single channel (H, W, 1)
        image = np.repeat(image, 3, axis=-1)
    elif image.shape[2] == 4:  # RGBA - drop alpha channel
        image = image[..., :3]
    elif image.shape[2] != 3:
        raise ValueError("Unsupported number of channels")

    # Resize image using custom function
    resized = resize_image(image, resize_to)

    # Reshape to pixel list
    pixels = resized.reshape(-1, 3)

    # Handle images with transparency
    pixels = pixels.astype(np.float32)  # KMeans works better with floats

    # Apply KMeans clustering
    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
    kmeans.fit(pixels)

    # Get dominant color
    counts = np.bincount(kmeans.labels_, minlength=k)
    dominant_color = kmeans.cluster_centers_[np.argmax(counts)]

    # Convert to integer RGB values
    r, g, b = np.round(dominant_color).astype(int)
    r, g, b = np.clip([r, g, b], 0, 255)  # Ensure valid color values

    return "#{:02x}{:02x}{:02x}".format(r, g, b)
