import numpy as np

def canny_edge_detection(image, low_threshold, high_threshold):
    """
    Perform Canny edge detection on a grayscale image (3 identical channels)

    Args:
        image: numpy array (H, W, 3) in any data type
        low_threshold: low threshold for hysteresis (0-255) scaled to image's range
        high_threshold: high threshold for hysteresis (0-255) scaled to image's range

    Returns:
        Binary edge image with same shape and dtype
    """
    # Validate inputs
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("Input must be a 3-channel image in (H, W, 3) format")

    # Extract first channel and convert to float32 for processing
    original_dtype = image.dtype
    gray = image[:, :, 0].astype(np.float32)

    # Step 1: Gaussian smoothing
    def gaussian_kernel(size=5, sigma=1.4):
        m = size // 2
        kernel = np.fromfunction(
            lambda x, y: np.exp(-((x - m) ** 2 + (y - m) ** 2) / (2 * sigma ** 2)),
            (size, size)
        )
        kernel /= kernel.sum()
        return kernel

    gauss_kernel = gaussian_kernel()
    pad_width = ((2, 2), (2, 2))
    padded_gray = np.pad(gray, pad_width, mode='reflect')
    windows = np.lib.stride_tricks.sliding_window_view(padded_gray, (5, 5))
    smoothed = np.sum(windows * gauss_kernel, axis=(-2, -1))

    # Step 2: Compute gradients with Sobel operators
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)

    pad_sobel = ((1, 1), (1, 1))
    padded_smoothed = np.pad(smoothed, pad_sobel, mode='reflect')
    windows_sobel = np.lib.stride_tricks.sliding_window_view(padded_smoothed, (3, 3))

    gx = np.sum(windows_sobel * sobel_x, axis=(-2, -1))
    gy = np.sum(windows_sobel * sobel_y, axis=(-2, -1))

    # Compute gradient magnitude and angles
    magnitude = np.hypot(gx, gy)
    angle_deg = np.degrees(np.arctan2(gy, gx)) % 180

    # Step 3: Non-maximum suppression
    direction_bins = np.zeros_like(angle_deg, dtype=np.uint8)
    direction_bins[(angle_deg >= 22.5) & (angle_deg < 67.5)] = 1
    direction_bins[(angle_deg >= 67.5) & (angle_deg < 112.5)] = 2
    direction_bins[(angle_deg >= 112.5) & (angle_deg < 157.5)] = 3

    magnitude_padded = np.pad(magnitude, 1, mode='constant')

    # Calculate neighbors for each direction
    neighbor1_0 = magnitude_padded[:-2, 1:-1]
    neighbor2_0 = magnitude_padded[2:, 1:-1]
    neighbor1_45 = magnitude_padded[:-2, 2:]
    neighbor2_45 = magnitude_padded[2:, :-2]
    neighbor1_90 = magnitude_padded[1:-1, 2:]
    neighbor2_90 = magnitude_padded[1:-1, :-2]
    neighbor1_135 = magnitude_padded[:-2, :-2]
    neighbor2_135 = magnitude_padded[2:, 2:]

    # Create suppression masks
    mask0 = (direction_bins == 0) & (magnitude >= neighbor1_0) & (magnitude >= neighbor2_0)
    mask45 = (direction_bins == 1) & (magnitude >= neighbor1_45) & (magnitude >= neighbor2_45)
    mask90 = (direction_bins == 2) & (magnitude >= neighbor1_90) & (magnitude >= neighbor2_90)
    mask135 = (direction_bins == 3) & (magnitude >= neighbor1_135) & (magnitude >= neighbor2_135)

    suppressed = magnitude * (mask0 | mask45 | mask90 | mask135)

    # Step 4: Double thresholding and hysteresis
    scale_factor = 1.0
    scaled = suppressed * scale_factor

    # Determine thresholds based on data type
    if np.issubdtype(original_dtype, np.integer):
        max_val = np.iinfo(original_dtype).max
        scaled_high = high_threshold
        scaled_low = low_threshold
    else:
        max_val = 1.0
        scaled_high = high_threshold / 255.0
        scaled_low = low_threshold / 255.0

    # Thresholding
    strong = scaled >= scaled_high
    weak = (scaled >= scaled_low) & (scaled < scaled_high)

    # Hysteresis with 8-connectivity
    padded_strong = np.pad(strong, 1, mode='constant')
    windows = np.lib.stride_tricks.sliding_window_view(padded_strong, (3, 3))
    dilated = np.any(windows, axis=(-2, -1))

    valid_weak = weak & dilated
    edges = (strong | valid_weak).astype(original_dtype)

    if np.issubdtype(original_dtype, np.integer):
        edges = edges * max_val
    else:
        edges = edges.astype(np.float32)

    return np.stack([edges, edges, edges], axis=-1)