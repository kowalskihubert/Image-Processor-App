import matplotlib.pyplot as plt
from transformations.filters.grayscale import is_grayscale

def create_histogram(image):
    """
    Create a histogram for the given image.

    Args:
        image: numpy array (H, W, 3) for RGB or grayscale

    Returns:
        A matplotlib figure object containing the histogram
    """
    if is_grayscale(image):
        # Grayscale image
        plt.figure(figsize=(10, 6))
        plt.hist(image[..., 0].ravel(), bins=256, color='gray', alpha=0.8)
        plt.title('Grayscale Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
    else:
        # RGB image
        plt.figure(figsize=(10, 6))
        colors = ('red', 'green', 'blue')
        for i, color in enumerate(colors):
            plt.hist(image[..., i].ravel(), bins=256, color=color, alpha=0.8, label=f'{color.capitalize()} Channel')
        plt.title('RGB Histogram')
        plt.xlabel('Pixel Intensity')
        plt.ylabel('Frequency')
        plt.legend()

    fig = plt.gcf()  # Get the current figure
    plt.close()  # Close the plot to prevent it from displaying immediately
    return fig