from PIL import Image, UnidentifiedImageError
from pathlib import Path
from matplotlib import pyplot as plt


def plot_image(image_source: Path | Image.Image | str) -> None:
    """Plot image by path or PIL Image object.

    Args:
        image_source (Path | Image.Image | str): Path to image or image.
    """
    if isinstance(image_source, (Path, str)):
        try:
            with Image.open(image_source) as image:
                plt.title(f'imgsize: {image.size}')
                plt.imshow(image)
                plt.show()
        except (FileNotFoundError, UnidentifiedImageError):
            print(f"Could not open image file: {image_source}")
            return
    elif isinstance(image_source, Image.Image):
        plt.title(f'imgsize: {image_source.size}')
        plt.imshow(image_source)
        plt.show()
