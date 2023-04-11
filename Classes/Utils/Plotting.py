from PIL import Image, UnidentifiedImageError
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib import patches
from typing import Dict, List

# TODO Refactor doc


def plot_image(image_source: Path | Image.Image | str, image_bboxs: Dict = None) -> None:
    fig, ax = plt.subplots(1)
    if isinstance(image_source, (Path, str)):
        try:
            with Image.open(image_source) as image:
                ax.set_title(f'imgsize: {image.size}')
                ax.imshow(image)
                plt.show()
        except (FileNotFoundError, UnidentifiedImageError):
            print(f"Could not open image file: {image_source}")
            return
    elif isinstance(image_source, Image.Image):
        ax.set_title(f'imgsize: {image_source.size}')
        ax.imshow(image_source)
        if image_bboxs:
            plot_bbox(bboxes=image_bboxs, ax=ax)
        plt.show()


def plot_bbox(bboxes: Dict, ax=None) -> plt.Axes:
    if not ax:
        fig, ax = plt.subplots(1)
        ax.imshow(Image.new('RGBA', (512, 512)))
    for class_name, list_bbox in bboxes.items():
        for bbox in list_bbox:
            x, y = bbox[0], bbox[1]
            w, h = bbox[2], bbox[3]
            box = patches.Rectangle((x, y), w, h, color='r',
                                    linestyle='-', fill=False,
                                    linewidth=1)
            ax.add_patch(box)
            ax.text(x, y + 15, class_name[:5], color='r')
