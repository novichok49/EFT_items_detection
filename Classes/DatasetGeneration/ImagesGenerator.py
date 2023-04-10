import pandas as pd
from pathlib import Path
from PIL import Image
import os
import numpy as np
from typing import List, Tuple, Dict
from Classes.Utils import GridPacker
from copy import deepcopy
import shutil


class ImagesGenerator:
    """The class contains functions for
        construct images from other images for training model.
    """

    def __init__(self,
                 df: pd.DataFrame,
                 path: str | Path,
                 grid_size: int = 64,
                 seed: int = None) -> None:
        """Class constructor

        Args:
            df (pd.DataFrame): The data containing information about the images.
            path (str | Path): The path to the directory
                containing the images dirs.
            grid_size (int, optional): The size of the grid
                for rescaling the images. Defaults to 64.
            seed (int, optional): The seed to use for random number
                generation for class. Defaults to None.
        """
        self.df = df
        self.path = Path(path)
        self.images_dirs = os.listdir(self.path)
        self.grid_size = grid_size
        self.seed = seed
        np.random.seed(seed)

    def rescale_images_by_grid(self,
                               images_dir: str,
                               overwrite: bool = True) -> None:
        """Rescale all images in the specified directory
            by the grid size and save them.

        Args:
            images_dir (str): Directory containing the images to be rescaled.
            overwrite (bool, optional): Whether to overwrite the original images.
                If False, the rescaled images will be saved in a new directory named
                "{images_dir}.rescaled". Defaults to True.
        """
        pathes = Path(self.path, images_dir).glob("*")
        if not overwrite:
            new_path = self.path / f'{images_dir}.rescaled'
            os.makedirs(new_path, exist_ok=True)
        for image_path in pathes:
            id = os.path.basename(image_path).split(sep=".")[0]
            new_size = (self.grid_size * self.df.loc[id, 'width'],
                        self.grid_size * self.df.loc[id, 'height'])
            image = Image.open(image_path)
            image = image.resize(new_size)
            if overwrite:
                image.save(image_path)
            else:
                filename = os.path.basename(image_path)
                image.save(new_path / filename)

    def rotate_images(self,
                      images_dir: str) -> None:
        """Rotate all images in the specified directory by 90 degrees
            counter-clockwise and save them in a new directory
            named "{images_dir}.rotated".

        Args:
            images_dir (str): The name of the directory containing
                the images to be rotated.
        """
        pathes = Path(self.path, images_dir).glob("*")
        new_path = self.path / f'{images_dir}.rotated'
        os.makedirs(new_path, exist_ok=True)
        for image_path in pathes:
            image = Image.open(image_path)
            image = image.rotate(-90, expand=True)
            filename = os.path.basename(image_path)
            image.save(new_path / filename)

    def generate(self,
                 size: Tuple[int, int],
                 samples: int,
                 im_sourses: List[str],
                 bg_sourses: List[str]) -> None:
        pass

    def merge_image_dirs(self,
                         im_sourses: List[str]):
        """Merge multiple directories containing images
            into a single directory named "Images".

        Args:
            im_sourses (List[str]): A list of directory names
            containing the images to be merged.
        """
        # TODO add path check for im_sourses
        im_id = 0
        save_path = Path(self.path / "Images")
        save_path.mkdir(exist_ok=True)
        for sourse in im_sourses:
            pathes = Path(self.path / sourse).glob('*')
            for path in pathes:
                class_id = os.path.basename(path).split('.')[0]
                shutil.copy(path, save_path / f'{class_id}.{im_id}.png')
                im_id += 1

    @classmethod
    def plot_grid_on_bg(cls,
                        grid_image: Image.Image,
                        bboxes: Dict,
                        background_image: Image.Image) -> Tuple[Image.Image, Dict]:
        """Plot a grid image on a background image and adjust the
            bounding box coordinates accordingly.

        Args:
            grid_image (Image.Image): The image of the grid to be plotted.
            bboxes (Dict): A dictionary containing bounding box
                information for the objects in the grid.
            background_image (Image.Image): The image of the background to
                plot the grid on.

        Returns:
             tuple containing the resulting image with the grid plotted on the
                background and the updated bounding box coordinates.
        """
        cutted_im = ImagesGenerator.cut_empty_parts(grid_image)
        paste_x = np.random.randint(
            0, background_image.size[0] - cutted_im.size[0])
        paste_y = np.random.randint(
            0, background_image.size[1] - cutted_im.size[1])

        background_image.paste(im=cutted_im,
                               box=(paste_x, paste_y),
                               mask=cutted_im.getchannel('A'))
        new_bboxes = deepcopy(bboxes)
        # Updating bbox coordinates
        for bbox in new_bboxes:
            new_bboxes[bbox][:, 0] += paste_x
            new_bboxes[bbox][:, 1] += paste_y
        return background_image, new_bboxes

    @classmethod
    def cut_empty_parts(cls,
                        image: Image.Image) -> Image.Image:
        """Remove the empty parts of an image based on
            its alpha channel and return the resulting cropped image.

        Args:
            image (Image.Image): The image to be cropped.

        Returns:
            The resulting cropped image.
        """
        alpha = image.getchannel('A')
        alpha_bbox = alpha.getbbox()
        res = image.crop(alpha_bbox)
        return res
