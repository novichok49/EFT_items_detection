import pandas as pd
from pathlib import Path
from PIL import Image
from os import makedirs
import numpy as np
from typing import List
# from GridPacker import GridPacker
from Classes import GridPacker

class ImagesGenerator:
    """The class contains functions for
    construct images from other images for training model.
    """
    GRID_SIZE = 64

    def __init__(self,
                 images_data: pd.DataFrame,
                 save_path: Path = Path('./Images/')) -> None:
        """Link class to dataframe with images information, create images default folder.

        Args:
            images_data (pd.DataFrame): Dataframe with images information.
            save_path (Path, optional): The path to the folder where
            the created images will be stored. Defaults to Path('./Images/').
        """
        self.data = images_data.copy(deep=True)
        self.save_path = save_path
        makedirs(self.save_path, exist_ok=True)

    def rescale_grid(self,
                     h_grid: str,
                     w_grid: str,
                     column: str,
                     name: str) -> pd.DataFrame:
        """Rescale images by in game ratio.

        Args:
            h_grid (str): Height of images in grid units.
            w_grid (str): Width of images in grid units.
            column (str): The dirrectory name for new rescaled
            images. 
            name (str): Name for new images files.

        Returns:
            pd.DataFrame: Dataframe contains new column with
            paths to new images.
        """
        h_list = self.data[h_grid].values
        w_list = self.data[w_grid].values
        paths = self.data[column]
        for i, filepath in enumerate(paths):
            image = Image.open(filepath)
            new_size = (ImagesGenerator.GRID_SIZE * w_list[i],
                        ImagesGenerator.GRID_SIZE * h_list[i])
            image = image.resize(new_size)
            makedirs(self.save_path / name, exist_ok=True)
            saved_filepath = self.save_path / \
                name / f"{name[:-1].lower()}_{i}.png"
            self.data.loc[i, name] = saved_filepath
            image.save(saved_filepath)
        return self.data

    def create_mask(self,
                    column: str,
                    name: str) -> pd.DataFrame:
        # TODO @novichok49 Дополнить документацию,
        # изменить имя функции
        """_summary_

        Args:
            column (str): _description_
            name (str): _description_

        Returns:
            pd.DataFrame: _description_
        """
        paths = self.data[column]
        for i, filepath in enumerate(paths):
            image = Image.open(filepath)
            image = image.split()[-1]
            image = image.convert('1')
            makedirs(self.save_path / name, exist_ok=True)
            saved_filepath = self.save_path / \
                name / f"{name[:-1].lower()}_{i}.png"
            self.data.loc[i, name] = saved_filepath
            image.save(saved_filepath)
        return self.data


    def create_image(self,
                     size: tuple,
                     images: list,
                     masks: list,
                     bg_image=None)-> tuple:
        pass