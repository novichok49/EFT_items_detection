import pandas as pd
from pathlib import Path
from PIL import Image
from os import makedirs,listdir
import numpy as np
from typing import List, Tuple
from Classes.DatasetGeneration import GridPacker

class ImagesGenerator:
    """The class contains functions for
    construct images from other images for training model.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 path: str | Path,
                 images_dirs: List[str],
                 grid_size: int) -> None:
        """_summary_

        Args:
            df (pd.DataFrame): Dataframe with image information.
            path (str | Path): The path to the folder where
            the created images will be stored.
            images_dirs (List[str]): 
            grid_size (int): Grid size for images.
        """
        self.df = df
        self.path = Path(path)
        self.images_dirs = images_dirs
        self.grid_size = grid_size

    def rescale_image_by_grid(self,
                              images_dir: str,
                              overwrite: bool) -> None:
        pathes = Path(self.path, images_dir).glob("*")
        for i, image_path in pathes:
            new_size = (self.grid_size * self.df['width'][i],
                        self.grid_size * self.df['height'][i])
            image = Image.open(image_path)
            image = image.resize(new_size)
            if overwrite:
                image.save(image_path)
            else:
                # image.save(self.path / f'{}')
                pass

    def create_mask(self,
                    images_dir: str,
                    overwrite) -> None:
        pass

    def generate(self,
                 size: Tuple[int, int],
                 samples: int,
                 background: List[Image.Image]) -> None:
        pass
    