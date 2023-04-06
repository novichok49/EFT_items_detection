import pandas as pd
from pathlib import Path
from PIL import Image
import os
import numpy as np
from typing import List, Tuple, Dict
from Classes.DatasetGeneration import GridPacker
from copy import deepcopy

class ImagesGenerator:
    """The class contains functions for
    construct images from other images for training model.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 path: str | Path,
                 grid_size: int=64,
                 seed: int=None) -> None:
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
        self.images_dirs = os.listdir(self.path) 
        self.grid_size = grid_size
        self.seed = seed
        np.random.seed(seed)

    def rescale_images_by_grid(self,
                              images_dir: str,
                              overwrite: bool=True) -> None:
        pathes = Path(self.path, images_dir).glob("*")
        if not overwrite:
            new_path = self.path / f'{images_dir}.rescaled'
            os.makedirs(new_path, exist_ok=True)
        for image_path in pathes:
            id = os.path.basename(image_path).split(sep=".")[0]
            new_size = (self.grid_size * self.df.loc[id, 'width'],
                        self.grid_size * self.df.loc[id,'height'])
            image = Image.open(image_path)
            image = image.resize(new_size)
            if overwrite:
                image.save(image_path)
            else:
                filename = os.path.basename(image_path)
                image.save(new_path / filename)


    def rotate_images(self,
                     images_dir: str)->None:
        pathes = Path(self.path, images_dir).glob("*")
        new_path = self.path / f'{images_dir}.rotated'
        os.makedirs(new_path, exist_ok=True)
        for image_path in pathes:
            image = Image.open(image_path)
            image = image.rotate(-90, expand=True)
            filename= os.path.basename(image_path)
            image.save(new_path / filename)

    def generate(self,
                 size: Tuple[int, int],
                 samples: int,
                 background: List[Image.Image]) -> None:
        pass
    
    @classmethod
    def plot_grid_on_bg(cls,
                        grid_image:Image.Image,
                        bboxes:Dict,
                        background_image:Image.Image):
        cutted_im = ImagesGenerator.cut_empty_parts(grid_image)
        paste_x = np.random.randint(0,background_image.size[0] - cutted_im.size[0])
        paste_y = np.random.randint(0,background_image.size[1] - cutted_im.size[1])
        
        background_image.paste(im=cutted_im,
                               box=(paste_x, paste_y),
                               mask=cutted_im.getchannel('A'))
        new_bboxes = deepcopy(bboxes)
        # Updating bbox coordinates 
        for bbox in new_bboxes:
            new_bboxes[bbox][:,0] += paste_x
            new_bboxes[bbox][:,1] += paste_y
        return background_image, new_bboxes
    
    @classmethod
    def cut_empty_parts(cls,
                        image:Image.Image):
        alpha = image.getchannel('A')
        alpha_bbox = alpha.getbbox()
        res = image.crop(alpha_bbox)
        return res