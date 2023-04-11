import pandas as pd
from pathlib import Path
from PIL import Image
import os
import numpy as np
from typing import List, Tuple, Dict
from Classes.Utils import GridPacker, ImageDirs, APIRequester
from copy import deepcopy
import shutil


class ImagesGenerator:
    def __init__(
            self,
            path: str | Path,
            image_dirs: List[str],
            grid_size: int = 64,
            seed: int = None) -> None:
        API = APIRequester()
        response = API.request(
            name='items',
            fields=['id', 'shortName', 'width', 'height'])
        np.random.seed(seed)

        self.path = Path(path)
        self.image_dirs = ImageDirs(path, image_dirs)
        self.image_info = pd.json_normalize(response).set_index('id')
        self.grid_size = grid_size

    def rescale_images_by_grid(self, dir: str) -> None:
        images = self.image_dirs[dir]
        for image_path in images:
            filename = os.path.basename(image_path)
            image_id = filename.split('.')[0]
            w = self.image_info.loc[image_id, 'width']
            h = self.image_info.loc[image_id, 'height']
            size = (w * self.grid_size, h * self.grid_size)
            image = Image.open(image_path)
            image = image.resize(size)
            image.save(image_path)

    def aug_rotate_images(self, dir: str) -> None:
        save_path = self.path / dir
        for image_path in self.image_dirs[dir]:
            filename = os.path.basename(image_path)
            new_filename = f'{filename[:-4]}.r.png'
            image = Image.open(image_path)
            image = image.rotate(-90, expand=True)
            image.save(save_path / new_filename)

    def generate(
            self,
            size: Tuple[int, int],
            samples: int,
            im_sourses: List[str],
            bg_sourse: str) -> None:
        pass

    # def merge_image_dirs(self,
    #                      im_sourses: List[str]):
    #     # TODO add path check for im_sourses
    #     im_id = 0
    #     save_path = Path(self.path / "Images")
    #     save_path.mkdir(exist_ok=True)
    #     for sourse in im_sourses:
    #         pathes = Path(self.path / sourse).glob('*')
    #         for path in pathes:
    #             class_id = os.path.basename(path).split('.')[0]
    #             shutil.copy(path, save_path / f'{class_id}.{im_id}.png')
    #             im_id += 1

    @classmethod
    def plot_grid_on_bg(
                    cls,
                    grid_image: Image.Image,
                    bboxes: Dict,
                    background_image: Image.Image) -> Tuple[Image.Image, Dict]:
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
    def cut_empty_parts(cls, image: Image.Image) -> Image.Image:
        alpha = image.getchannel('A')
        alpha_bbox = alpha.getbbox()
        res = image.crop(alpha_bbox)
        return res
