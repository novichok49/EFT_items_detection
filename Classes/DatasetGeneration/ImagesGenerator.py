import pandas as pd
from pathlib import Path
from PIL import Image
import os
import numpy as np
from typing import List, Tuple, Dict
from Classes.Utils import GridPacker, APIRequester, ImagesDir
from copy import deepcopy
# adadsd


class ImagesGenerator:
    def __init__(
            self,
            path: str | Path,
            class_field: str = 'normalizedName',
            grid_size: int = 64,
            seed: int = None):
        """"""
        self._path = Path(path)
        if not self._path.exists():
            raise FileNotFoundError(f'No such directory: {self._path}')
        np.random.seed(seed)
        self._class_field = class_field
        response = APIRequester.post(
            name='items',
            fields=[self._class_field, 'width', 'height'])

        self._grid_size = grid_size
        self._image_dirs = {dir.name: ImagesDir(dir)
                            for dir in self._path.iterdir()
                            if dir.is_dir()}
        self._image_info = pd.json_normalize(
            response).set_index(self._class_field)
        
    def __getitem__(self, key:str):
        return self._image_dirs[key]

    def rescale_images_by_grid(self, dir: str) -> None:
        for image_path, class_id in self._image_dirs[dir]:
            class_name = self._image_dirs[dir].decode_id(class_id)
            w = self._image_info.loc[class_name, 'width']
            h = self._image_info.loc[class_name, 'height']
            size = (w * self._grid_size, h * self._grid_size)
            image = Image.open(image_path)
            image = image.resize(size)
            image.save(image_path)

    def aug_rotate_images(self, dir: str) -> None:
        save_dir_name = f'{dir}_r90'
        save_dir = Path(self._path / save_dir_name)
        save_dir.mkdir(exist_ok=True)
        self._image_dirs[save_dir_name] = ImagesDir(save_dir)
        for image_path, class_id in self._image_dirs[dir]:
            class_name = self._image_dirs[dir].decode_id(class_id)
            image = Image.open(image_path)
            image = image.rotate(-90, expand=True)
            self._image_dirs[save_dir_name].add_image(image, class_name)
        self._image_dirs[save_dir_name].save_state()
            

    def generate_dataset(
            self,
            grid_im_size: Tuple[int, int],
            samples_on_image: int,
            samples: int,
            im_dir: str,
            bg_dir: str) -> None:
        # im_pathes = deepcopy(self.image_dirs[im_dir])
        # k = int(np.ceil(len(im_pathes) / samples_on_image))
        im_pathes = deepcopy(self._image_dirs[im_dir])
        w = grid_im_size[0]
        h = grid_im_size[1]
        grid_packer = GridPacker(w, h, self._grid_size)
        for i in range(samples):
            sample = np.random.choice(im_pathes, samples_on_image)

    # def rename_images(self, dir):
    #     index = 0
    #     path = self._path / dir
    #     for image_path in self._image_dirs[dir]:
    #         filename = os.path.basename(image_path)
    #         image_id = filename.split('.')[0]
    #         class_code = self._image_info.loc[image_id]['class_code']
    #         new_name = f'{index}_{class_code}_.png'
    #         os.rename(image_path, path / new_name)
    #         index += 1
    #     self._image_dirs[dir].update_filepathes()

    @staticmethod
    def plot_grid_on_bg(
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

    @staticmethod
    def cut_empty_parts(image: Image.Image) -> Image.Image:
        alpha = image.getchannel('A')
        alpha_bbox = alpha.getbbox()
        res = image.crop(alpha_bbox)
        return res

    @property
    def image_dirs(self):
        return list(self._image_dirs.keys())
