import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
from Classes.Utils import GridPacker, APIRequester, ImagesDir
from copy import deepcopy
from .TarkovItemsDataset import TarkovItemsDataset


class ImagesGenerator:
    def __init__(
            self,
            path: str | Path,
            class_field: str = 'normalizedName',
            grid_size: int = 64,
            seed: int = None):
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
        self._image_info = pd.json_normalize(response)\
            .set_index(self._class_field)

    def __getitem__(self, key: str):
        return self._image_dirs[key]

    def rescale_images_by_grid(self, dir: str) -> None:
        """
        Rescale images in `dir` folder and overwrite.

        Arguments:
            `dir` -- Directory name in `image_dirs` property.
        """
        for image_path, class_id in self._image_dirs[dir]:
            class_name = self._image_dirs[dir].decode_id(class_id)
            w = self._image_info.loc[class_name, 'width']
            h = self._image_info.loc[class_name, 'height']
            size = (w * self._grid_size, h * self._grid_size)
            image = Image.open(image_path)
            image = image.resize(size)
            image.save(image_path)

    def aug_rotate_images(self, dir: str) -> None:
        """
        Rotate all images in the `dir` 90 degrees clockwise
        and save them in a new folder named `dir_r90`.

        Arguments:
            `dir` -- Directory name in `image_dirs` property.
        """
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
            size: Tuple[int, int],
            classes_on_image: int,
            count_base_images: int,
            base_dir: str,
            backgrounds_dir: str | Path,
            dataset_name: str = 'Mydataset') -> None:
        im_id = 0
        dataset_path = self._path / dataset_name
        dataset_path.mkdir(exist_ok=True)
        backgrounds_dir = Path(backgrounds_dir)
        bg_ims = [bg_im for bg_im in backgrounds_dir.iterdir()]
        packer = GridPacker(512, size[1] - 1, self._grid_size)
        labels_map = self._image_dirs[base_dir].decode_map
        dataset = TarkovItemsDataset(dataset_path, labels_map)
        for _ in range(count_base_images):
            base_imgs = self._image_dirs[base_dir][:]
            np.random.shuffle(base_imgs)
            slice_range = int(np.ceil(len(base_imgs) / classes_on_image))
            for i in range(slice_range):
                start = i * classes_on_image
                stop = i * classes_on_image + classes_on_image
                # Grid pack images
                im_slice = base_imgs[start:stop]
                grid_im, bboxes, labels = packer.pack(im_slice)
                # Open random background
                bg_im = Image.open(np.random.choice(bg_ims, 1)[0])
                # Add grid image on background
                gen_im, bboxes = ImagesGenerator.plot_grid_on_bg(
                    grid_image=grid_im,
                    bboxes=bboxes,
                    background_image=bg_im)
                # Resize image and bboxes
                x_scale = size[0] / gen_im.size[0] 
                y_scale = size[1] / gen_im.size[1]
                gen_im = gen_im.resize(size=size)
                for bbox in bboxes:
                    bbox[0] = int(bbox[0] * x_scale)
                    bbox[1] = int(bbox[1] * y_scale)
                    bbox[2] = int(bbox[2] * x_scale)
                    bbox[3] = int(bbox[3] * y_scale)
                # Save image and bboxes
                filename = f'{im_id}.png'
                gen_im.save(dataset_path / filename)
                dataset.add_image(filename, bboxes, labels)
                im_id += 1
        dataset.save()

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
        #TODO Optimize
        new_bboxes = deepcopy(bboxes)
        for bbox in new_bboxes:
            bbox[0] += paste_x
            bbox[1] += paste_y
            bbox[2] += paste_x
            bbox[3] += paste_y
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
