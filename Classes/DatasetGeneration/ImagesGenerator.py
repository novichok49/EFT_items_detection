import pandas as pd
from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict
from Classes.Utils import GridPacker, APIRequester, BaseImages
from copy import deepcopy
from .TarkovItemsDataset import TarkovItemsDataset
from tqdm import tqdm

class ImagesGenerator:
    def __init__(
            self,
            base_images_path: str | Path,
            class_field: str = 'normalizedName',
            grid_size: int = 64,
            seed: int = None):
        self.path = Path(base_images_path)
        if not self.path.exists():
            raise FileNotFoundError(f'No such directory: {self.path}')
        np.random.seed(seed)
        self._class_field = class_field
        response = APIRequester.post(
            name='items',
            fields=[self._class_field, 'width', 'height'])

        self._grid_size = grid_size
        self.image_dir = BaseImages(self.path)
        self._image_info = pd.json_normalize(response)\
            .set_index(self._class_field)

    def __getitem__(self, key: str):
        return self.image_dir

    def generate_dataset(
            self,
            size: Tuple[int, int],
            classes_on_image: int,
            count_base_images: int,
            base_dir: str,
            backgrounds_dir: str | Path,
            dataset_path: str = 'Mydataset') -> None:
        im_id = 0
        dataset_path = dataset_path
        dataset_path.mkdir(exist_ok=True)
        backgrounds_dir = Path(backgrounds_dir)
        bg_ims = [bg_im for bg_im in backgrounds_dir.iterdir()]
        grid_packer = GridPacker(512, size[1] - 1, self._grid_size)
        # 0 index need for __background__ class
        self.image_dir.create_labels(sep=1)
        labels_map = self.image_dir.get_decode_map()
        # Add background class
        labels_map[0] = "__background__"
        dataset = TarkovItemsDataset(dataset_path, labels_map)
        for _ in tqdm(range(count_base_images), desc='Generation'):
            base_imgs = self.image_dir[:]
            np.random.shuffle(base_imgs)
            slice_range = int(np.ceil(len(base_imgs) / classes_on_image))
            for i in range(slice_range):
                start = i * classes_on_image
                stop = i * classes_on_image + classes_on_image
                # Grid pack images
                im_slice = base_imgs[start:stop]
                images = [image for image, _ in im_slice]
                labels = [label for _, label in im_slice]
                grid_im, bboxes, labels = grid_packer.pack_images(images, labels)
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
        paste_x = np.random.randint(
            0, background_image.size[0] - grid_image.size[0])
        paste_y = np.random.randint(
            0, background_image.size[1] - grid_image.size[1])

        background_image.paste(im=grid_image,
                               box=(paste_x, paste_y),
                               mask=grid_image.getchannel('A'))
        #TODO Optimize
        new_bboxes = deepcopy(bboxes)
        for bbox in new_bboxes:
            bbox[0] += paste_x
            bbox[1] += paste_y
            bbox[2] += paste_x
            bbox[3] += paste_y
        return background_image, new_bboxes
