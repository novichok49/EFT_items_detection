from PIL import Image
from PIL.Image import Resampling
from typing import Tuple
from io import BytesIO
from pathlib import Path
import requests
from tqdm import tqdm
import pandas as pd
from typing import List, Tuple, Dict
import numpy as np
# from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder 
from .Utils import ApiService


class BackgroundImageEditor:
    COLORS = {
        'black': (0, 0, 0, 77),
        'blue': (28, 65, 86, 77),
        'default': (127, 127, 127, 77),
        'green': (21, 45, 0, 77),
        'grey': (29, 29, 29, 77),
        'orange': (60, 25, 0, 77),
        'red': (109, 36, 24, 77),
        'violet': (76, 42, 85, 77),
        'yellow': (104, 102, 40, 77),
        'border': (73, 81, 84, 77),
    }

    def __init__(self, cell_grid: Image.Image) -> None:
        cell_grid.putalpha(255)
        self.cell_grid = cell_grid

    def generate_background(self,
                            size: Tuple[int, int],
                            color: Tuple[int, int, int, int]) -> Image.Image:
        """
        Create image backgruond by filling it with cell_grid.

        Arguments:
            size -- output image size
            color -- Background color name 

        Returns:
            PIL image.
        """
        x_step, y_step = self.cell_grid.size
        tmp_image = Image.new('RGBA', self.cell_grid.size, color)
        tmp_image = Image.alpha_composite(self.cell_grid, tmp_image)
        new_image = Image.new('RGBA', size=size, color=color)
        for i in range(0, size[0] - 2, x_step):
            for j in range(0, size[1] - 1, y_step):
                new_image.paste(tmp_image, (i, j + 1))
        return new_image

    @classmethod
    def add_border(cls, image: Image.Image):
        border = Image.new('RGBA', image.size, cls.COLORS["border"])
        border.putalpha(255)
        border.paste(image.crop((1, 1, image.size[0] - 1, image.size[1] - 1)),
                     (1, 1))
        return border

    def add_background_grid(self, image, color):
        
        if not color in BackgroundImageEditor.COLORS:
            raise KeyError(f'Color named: "{color}" not support.')
        new_image = self.generate_background(
            size=image.size,
            color=BackgroundImageEditor.COLORS[color])
        new_image = BackgroundImageEditor.add_border(new_image)
        new_image.alpha_composite(image)
        return new_image


class GridPacker:
    """
    Class for packing images into one image
    """

    def __init__(self, width: int, height: int, cell_size: int, p_val: float = 0.5):
        """
        Arguments:
            `width` -- Output image width\n
            `height` -- Output image height\n
            `cell_size` -- The size of one cell in the grid for input images\n
            `p_val` -- Probability to rotate the image.
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size - 1
        self.p_val = p_val
        grid_shape = (height // cell_size, width // cell_size, )
        self.grid = np.zeros(grid_shape, dtype=bool)

    def find_best_position(self, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Finding the first position in the grid for an image with `size`

        Arguments:
            `size` -- Insetring image

        Returns:
            Position coordinates in format (x, y, w, h) or `None` if can't
            insert image
        """
        w, h = size
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.check_position(i, j, w, h):
                    x = j * self.cell_size
                    y = i * self.cell_size
                    return (x, y, w, h)
        # If can't insert image on grid
        return None

    def check_position(self, i: int, j: int, w: int, h: int) -> bool:
        """
        Checking for position `i`, `j` is it possible to insert an image

        Arguments:
            `i` -- Grid row index\n
            `j` -- Grid col index\n
            `w` -- Width of inserting image\n
            `h` -- Height of inserting image

        Returns:
            `True` if can insert image on (`i`, `j`) else `False`
        """
        w, h = (w // self.cell_size, h // self.cell_size)
        for row in range(i, i + h):
            for col in range(j, j + w):
                if (row >= len(self.grid) or
                    col >= len(self.grid[0]) or
                        self.grid[row, col]):
                    return False
        return True

    def update_grid(self, found_position: Tuple[int, int, int, int]) -> None:
        """
        Update grid by found position

        Arguments:
            `found_position` -- Found position in format (x, y, w, h)
        """
        x, y, w, h = found_position
        for row in range(y // self.cell_size, (y + h) // self.cell_size):
            for col in range(x // self.cell_size, (x + w) // self.cell_size):
                self.grid[row][col] = True

    def reset_grid(self):
        """
        Reset `self.grid` to `False` values
        """
        self.grid = np.full_like(self.grid, False)

    def cut_empty_parts(self, image: Image.Image) -> Image.Image:
        """
        Cutting out empty parts from `image` using grid.

        Arguments:
            `image` -- PIL image.

        Returns:
            Croped PIL image.
        """
        rows_id = np.all(self.grid == False, axis=1)
        cols_id = np.all(self.grid == False, axis=0)
        rows_id = np.sum(rows_id)
        cols_id = np.sum(cols_id)
        r = self.grid.shape[0]
        c = self.grid.shape[1]
        row_crop = (r - rows_id) * (self.cell_size) + 1
        col_crop = (c - cols_id) * (self.cell_size) + 1
        return image.crop((0, 0, col_crop, row_crop))

    def rotate_p_val(self, image: Image.Image):
        """
        Rotate image by p_val

        Arguments:
            `image` -- Image to rotate.

        Returns:
            Rotated `image` or not.
        """
        if np.random.uniform() < self.p_val:
            return image.transpose(Image.ROTATE_270)
        else:
            return image

    def pack_images(self,
                    images: List[Image.Image | Path],
                    labels: List[int]) -> Tuple[Image.Image, List, List]:
        """
        Insert `images` on new image 

        Arguments:
            `images` -- PIL images or Pathes\n
            `labels` -- List labels for images 

        Returns:
            New image, boxes list, and labels list
        """
        out_image = Image.new('RGBA', (self.width, self.height))
        boxes = []
        out_labels = []
        # If images has Path Instead of Image.Image open all Images
        if all([isinstance(im, Path) for im in images]):
            images = [Image.open(path) for path in images]
        for image, class_id in zip(images, labels):
            image = self.rotate_p_val(image)
            found_position = self.find_best_position(image.size)
            if found_position:
                x, y, w, h = found_position
                out_image.paste(image, (x, y))
                if class_id:
                    # bbox in format [x1, y1, x2, y2]
                    box = [x, y, x + w - 1, y + h - 1]
                    boxes.append(box)
                    out_labels.append(class_id)
                self.update_grid(found_position)
        out_image = self.cut_empty_parts(out_image)
        self.reset_grid()
        boxes = np.array(boxes, dtype=np.float64)
        out_labels = np.array(out_labels)
        return (out_image, boxes, out_labels)
    

class Manager(pd.DataFrame):
    IMAGES_COLUMN: str = 'PIL_images'
    LABEL_COLUMN: str = 'label'
    VISIBLE_COLUMN: str = 'visible'

    def __init__(self,
                 images_path: str | Path) -> None:
        path = Path(images_path)
        data = ApiService.items(['id',
                                 'shortName',
                                 'backgroundColor',
                                 'baseImageLink'])
        data = pd.json_normalize(data).set_index('id')
        data[Manager.VISIBLE_COLUMN] = True
        super().__init__(data=data)
        self.path = path

    def open_images(self) -> None:
        self[Manager.IMAGES_COLUMN] = self.index.to_series().apply(lambda x: Image.open(self.path / f'{x}.png'))
    
    def to_csv(self) -> None:
        if Manager.IMAGES_COLUMN in self.columns:
            self.drop(columns=Manager.IMAGES_COLUMN).to_csv(self.path / '.csv')
        else:
            self.to_csv(self.path / '.csv')

    @classmethod
    def from_csv(cls, file_path: Path | str, id_column: str, path_column: str) -> 'Manager':
        data = pd.read_csv(file_path)
        return cls(data, id_column, path_column)

    def create_labels(self, start_from: int = 0) -> None:
        """
        Creates label encoding for visible images use `Manager.VISIBLE_COLUMN`
        if not found use all images.

        Keyword Arguments:
            `start_from` -- The separation value to add to the encoded labels.
            If you need start labeling from 1 set sep to 1. (default: {0})
        """
        if Manager.VISIBLE_COLUMN not in self.columns:
            self[Manager.VISIBLE_COLUMN] = True
        encoder = OrdinalEncoder(handle_unknown='use_encoded_value',
                                 unknown_value=-1)
        encoder.fit(self[self[Manager.VISIBLE_COLUMN]].index.to_numpy().reshape(-1, 1))
        labels = encoder.transform(self.index.to_numpy().reshape(-1, 1))
        self[Manager.LABEL_COLUMN] = labels + start_from

    def set_visible_images(self, visible: Dict[int, bool]) -> None:
        """
        Sets the visibility of images.

        Arguments:
            `visible` -- A dictionary mapping class
            names to their visibility status.
        """
        def fc(row): return visible.get(row[self.label_column], False)
        self[Manager.VISIBLE_COLUMN] = self[Manager.VISIBLE_COLUMN].apply(fc, axis=1)

    def get_decode_map(self) -> Dict[int, str]:
        """
        Retrieves a dictionary mapping encoded index
        to their corresponding class names.

        Returns:
            A dictionary mapping encoded index to class names.
        """
        if self.label_column not in self.columns:
            self.create_labels()
        visible = self[self.label_column] == True
        decode_map = {k: v for k, v in zip(self.index.values,
                                           self.loc[visible, self.label_column])}
        return decode_map

    def get_encode_map(self) -> Dict[str, int]:
        """
        Retrieves a dictionary mapping class names
        to their corresponding encoded index.

        Returns:
            A dictionary mapping class names to encoded index.
        """
        if self.label_column not in self.columns:
            self.create_labels()
        visible = self[self.label_column] == True
        encode_map = {v: k for k, v in zip(self.index.values,
                                           self.loc[visible, self.label_column])}
        return encode_map

    def add_image(image_sourse: Path | Image.Image | str, visible: bool = True):
        pass

    def drop_image(id)-> Image.Image:
        pass

    def add_image_from_path():
        pass
    
    def add_image_from_PIL():
        pass

    def download_images(self):
        preprocessing = BackgroundImageEditor(Image.open('/home/yaroslav/Documents/Projects/EFT_items_detection/src/Data/grid_cell.png'))
        for id, _, bg_color, link, _ in tqdm(self.itertuples(index=True),
                                       desc='Downloading images',
                                       position=0,
                                       total=len(self),
                                       leave=True):
            response = requests.get(link)
            if response.status_code == 200:
                image = Image.open(
                    BytesIO(response.content)).convert(mode='RGBA')
                image = preprocessing.add_background_grid(
                    image=image, color=bg_color)
                image_name = f'{id}.png'
                image.save(self.path / image_name)
            # else:
                # self._failed_download_links.append(link)

    @property
    def images(self):
        if Manager.IMAGES_COLUMN not in self.columns:
            self.open_images()
        return self[Manager.IMAGES_COLUMN].values


class Similarity:
    """
    Class for searching similar images using average hashing.
    """

    def __init__(self,
                 hash_size: int = 8,
                 resampling: Resampling = Resampling.LANCZOS,
                 threshold: int = 5) -> None:
        """
        Init resampling params and threshold param.

        Keyword Arguments:
            `hash_size` -- Resample image to (hash_size, hash_size)
             size. (default: {8})\n
            `resampling` -- Resampling method. (default: {Resampling.LANCZOS})\n
            `threshold` -- Diff between the hashes of two images
            in which the images are considered the same. (default: {5})\n
        """
        self.hash_size = hash_size
        self.resampling = resampling
        self.threshold = threshold

    def avg_hash(self, image: Image.Image) -> int:
        """
         Calculate average hash for `image`.

         Arguments:
             `image` -- Image whose hash will be calculated.

         Returns:
             Hash as int number. 
        """
        image = image.convert('L')\
            .resize((self.hash_size, self.hash_size), self.resampling)
        pixels = list(image.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join(["1" if pixel >= avg else "0" for pixel in pixels])
        return int(bits, 2)

    def similarity(self, im2_hash: int, im1_hash: int):
        """
        Count diff bits between two hashes, and check this
        less than`self.threshold`.

        Arguments:
            `im2_hash` -- First image\n
            `im1_hash` -- Second image\n

        Returns:
            True if two hashes are similar otherwise False.
        """
        if bin(im1_hash ^ im2_hash).count('1') <= self.threshold:
            return True
        else:
            return False

    def __call__(self, images: List[Image.Image]) -> List[tuple]:
        """
        Calculate similarity each to each images in `images`,
        excluding pair like (1, 2) and (2, 1) or (1, 1). 

        Arguments:
            `images` -- Images for similarity checking.

        Returns:
            List contains pairs of ids similar images in `images`.
        """
        hashes = []
        similar_subsets = []
        ids = [i for i in range(len(images))]
        for id, image in zip(ids, images):
            hashes.append((id, self.avg_hash(image)))

        for im1_id, hash1 in hashes:
            for im2_id, hash2 in hashes:
                if self.similarity(hash1, hash2):
                    is_merged = False
                    similar = {im1_id, im2_id}
                    for subset in similar_subsets:
                        if not similar.isdisjoint(subset):
                            subset.update(similar)
                            is_merged = True
                    if not is_merged:
                        similar_subsets.append(similar)
        return similar_subsets
    

    # @staticmethod
    # def merge_overlapping_subsets(set_list: List):
    #     subsets = []
    #     for pair in set_list:
    #         pair: set
    #         merged = False

    #         for subset in subsets:
    #             subset: set
    #             if not pair.isdisjoint(subset):
    #                 subset.update(pair)
    #                 merged  = True
    #                 break
    #         if not merged:
    #             subsets.append(pair)
    #     return subsets


class Downloader:
    IMG_LINK_FIELDS: List[str] = ["iconLink", "gridImageLink",
                                  "baseImageLink", "image8xLink",
                                  "inspectImageLink", "image512pxLink"]
    def __init__(self, link_field: str='gridImageLink'):
        """
        Make request to API for images links and save response.

        Arguments:
            `link_fields` -- Fields containing link to images download\n
            `class_field` -- Field containing image class name

        Raises:
            Exception: Bad name field in link_fields see support fields
                in `IMG_LINK_FIELDS`
        """
        if link_field not in Downloader.IMG_LINK_FIELDS:
            raise NameError(f'link_field {link_field} not found.')
        self._image_field = link_field
        fields = ['id', 'backgroundColor', link_field]
        self._images_data = ApiService.items(fields=fields)
        self._images_data = pd.json_normalize(self._images_data)
        self._failed_download_links = []

    def download(self, path: str | Path) -> Manager:
            """
            Dowload images from fields to BaseImages objects.

            Arguments:
                `path` -- Downloading path.

            Returns:
                Manager class with downloaded images.
            """
            path = Path(path)
            path.mkdir(parents=True, exist_ok=True)
            self._images_data['path'] = None
            preprocessing = BackgroundImageEditor(Image.open(
                './src/Data/grid_cell.png'))
            for id, bg_color, link, _ in tqdm(self._images_data.itertuples(index=False),
                                                desc=f'Downloading {self._image_field}',
                                                position=0,
                                                total=len(self._images_data),
                                                leave=True):
                response = requests.get(link)
                if response.status_code == 200:
                    image = Image.open(
                        BytesIO(response.content)).convert(mode='RGBA')
                    image = preprocessing.add_background_grid(
                        image=image, color=bg_color)
                    image_name = f'{id}.png'
                    image.save(path / image_name)
                    self._images_data.loc[self._images_data["id"]
                                            == id, 'path'] = image_name
                else:
                    self._failed_download_links.append(link)
            image_manager = Manager(images_path=path)
            image_manager.to_csv()
            return image_manager
