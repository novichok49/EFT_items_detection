from pathlib import Path
from PIL import Image
import numpy as np
from typing import List, Tuple
from Classes.Utils import GridPacker, BaseImages
from tqdm import tqdm
from multiprocessing import Pool
import yaml

class ImagesGenerator:
    """
    Generate dataset from backgrounds and base images.
    """

    def __init__(
            self,
            base_images_path: str | Path,
            backgrounds_path: str | Path,
            grid_size: int = 64,
            seed: int = 0) -> None:
        """
        Config generation params.

        Arguments:
            `base_images_path` -- Path to base images. It must be created
            by `BaseImages` class.\n
            `backgrounds_path` -- Path to backround images.

        Keyword Arguments:
            grid_size -- One cell cize. (default: {64})
            seed -- Random seed. (default: {0})

        Raises:
            FileNotFoundError: If base images path or background path not found.
        """
        self.base_images_path = Path(base_images_path)
        self.backgrounds_path = Path(backgrounds_path)
        if not self.base_images_path.exists():
            raise FileNotFoundError(f'No such base images directory:\
                                    {self.base_images_path}')
        elif not self.backgrounds_path.exists():
            raise FileNotFoundError(f'No such backgrounds directory:\
                                    {self.backgrounds_path}')
        np.random.seed(seed)
        self.grid_size = grid_size
        self.image_dir = BaseImages(self.base_images_path)

    def generate_dataset(
            self,
            images_size: Tuple[int, int],
            classes_on_image: int,
            count_base_images: int,
            dataset_path: str | Path,
            n_jobs: int) -> None:
        """
        Generate dataset from `BaseImages` and save them.

        Arguments:
            `images_size` -- Output_image size.\n
            `classes_on_image` -- Num samples in one image.\n
            `count_base_images` -- Number of examples
            of each classes in the dataset.\n
            dataset_path -- Path to save dataset.\n
            n_jobs -- Count subprocess for generation.
        """
        self.images_size = images_size
        dataset_path = Path(dataset_path)
        self.im_path = dataset_path.joinpath('images')
        self.lab_path = dataset_path.joinpath('labels')
        dataset_path.mkdir(exist_ok=True)
        self.im_path.mkdir(exist_ok=True)
        self.lab_path.mkdir(exist_ok=True)

        self.backgrounds = [image
                            for image in self.backgrounds_path.iterdir()]
        self.image_dir.create_labels(sep=1)
        self.labels_map = self.image_dir.get_decode_map()
        self.labels_map[0] = "__background__"

        kerneldt = np.dtype([('PILImage', object), ('Label', object)])
        self.base_images = np.array(self.image_dir[:], dtype=kerneldt)

        sub_samples_ids = ImagesGenerator.get_rand_subsamples(self.base_images,
                                                              count_base_images,
                                                              classes_on_image)
        with Pool(n_jobs) as pool:
            with tqdm(total=len(sub_samples_ids), desc='Generate') as bar:
                for _ in pool.imap(self.generate, enumerate(sub_samples_ids)):
                    bar.update()
        # Save class labels to YAML file
        with open(dataset_path / 'classes.yaml', 'w') as file:
            yaml.dump(self.labels_map, file)

    def generate(self, item: Tuple[int, np.ndarray]) -> None:
        """
        Generate one image use sample indexs and save it to file.

        Arguments:
            item -- Tuple contains image id, indexes base images
        """
        image_id, image_ids = item
        sample = self.base_images[image_ids]
        images = [image for image, _ in sample]
        labels = [label for _, label in sample]
        grid_packer = GridPacker(width=512,
                                 height=self.images_size[1] - 1,
                                 cell_size=self.grid_size)
        grid_image, boxes, labels = grid_packer.pack_images(images=images,
                                                            labels=labels)
        bg_image = Image.open(np.random.choice(self.backgrounds, 1)[0])
        gen_image, boxes = ImagesGenerator.plot_grid_on_bg(
            grid_image=grid_image,
            boxes=boxes,
            background_image=bg_image
        )
        gen_image = gen_image.resize(size=self.images_size)
        image_name = f'{image_id}.png'.zfill(10)
        gen_image.save(self.im_path / image_name)
        lab_name = f'{image_id}.txt'.zfill(10)
        with open(self.lab_path / lab_name, 'w') as file:
            for i in range(len(labels)):
                file.write(f'{labels[i]} ' +
                           ' '.join(map(str, boxes[i])) + '\n')

    @staticmethod
    def get_rand_subsamples(images: np.ndarray,
                            count_base_images: int,
                            classes_on_image: int,) -> List[np.ndarray]:
        """
        Create list of `np.ndarray` with indexes of subsamples

        Arguments:
            images -- images for subsumpling.\n
            count_base_images -- See generate_dataset args.\n
            classes_on_image -- See generate_dataset args.

        Returns:
            List `np.ndarray` with indexes of subsamples
        """
        ids = np.arange(len(images))
        slice_range = int(np.ceil(len(ids) / classes_on_image))
        sub_samples_ids = []
        for _ in range(count_base_images):
            np.random.shuffle(ids)
            for i in range(slice_range):
                start = i * classes_on_image
                stop = i * classes_on_image + classes_on_image
                sub_samples_ids.append(ids[start:stop])
        return sub_samples_ids

    @staticmethod
    def plot_grid_on_bg(
            grid_image: Image.Image,
            boxes: np.ndarray,
            background_image: Image.Image) -> Tuple[Image.Image, np.ndarray]:
        """
        Paste generated image on background image.

        Arguments:
            grid_image -- Generated PIL image.\n
            boxes -- array bound boxes for classes.\n
            background_image -- Using background.

        Returns:
            Tuple with new image and normalized bound boxes.
        """
        paste_x = np.random.randint(
            0, background_image.size[0] - grid_image.size[0])
        paste_y = np.random.randint(
            0, background_image.size[1] - grid_image.size[1])
        background_image.paste(im=grid_image,
                               box=(paste_x, paste_y),
                               mask=grid_image.getchannel('A'))
        # Changing bound boxes coordinates and normalizing them.
        boxes[:, 0] = (paste_x + boxes[:, 0]) / background_image.size[0]
        boxes[:, 1] = (paste_y + boxes[:, 1]) / background_image.size[1]
        boxes[:, 2] = (paste_x + boxes[:, 2]) / background_image.size[0]
        boxes[:, 3] = (paste_y + boxes[:, 3]) / background_image.size[1]
        return background_image, boxes
