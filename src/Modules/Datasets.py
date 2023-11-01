from Images import GridPacker, Manager
from typing import List, Tuple
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from multiprocessing import Pool
from torch.utils.data import Dataset
from torchvision import transforms as T
import torch
import yaml


class Generator:
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
        self.images_manager = Manager.from_csv(self.base_images_path,
                                               'Id',
                                               'Path')

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
        self.images_manager.create_labels(sep=1)
        self.labels_map = self.images_manager.get_decode_map()
        self.labels_map[0] = "__background__"

        kerneldt = np.dtype([('PILImage', object), ('Label', object)])
        self.base_images = np.array(self.images_manager[:], dtype=kerneldt)

        sub_samples_ids = Generator.get_rand_subsamples(self.base_images,
                                                              count_base_images,
                                                              classes_on_image)
        with Pool(n_jobs) as pool:
            with tqdm(total=len(sub_samples_ids), desc='Generate') as bar:
                #BUG Multiprocessing random (different process have same random)
                for _ in pool.imap(self.generate, enumerate(sub_samples_ids)):
                    bar.update()
        # Save class labels to YAML file
        dataset_info = {'nc': len(self.labels_map),
                        'names': [self.labels_map[s] for s in sorted(self.labels_map)]}
        with open(dataset_path / 'classes.yaml', 'w') as file:
            yaml.dump(dataset_info, file)

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
        gen_image, boxes = Generator.plot_grid_on_bg(
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
        #FIXME 2 param in paste_x, paste_y + 1 check 
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
    

class TarkovItemsDataset(Dataset):
    def __init__(self,
                 dataset_path: str | Path,
                #  labels_map: Dict = None,
                 transforms=None) -> None:
        self.dataset_path = Path(dataset_path)
        self.transforms = transforms

    def __getitem__(self, index):
        target = {}
        image_name = f'{index}.png'.zfill(10)
        labels_name = f'{index}.txt'.zfill(10)
        labels, boxes = self.read_bound_boxes(self.dataset_path / 'labels' / labels_name)
        image = Image.open(self.dataset_path / 'images' / image_name).convert('RGB')
        image = T.ToTensor()(image)
        if self.transforms:
            transform = T.Compose(self.transforms)
            image = transform(image)
        w, h = image.shape[2], image.shape[1]
        labels = torch.tensor(labels, dtype=torch.int64)
        boxes = torch.tensor(boxes, dtype=torch.float32)
        boxes[::, 0::2] *= w
        boxes[::, 1::2] *= h
        iscrowd = torch.zeros((boxes.shape[0]), dtype=torch.int8)
        image_id = torch.tensor([index], dtype=torch.int64)
        target['boxes'] = boxes
        target['labels'] = labels
        target['area'] = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['iscrowd'] = iscrowd
        target['image_id'] = image_id
        return image, target

    def __len__(self):
        return len(list((self.dataset_path / 'images').glob('*')))

    def read_bound_boxes(self, labels_path: Path) -> Tuple[list, list]:
        labels = []
        boxes = []
        with open(labels_path, 'r') as file:
            for line in file:
                one_line = line.strip().split(sep=' ')
                labels.append(int(one_line[0]))
                boxes.append([float(x) for x in one_line[1:]])
        return labels, boxes