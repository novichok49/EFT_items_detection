from torch.utils.data import Dataset
from typing import Tuple
import os
from PIL import Image

# TODO Add doc


class ImagesDataset(Dataset):

    def __init__(self, path: str) -> None:
        self.path = path
        self.image_filenames = os.listdir(self.path)
        self.label_map = {}
        self.class_index = 0

    def __getitem__(self, index) -> Tuple[Image.Image, str]:
        filename = self.image_filenames[index]
        image_path = os.path.join(self.path, filename)
        image = Image.open(image_path)
        class_name = filename.split('_')[0]
        class_code = self.encode(class_name)
        return image, class_code

    def __len__(self) -> int:
        return len(self.image_filenames)

    def encode(self, class_name):
        if class_name in self.label_map:
            code = self.label_map[class_name]
        else:
            code = self.class_index
            self.label_map[class_name] = self.class_index
            self.class_index += 1
        return code
