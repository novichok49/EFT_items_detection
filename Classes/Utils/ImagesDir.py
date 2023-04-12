from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image
import json


class ImagesDir:
    def __init__(self, path: str | Path):
        self.dir_path = Path(path)
        if not self.dir_path.exists():
            raise FileNotFoundError(
                f'No such file or directory: {self.dir_path}')
        # Load class info from file
        label_path = self.dir_path / 'ImagesDir.json'
        if label_path.exists():
            with open(label_path, 'r') as file:
                dir_info = json.load(file)
            encode_map = dir_info['label_map']
            decode_map = {val: key for key, val in encode_map.items()}
            last_image_id = dir_info['last_image_id']
            last_class_id = dir_info['last_class_id']
            im_files = [file
                        for file in self.dir_path.iterdir()
                        if (not file.is_dir()) and (file.suffix == '.png')]
        else:
            encode_map = {}
            decode_map = {}
            last_image_id = 0
            last_class_id = 0
            im_files = []

        self._encode_map = encode_map
        self._decode_map = decode_map
        self.last_image_id = last_image_id
        self.last_class_id = last_class_id
        self.im_files = im_files

    def add_image(self, image: Image.Image, class_name: str) -> None:
        if class_name in self._encode_map:
            class_id = self._encode_map[class_name]
        else:
            self._encode_map[class_name] = self.last_class_id
            self._decode_map[self.last_class_id] = class_name
            class_id = self.last_class_id
            self.last_class_id += 1

        save_name = f'{self.last_image_id}_{class_id}.png'
        save_path = self.dir_path / save_name

        image.save(save_path)
        self.im_files.append(save_path)

        self.last_image_id += 1

    def decode_id(self, class_id: int) -> str:
        return self._decode_map[class_id]
    
    def save_info(self):
        self.__del__()

    def __getitem__(self, index) -> Tuple[Image.Image, int]:
        file = self.im_files[index]
        class_id = int(file.stem.split('_')[1])
        image = Image.open(file)
        return image, class_id

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.im_files):
            result = self.__getitem__(self.index)
            self.index += 1
            return result
        else:
            raise StopIteration

    def __del__(self):
        # Save class info into file
        data = {'label_map': self._encode_map,
                'last_class_id': self.last_class_id,
                'last_image_id': self.last_image_id}
        with open(self.dir_path / 'ImagesDir.json', 'w') as file:
            json.dump(data, file)

    @property
    def encode_map(self):
        return self._encode_map