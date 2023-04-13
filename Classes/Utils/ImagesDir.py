from pathlib import Path
from typing import Tuple, List, Dict
from PIL import Image
import json


class ImagesDir:
    def __init__(self, path: str | Path):
        self._dir_path = Path(path)
        if not self._dir_path.exists():
            raise FileNotFoundError(
                f'No such file or directory: {self._dir_path}')
        # Load class info from file if exist
        class_info = ImagesDir.__load_state(self._dir_path)
        encode_map, decode_map, last_image_id, last_class_id, im_files = class_info
        self._encode_map = encode_map
        self._decode_map = decode_map
        self._last_image_id = last_image_id
        self._last_class_id = last_class_id
        self._im_files = im_files

    def add_image(self, image: Image.Image, class_name: str) -> None:
        if class_name in self._encode_map:
            class_id = self._encode_map[class_name]
        else:
            self._encode_map[class_name] = self._last_class_id
            self._decode_map[self._last_class_id] = class_name
            class_id = self._last_class_id
            self._last_class_id += 1

        save_name = f'{self._last_image_id}_{class_id}.png'

        image.save(self._dir_path / save_name)
        self._im_files.append(Path(save_name))

        self._last_image_id += 1

    def decode_id(self, class_id: int) -> str:
        return self._decode_map[class_id]

    def save_state(self):
        self.__del__()

    def __getitem__(self, image_id: slice | int) -> Tuple[Image.Image, int]:
        if isinstance(image_id, slice):
            files = self._im_files[image_id.start:image_id.stop:image_id.step]
            items_list = []
            for file in files:
                class_id = int(file.stem.split('_')[1])
                imagepath = self._dir_path / file
                items_list.append((imagepath, class_id))
            return items_list
        elif isinstance(image_id, int):
            file = self._im_files[image_id]
            class_id = int(file.stem.split('_')[1])
            imagepath = self._dir_path / file
            return imagepath, class_id
        else:
            raise TypeError(f'Unsupported index type: {type(image_id)}.')

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self._im_files):
            result = self.__getitem__(self.index)
            self.index += 1
            return result
        else:
            raise StopIteration

    def __del__(self):
        # Save class info into file
        data = {'label_map': self._encode_map,
                'last_class_id': self._last_class_id,
                'last_image_id': self._last_image_id}
        with open(self._dir_path / 'ImagesDir.json', 'w') as file:
            json.dump(data, file)

    @property
    def encode_map(self):
        return self._encode_map
    
    @property
    def decode_map(self):
        return self._decode_map

    @classmethod
    def __path_sort(cls, x):
        try:
            res = int(x.stem.split('_')[0])
        except ValueError:
            # if name is not in return max int64
            res = 9223372036854775807
        return res

    @classmethod
    def __load_state(cls, path: Path):
        label_path = path / 'ImagesDir.json'
        if label_path.exists():
            with open(label_path, 'r') as file:
                dir_info = json.load(file)
            encode_map = dir_info['label_map']
            decode_map = {val: key for key, val in encode_map.items()}
            last_image_id = dir_info['last_image_id']
            last_class_id = dir_info['last_class_id']
            im_files = [Path(file.name)
                        for file in sorted(path.iterdir(), key=cls.__path_sort)
                        if (not file.is_dir()) and (file.suffix == '.png')]
        else:
            encode_map = {}
            decode_map = {}
            last_image_id = 0
            last_class_id = 0
            im_files = []
        return encode_map, decode_map, last_image_id, last_class_id, im_files
