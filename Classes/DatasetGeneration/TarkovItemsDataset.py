from pathlib import Path
from typing import Dict, List, Iterable
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import json


class TarkovItemsDataset(Dataset):
    def __init__(self,
                 images_path: str | Path,
                 labels_map: Dict = None,
                 transforms=None) -> None:
        self.images_path = Path(images_path)
        data = self.__try_load(self.images_path)
        if labels_map != None:
            self.labels_map = labels_map
        else:
            self.labels_map = data['labels_map']

        self.images_map = data['images_map']
        self.images = data['images']

    def __getitem__(self, index):
        target = {}
        file_name = self.images_map[index]
        image_path = self.images_path / file_name
        image = Image.open(image_path).convert('RGB')
        image_data = self.images[index]
        transform = T.ToTensor()
        image = transform(image)
        labels = torch.tensor(image_data['labels'], dtype=torch.int64)
        bboxes = torch.tensor(image_data['bboxes'], dtype=torch.float16)
        target['bboxes'] = bboxes
        target['labels'] = labels
        return image, target

    def __len__(self):
        return len(self.images)

    def __del__(self):
        self.save()

    def add_image(self,
                  image_name: str,
                  bboxes: List,
                  labels: List) -> None:
        image_data = {'bboxes': bboxes,
                      'labels': labels}
        id = len(self.images)
        self.images_map[id] = image_name
        self.images[id] = image_data

    def decode_labels(self, ids: Iterable):
        labels = [self.labels_map[int(id)]
                  for id in ids]
        return labels

    def save(self) -> None:
        data = {'images': self.images,
                'images_map': self.images_map,
                'labels_map': self.labels_map}
        file_path = self.images_path / '.json'
        with open(file_path, 'w') as file:
            json.dump(data, file)

    def get_image(self, index):
        target = {}
        file_name = self.images_map[index]
        image_path = self.images_path / file_name
        image = Image.open(image_path).convert('RGB')
        image_data = self.images[index]
        transform = T.ToTensor()
        # image = transform(image)
        # labels = torch.tensor(image_data['labels'], dtype=torch.int64)
        # bboxes = torch.tensor(image_data['bboxes'], dtype=torch.float16)
        target['bboxes'] = image_data['bboxes']
        target['labels'] = image_data['labels']
        return image, target

    def __try_load(self, path: Path) -> Dict[str, Dict]:
        def parse_int_keys(key):
            try:
                res = int(key)
            except:
                res = key
            return res

        def hook(dct):
            return {parse_int_keys(k): v for k, v in dct.items()}

        file_path = path / '.json'
        if file_path.exists():
            with open(file_path, 'r') as file:
                data = json.load(file, object_hook=hook)
            return {'images': data['images'],
                    'images_map': data['images_map'],
                    'labels_map': data['labels_map']}
        else:
            return {'images': {},
                    'images_map': {},
                    'labels_map': {}}
