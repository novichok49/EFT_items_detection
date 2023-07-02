from pathlib import Path
from typing import Dict, List, Iterable, Tuple
import torch
from torch.utils.data import Dataset
from torchvision import transforms as T
from PIL import Image
import json
import yaml

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

