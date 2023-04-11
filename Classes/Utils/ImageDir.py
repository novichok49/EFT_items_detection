from pathlib import Path
import os
from typing import Tuple
from PIL import Image

class ImageDir:
    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Dir {path} not found')
        self.path = path
        self.image_filenames = os.listdir(self.path)

    def __getitem__(self, index) -> Tuple:
        filename = self.image_filenames[index]
        image_path = self.path / filename
        return image_path
    
    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        if self.index < len(self.image_filenames):
            result = self.__getitem__(self.index)
            self.index += 1
            return result
        else:
            raise StopIteration
    
    def update_filepathes(self):
        self.image_filenames = os.listdir(self.path)