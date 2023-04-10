from pathlib import Path
from typing import List
import os
from .ImageDir import ImageDir

class ImageDirs:
    def __init__(
            self,
            path: str | Path,
            image_dirs: List[str]) -> None:
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f'Dir {path} not found.')
        self.path = path
        self.dirs = {}
        for dir in image_dirs:
            self.dirs[dir]=ImageDir(self.path / dir)
        
    def __getitem__(self, dir_name) -> ImageDir:
        if dir_name in self.dirs:
            return self.dirs[dir_name]
        else:
            raise KeyError(f'{dir_name} key not found.')
    
    def __iter__(self):
        self.index = 0
        self.dir_names = list(self.dirs.keys())
        return self
    
    def __next__(self):
        if self.index < len(self.dir_names):
            result = self.__getitem__(self.dir_names[self.index])
            self.index += 1
            return result
        else:
            raise StopIteration
    
