import pandas as pd
from pathlib import Path
from PIL import Image
from os import makedirs
import numpy as np
from typing import List, Tuple
from Classes import GridPacker

class ImagesGenerator:
    """The class contains functions for
    construct images from other images for training model.
    """
    def __init__(self,
                 df: pd.DataFrame,
                 path: str | Path,
                 images_dirs: List[str],
                 grid_size: int) -> None:
        pass

    def rescale_image_by_grid(self,
                              images_dir: str,
                              overwrite: bool) -> None:
        pass

    def create_mask(self,
                    images_dir: str,
                    overwrite) -> None:
        pass

    def generate(self,
                 size: Tuple[int, int],
                 samples: int,
                 background: List[Image.Image]) -> None:
        pass
    