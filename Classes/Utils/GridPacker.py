from typing import List, Tuple, Dict
from PIL import Image
from numpy import array
from pathlib import Path
import numpy as np


class GridPacker:
    """
    Class for packing images into one image
    """

    def __init__(self, width: int, height: int, cell_size: int):
        """
        Arguments:
            `width` -- Output image width\n
            `height` -- Output image height\n
            `cell_size` -- The size of one cell in the grid for input images\n
        """
        self.width = width
        self.height = height
        self.cell_size = cell_size - 1
        grid_shape = (height // cell_size, width // cell_size, )
        self.grid = np.zeros(grid_shape, dtype=bool)

    def find_best_position(self, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        """
        Finding the first position in the grid for an image with `size`

        Arguments:
            `size` -- Insetring image

        Returns:
            Position coordinates in format (x, y, w, h) or `None` if can't
            insert image
        """
        w, h = size
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.check_position(i, j, w, h):
                    x = j * self.cell_size
                    y = i * self.cell_size
                    return (x, y, w, h)
        # If can't insert image on grid
        return None

    def check_position(self, i: int, j: int, w: int, h: int) -> bool:
        """
        Checking for position `i`, `j` is it possible to insert an image

        Arguments:
            `i` -- Grid row index\n
            `j` -- Grid col index\n
            `w` -- Width of inserting image\n
            `h` -- Height of inserting image

        Returns:
            `True` if can insert image on (`i`, `j`) else `False`
        """
        w, h = (w // self.cell_size, h // self.cell_size)
        for row in range(i, i + h):
            for col in range(j, j + w):
                if (row >= len(self.grid) or
                    col >= len(self.grid[0]) or
                        self.grid[row, col]):
                    return False
        return True

    def update_grid(self, found_position: Tuple[int, int, int, int]) -> None:
        """
        Update grid by found position

        Arguments:
            `found_position` -- Found position in format (x, y, w, h)
        """
        x, y, w, h = found_position
        for row in range(y // self.cell_size, (y + h) // self.cell_size):
            for col in range(x // self.cell_size, (x + w) // self.cell_size):
                self.grid[row][col] = True

    def reset_grid(self):
        """
        Reset `self.grid` to `False` values
        """
        self.grid = np.full_like(self.grid, False)

    def cut_empty_parts(self, image: Image.Image) -> Image.Image:
        """
        Cutting out empty parts from `image` using grid.

        Arguments:
            `image` -- PIL image.

        Returns:
            Croped PIL image.
        """
        rows_id = np.all(self.grid == False, axis=1)
        cols_id = np.all(self.grid == False, axis=0)
        rows_id = np.sum(rows_id)
        cols_id = np.sum(cols_id)
        r = self.grid.shape[0]
        c = self.grid.shape[1]
        row_crop = (r - rows_id) * (self.cell_size) + 1
        col_crop = (c - cols_id) * (self.cell_size) + 1
        return image.crop((0, 0, col_crop, row_crop))

    def pack_images(self,
                    images: List[Image.Image | Path],
                    labels: List[int]) -> Tuple[Image.Image, List, List]:
        """
        Insert `images` on new image 

        Arguments:
            `images` -- PIL images or Pathes\n
            `labels` -- List labels for images 

        Returns:
            New image, boxes list, and labels list
        """
        out_image = Image.new('RGBA', (self.width, self.height))
        boxes = []
        out_labels = []
        # If images has Path Instead of Image.Image open all Images
        if all([isinstance(im, Path) for im in images]):
            images = [Image.open(path) for path in images]
        for image, class_id in zip(images, labels):
            # add aug rotate
            
            found_position = self.find_best_position(image.size)
            if found_position:
                x, y, w, h = found_position
                out_image.paste(image, (x, y))
                # class_id None if class visible set to False 
                if class_id:
                    # bbox in format [x1, y1, x2, y2]
                    box = [x, y, x + w - 1, y + h - 1]
                    boxes.append(box)
                    out_labels.append(class_id)
                self.update_grid(found_position)
        out_image = self.cut_empty_parts(out_image)
        self.reset_grid()
        return (out_image, boxes, out_labels)
