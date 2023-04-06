from typing import List, Tuple, Dict
from PIL import Image
from numpy import array 

class GridPacker:
    def __init__(self, width: int, height: int, cell_size: int):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = [[None for _ in range(width // cell_size)] for _ in range(height // cell_size)]

    def find_best_position(self, size: Tuple[int, int]) -> Tuple[int, int, int, int]:
        w, h = size
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.check_position(i, j, w, h):
                    return (j * self.cell_size, i * self.cell_size, w, h)
        return None

    def check_position(self, i: int, j: int, w: int, h: int) -> bool:
        for row in range(i, i + h // self.cell_size):
            for col in range(j, j + w // self.cell_size):
                if row >= len(self.grid) or col >= len(self.grid[0]) or self.grid[row][col] is not None:
                    return False
        return True

    def update_grid(self, x: int, y: int, w: int, h: int) -> None:
        for row in range(y // self.cell_size, (y + h) // self.cell_size):
            for col in range(x // self.cell_size, (x + w) // self.cell_size):
                self.grid[row][col] = (x, y, w, h)

    def pack(self, images: List[Tuple]) -> Tuple[Image.Image, Dict]:
        out_image = Image.new('RGBA', (self.width, self.height))
        bboxs = {}
        for image, class_name in images:
            position = self.find_best_position(image.size)
            if position is not None:
                x, y, w, h = position
                out_image.paste(image, (x, y))
                bbox = [x, y, image.size[0] - 1, image.size[1] - 1]
                if class_name in bboxs:
                    bboxs[class_name].append(bbox)
                else:
                    bboxs[class_name] = [bbox]
                self.update_grid(x, y, w, h)
        for bbox in bboxs:
            bboxs[bbox] = array(bboxs[bbox])
        return (out_image, bboxs)