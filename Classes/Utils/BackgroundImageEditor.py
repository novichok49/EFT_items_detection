from PIL import Image
from typing import Tuple


class BackgroundImageEditor:
    colors = {
        'black': (0, 0, 0, 77),
        'blue': (28, 65, 86, 77),
        'default': (127, 127, 127, 77),
        'green': (21, 45, 0, 77),
        'grey': (29, 29, 29, 77),
        'orange': (60, 25, 0, 77),
        'red': (109, 36, 24, 77),
        'violet': (76, 42, 85, 77),
        'yellow': (104, 102, 40, 77),
        'border': (73, 81, 84, 77),
    }

    def __init__(self, cell_grid: Image.Image) -> None:
        cell_grid.putalpha(255)
        self.cell_grid = cell_grid

    def generate_background(self,
                            size: Tuple[int, int],
                            color: Tuple[int, int, int, int]) -> Image.Image:
        x_step, y_step = self.cell_grid.size
        tmp_image = Image.new('RGBA', self.cell_grid.size, color)
        tmp_image = Image.alpha_composite(self.cell_grid, tmp_image)
        new_image = Image.new('RGBA', size=size, color=color)
        for i in range(0, size[0] - 2, x_step):
            for j in range(0, size[1] - 1, y_step):
                new_image.paste(tmp_image, (i, j + 1))
        return new_image

    @classmethod
    def add_border(cls, image: Image.Image):
        border = Image.new('RGBA', image.size, cls.colors["border"])
        border.putalpha(255)
        border.paste(image.crop((1, 1, image.size[0] - 1, image.size[1] - 1)),
                     (1, 1))
        return border

    def add_background_grid(self, images, colors):
        for image, color in zip(images, colors):
            if not color in BackgroundImageEditor.colors:
                raise KeyError(f'Color named: "{color}" not support.')

            grid_image = Image.open('./grid_cell.png')
            new_image = self.generate_background(
                size=image.size,
                color=BackgroundImageEditor.colors[color])
            new_image = BackgroundImageEditor.add_border(new_image)
            new_image.alpha_composite(image)
            yield new_image
        # return new_image
