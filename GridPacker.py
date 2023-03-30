from PIL import Image

class GridPacker:
    def __init__(self, width, height, cell_size=64):
        self.width = width
        self.height = height
        self.cell_size = cell_size
        self.grid = [[None for _ in range(width // cell_size)] for _ in range(height // cell_size)]

    def find_best_position(self, size):
        w, h = size
        for i in range(len(self.grid)):
            for j in range(len(self.grid[0])):
                if self.check_position(i, j, w, h):
                    return (j*self.cell_size, i*self.cell_size, w, h)
        return None

    def check_position(self, i, j, w, h):
        for row in range(i, i + h // self.cell_size):
            for col in range(j, j + w // self.cell_size):
                if row >= len(self.grid) or col >= len(self.grid[0]) or self.grid[row][col] is not None:
                    # Position is outside the grid or overlaps with another image
                    return False
        return True

    def update_grid(self, x, y, w, h):
        for row in range(y // self.cell_size, (y + h) // self.cell_size):
            for col in range(x // self.cell_size, (x + w) // self.cell_size):
                self.grid[row][col] = (x, y, w, h)

    def pack(self, images):
        output = Image.new('RGBA', (self.width, self.height))
        for image in images:
            position = self.find_best_position(image.size)
            if position is not None:
                x, y, w, h = position
                output.paste(image, (x, y))
                self.update_grid(x, y, w, h)
        return output