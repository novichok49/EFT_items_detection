from PIL import Image
from PIL.Image import Resampling
from pathlib import Path
from typing import List


class ImageSimilarity():
    """
    Class for searching similar images using average hashing.
    """

    def __init__(self,
                 hash_size: int = 8,
                 resampling: Resampling = Resampling.LANCZOS,
                 threshold: int = 5) -> None:
        """
        Init resampling params and threshold param.

        Keyword Arguments:
            `hash_size` -- Resample image to (hash_size, hash_size)
             size. (default: {8})\n
            `resampling` -- Resampling method. (default: {Resampling.LANCZOS})\n
            `threshold` -- Diff between the hashes of two images
            in which the images are considered the same. (default: {5})\n
        """
        self.hash_size = hash_size
        self.resampling = resampling
        self.threshold = threshold

    def avg_hash(self, image: Image.Image) -> int:
        """
         Calculate average hash for `image`.

         Arguments:
             `image` -- Image whose hash will be calculated.

         Returns:
             Hash as int number. 
        """
        image = image.convert('L')\
            .resize((self.hash_size, self.hash_size), self.resampling)
        pixels = list(image.getdata())
        avg = sum(pixels) / len(pixels)
        bits = "".join(["1" if pixel >= avg else "0" for pixel in pixels])
        return int(bits, 2)

    def similarity(self, im2_hash: int, im1_hash: int):
        """
        Count diff bits between two hashes, and check this
        less than`self.threshold`.

        Arguments:
            `im2_hash` -- First image\n
            `im1_hash` -- Second image\n

        Returns:
            True if two hashes are similar otherwise False.
        """
        if bin(im1_hash ^ im2_hash).count('1') <= self.threshold:
            return True
        else:
            return False

    def similar_pairs(self, images: List[Image.Image]) -> List[tuple]:
        """
        Calculate similarity each to each images in `images`,
        excluding pair like (1, 2) and (2, 1) or (1, 1). 

        Arguments:
            `images` -- Images for similarity checking.

        Returns:
            List contains pairs of ids similar images in `images`.
        """
        hashes = []
        ids = [i for i in range(len(images))]
        for id, image in zip(ids, images):
            hashes.append((id, self.avg_hash(image)))

        pairs = []
        for im1_id, hash1 in hashes:
            for im2_id, hash2 in hashes:
                if self.similarity(hash1, hash2) and (im1_id != im2_id):
                    pair = frozenset(((im1_id, im2_id)))
                    pairs.append(pair)
        similares = list(set(pairs))
        similares = [tuple(pair) for pair in similares]
        return similares
