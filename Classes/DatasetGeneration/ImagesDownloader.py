from PIL import Image
from pathlib import Path
from io import BytesIO
import requests
from Classes.Utils import APIRequester, ImagesDir
from typing import List, Dict
from tqdm import tqdm


class ImagesDownloader:
    """
    Class for download images from tarkov.dev API.
    """
    IMG_LINK_FIELDS: List[str] = ["iconLink", "gridImageLink",
                                  "baseImageLink", "image8xLink",
                                  "inspectImageLink", "image512pxLink"]

    def __init__(self, link_fields: List[str], class_field: str = 'normalizedName'):
        """
        Make request to API for images links and save response.

        Arguments:
            `link_fields` -- Fields containing link to images download\n
            `class_field` -- Field containing image class name

        Raises:
            Exception: Bad name field in link_fields see support fields
                in `IMG_LINK_FIELDS`
        """
        if not all(field in ImagesDownloader.IMG_LINK_FIELDS for field in link_fields):
            raise Exception("Bad field in link_fields.")
        self._image_fields = link_fields
        self._class_field = class_field
        fields = [class_field]
        fields.extend(self._image_fields)
        self._images_data = APIRequester.post(name='items', fields=fields)
        self._failed_download_links = {key: [] for key in self._image_fields}

    def download(self, path: str | Path) -> Dict[str, ImagesDir]:
        """
        Dowload images from fields to ImagesDir objects.

        Arguments:
            `path` -- Downloading path.

        Returns:
            Dict ImagesDir objects with downloaded images.
        """
        im_dirs = {}
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)

        for field in self._image_fields:
            field_path = base_path / field
            field_path.mkdir(exist_ok=True)
            im_dirs[field] = ImagesDir(field_path)
            for item in tqdm(self._images_data,
                             desc=f'Downloading {field}',
                             position=0,
                             leave=True):
                link = item[field]
                response = requests.get(link)
                if response.status_code == 200:
                    item_class = item[self._class_field]
                    image = Image.open(
                        BytesIO(response.content)).convert(mode='RGBA')
                    im_dirs[field].add_image(image, item_class)
                else:
                    self._failed_download_links[field].append(link)
        im_dirs[field].save_state()
        return im_dirs

    @property
    def failed_download_links(self):
        """
        Dict links that failed to load the image.
        """
        return self._failed_download_links
