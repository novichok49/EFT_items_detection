from PIL import Image
from pathlib import Path
from io import BytesIO
import requests
from Classes.Utils import APIRequester, BaseImages
from typing import List, Dict
from tqdm import tqdm


class ImagesDownloader:
    """
    Class for download images from tarkov.dev API.
    """
    IMG_LINK_FIELDS: List[str] = ["iconLink", "gridImageLink",
                                  "baseImageLink", "image8xLink",
                                  "inspectImageLink", "image512pxLink"]

    def __init__(self, link_field: str, class_field: str = 'normalizedName'):
        """
        Make request to API for images links and save response.

        Arguments:
            `link_fields` -- Fields containing link to images download\n
            `class_field` -- Field containing image class name

        Raises:
            Exception: Bad name field in link_fields see support fields
                in `IMG_LINK_FIELDS`
        """
        if not (link_field in ImagesDownloader.IMG_LINK_FIELDS):
            raise Exception('Bad field')
        self._image_field = link_field
        self._class_field = class_field
        fields = [class_field]
        fields.append(self._image_field)
        self._images_data = APIRequester.post(name='items', fields=fields)
        self._failed_download_links = []

    def download(self, path: str | Path) -> Dict[str, BaseImages]:
        """
        Dowload images from fields to BaseImages objects.

        Arguments:
            `path` -- Downloading path.

        Returns:
            Dict BaseImages objects with downloaded images.
        """
        base_path = Path(path)
        base_path.mkdir(parents=True, exist_ok=True)
        self.images_dir = BaseImages(base_path)
        for item in tqdm(self._images_data,
                            desc=f'Downloading {self._image_field}',
                            position=0,
                            leave=True):
            link = item[self._image_field]
            response = requests.get(link)
            if response.status_code == 200:
                item_class = item[self._class_field]
                image = Image.open(
                    BytesIO(response.content)).convert(mode='RGBA')
                self.images_dir.add_image(image, item_class)
            else:
                self._failed_download_links.append(link)
        self.images_dir.save_state()
        return self.images_dir

    @property
    def failed_download_links(self):
        """
        Dict links that failed to load the image.
        """
        return self._failed_download_links
